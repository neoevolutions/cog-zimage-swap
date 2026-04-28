#!/usr/bin/env python3
"""vast.ai provisioning + tunnel + cost guardrails for cog-zimage.

Subcommands:
  setup           Register your local SSH public key with vast.ai (one-time).
  find            Dry-run: list top-N matching offers per deploy.yaml filters. Free.
  run             Provision an instance, open SSH tunnel on local_port, arm guardrails.
  destroy <id>    Kill an instance.
  status          Show your active instances.

Reads .env from the project root for VAST_API_KEY. Falls back to ~/.vast_api_key
or the VAST_API_KEY env var. Never logs the key.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEPLOY_DIR = Path(__file__).resolve().parent
DEPLOY_YAML = DEPLOY_DIR / "deploy.yaml"
ENV_FILE = PROJECT_ROOT / ".env"
TUNNEL_PID_FILE = DEPLOY_DIR / ".tunnel.pid"


# -------------------- env / config --------------------

def load_env() -> None:
    """Populate os.environ from project .env without overwriting existing values."""
    if not ENV_FILE.exists():
        return
    for raw in ENV_FILE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def get_api_key() -> str:
    key = os.environ.get("VAST_API_KEY")
    if key:
        return key
    legacy = Path.home() / ".vast_api_key"
    if legacy.exists():
        return legacy.read_text().strip()
    sys.exit("VAST_API_KEY missing. Add it to .env or run `vastai set api-key <key>`.")


def load_config() -> dict:
    if not DEPLOY_YAML.exists():
        sys.exit(f"missing {DEPLOY_YAML}")
    return yaml.safe_load(DEPLOY_YAML.read_text())


def expand_path(p: str) -> str:
    return str(Path(os.path.expanduser(p)).resolve())


# -------------------- vastai CLI shim --------------------

def vastai(*args: str, json_output: bool = True, check: bool = True) -> object:
    """Call the vastai CLI. Returns parsed JSON when json_output=True, else raw stdout str."""
    cmd = ["vastai", *args]
    if json_output:
        cmd.append("--raw")
    env = {**os.environ, "VAST_API_KEY": get_api_key()}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    if check and result.returncode != 0:
        sys.exit(f"vastai {' '.join(args)} failed (exit {result.returncode}):\n{result.stderr}")
    if json_output:
        try:
            return json.loads(result.stdout) if result.stdout.strip() else None
        except json.JSONDecodeError:
            sys.exit(f"vastai {' '.join(args)} returned non-JSON output:\n{result.stdout[:500]}")
    return result.stdout


# -------------------- offer search --------------------

def build_offer_query(cfg: dict) -> str:
    parts: list[str] = ["rentable=true"]
    if cfg.get("gpu_name"):
        parts.append(f"gpu_name in [{','.join(cfg['gpu_name'])}]")
    if cfg.get("min_vram_gb"):
        # vast.ai CLI inconsistency: search filter takes GB, response field returns MB.
        parts.append(f"gpu_ram>={cfg['min_vram_gb']}")
    if cfg.get("min_inet_down_mbps"):
        parts.append(f"inet_down>={cfg['min_inet_down_mbps']}")
    if cfg.get("min_reliability"):
        parts.append(f"reliability2>={cfg['min_reliability']}")
    if cfg.get("max_dph"):
        parts.append(f"dph_total<={cfg['max_dph']}")
    if cfg.get("cuda_min"):
        parts.append(f"cuda_max_good>={cfg['cuda_min']}")
    if cfg.get("disk_gb"):
        parts.append(f"disk_space>={cfg['disk_gb']}")
    if cfg.get("geolocation"):
        parts.append(f"geolocation in [{','.join(cfg['geolocation'])}]")
    if "verified" in cfg:
        parts.append(f"verified={'true' if cfg['verified'] else 'false'}")
    if cfg.get("min_direct_port_count"):
        # Excludes hosts that report mapped ports in metadata but can't
        # actually route inbound traffic to them (NAT-restricted home setups).
        parts.append(f"direct_port_count>={cfg['min_direct_port_count']}")
    # interruptible is enforced at create-time (on-demand = omit --bid-price),
    # not at search-time. Don't add it to the query — there's no equivalent filter.
    return " ".join(parts)


def search_offers(cfg: dict, limit: int = 5) -> list[dict]:
    query = build_offer_query(cfg)
    offers = vastai("search", "offers", query, "-o", "dph_total")
    if not isinstance(offers, list):
        return []
    return offers[:limit]


def fmt_offer(o: dict) -> str:
    return (
        f"id={o.get('id'):>10}  "
        f"{o.get('gpu_name', '?'):<16}  "
        f"vram={o.get('gpu_ram', 0) / 1024:>5.1f}GB  "
        f"${o.get('dph_total', 0):.3f}/hr  "
        f"rel={o.get('reliability2', 0) * 100:>5.1f}%  "
        f"down={o.get('inet_down', 0):>4.0f}Mbps  "
        f"{o.get('geolocation', '?')}"
    )


# -------------------- provisioning --------------------

def _onstart_cmd() -> str:
    """Boot cog's HTTP server inside the container.

    vast.ai never runs the docker image's ENTRYPOINT/CMD — it only runs its
    own SSH bootstrap shim and expects the user to invoke their workload via
    --onstart-cmd. Without this, the container is a dead box: no cog, no
    ComfyUI (cog's setup() is what spawns ComfyUI as a subprocess).

    The command:
      - cd /src so cog finds predict.py and workflows/headswap.json
      - setsid -f detaches into a new session so vast.ai's bootstrap returns
      - redirects to /var/log/cog.log so failures are diagnosable via SSH
      - python comes from pyenv shims set up at image build time
    """
    return (
        "setsid -f bash -c "
        "'cd /src; exec python -m cog.server.http "
        "--host 0.0.0.0 --port 5000 "
        "> /var/log/cog.log 2>&1 < /dev/null'"
    )


def create_instance(offer_id: int, image: str, disk_gb: int) -> int:
    print(f">>> creating instance from offer {offer_id} with image {image}")
    out = vastai(
        "create", "instance", str(offer_id),
        "--image", image,
        "--disk", str(disk_gb),
        "--ssh",
        # --direct bypasses vast.ai's SSH gateway proxy. The host's actual IP
        # is used for both SSH and mapped ports. Avoids the gateway's per-instance
        # SSH-key cache lag (which produced "Permission denied (publickey)" on
        # ssh3 for keys that worked on ssh2 — same key, different cache state).
        # Requires the offer's direct_port_count >= ports we map; deploy.yaml's
        # min_direct_port_count filter selects for that.
        "--direct",
        # vastai bundles env vars + port mappings into a single --env arg
        # (docker-style flags inside one quoted string).
        # cog API on 5000; ComfyUI UI on 8188.
        "--env", "-p 5000:5000 -p 8188:8188",
        "--onstart-cmd", _onstart_cmd(),
    )
    if not isinstance(out, dict) or "new_contract" not in out:
        sys.exit(f"vastai create returned unexpected payload: {out}")
    return int(out["new_contract"])


def public_endpoints(instance_id: int) -> dict[str, str]:
    """Return {container_port: 'http://host:port'} from vast.ai port mappings."""
    inst = get_instance(instance_id)
    public_ip = inst.get("public_ipaddr")
    ports = inst.get("ports") or {}
    out: dict[str, str] = {}
    for cport, mapping in ports.items():
        # cport like '5000/tcp'; mapping is a list of {HostIp, HostPort} or None
        if not mapping or not public_ip:
            continue
        host_port = mapping[0].get("HostPort") if isinstance(mapping, list) else None
        if host_port:
            out[cport.split("/")[0]] = f"http://{public_ip}:{host_port}"
    return out


def get_instance(instance_id: int) -> dict:
    out = vastai("show", "instance", str(instance_id))
    if isinstance(out, list):
        return out[0] if out else {}
    return out or {}


TERMINAL_FAILURE_STATUSES = {"exited", "offline", "error"}


def wait_for_ssh(instance_id: int, timeout_s: int) -> tuple[str, int]:
    """Poll vast.ai until the instance is running and SSH is reachable.

    Image pulls on slow hosts (10+ GB cog images) can legitimately take
    20+ minutes. We tolerate that and bail early only on terminal failures.
    Reports actual_status and status_msg changes as they happen so the user
    can see progress (e.g. "Pulling image", "Container started").
    """
    started_at = time.time()
    deadline = started_at + timeout_s
    last_status: str | None = None
    last_msg: str | None = None
    while time.time() < deadline:
        inst = get_instance(instance_id)
        status = inst.get("actual_status") or "?"
        msg = (inst.get("status_msg") or "").strip()
        if status != last_status or msg != last_msg:
            elapsed = int(time.time() - started_at)
            head = f">>> [{elapsed:>4}s] instance {instance_id}: status={status}"
            print(f"{head} {msg[:160]}" if msg else head)
            last_status = status
            last_msg = msg
        if status == "running" and inst.get("ssh_host") and inst.get("ssh_port"):
            return inst["ssh_host"], int(inst["ssh_port"])
        if status in TERMINAL_FAILURE_STATUSES:
            sys.exit(
                f"instance {instance_id} reached terminal status '{status}' "
                f"after {int(time.time() - started_at)}s — destroy and retry "
                f"(probably a different host).\nstatus_msg: {msg}"
            )
        time.sleep(10)
    sys.exit(
        f"instance {instance_id} did not reach running+ssh within {timeout_s}s "
        f"(last status: {last_status}). Use `deploy.py status` to check; "
        f"`deploy.py destroy {instance_id}` to clean up."
    )


def open_tunnel(ssh_host: str, ssh_port: int, ssh_key: str, local_port: int) -> int:
    # Forward both cog's HTTP API (5000) and ComfyUI's UI (8188) — the latter
    # so workflow JSON re-export can be done in a browser without a separate
    # SSH command. ComfyUI is spawned by cog setup() and binds 0.0.0.0:8188
    # inside the container; localhost:8188 from inside the SSH session reaches it.
    cmd = [
        "ssh",
        "-i", expand_path(ssh_key),
        "-p", str(ssh_port),
        "-N", "-f",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ServerAliveInterval=30",
        "-L", f"{local_port}:localhost:5000",
        "-L", "8188:localhost:8188",
        f"root@{ssh_host}",
    ]
    print(f">>> opening tunnel: {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.run(cmd, check=True)
    pgrep = subprocess.run(
        ["pgrep", "-f", f"-L {local_port}:localhost:5000"],
        capture_output=True, text=True,
    )
    pid = int(pgrep.stdout.strip().split("\n")[0]) if pgrep.stdout.strip() else 0
    if pid:
        TUNNEL_PID_FILE.write_text(str(pid))
    return pid


def verify_public_reachability(instance_id: int, timeout_s: int = 90) -> None:
    """Probe ComfyUI's public URL to confirm port mapping actually routes.

    vast.ai populates `ports` in instance metadata as soon as the container
    starts, but on hosts behind restrictive NAT the mapped port isn't actually
    routable from the open internet. Catches that here rather than at first
    use. Warns + continues rather than fails — local SSH tunnel still works.
    """
    public = public_endpoints(instance_id)
    url = public.get("8188")
    if not url:
        print(
            "!!! no public 8188 mapping in vast.ai metadata. The host may not "
            "support port mapping; SSH tunnel still works. Consider destroying "
            "and picking a different offer-id.",
            file=sys.stderr,
        )
        return
    probe = f"{url}/system_stats"
    deadline = time.time() + timeout_s
    last_err = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(probe, timeout=5) as r:
                if r.status == 200:
                    print(f">>> public ComfyUI reachable: {url}")
                    return
        except Exception as e:
            last_err = type(e).__name__
        time.sleep(5)
    print(
        f"!!! public URL {url} not reachable in {timeout_s}s (last: {last_err}). "
        f"Host's NAT/firewall is blocking the mapped port. SSH tunnel still works "
        f"(localhost:8188). Destroy + pick a different offer if you need public access.",
        file=sys.stderr,
    )


def health_check(local_port: int, timeout_min: int) -> None:
    url = f"http://localhost:{local_port}/health-check"
    deadline = time.time() + timeout_min * 60
    print(f">>> health-checking {url} (timeout {timeout_min}m)")
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    print(">>> instance reports ready")
                    return
        except (urllib.error.URLError, ConnectionRefusedError, TimeoutError):
            pass
        time.sleep(10)
    sys.exit(f"instance did not become healthy within {timeout_min}m — destroy and retry")


def schedule_self_destruct(instance_id: int, ssh_host: str, ssh_port: int,
                           ssh_key: str, max_hours: float, api_key: str) -> None:
    """Run on the remote box: sleep, then DELETE the instance via vast.ai REST.

    Retries with backoff because sshd often isn't accepting connections in the
    first few seconds after vast.ai reports actual_status=running. If all retries
    fail, warn but keep going — the local atexit handler still cleans up on
    normal exit; only laptop-dies-mid-session leaks money in that case.
    """
    seconds = int(max_hours * 3600)
    url = f"https://console.vast.ai/api/v0/instances/{instance_id}/?api_key={api_key}"
    # Use setsid -f so the inner sleep+curl is detached from the SSH session;
    # otherwise SSH waits for the descriptor to close.
    remote = (
        f"setsid -f sh -c "
        f"'sleep {seconds}; curl -s -X DELETE \"{url}\"' "
        f"</dev/null >/tmp/self-destruct.log 2>&1"
    )
    base_cmd = [
        "ssh",
        "-i", expand_path(ssh_key),
        "-p", str(ssh_port),
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=10",
        f"root@{ssh_host}",
        remote,
    ]
    last_err = ""
    for attempt in range(1, 7):  # ~60s total worst case
        try:
            subprocess.run(base_cmd, check=True, capture_output=True, text=True, timeout=20)
            print(f">>> server-side self-destruct armed: instance {instance_id} dies in {max_hours}h")
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            last_err = (getattr(e, "stderr", "") or "").strip() or str(e)
            time.sleep(min(2 * attempt, 15))
    print(
        f"!!! could not arm server-side self-destruct after 6 attempts "
        f"(last error: {last_err[:200]}). Local atexit cleanup is still active; "
        f"if this script exits cleanly the instance will be destroyed. "
        f"Don't close your laptop without Ctrl-C first.",
        file=sys.stderr,
    )


def register_atexit_cleanup(instance_id: int) -> None:
    def _cleanup():
        try:
            print(f"\n>>> atexit: destroying instance {instance_id}")
            vastai("destroy", "instance", "-y", str(instance_id), json_output=False, check=False)
        except Exception as e:
            print(f"!!! atexit cleanup failed: {e}", file=sys.stderr)
    atexit.register(_cleanup)
    signal.signal(signal.SIGINT, lambda *_: sys.exit(130))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))


# -------------------- subcommands --------------------

def cmd_setup(args: argparse.Namespace) -> int:
    cfg = load_config()
    pub_key_path = Path(expand_path(os.environ.get("SSH_KEY_PATH", cfg["ssh_key"]) + ".pub"))
    if not pub_key_path.exists():
        sys.exit(f"public key not found: {pub_key_path}")
    pub = pub_key_path.read_text().strip()
    print(f">>> registering {pub_key_path} with vast.ai")
    vastai("create", "ssh-key", pub, json_output=False)
    print(">>> done. Verify with: vastai show ssh-keys")
    return 0


def cmd_find(args: argparse.Namespace) -> int:
    cfg = load_config()
    offers = search_offers(cfg, limit=args.limit)
    if not offers:
        print("no offers match deploy.yaml filters; loosen something")
        return 1
    print(f">>> top {len(offers)} offers (by dph_total ascending):")
    for o in offers:
        print("  " + fmt_offer(o))
    print()
    print("To launch a specific one: python3 deploy/deploy.py run --offer-id <id>")
    return 0


def fetch_offer(offer_id: int) -> dict:
    """Look up a specific offer by id, ignoring deploy.yaml filters."""
    out = vastai("search", "offers", f"id={offer_id} rentable=true")
    if not isinstance(out, list) or not out:
        sys.exit(f"offer {offer_id} not found or no longer rentable")
    return out[0]


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config()
    image = args.image or cfg.get("image") or os.environ.get("GHCR_IMAGE")
    if not image:
        sys.exit("no image specified (deploy.yaml `image:` or --image or $GHCR_IMAGE)")

    if args.offer_id:
        offer = fetch_offer(args.offer_id)
        print(f">>> using requested offer: {fmt_offer(offer)}")
    else:
        offers = search_offers(cfg, limit=10)
        if not offers:
            sys.exit("no offers match filters")
        offer = offers[0]
        print(f">>> picked cheapest: {fmt_offer(offer)}")

    instance_id = create_instance(int(offer["id"]), image, cfg["disk_gb"])
    register_atexit_cleanup(instance_id)
    provision_timeout_s = int(cfg.get("provision_timeout_min", 30)) * 60
    ssh_host, ssh_port = wait_for_ssh(instance_id, timeout_s=provision_timeout_s)

    schedule_self_destruct(
        instance_id, ssh_host, ssh_port,
        cfg["ssh_key"], cfg["max_session_hours"], get_api_key(),
    )

    open_tunnel(ssh_host, ssh_port, cfg["ssh_key"], cfg["local_port"])
    health_check(cfg["local_port"], cfg["health_check_timeout_min"])
    verify_public_reachability(instance_id, timeout_s=90)

    public = public_endpoints(instance_id)

    print()
    print(f"  instance:   {instance_id}")
    print(f"  ssh:        ssh -i {cfg['ssh_key']} -p {ssh_port} root@{ssh_host}")
    print(f"  tunnel:     http://localhost:{cfg['local_port']}  (cog API)")
    print(f"              http://localhost:8188             (ComfyUI UI)")
    if public.get("5000"):
        print(f"  public api: {public['5000']}")
    if public.get("8188"):
        print(f"  public ui:  {public['8188']}")
    if not public:
        print("  public:     (port mapping not yet visible in vast.ai metadata; "
              "re-check with `deploy.py status` in ~30s)")
    print(f"  destroy:    python deploy/deploy.py destroy {instance_id}")
    print(f"  self-destruct: armed for {cfg['max_session_hours']}h")
    print()
    if args.detach:
        atexit.unregister(_atexit_cleanup_for(instance_id))  # let it live
        return 0
    print("Press Ctrl-C to destroy and exit.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        return 0


def _atexit_cleanup_for(instance_id: int):
    """Stub for symmetry; atexit cannot truly unregister without keeping a ref."""
    return None


def cmd_destroy(args: argparse.Namespace) -> int:
    print(f">>> destroying instance {args.instance_id}")
    vastai("destroy", "instance", "-y", str(args.instance_id), json_output=False)
    if TUNNEL_PID_FILE.exists():
        try:
            pid = int(TUNNEL_PID_FILE.read_text().strip())
            os.kill(pid, signal.SIGTERM)
        except (ValueError, ProcessLookupError):
            pass
        TUNNEL_PID_FILE.unlink(missing_ok=True)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    out = vastai("show", "instances")
    if not isinstance(out, list):
        print(out)
        return 0
    if not out:
        print("no active instances")
        return 0
    for inst in out:
        print(f"  id={inst.get('id')}  status={inst.get('actual_status')}  "
              f"gpu={inst.get('gpu_name')}  "
              f"image={inst.get('image_uuid', inst.get('image'))}  "
              f"dph={inst.get('dph_total')}")
        public = public_endpoints(int(inst["id"]))
        for cport, url in public.items():
            label = "ComfyUI UI" if cport == "8188" else "cog API" if cport == "5000" else cport
            print(f"    {label}: {url}")
    return 0


# -------------------- main --------------------

def main(argv: list[str] | None = None) -> int:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("setup", help="Register your SSH public key with vast.ai (one-time)").set_defaults(func=cmd_setup)

    p_find = sub.add_parser("find", help="List top matching offers (dry-run)")
    p_find.add_argument("--limit", type=int, default=5)
    p_find.set_defaults(func=cmd_find)

    p_run = sub.add_parser("run", help="Provision instance, tunnel, ready")
    p_run.add_argument("--image", help="Override deploy.yaml image:")
    p_run.add_argument("--offer-id", type=int, default=None,
                       help="Use a specific offer id (from `find`) instead of the cheapest match.")
    p_run.add_argument("--detach", action="store_true",
                       help="Exit after ready instead of holding ownership; instance survives.")
    p_run.set_defaults(func=cmd_run)

    p_destroy = sub.add_parser("destroy", help="Destroy an instance by id")
    p_destroy.add_argument("instance_id", type=int)
    p_destroy.set_defaults(func=cmd_destroy)

    sub.add_parser("status", help="Show your vast.ai instances").set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
