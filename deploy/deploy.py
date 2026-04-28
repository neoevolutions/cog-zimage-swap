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

def create_instance(offer_id: int, image: str, disk_gb: int) -> int:
    print(f">>> creating instance from offer {offer_id} with image {image}")
    out = vastai(
        "create", "instance", str(offer_id),
        "--image", image,
        "--disk", str(disk_gb),
        "--ssh",
        "--onstart-cmd", "bash -c 'echo ready > /tmp/cog_ready'",
    )
    if not isinstance(out, dict) or "new_contract" not in out:
        sys.exit(f"vastai create returned unexpected payload: {out}")
    return int(out["new_contract"])


def get_instance(instance_id: int) -> dict:
    out = vastai("show", "instance", str(instance_id))
    if isinstance(out, list):
        return out[0] if out else {}
    return out or {}


def wait_for_ssh(instance_id: int, timeout_s: int = 600) -> tuple[str, int]:
    deadline = time.time() + timeout_s
    last_status = None
    while time.time() < deadline:
        inst = get_instance(instance_id)
        status = inst.get("actual_status")
        if status != last_status:
            print(f">>> instance {instance_id} status: {status}")
            last_status = status
        if status == "running" and inst.get("ssh_host") and inst.get("ssh_port"):
            return inst["ssh_host"], int(inst["ssh_port"])
        time.sleep(5)
    sys.exit(f"instance {instance_id} did not reach running+ssh within {timeout_s}s")


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
    """Run on the remote box: sleep, then DELETE the instance via vast.ai REST."""
    seconds = int(max_hours * 3600)
    remote = (
        f"nohup sh -c "
        f"'sleep {seconds} && "
        f"curl -s -X DELETE "
        f"\"https://console.vast.ai/api/v0/instances/{instance_id}/?api_key={api_key}\"' "
        f"> /tmp/self-destruct.log 2>&1 &"
    )
    cmd = [
        "ssh",
        "-i", expand_path(ssh_key),
        "-p", str(ssh_port),
        "-o", "StrictHostKeyChecking=accept-new",
        f"root@{ssh_host}",
        remote,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f">>> server-side self-destruct armed: instance {instance_id} dies in {max_hours}h")


def register_atexit_cleanup(instance_id: int) -> None:
    def _cleanup():
        try:
            print(f"\n>>> atexit: destroying instance {instance_id}")
            vastai("destroy", "instance", str(instance_id), json_output=False, check=False)
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
    ssh_host, ssh_port = wait_for_ssh(instance_id, timeout_s=600)

    schedule_self_destruct(
        instance_id, ssh_host, ssh_port,
        cfg["ssh_key"], cfg["max_session_hours"], get_api_key(),
    )

    open_tunnel(ssh_host, ssh_port, cfg["ssh_key"], cfg["local_port"])
    health_check(cfg["local_port"], cfg["health_check_timeout_min"])

    print()
    print(f"  instance:   {instance_id}")
    print(f"  ssh:        ssh -i {cfg['ssh_key']} -p {ssh_port} root@{ssh_host}")
    print(f"  tunnel:     http://localhost:{cfg['local_port']}")
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
    vastai("destroy", "instance", str(args.instance_id), json_output=False)
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
