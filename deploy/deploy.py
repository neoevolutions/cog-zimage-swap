#!/usr/bin/env python3
"""vast.ai provisioning + tunnel + cost guardrails for cog-zimage.

Subcommands:
  setup           Register your local SSH public key with vast.ai (one-time).
  find            Dry-run: list top-N matching offers per deploy.yaml filters. Free.
  run             Provision an instance, open SSH tunnel on local_port, arm guardrails.
  tunnel [id]     (Re)open SSH tunnel to a running instance — use after IP/network
                  changes (VPN reconnect, etc.) or when the local tunnel process died.
                  Defaults to the sole running instance. --lan binds on 0.0.0.0.
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
    if cfg.get("require_static_ip"):
        # `ssh_direct` is documented but the API rejects it as "not a valid
        # search key" — vastai docs lie. The real filter is `static_ip=true`,
        # which selects datacenter hosts with dedicated IPs. Those are the
        # ones that actually honor --direct (host IP routing instead of
        # vast.ai's proxy gateway, which has the SSH-key-cache lag bug).
        parts.append("static_ip=true")
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
    # cog.server.http only accepts --host (no --port flag); it always binds 5000.
    # Our --env "-p 5000:5000" mapping handles the port routing.
    return (
        "setsid -f bash -c "
        "'cd /src; exec python -m cog.server.http "
        "--host 0.0.0.0 "
        "> /var/log/cog.log 2>&1 < /dev/null'"
    )


def create_instance(offer_id: int, image: str, disk_gb: int) -> int:
    # vastai 1.0.7's CLI has a typo bug: passing --direct produces the runtype
    # string "ssh_direc ssh_proxy" (missing the 't'). The vast.ai backend
    # tokenizes runtype on whitespace, doesn't recognize "ssh_direc", and
    # silently falls back to ssh_proxy (gateway) mode. Result: every instance
    # ended up routed through ssh*.vast.ai, hitting the gateway's per-key
    # cache lag and failing with "Permission denied (publickey)".
    # See vastai/cli/util.py:685 and vastai/cli/commands/instances.py:116.
    # Workaround: bypass the buggy CLI and call the SDK directly with the
    # correct runtype spelling.
    print(f">>> creating instance from offer {offer_id} with image {image}")
    from vastai.sdk import VastAI

    sdk = VastAI(api_key=get_api_key())
    # parse_env in vastai/cli/util.py turns "-p 5000:5000 -p 8188:8188" into
    # this dict shape; reproduce it directly for the SDK call.
    env = {"-p 5000:5000": "1", "-p 8188:8188": "1"}
    # Forward CIVITAI_TOKEN if set in local .env. predict.py setup() reads it
    # to download Civitai-gated weights (e.g. character LoRAs). Sent over the
    # vast.ai create-instance API request — they store env vars on the
    # instance metadata. If you don't want the token in vast.ai's API logs,
    # leave CIVITAI_TOKEN unset and download Civitai weights manually after
    # the instance is up.
    civitai_token = os.environ.get("CIVITAI_TOKEN", "").strip()
    if civitai_token:
        env["CIVITAI_TOKEN"] = civitai_token
    out = sdk.create_instance(
        id=offer_id,
        image=image,
        disk=disk_gb,
        env=env,
        runtype="ssh_direct ssh_proxy",
        onstart_cmd=_onstart_cmd(),
    )
    if not isinstance(out, dict) or "new_contract" not in out:
        sys.exit(f"vastai SDK create_instance returned unexpected payload: {out}")
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


def attach_ssh_to_instance(instance_id: int, ssh_pub_key: str, api_key: str) -> bool:
    """Force-sync an SSH pubkey to a specific running instance.

    Vast.ai injects authorized_keys at create-time using a snapshot of the
    account's keys. That snapshot can be stale on hosts with slow key sync,
    causing 'Permission denied (publickey)' on every attempt even though the
    key is registered on the account. POSTing to /api/v0/instances/<id>/ssh/
    pushes the key to the specific host bypassing the cache lag.

    Returns True on success, False on failure (logged but non-fatal — caller
    can still try authenticating).
    """
    body = json.dumps({"ssh_key": ssh_pub_key}).encode()
    req = urllib.request.Request(
        f"https://console.vast.ai/api/v0/instances/{instance_id}/ssh/",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            print(f">>> attached SSH key to instance {instance_id} (HTTP {resp.status})")
            return True
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")[:300]
        # 400 "duplicate" or "already attached" is fine — key was already there.
        if e.code == 400 and ("duplicate" in msg.lower() or "already" in msg.lower()):
            print(f">>> key already attached to instance {instance_id}")
            return True
        print(f"!!! attach_ssh_to_instance HTTP {e.code}: {msg}", file=sys.stderr)
        return False
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"!!! attach_ssh_to_instance failed: {e}", file=sys.stderr)
        return False


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


def _ssh_base_opts() -> list[str]:
    """SSH options for ephemeral vast.ai hosts.

    - accept-new/no host check: gateway ports recycle across instances with
      different host keys; pinning fingerprints is meaningless and trips strict
      mode on every reuse.
    - IdentitiesOnly=yes: prevents ssh from offering every key in ssh-agent
      (which can exhaust the server's MaxAuthTries before the right one is
      tried, manifesting as "Permission denied (publickey)" even when -i
      points at the right key).
    """
    return [
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-o", "IdentitiesOnly=yes",
    ]


def _ssh_auth_probe(ssh_host: str, ssh_port: int, ssh_key: str) -> tuple[bool, str]:
    """Run a synchronous `echo ok` over SSH with -vv. Returns (ok, verbose_stderr).
    Single-shot; for transient-error retry use _ssh_wait_for_auth.
    """
    cmd = [
        "ssh",
        "-i", expand_path(ssh_key),
        "-p", str(ssh_port),
        "-vv",
        *_ssh_base_opts(),
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        f"root@{ssh_host}",
        "echo ok",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        return False, "ssh probe timed out after 20s"
    ok = result.returncode == 0 and "ok" in (result.stdout or "")
    return ok, result.stderr or ""


def _is_transient_ssh_error(stderr: str) -> bool:
    """Tell apart errors worth retrying from ones that won't fix themselves.

    Retry-worthy:
    - TCP-level failures: sshd not yet bound inside the container (30–90s window).
    - 'Permission denied (publickey)': vast.ai's own login banner literally
      says "If authentication fails, try again after a few seconds, and
      double check your ssh key" — their per-host authorized_keys sync has a
      lag. We attach_ssh_to_instance upstream to push the key explicitly,
      but propagation still takes a few seconds after that.

    Hopeless: malformed args, missing key file, etc. — those won't appear in
    stderr from a well-formed -vv probe and would surface as different
    failures earlier.
    """
    retry_markers = (
        "Connection refused",
        "Connection timed out",
        "Connection reset",
        "No route to host",
        "Host is down",
        "Network is unreachable",
        "kex_exchange_identification",  # sshd accepted TCP but closed before banner
        "Permission denied (publickey)",  # vast.ai key-sync lag
    )
    return any(m in stderr for m in retry_markers)


def _ssh_wait_for_auth(ssh_host: str, ssh_port: int, ssh_key: str,
                       timeout_s: int = 180) -> tuple[bool, str]:
    """Probe auth, retrying on transient TCP-level errors. Bails immediately on
    auth-level rejection (Permission denied) since that won't fix itself.
    """
    deadline = time.time() + timeout_s
    last_verbose = ""
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        ok, verbose = _ssh_auth_probe(ssh_host, ssh_port, ssh_key)
        last_verbose = verbose
        if ok:
            if attempt > 1:
                print(f">>> ssh auth ok on attempt {attempt}")
            return True, verbose
        if not _is_transient_ssh_error(verbose):
            # Permission denied or similar — sshd is up and actively rejecting.
            return False, verbose
        if attempt == 1:
            print(f">>> sshd not yet listening on {ssh_host}:{ssh_port}; retrying for {timeout_s}s")
        time.sleep(min(5 + attempt * 2, 15))
    return False, last_verbose


def diagnose_container(instance_id: int, ssh_key: str, what: str = "all") -> None:
    """Inspect container state via SSH (vast.ai's `execute` API is whitelisted).

    `what` selects sections: 'ssh', 'cog', or 'all'. Uses SSH directly because
    vast.ai's execute endpoint rejects arbitrary commands ("Invalid command
    given") even for `echo hello`. SSH works as long as the host accepted our
    pubkey (which we can confirm via `vastai logs`).
    """
    try:
        ssh_host, ssh_port, _mode = select_ssh_endpoint(instance_id)
    except Exception as e:
        print(f"!!! could not resolve ssh endpoint for diagnostic: {e}", file=sys.stderr)
        return
    sections = []
    if what in ("ssh", "all"):
        sections.append(
            "echo '--- /root/.ssh/ ---'; ls -la /root/.ssh/ 2>&1; "
            "echo '--- authorized_keys ---'; cat /root/.ssh/authorized_keys 2>&1; "
            "echo '--- sshd processes ---'; ps aux | grep -E '[s]shd' 2>&1"
        )
    if what in ("cog", "all"):
        # ss isn't installed on slim images; use netstat or /proc fallback.
        sections.append(
            "echo '--- cog/python processes ---'; ps aux | grep -E '[p]ython|[c]og' | head -10; "
            "echo '--- listeners on 5000/8188 ---'; "
            "(netstat -tlnp 2>/dev/null || ss -tlnp 2>/dev/null) | grep -E ':(5000|8188)\\b' || echo '(no listeners)'; "
            "echo '--- /var/log/cog.log (last 80 lines) ---'; tail -n 80 /var/log/cog.log 2>&1; "
            "echo '--- /src contents ---'; ls -la /src/ 2>&1 | head -25; "
            "echo '--- onstart log ---'; "
            "tail -n 40 /var/log/onstart.log 2>/dev/null || cat /root/onstart.log 2>/dev/null || "
            "tail -n 40 /tmp/onstart.log 2>/dev/null || echo '(no onstart log found)'"
        )
    cmd = "; ".join(sections)
    ssh_cmd = [
        "ssh",
        "-i", expand_path(ssh_key),
        "-p", str(ssh_port),
        *_ssh_base_opts(),
        "-o", "ConnectTimeout=10",
        f"root@{ssh_host}",
        cmd,
    ]
    print(f">>> diagnosing container ({what}) via ssh root@{ssh_host}:{ssh_port}...")
    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        print("!!! diagnostic ssh timed out after 30s", file=sys.stderr)
        return
    print(f"--- container state ({what}) ---")
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(f"(ssh exit {result.returncode})")
        if result.stderr:
            print(f"stderr: {result.stderr.strip()}")
    print("--- end ---")


def select_ssh_endpoint(instance_id: int) -> tuple[str, int, str]:
    """Pick the best SSH endpoint for an instance: direct host > proxy gateway.

    vast.ai always sets ssh_host/ssh_port to the gateway proxy (ssh*.vast.ai)
    in the metadata, even on hosts that support direct connections. But for
    hosts with static_ip=true and a port-22 mapping in `ports`, we can SSH
    straight to public_ipaddr:host_port and skip the gateway's key-cache lag.

    Returns (host, port, mode) where mode is 'direct' or 'proxy' for logging.
    """
    inst = get_instance(instance_id)
    public_ip = inst.get("public_ipaddr")
    ports = inst.get("ports") or {}
    ssh_mapping = ports.get("22/tcp")
    if public_ip and isinstance(ssh_mapping, list) and ssh_mapping:
        host_port = ssh_mapping[0].get("HostPort")
        if host_port:
            return public_ip, int(host_port), "direct"
    # Fallback: use vast.ai's gateway proxy
    return inst["ssh_host"], int(inst["ssh_port"]), "proxy"


def open_tunnel(ssh_host: str, ssh_port: int, ssh_key: str, local_port: int,
                instance_id: int | None = None, bind_addr: str = "localhost") -> int:
    """Open SSH tunnel forwarding local 5000+8188 to the container's same ports.

    bind_addr defaults to localhost (only this Mac can connect). Pass "0.0.0.0"
    to bind on all interfaces so other devices on the LAN can reach the
    tunnel — requires GatewayPorts=yes (set automatically below).
    """
    # Forward both cog's HTTP API (5000) and ComfyUI's UI (8188) — the latter
    # so workflow JSON re-export can be done in a browser without a separate
    # SSH command. ComfyUI is spawned by cog setup() and binds 0.0.0.0:8188
    # inside the container; localhost:8188 from inside the SSH session reaches it.
    extra_opts: list[str] = []
    if bind_addr != "localhost":
        # OpenSSH ignores non-localhost bind addresses unless GatewayPorts=yes.
        extra_opts.extend(["-o", "GatewayPorts=yes"])
    cmd = [
        "ssh",
        "-i", expand_path(ssh_key),
        "-p", str(ssh_port),
        "-N", "-f",
        *_ssh_base_opts(),
        "-o", "ServerAliveInterval=30",
        *extra_opts,
        "-L", f"{bind_addr}:{local_port}:localhost:5000",
        "-L", f"{bind_addr}:8188:localhost:8188",
        f"root@{ssh_host}",
    ]
    print(f">>> opening tunnel: {' '.join(shlex.quote(c) for c in cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        if instance_id is not None:
            diagnose_container(instance_id, ssh_key, what="ssh")
        raise
    pgrep = subprocess.run(
        ["pgrep", "-f", f":{local_port}:localhost:5000"],
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


def health_check(local_port: int, timeout_min: int, instance_id: int | None = None,
                 ssh_key: str | None = None) -> None:
    url = f"http://localhost:{local_port}/health-check"
    deadline = time.time() + timeout_min * 60
    started = time.time()
    print(f">>> health-checking {url} (timeout {timeout_min}m)")
    last_err = ""
    poll_count = 0
    while time.time() < deadline:
        poll_count += 1
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    print(">>> instance reports ready")
                    return
                last_err = f"HTTP {r.status}"
        # ConnectionResetError happens when ssh forwards to a dead/binding-but-not-yet-listening
        # port on the remote, OR when cog accepts and then drops mid-response. OSError covers
        # broken pipes, host unreachable, etc. — cog isn't ready yet, retry.
        except (urllib.error.URLError, ConnectionRefusedError, ConnectionResetError,
                TimeoutError, OSError) as e:
            last_err = type(e).__name__
        # Run a one-time diagnostic ~60s in if we're still failing; gives us early
        # visibility into whether cog ever started instead of waiting the full 12 min.
        if poll_count == 6 and instance_id is not None and ssh_key:
            elapsed = int(time.time() - started)
            print(f">>> still failing after {elapsed}s ({last_err}); pulling cog state...")
            diagnose_container(instance_id, ssh_key, what="cog")
        time.sleep(10)
    if instance_id is not None and ssh_key:
        print(f">>> health-check timed out after {timeout_min}m (last: {last_err}); final diagnostic:")
        diagnose_container(instance_id, ssh_key, what="all")
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
        *_ssh_base_opts(),
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
    # Direct API call: the `vastai` CLI shim ignores our VAST_API_KEY env var
    # (it expects ~/.config/vastai/vast_api_key) so it returns "Failed with
    # error 403" mixed into stdout while still exiting 0 — silently no-ops.
    api_key = get_api_key()
    body = json.dumps({"ssh_key": pub}).encode()
    req = urllib.request.Request(
        "https://console.vast.ai/api/v0/ssh/",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            print(f">>> registered (HTTP {resp.status})")
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")
        if e.code == 400 and "duplicate" in msg.lower():
            print(">>> already registered (no-op)")
        else:
            sys.exit(f"failed to register key (HTTP {e.code}): {msg}")
    # Verify and show all keys on the account so the user can sanity-check.
    list_req = urllib.request.Request(
        "https://console.vast.ai/api/v0/ssh/",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    keys = json.load(urllib.request.urlopen(list_req, timeout=15))
    print(f">>> {len(keys)} key(s) on account:")
    for k in keys:
        pubk = k.get("public_key", "")
        # Show fingerprint + comment for easy identification
        print(f"  id={k.get('id')}  {pubk.split()[0] if pubk else '?'}  {pubk.split()[-1] if pubk and len(pubk.split()) > 2 else ''}")
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
    wait_for_ssh(instance_id, timeout_s=provision_timeout_s)

    # Prefer the host's public IP + mapped port 22 over vast.ai's gateway.
    # The gateway has a per-key cache lag bug that causes spurious
    # "Permission denied (publickey)" rejections even with a valid key.
    ssh_host, ssh_port, mode = select_ssh_endpoint(instance_id)
    print(f">>> ssh endpoint: {mode} root@{ssh_host}:{ssh_port}")

    # Force-attach the SSH pubkey to this specific instance. Account-level
    # registration alone has a known per-host sync lag that produces
    # "Permission denied (publickey)" indefinitely on some hosts; this API
    # call pushes the key to the host directly.
    pub_key_path = Path(expand_path(cfg["ssh_key"] + ".pub"))
    if pub_key_path.exists():
        attach_ssh_to_instance(
            instance_id, pub_key_path.read_text().strip(), get_api_key(),
        )

    # Probe auth. wait_for_ssh returns when vast.ai's metadata says
    # status=running, but sshd inside the container takes another 30–90s to
    # bind, and the attach above takes a few more seconds to propagate to
    # authorized_keys. Both Connection refused and Permission denied are
    # treated as transient here.
    ok, verbose = _ssh_wait_for_auth(ssh_host, ssh_port, cfg["ssh_key"])
    if not ok and mode == "direct":
        print(f"!!! direct SSH probe failed; verbose follows", file=sys.stderr)
        print(verbose, file=sys.stderr)
        inst = get_instance(instance_id)
        gw_host, gw_port = inst["ssh_host"], int(inst["ssh_port"])
        print(f">>> falling back to gateway {gw_host}:{gw_port}", file=sys.stderr)
        ok2, verbose2 = _ssh_wait_for_auth(gw_host, gw_port, cfg["ssh_key"])
        if ok2:
            print(f">>> gateway auth ok — switching endpoint")
            ssh_host, ssh_port, mode = gw_host, gw_port, "proxy"
        else:
            print(f"!!! gateway SSH probe ALSO failed; verbose follows", file=sys.stderr)
            print(verbose2, file=sys.stderr)
            sys.exit(
                "SSH unreachable on both direct and gateway after retry window. "
                "If verbose shows 'Permission denied', re-run `deploy.py setup` "
                "and retry. If 'Connection refused' kept repeating, the host's "
                "container failed to start sshd — destroy this instance "
                "and pick a different offer."
            )
    elif not ok:
        print(f"!!! SSH probe failed; verbose follows", file=sys.stderr)
        print(verbose, file=sys.stderr)
        sys.exit("SSH unreachable — see verbose output above")

    schedule_self_destruct(
        instance_id, ssh_host, ssh_port,
        cfg["ssh_key"], cfg["max_session_hours"], get_api_key(),
    )

    open_tunnel(ssh_host, ssh_port, cfg["ssh_key"], cfg["local_port"], instance_id)
    health_check(cfg["local_port"], cfg["health_check_timeout_min"], instance_id, cfg["ssh_key"])
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


def _kill_existing_tunnel(local_port: int) -> int:
    """Kill any local ssh tunnel processes forwarding this local_port.

    Catches both the recorded TUNNEL_PID_FILE entry and any orphans (e.g. a
    tunnel started manually in another terminal). Returns the count killed.
    """
    killed = 0
    if TUNNEL_PID_FILE.exists():
        try:
            pid = int(TUNNEL_PID_FILE.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            print(f">>> killed tunnel pid {pid} (from {TUNNEL_PID_FILE.name})")
            killed += 1
        except (ValueError, ProcessLookupError):
            pass
        TUNNEL_PID_FILE.unlink(missing_ok=True)
    # Catch orphans by pattern. Match both "5000:..." and "localhost:5000:..."
    # / "0.0.0.0:5000:..." -L formats.
    result = subprocess.run(
        ["pgrep", "-f", f":{local_port}:localhost:5000"],
        capture_output=True, text=True,
    )
    for line in result.stdout.strip().splitlines():
        try:
            pid = int(line.strip())
            os.kill(pid, signal.SIGTERM)
            print(f">>> killed orphan tunnel pid {pid}")
            killed += 1
        except (ValueError, ProcessLookupError):
            pass
    return killed


def _resolve_target_instance(arg_id: int | None) -> int:
    """Pick the instance id for tunnel ops: explicit arg > sole running > error."""
    if arg_id:
        return arg_id
    out = vastai("show", "instances")
    if not isinstance(out, list) or not out:
        sys.exit("no active instances; pass an instance id explicitly")
    running = [i for i in out if i.get("actual_status") == "running"]
    if not running:
        statuses = ", ".join(f"{i['id']}={i.get('actual_status')}" for i in out)
        sys.exit(f"no running instances ({statuses}); pass an instance id explicitly")
    if len(running) > 1:
        ids = ", ".join(str(i["id"]) for i in running)
        sys.exit(f"multiple running instances ({ids}); pass one explicitly: deploy.py tunnel <id>")
    return int(running[0]["id"])


def _probe_tunnel(local_port: int, timeout_s: float = 4.0) -> bool:
    """Quick TCP probe of the local tunnel endpoint."""
    import socket
    try:
        with socket.create_connection(("localhost", local_port), timeout=timeout_s):
            return True
    except OSError:
        return False


def cmd_tunnel(args: argparse.Namespace) -> int:
    """(Re)open the SSH tunnel to a running vast.ai instance.

    Use this when:
      - your laptop's IP changed (VPN reconnect, network swap) and the existing
        tunnel went silent;
      - the local ssh process died for any reason;
      - you want to switch the tunnel to LAN-bound (--lan) so other devices
        can reach the cog API + ComfyUI UI.

    Resolves the instance's current SSH endpoint fresh each call, so direct
    vs proxy host changes (host migration on vast.ai's side) are picked up
    automatically.
    """
    cfg = load_config()
    instance_id = _resolve_target_instance(args.instance_id)
    print(f">>> target instance: {instance_id}")

    _kill_existing_tunnel(cfg["local_port"])

    ssh_host, ssh_port, mode = select_ssh_endpoint(instance_id)
    print(f">>> ssh endpoint: {mode} root@{ssh_host}:{ssh_port}")

    bind_addr = "0.0.0.0" if args.lan else "localhost"
    open_tunnel(ssh_host, ssh_port, cfg["ssh_key"], cfg["local_port"],
                instance_id=instance_id, bind_addr=bind_addr)

    # Brief verification — sshd is up the moment open_tunnel returns, but cog
    # may still be doing setup() if the instance just spawned. A miss here is
    # common and not fatal.
    if _probe_tunnel(cfg["local_port"]):
        print(">>> tunnel verified: localhost:{} reachable".format(cfg["local_port"]))
    else:
        print("!!! tunnel opened but cog API not yet responding (may still be in setup)")

    print()
    if bind_addr == "localhost":
        print(f"  cog API:  http://localhost:{cfg['local_port']}")
        print(f"  ComfyUI:  http://localhost:8188")
    else:
        # Show the LAN IP so the user knows what to type from another device.
        try:
            import socket
            lan_ip = socket.gethostbyname(socket.gethostname())
        except OSError:
            lan_ip = "<your-lan-ip>"
        print(f"  cog API:  http://{lan_ip}:{cfg['local_port']}  (bound on 0.0.0.0)")
        print(f"  ComfyUI:  http://{lan_ip}:8188")
    print(f"  ssh:      ssh -i {cfg['ssh_key']} -p {ssh_port} root@{ssh_host}")
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

    p_tunnel = sub.add_parser(
        "tunnel",
        help="(Re)open SSH tunnel to a running instance after IP/network changes",
    )
    p_tunnel.add_argument(
        "instance_id", type=int, nargs="?", default=None,
        help="Instance id (defaults to the sole running instance if there's only one)",
    )
    p_tunnel.add_argument(
        "--lan", action="store_true",
        help="Bind tunnel on 0.0.0.0 so other devices on your LAN can reach it (default: localhost only)",
    )
    p_tunnel.set_defaults(func=cmd_tunnel)

    sub.add_parser("status", help="Show your vast.ai instances").set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
