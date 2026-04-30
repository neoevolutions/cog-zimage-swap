# Replicating this architecture

If you're building something with the same shape as this project — Cog
container wrapping a ComfyUI workflow with custom nodes, deployed to a
discount cloud GPU marketplace, with private/gated weights — here's the
distilled experience. Each section below is a decision or a gotcha that cost
real time the first time around. The goal is to give a future-you (or
someone setting up a sister project) a 1-day path through what took
multiple debugging sessions to solve.

## The four big architectural decisions

### 1. Decouple the API from the workflow graph via named handles

Cog's predictor signature (`target_image`, `prompt`, `seed`, `lora`) needs
to write into specific ComfyUI nodes. The naive approach is to hard-code
node IDs (`workflow["958"]["inputs"]["image"] = ...`). That breaks every
time the workflow author rearranges the graph or you re-export.

Instead: use ComfyUI's `_meta.title` field as a stable handle. The workflow
JSON has nodes like:

```json
"958": {
  "class_type": "LoadImage",
  "_meta": {"title": "INPUT_TARGET_IMAGE"},
  "inputs": {"image": "..."}
}
```

The patcher (`workflow_patch.py`) walks the workflow looking for nodes by
`_meta.title`, not by ID. The author can rearrange/duplicate/replace nodes
freely as long as the titles survive. This is the **single most important
pattern** for keeping the API contract stable across workflow edits.

The patcher should also:

- Treat handle absence as a hard error (fail fast at predict time, not
  silently produce wrong outputs).
- Allow optional handles (`LORA_LOADER` here) — bypass-by-default if
  unset.
- Skip dict keys starting with `_` so JSON `_comment` annotations don't
  break the lookup.

Unit-test the patcher *without* importing Cog. Keep it pure-Python.

### 2. Runtime weight downloads, not build-time baking

Tempting to bake all weights into the Docker image so cold start is fast.
Three problems:

1. Cog COPYs `/src` into the container *after* `run:` steps complete. So
   build steps can't reference your `weights.json` or download script. You'd
   have to inline the URL list as bash. Brittle and ugly.
2. Gated weights (Civitai, private HF repos) need credentials at build
   time. That's BuildKit secrets, which is workable but adds infra.
3. The image becomes 12+ GB, slow to push to a registry and slow to pull on
   marketplace hosts.

Better: **fetch weights at instance startup, in the Cog `setup()` method**.

```python
def setup(self):
    self._download_weights()       # idempotent; skips files already present
    self.comfyui_proc = self._start_comfyui()
    self._wait_for_comfyui(...)
```

Image stays ~2 GB. Cold start pays a one-time download (3–5 min for ~10 GB
on a fast cloud host). Weights can change without rebuilding. Tokens stay
in env vars, never in image layers.

The download script should:

- Be idempotent (skip files that exist + have plausibly-correct size). Use
  `pget` if available (parallel chunked download), fall back to `curl`.
- Treat sentinel URLs (`TODO_FIND_HF_URL`, `MULTI_FILE_BUNDLE`) as "skip,
  log, move on" — never silently fail the whole startup.
- Send credentials as `Authorization: Bearer …` headers, never as
  `?token=…` query strings. Query-string tokens leak into HTTP redirect
  chains, server logs, and tools like `pget` that echo URLs.

### 3. Direct host access > marketplace SSH gateway

Cloud GPU marketplaces typically front their hosts with an SSH gateway
proxy (`ssh*.vast.ai`, `ssh.runpod.io`, etc.). The proxy:

- Authenticates against your account-registered SSH keys.
- Forwards to the actual instance.
- Caches authorized_keys per-instance on a delay.

That cache lag bites: spawn a fresh instance, the proxy hasn't synced your
key yet, you get `Permission denied (publickey)` even though everything is
correct. Sometimes for 10s, sometimes for minutes.

Sidestep the proxy. The host's instance metadata exposes a `public_ipaddr`
and a `ports["22/tcp"]` mapping when the offer was direct-mode-capable
(usually `static_ip=true` filter selects for these). Connect directly:

```python
def select_ssh_endpoint(instance_id):
    inst = get_instance(instance_id)
    public_ip = inst.get("public_ipaddr")
    ports = inst.get("ports") or {}
    ssh_mapping = ports.get("22/tcp")
    if public_ip and ssh_mapping:
        host_port = ssh_mapping[0].get("HostPort")
        if host_port:
            return public_ip, int(host_port), "direct"
    return inst["ssh_host"], int(inst["ssh_port"]), "proxy"  # fallback
```

Filter offers for direct-capable hosts at search time (`static_ip=true` is
the working filter on vast.ai — `ssh_direct=true` is documented but the
API rejects it).

### 4. Keep diagnostics SSH-based, not CLI-based

Marketplace CLIs sometimes provide an `execute` command for running
arbitrary shell commands on an instance via the API. Tempting for
diagnostics. But (at least on vast.ai) it's whitelist-restricted —
`execute` returns HTTP 400 "Invalid command given" for anything that isn't
a small set of pre-approved commands.

Build your diagnostic to use SSH directly (it's already configured for the
tunnel). Run a multi-line shell command to dump:

- `/root/.ssh/authorized_keys` and dir perms
- listeners on your service ports (`netstat -tlnp` or `ss -tlnp` —
  whichever exists; slim images often lack `ss`)
- last 80 lines of your service's log (`tail /var/log/cog.log`)
- whether your runtime processes are alive (`ps aux | grep python`)

Wire the diagnostic into your tunnel-open and health-check failure paths
so when something goes wrong, you get actionable context immediately
without paying another debug round-trip.

## Specific gotchas worth saving you a session each

### vastai CLI 1.0.7 typo: `ssh_direc` instead of `ssh_direct`

In `vastai/cli/util.py:685`, the runtype string is built as
`'ssh_direc ssh_proxy'` (missing `t`). The vast.ai backend tokenizes
runtype on whitespace, doesn't recognize the typo, and silently falls back
to proxy mode. So `vastai create instance --direct` produces a
gateway-routed instance.

Workaround: bypass the CLI for instance creation. Call the SDK directly
with the corrected runtype string:

```python
from vastai.sdk import VastAI
sdk = VastAI(api_key=...)
sdk.create_instance(
    id=offer_id, image=..., disk=..., env={...},
    runtype="ssh_direct ssh_proxy",  # the actual correct token
    onstart_cmd=...,
)
```

Check whether this is fixed before adopting the workaround in your project.

### SSH key passphrases break batch-mode tunnels silently

`ssh -N -f` (background, no command) can't prompt for a key passphrase. If
your private key is encrypted, the tunnel attempt fails with
`Permission denied (publickey)` — the same error you'd get for a wrong key,
which sends you down the wrong debugging path.

Two fixes:

- **Per-session**: `ssh-add ~/.ssh/id_rsa_<name>` once per terminal.
- **Permanent on macOS**: in `~/.ssh/config` for the marketplace host:
  ```
  Host *.vast.ai
      IdentityFile ~/.ssh/id_rsa_vastai
      UseKeychain yes
      AddKeysToAgent yes
  ```
  Then `ssh-add --apple-use-keychain ~/.ssh/id_rsa_vastai` once. macOS
  Keychain auto-unlocks the key on every login.

### Always force `IdentitiesOnly=yes` on tunnel SSH

When `ssh-agent` has multiple keys and you pass `-i somekey`, ssh tries
`somekey` *plus* every key in the agent. If the right key isn't tried
within `MaxAuthTries` (server-side, often 6), authentication fails with —
again — `Permission denied (publickey)`. Same misleading error.

Add `-o IdentitiesOnly=yes` to all SSH calls so only the explicitly
requested key is offered.

### Marketplace gateway ports recycle host keys

Gateway ports get reassigned to new instances over time, so the host key
on `ssh*.vast.ai:NNNN` changes between runs. `StrictHostKeyChecking=accept-new`
adds the first one to `~/.ssh/known_hosts`, then on the next instance refuses
to connect because the fingerprint changed.

For ephemeral compute it's safe to disable strict checking entirely:

```
-o StrictHostKeyChecking=no
-o UserKnownHostsFile=/dev/null
-o LogLevel=ERROR     # suppress "Permanently added ..." spam
```

### `wait_for_ssh` returning early — sshd not actually listening yet

`actual_status == "running"` in vast.ai's instance metadata means the
container has been *created*, not that sshd is *bound*. The container's
sshd takes another 30–90s to start, during which both the host's direct
port and the gateway port return `Connection refused` at TCP layer. If
your provisioning script reads metadata, sees status=running, and
immediately opens a tunnel, you get an opaque "Connection refused" with
no clue that you should have just waited.

Fix: do a synchronous SSH probe (`ssh ... echo ok`) before opening the
tunnel, with a retry loop that explicitly tolerates transient errors:

```python
TRANSIENT = (
    "Connection refused", "Connection timed out", "Connection reset",
    "No route to host", "kex_exchange_identification",
    "Permission denied (publickey)",  # see next gotcha
)
def wait_for_ssh_auth(host, port, key, timeout_s=180):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        result = subprocess.run(["ssh", "-vv", "-i", key, ...], capture_output=True, text=True)
        if result.returncode == 0:
            return True
        if not any(m in result.stderr for m in TRANSIENT):
            return False  # genuine config error, no point retrying
        time.sleep(...)
    return False
```

Use `-vv` so when the loop does eventually time out, the captured stderr
shows the actual rejection reason (algorithm negotiation, fingerprint of
the key offered, server response) rather than the bare top-level error.

### Per-host `authorized_keys` cache lag — register *and* attach

The single most-confusing failure mode on vast.ai. Symptom: SSH key is
registered on your account (`GET /api/v0/ssh/` confirms it, fingerprint
matches the local pubkey), the verbose SSH log shows the right key is
offered with a sane signature algorithm, the algorithm negotiation
succeeds — and the server still rejects with `Permission denied
(publickey)`. The login banner says, verbatim:

> Welcome to vast.ai. If authentication fails, try again after a few
> seconds, and double check your ssh key.

Root cause hypothesis: vast.ai writes `authorized_keys` on the host at
instance create-time using a snapshot of the account's keys. On hosts
with slow or stalled key sync, that snapshot can be empty or stale, and
no amount of waiting fixes it because the sync isn't continuous —
account-level `POST /api/v0/ssh/` only updates *future* instances.

Two-part fix that worked:

1. **Force-attach the key to the specific running instance.** Vast.ai
   exposes `POST /api/v0/instances/<id>/ssh/` with body
   `{"ssh_key": "<pub key one-liner>"}` — call it after instance create
   but before opening the tunnel. This pushes the key to that host's
   `authorized_keys` directly, bypassing the cache:

   ```python
   urllib.request.urlopen(urllib.request.Request(
       f"https://console.vast.ai/api/v0/instances/{instance_id}/ssh/",
       data=json.dumps({"ssh_key": pub}).encode(),
       method="POST",
       headers={"Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"},
   ))
   ```

   Treat HTTP 400 with "duplicate" or "already" in the response as
   success — means the key was already on the host. Treat the call as
   non-fatal: if it fails, the auth probe loop below still has a chance.

2. **Treat `Permission denied (publickey)` as transient in the probe
   loop**, not just TCP-level errors. The vast.ai banner explicitly tells
   you to retry; propagation after the attach call takes a few seconds.
   180s of retry is plenty for both the sshd-not-bound-yet window and the
   key-sync-after-attach window.

Independent of these: also register the key with `POST /api/v0/ssh/`
*before* creating any instance, so the create-time snapshot has it.

If the probe still fails after 180s on both direct and gateway, the
verbose output will show the actual rejection reason — at that point
it's likely the host itself is misconfigured and you should destroy the
instance and pick a different offer.

(Also: don't trust the `vastai` CLI for SSH key registration. It expects
the API key in `~/.config/vastai/vast_api_key`, not the `VAST_API_KEY`
env var. If you set only the env var, `vastai create ssh-key` returns
exit 0 with "Failed with error 403" mixed into stdout — silently no-ops.
Use the API directly.)

### `cog.server.http` doesn't accept `--port`

`cog.server.http` only takes `--host`. It always binds 5000. Passing
`--port 5000` makes argparse error out and the cog process exits silently.
Symptom: container is "running", SSH tunnel opens fine, but `localhost:5000`
gives `ConnectionResetError` because nothing is listening on the remote
end. Tunnel forwards traffic to a dead port and SSH closes the channel.

Just don't pass `--port`.

### `ConnectionResetError` ≠ `ConnectionRefusedError`

If your local-port health-check loop catches only `URLError`,
`ConnectionRefusedError`, and `TimeoutError`, it crashes on the first
`ConnectionResetError`. Forwarding to a dead port through SSH gives reset,
not refused. Catch `OSError` (its parent) and keep polling.

### VPN exits get blackholed by some APIs

Anti-fraud blocking on commercial VPN exits hits SaaS APIs but leaves
public/marketing pages alone. Symptom: `console.vast.ai` returns 403 or TCP
times out, but `vast.ai` (marketing site) and `status.vast.ai` work fine.
Cloudflare also works. So it's not your network — it's the vendor blocking
your VPN exit's IP range at their API edge.

Fix: split-tunnel the API host outside your VPN. Most clients support
adding individual hosts. The rule typically resets when your VPN reconnects
to a different exit, so you may need to re-add it occasionally.

When debugging connection failures to a vendor's API, check your public IP
range against known commercial VPN provider ranges (Multacom, M247,
Datacamp, OVH, etc.) before assuming a vendor outage.

### Custom node + transformers version drift

Custom nodes pinned to specific transformers/diffusers/whatever versions
break when ComfyUI's base requirements move. Concrete example here:
`1038lab/ComfyUI-JoyCaption` calls
`image.resize(self.target_size, Image.Resampling.LANCZOS)`. The
`target_size` came from a transformers image-processor. Newer transformers
returns a `SizeDict` object, not an `(int, int)` tuple, and PIL's resize
raises `TypeError: 'SizeDict' object cannot be interpreted as an integer`.

Defensive practices:

- Pin custom-node revisions by SHA in `custom_nodes.json`, not by tag or
  ComfyUI Registry version.
- Pin transformers/diffusers/torch versions in `cog.yaml::python_packages`,
  even if they're not the bleeding edge.
- Where possible, design the workflow so a buggy node can be bypassed
  rather than blocking all inference. (Here: rewire the prompt source to a
  user-supplied text node, leaving JoyCaption orphaned.)

### ComfyUI executor pulls every output node, not just the one you care about

If you delete the connection from JoyCaption to your CLIPTextEncode but
leave a `PreviewAny` node still wired to the JoyCaption output, ComfyUI
will still execute JoyCaption to feed the preview. Output nodes
(`PreviewImage`, `PreviewAny`, `SaveImage`, etc.) are always evaluated.

To truly skip a subgraph: rewire *all* downstream consumers to a different
source, then the subgraph becomes orphaned and the executor skips it.

### `PreviewImage` output isn't reachable from Cog

`PreviewImage` writes to ComfyUI's `temp/` and reports `type: temp` in the
prompt history. You can't return a `Path` from those — Cog needs a stable
`output/` filename.

Convert your final-output node to `SaveImage` with a `filename_prefix`. The
result lands in `/opt/comfyui/output/` and `SaveImage` reports
`type: output` with a stable filename your predict step can return.

### CLIPLoaderGGUF scans `clip/`, not `text_encoders/`

GGUF text encoders go in `models/clip/`, not `models/text_encoders/`,
because that's where ComfyUI-GGUF's `CLIPLoaderGGUF` looks. The naming is
misleading — Qwen3 is a text encoder, not CLIP. Verify the scan path of
your loader nodes by reading their source, not by inferring from their
class name.

### VAE filename must match the workflow widget value

`VAELoader` reads a string filename from a dropdown widget. Whatever value
was saved in the workflow JSON (`"vae_name": "z_image_turbo_vae.safetensors"`)
must exist verbatim in `models/vae/`. If your source on Hugging Face is
named `vae/diffusion_pytorch_model.safetensors`, **rename on download** to
the workflow's expected filename. Don't change the workflow — the workflow
is your stable contract.

Same principle for any node that loads by filename: checkpoints, LoRAs,
VAEs, segmentation models. The download step is where the rename happens.

## Project structure that worked

```
project/
├── README.md                  # what + how, links to deeper docs
├── cog.yaml                   # build config (deps + custom nodes only; no weights)
├── predict.py                 # Cog entry point; calls download_weights in setup()
├── workflow_patch.py          # pure named-handle substitution; unit-tested
├── workflows/
│   ├── headswap.json          # the validated workflow with handles tagged
│   └── originals/             # gitignored; Civitai workflows etc.
├── weights.json               # URL/dest declarations, with auth tags
├── custom_nodes.json          # pinned custom-node SHAs (source of truth for cog.yaml)
├── scripts/
│   └── download_weights.py    # runtime fetcher
├── deploy/
│   ├── deploy.py              # vast.ai provisioning, tunnel, diagnostics, cost guardrails
│   ├── deploy.yaml            # filters + image refs
│   ├── predict_client.py      # local CLI to hit the cog HTTP API
│   └── requirements.txt
├── tests/
│   └── test_dry_run.py        # named-handle patcher tests, no Cog needed
├── docs/
│   ├── architecture-replication.md   # this file
│   └── session-state-*.md            # snapshot at each major milestone
├── .github/workflows/
│   └── build.yml              # cog build + push to GHCR
├── .env.example               # template; .env is gitignored
├── .gitignore                 # see below
└── LICENSE
```

`.gitignore` essentials:

```
# secrets
.env
*.tunnel.pid

# upstream workflows we don't redistribute
Character-Head-Swap*.json   # whatever the source filename pattern is
characterHeadSwap*

# weights cache (never commit; resolved at runtime)
models/
weights_cache/
loras/

# dev tooling state
__pycache__/
*.pyc
.cog/
.DS_Store
.venv/
.claude/
.dora/
```

## Starting a sibling project: copy list

When you spin up a second project with the same shape — same marketplace,
same Cog/ComfyUI base, different workflow — copy the workflow-agnostic
infrastructure verbatim and start fresh on everything else. After ~3–6
months of both running, look at what stayed identical and extract a shared
package then. Premature abstraction is worse than duplication. The trigger
to extract is a third consumer, not the second.

### Copy verbatim (workflow-agnostic)

| Path | Why |
|------|-----|
| `deploy/` (whole directory) | Marketplace provisioning, SSH endpoint selection, tunnel, diagnostics, cost guardrails — none of this depends on what the workflow does. |
| `scripts/download_weights.py` | Idempotent runtime fetcher with anonymous-HF + Civitai-Bearer auth. Reads any `weights.json` shape. |
| `.github/workflows/build.yml` | Cog build + GHCR push. Image tag conventions match across projects. |
| `.gitignore` | Same secret/cache patterns apply (`.env`, `models/`, source workflow file patterns). Adjust `Character-Head-Swap*.json` to your new workflow's filename pattern. |
| `.env.example` | Same env vars apply (`VAST_API_KEY`, `CIVITAI_TOKEN`, `GHCR_*`, `SSH_KEY_PATH`). |
| `docs/architecture-replication.md` (this file) | So the gotchas don't have to be relearned. Add a project-specific addendum at the bottom for what's different. |

### Start fresh (workflow-shaped)

| Path | Why |
|------|-----|
| `cog.yaml` | Different custom-node set, different python deps, different CUDA version maybe. Reuse the `run:` step structure (one node-clone per layer for cache hits) but rebuild the list. |
| `predict.py` | Different `predict()` signature (training takes a dataset dir, returns a `.safetensors`; inference takes an image, returns an image). Reuse the `setup()` skeleton (download_weights then spawn ComfyUI). |
| `workflow_patch.py` | The named-handle *pattern* transfers; the specific `HANDLE_*` constants don't. For a trainer: `DATASET_DIR`, `LORA_OUTPUT_PATH`, `TRAINING_STEPS`, `LEARNING_RATE`. |
| `weights.json` | Different model URLs, different dest paths. Same schema (with `auth` field). |
| `custom_nodes.json` | Different nodes for different workflows. Keep the schema. |
| `workflows/*.json` | Workflow JSON is workflow-specific by definition. |
| `tests/test_dry_run.py` | Handle-specific assertions; rewrite around the new handle set. Keep the structure (no Cog import; pure unit tests). |
| `deploy/deploy.yaml` | Image ref points to the new GHCR image. Filter values may need tuning (training likely needs more VRAM, longer `max_session_hours`, etc.). |
| `LICENSE` | Same license, but copyright year/holder may differ. |

### Adapt before reusing

| Path | What to change |
|------|----------------|
| `deploy/deploy.py::cmd_run` | The `predict_client.py` call signature, the cost-guardrail values for training-shaped workloads (e.g. `max_session_hours: 24` not `4`), and possibly the `health_check` URL if the trainer exposes a different ready signal. |
| `deploy/deploy.yaml` | Adjust GPU filters: training often needs more VRAM (24 GB+ for SDXL-class training, 48 GB+ for some cases) and tolerates higher hourly rates because runs are longer. |
| `deploy/predict_client.py` | Different request shape entirely. May want progress-streaming support for long-running training. |

### When to extract a shared library

Stop copying when you hit a third consumer or when the same change has to be
made in three places. At that point, extract `deploy/`,
`scripts/download_weights.py`, and the named-handle pattern into a
`cog-<family>-common` package. Versioning options, cheapest first:

- **Git submodule** — zero infra, but submodule UX is rough.
- **Private PyPI / GitHub-installed package** — clean, but you need somewhere
  to host it.
- **Vendored copy with a `make sync-common` target** — pragmatic middle
  ground if both repos are yours and you can tolerate manual sync.

Don't do this on day one. The shape of "what's truly shared" only becomes
clear after the second project has lived a few months on its own.

## Cost-discipline patterns

When iterating against a paid GPU host, every careless minute costs money.
Patterns that paid back:

- **Server-side self-destruct timer**, armed at provision time: SSH a
  `setsid -f sh -c 'sleep 14400; curl -X DELETE …'` onto the host. If your
  laptop dies, the instance still suicides on schedule.
- **Local atexit cleanup**: register a `vastai destroy instance -y …`
  handler so Ctrl-C in your provisioning script tears down the host.
- **Pre-ready health-check timeout**: don't let a stuck setup() rack up
  hours of charges. 12 minutes is enough for a 10 GB weight download on a
  fast host; bail and pick a different offer if it doesn't beat that.
- **Refuse to spawn until you have a workflow that runs**: the temptation
  to "just spin up an instance and figure it out" leads to multi-hour
  debugging sessions on the meter. Validate as much as possible locally
  (the dry-run pattern) first.
- **Use `tunnel` to recover from network blips, not `run`**: `run` spins
  up a new instance. `tunnel` reattaches to the existing one. After a VPN
  reconnect or laptop sleep, `tunnel` is what you want.

## What I'd do differently next time

1. **Set up a recurring secrets scanner before any commits.** Even though
   we got lucky and nothing leaked into git, the live tokens did leak into
   the conversation transcript via `pget` URL echoing and a vastai CLI
   stack trace. Recurring scanner catches both `?token=` patterns in
   committed code and enforces the Bearer-header convention.

2. **Validate workflow runs against the upstream image before committing
   to a custom one.** The upstream `vastai/comfy:latest` image has all the
   networking and SSH plumbing already working. Spawn that, manually drop
   the workflow + weights in, prove the workflow itself runs, *then* build
   your custom Cog wrapper. Half the SSH/tunnel debugging would have been
   shortcut by this.

3. **Pin everything immediately, not "before production".** Custom-node
   SHAs, transformers version, torch version, every weight commit_sha.
   "I'll pin it later" became "we hit the unpinned-version-broke incident
   before we pinned anything."

4. **Treat `cog COPY-after-run` as a known constraint and design around
   it from the start.** Don't try to be clever in `cog.yaml`'s `run:`
   steps; do everything that needs `/src` access at runtime in `setup()`
   instead.

5. **Document the named-handle convention in the predictor's docstring,
   not just in the README.** It's the load-bearing contract for the whole
   API; future-you reading `predict.py` should see it immediately.
