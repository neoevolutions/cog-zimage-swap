# Session state — 2026-04-29

End-of-session snapshot of what's been validated, what's captured in source, and what still needs to be baked into the image before the project can be respawned cleanly from `ghcr.io/neoevolutions/cog-zimage-swap:phase0`.

## Validation milestone reached

The headswap workflow ran end-to-end on a vast.ai instance and produced a final head-swap image (node 984 "Face/Head Swap Output" — `ComfyUI_temp_kdvmk_00001_.png` in ComfyUI's history).

Pipeline confirmed working:

- **SAM3** segmentation: face mask top score 0.947, hair mask top score 0.945
- **Qwen3-4B GGUF** text encoder loaded (3.2GB on GPU via CLIPLoaderGGUF, type=lumina2)
- **Z-Image Turbo GGUF** diffusion model loaded
- **d4ra character LoRA** applied via Power Lora Loader (rgthree)
- **Z-Image VAE** encode/decode
- **Inpaint Crop + Stitch** for final compositing
- **JoyCaption fully bypassed** (incompatible with current transformers version; `image.resize(SizeDict, ...)` crash)

Test instance: `35794510` (kept alive for further iteration; ssh `root@142.169.249.42:12590`).

## ✅ Captured in source

### `deploy/deploy.py`

- **`vastai create instance` typo bypass** — calls `vastai.sdk.VastAI.create_instance()` directly with `runtype="ssh_direct ssh_proxy"` (correct spelling). The CLI in `vastai==1.0.7` builds `"ssh_direc ssh_proxy"` (missing `t`); the backend silently drops the unrecognized token and falls back to gateway-proxy mode.
- **`select_ssh_endpoint()`** — prefers `public_ipaddr` + `ports["22/tcp"]` (direct host) over `ssh_host`/`ssh_port` (vast.ai gateway). Bypasses the gateway's per-key cache lag bug.
- **SSH options for ephemeral hosts** — `StrictHostKeyChecking=no`, `UserKnownHostsFile=/dev/null`, `IdentitiesOnly=yes`, `LogLevel=ERROR`. Required because gateway ports are recycled across instances with different host keys, and `IdentitiesOnly` prevents `MaxAuthTries` exhaustion when ssh-agent has multiple keys.
- **`_onstart_cmd()`** — removed `--port 5000` (cog v0.10+ rejects this flag; it always binds 5000).
- **`diagnose_container()`** — SSH-based remote diagnostic (vast.ai's `execute` API rejects non-whitelisted commands with HTTP 400).
- **`health_check()`** — tolerates `ConnectionResetError`/`OSError`, runs cog-state diagnostic at the 60s mark and on final timeout.

### `workflows/headswap_test.json`

Live-validated workflow with:
- Node 944 (`CLIPTextEncode`) — `text` input rewired from `["1070", 0]` → `["1071", 0]` (skip JoyCaption).
- Node 961 (`PreviewAny` "Prompt Preview") — `source` input rewired from `["1070", 0]` → `["1071", 0]` (so JoyCaption isn't pulled in transitively as an output dependency).
- Node 1086 (`Power Lora Loader`) — `lora_1.lora` set to `d4ra.safetensors`.
- Nodes 1017, 1018 (JoyCaption), 1070 (Text Concatenate) — orphaned, won't execute.

## ✅ NOW captured — applied at end of session

Everything below was added before destroy, so a fresh respawn from a freshly built image will work without any manual fixes. Commit and push to trigger the GHA build.

### `cog.yaml`

- Added `llama-cpp-python` to `python_packages`.
- Decided NOT to bake weights at build time. cog COPYs `/src` into the container *after* `run:` completes, so build steps can't reference `scripts/download_weights.py` or `weights.json`. Build-time baking would also require BuildKit secret handling for Civitai auth. Runtime download (next bullet) avoids both, keeps the image ~2GB instead of ~12GB, and lets weights change without rebuilds. The trade-off: first inference on a fresh instance pays ~3-5 minutes for downloads.

### `scripts/download_weights.py` (NEW)

Reads `weights.json`, downloads each entry to its `dest`, skips files already present (size-checked against `approx_size_gb` so half-finished downloads get refetched). Uses `pget` if available, else `curl`. Civitai auth via `Authorization: Bearer $CIVITAI_TOKEN` header — never query string, so the token doesn't leak into redirect chains, server logs, or pget output. Sentinel URLs (`MULTI_FILE_BUNDLE`, `TODO_FIND_HF_URL`, anything matching `TODO_*`) are skipped.

### `predict.py`

`setup()` now invokes `scripts/download_weights.py` before launching ComfyUI. Idempotent. If downloads fail, ComfyUI starts anyway — the workflow surfaces missing-file errors as clear validation messages rather than a silent hang.

### `deploy/deploy.py`

`create_instance` now forwards `CIVITAI_TOKEN` (read from local `.env`) into the vast.ai container's env. The token rides in the create-instance API request body. If you don't want the token in vast.ai's API logs, just unset it in `.env` and download Civitai weights manually after the instance is up.

### `weights.json`

- Fixed VAE URL: `https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/resolve/main/vae/diffusion_pytorch_model.safetensors` (was a 404 path). Renamed on download to `z_image_turbo_vae.safetensors` to match what the workflow's VAELoader nodes expect.
- Resolved SAM3 URL: `https://huggingface.co/yolain/sam3-safetensors/resolve/main/sam3-fp16.safetensors` (from `ComfyUI-Easy-Sam3` README).
- Fixed SAM3 dest path: `models/sam3/` (was `models/sam/`).
- Fixed Qwen3-4B dest path: `models/clip/Qwen3-4B-UD-Q5_K_XL.gguf` (was `models/text_encoders/`). CLIPLoaderGGUF scans `clip/`.
- Added `d4ra` Civitai entry with `auth: "civitai"` flag.
- Marked JoyCaption deferred via `MULTI_FILE_BUNDLE` sentinel; download_weights.py skips it.

### `workflows/headswap.json`

Replaced placeholder. Now contains the validated workflow with:

- Named handles tagged in `_meta.title`:
  - 958 → `INPUT_TARGET_IMAGE`
  - 1071 → `INPUT_PROMPT`
  - 1022 → `SEED_INPUT`
  - 1086 → `LORA_LOADER`
  - 984 → `OUTPUT_IMAGE`
- Node 984 converted from `PreviewImage` to `SaveImage` with `filename_prefix: "headswap"` so output lands in `/opt/comfyui/output/` and `predict.py` can return it as a `Path`.
- JoyCaption nodes (1017, 1018, 1070) orphaned — no consumers, executor skips them.
- LoRA filename set to `d4ra.safetensors`.

All 7 dry-run tests in `tests/test_dry_run.py` pass against the new workflow.

## ❌ User-environment, not project

These can't be captured in source — must be present in each new shell/machine.

- **VPN split-tunnel rule for `console.vast.ai`** — vast.ai blackholes known commercial VPN exits at the API edge (`HTTP 403`); marketing site, status page, and Cloudflare all work, only the API is blocked. Required IPs (currently): `100.28.157.160`, `54.173.192.99`, `3.86.37.17`. Rule resets when VPN reconnects to a different exit.
- **`ssh-add ~/.ssh/id_rsa_vastai`** — the key has a passphrase. Loading into `ssh-agent` once per terminal (or via macOS Keychain integration: `UseKeychain yes` + `AddKeysToAgent yes` in `~/.ssh/config`, then `ssh-add --apple-use-keychain ~/.ssh/id_rsa_vastai` once) lets `ssh -N -f` background tunnels work. Without the agent, `Permission denied (publickey)` because batch-mode ssh can't prompt for passphrase.
- **`CIVITAI_TOKEN` in `.env`** — already added. Required for `d4ra.safetensors` download (and any other Civitai assets that require login). Must be passed as a GitHub Actions secret for image build.

## Sequence to destroy and respawn cleanly

1. **Commit and push** the changes from this session:
   - `cog.yaml`, `weights.json`, `workflows/headswap.json`, `predict.py`, `deploy/deploy.py`, `scripts/download_weights.py` (new), `docs/session-state-2026-04-29.md` (this file)
   - The push triggers GHA build of `ghcr.io/neoevolutions/cog-zimage-swap:phase0` (~12 min).
2. **Destroy the live instance**: `python3 deploy/deploy.py destroy 35794510` (or whatever the live id is when you act on this).
3. **Respawn**: `python3 deploy/deploy.py run`.
   - Cold-start budget: image pull ~3-5 min + weight downloads ~3-5 min + ComfyUI boot ~30s = up to ~10 min before health-check passes.
   - `deploy.yaml` has `provision_timeout_min: 30` and `health_check_timeout_min: 12` — comfortable for this.
4. **Verify**: queue the same workflow you just validated. Output should land in `/opt/comfyui/output/headswap_*.png` (now `SaveImage`, not preview). The cog API at `localhost:5000` can also drive it: `curl -X POST http://localhost:5000/predictions -d '{"input": {"target_image": "...", "prompt": "...", "seed": 42}}'`.

If anything fails on respawn, `deploy.py` runs the SSH-based `diagnose_container` automatically and dumps cog/SSH state.

## Open known issues (not blockers)

- `vast.ai` host's NAT does not route the ports vast.ai mapped externally — public URLs fail probe. SSH tunnel (the actual access path) is unaffected. Different host or ignore.
- `ComfyUI-JoyCaption` (1038lab) has a transformers-version bug (`SizeDict` not iterable as int). Pin or patch before re-enabling.
- `vast.ai`'s `execute` API is whitelisted; arbitrary commands return HTTP 400 "Invalid command given". Diagnostic must use SSH directly (already implemented).
- `weights.json` `commit_sha` pinning is still scaffolding-only.
