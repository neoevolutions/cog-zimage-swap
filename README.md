# cog-zimage-swap

Replicate-style API for the Z-Image Turbo + SAM3 character headswap workflow,
packaged as a Cog container, deployable to vast.ai cloud GPUs.

Workflow credit: [Character Head Swap V1 — Civitai](https://civitai.com/models/2478306/character-head-swap-v1-low-vram-or-comfyui-z-image-turbo-sam3).

- [`docs/INITIAL_PLAN.md`](docs/INITIAL_PLAN.md) — the original architectural plan and rationale
- [`docs/architecture-replication.md`](docs/architecture-replication.md) — lessons learned, patterns to reuse, gotchas to avoid
- [`docs/session-state-2026-04-29.md`](docs/session-state-2026-04-29.md) — snapshot at end of validation milestone

## Status

End-to-end head-swap inference validated on a vast.ai RTX 3090. The full
pipeline runs: SAM3 face+hair segmentation → Qwen3-4B GGUF text encoder →
Z-Image Turbo GGUF + character LoRA via Power Lora Loader → VAE decode →
inpaint crop+stitch.

| Component | State |
|-----------|-------|
| `predict.py` | Cog predictor; spawns headless ComfyUI, runs `scripts/download_weights.py` in setup() |
| `workflow_patch.py` | Pure named-handle substitution (Cog-free, unit-testable) |
| `workflows/headswap.json` | Validated workflow with named handles tagged |
| `scripts/download_weights.py` | Runtime weight fetcher (pget+curl, Civitai Bearer auth) |
| `weights.json` | URL/dest declarations for runtime download |
| `custom_nodes.json` / `cog.yaml` | Pinned custom-node revisions; `llama-cpp-python` for GGUF |
| `deploy/deploy.py` | vast.ai provisioning, SSH tunnel, cost guardrails, diagnostics |
| `tests/test_dry_run.py` | Named-handle substitution tests, runnable on Mac without Cog |
| JoyCaption | **Deferred** — node has a transformers-version bug (`SizeDict` not iterable as int). Workflow bypasses it; user-supplied prompt feeds CLIPTextEncode directly. |

## The named-handle contract

The workflow API surface is decoupled from the graph by `_meta.title`. The LLM
or human editing the workflow can rearrange the graph as long as the titles
survive. `predict.patch_workflow` rejects any workflow missing the four
required handles.

| Handle title          | Node type                 | Field patched          | Required |
|-----------------------|---------------------------|------------------------|----------|
| `INPUT_TARGET_IMAGE`  | LoadImage                 | `inputs.image`         | yes      |
| `INPUT_PROMPT`        | Text Multiline (or any)   | `inputs.text`          | yes      |
| `SEED_INPUT`          | KSampler                  | `inputs.seed`          | yes      |
| `LORA_LOADER`         | Power Lora Loader (rgthree) | `inputs.lora_*.{on,strength,lora}` (bypass-by-default) | no |
| `OUTPUT_IMAGE`        | SaveImage                 | sentinel — output is collected from `output/`         | yes |

## How the image works

Weights are **not baked in at build time**. cog COPYs `/src` into the
container *after* `run:` steps complete, so the build can't reference
`scripts/download_weights.py` or `weights.json`. Instead:

1. `cog build` produces a ~2 GB image with cog + ComfyUI + custom nodes only.
2. On instance startup, `predict.py` `setup()` invokes
   `scripts/download_weights.py`, which reads `weights.json` and pulls each
   entry to its `dest` (idempotent — skips files already present).
3. Anonymous HF entries fetch with no credentials. Civitai-gated entries
   require `CIVITAI_TOKEN` in the container env (forwarded by `deploy.py`
   from the local `.env`). Token is sent as `Authorization: Bearer …`, never
   as `?token=…` — query strings leak into redirect chains, server logs, and
   `pget` output.
4. ComfyUI starts. Cold-start budget on a fresh vast.ai 3090: image pull
   3–5 min + weight downloads 3–5 min for ~10 GB + ComfyUI boot ~30s.

JoyCaption stays deferred until a non-buggy revision is available. The
workflow rewires `CLIPTextEncode` to read the user prompt directly from
`Text Multiline`, so the JoyCaption nodes are orphaned and skipped at
execution time.

## Workflow editing

The committed `workflows/headswap.json` was exported from the live ComfyUI
running on vast.ai with the named handles tagged and node 984 converted from
`PreviewImage` to `SaveImage`. To edit it:

1. Drag the file onto a ComfyUI canvas (any instance with the same custom
   nodes installed). Open `http://localhost:8188` after `deploy.py run`.
2. Make changes. Keep the `_meta.title` values intact on the four required
   handle nodes.
3. **Settings → Enable Dev mode Options** → top menu **Workflow → Export
   (API)**. Replace `workflows/headswap.json` with the export.
4. `python3 tests/test_dry_run.py` should still pass — it verifies all
   required handles are present.

The original Civitai source file is **not** redistributed in this repo.
`Character-Head-Swap*.json` is gitignored so a local copy won't accidentally
land in a commit.

## Dry-run

With Cog and Docker:

```
COG_SKIP_COMFYUI=1 cog predict -i target_image=@test.jpg -i prompt="young man with glasses" -i dry_run=true
```

Without Cog (Mac with stock Python — recommended for fast iteration):

```
python3 tests/test_dry_run.py
```

Both paths exercise the named-handle patcher against the workflow JSON.

## Build + deploy

### Build (GitHub Actions)

Local `cog build` on Apple Silicon hits a Rosetta 2 + Docker Desktop RAM
ceiling and tends to hang. The image build runs on `ubuntu-latest` (native
amd64) instead.

Triggers:

- **Manual:** Actions tab → "build-and-push" → Run workflow → tag `phase0`
- **On tag:** `git tag v0.1.0 && git push --tags` → builds and pushes
  `ghcr.io/neoevolutions/cog-zimage-swap:0.1.0`

The workflow runs `cog build` and `docker push`. `GITHUB_TOKEN` already has
`write:packages` so no PAT setup is needed for CI itself. For local pushes,
you'll need a PAT — see `.env.example`. `CIVITAI_TOKEN` is **not** needed
at build time (weights download at runtime).

### Pre-commit secrets scanning

Before your first commit on a fresh clone, install the pre-commit hook so
secrets never reach git:

```
brew install pre-commit gitleaks   # or: pip install pre-commit + gitleaks binary
pre-commit install                 # writes .git/hooks/pre-commit
```

Hook config: `.pre-commit-config.yaml`. Rules: gitleaks defaults plus
`.gitleaks.toml` (covers Civitai/vastai URL-query-string token leaks
specifically — see [`docs/architecture-replication.md`](docs/architecture-replication.md)
on why query-string credentials are dangerous). The hook runs on every
commit regardless of where it's triggered (Claude Code, terminal, IDE).

Override only when you know the finding is a false positive:
`git commit --no-verify`. Otherwise, fix the leak.

### Deploy (vast.ai)

One-time setup:

```
cp .env.example .env             # populate VAST_API_KEY (and optionally CIVITAI_TOKEN)
pip install -r deploy/requirements.txt
python deploy/deploy.py setup    # registers ~/.ssh/id_rsa_vastai.pub with vast.ai
```

If your `~/.ssh/id_rsa_vastai` has a passphrase: load it into your shell's
ssh-agent once per terminal (`ssh-add ~/.ssh/id_rsa_vastai`). Otherwise
batch-mode SSH (`-N -f`) can't unlock it and `deploy.py` fails with
`Permission denied (publickey)`. macOS Keychain integration via
`UseKeychain yes` + `AddKeysToAgent yes` in `~/.ssh/config` makes this
once-and-for-all.

Daily loop:

```
python deploy/deploy.py find                # dry-run; lists top 5 matching offers, free
python deploy/deploy.py run                 # provisions, tunnels localhost:5000+8188, arms guardrails
python deploy/predict_client.py --image test.jpg --prompt "young woman with glasses"
# Ctrl-C in the run terminal destroys the instance.
python deploy/deploy.py tunnel              # (re)open SSH tunnel after VPN reconnect / IP change / disconnect
python deploy/deploy.py tunnel --lan        # bind tunnel on 0.0.0.0 so other devices can reach it
python deploy/deploy.py destroy <id>        # if the run script died but the instance lives
python deploy/deploy.py status              # list active instances
```

Filters and image refs live in `deploy/deploy.yaml`. Cost guardrails
(server-side self-destruct timer, pre-ready health-check, weight-download
timeout) are wired in `deploy.py`.

The `tunnel` subcommand re-resolves the SSH endpoint from scratch each call
so it picks up vast.ai-side host migrations and direct-vs-proxy changes
automatically. Use it when the original tunnel goes silent — most commonly
after a VPN reconnect.

## VPN / network gotchas

- **vast.ai's API (`console.vast.ai`) blackholes known commercial VPN exits**
  at the network edge — public site, status page, and Cloudflare all work,
  only the API returns silent timeouts or 403. If `deploy.py` calls hang or
  return 403, add a split-tunnel rule for `console.vast.ai` (or its A
  records) in your VPN client. The rule resets when the VPN reconnects to a
  different exit.
- **vast.ai's SSH gateway has a per-key cache lag bug**, so we go direct to
  the host's public IP via `select_ssh_endpoint` whenever possible. This
  needs offers with `static_ip=true` and a populated port-22 mapping —
  `deploy.yaml` already filters for this.
- **vastai 1.0.7 has a typo bug** in `--direct` runtype (writes `ssh_direc`
  instead of `ssh_direct`), which makes the API silently fall back to proxy
  mode. `deploy.py::create_instance` bypasses the buggy CLI by calling the
  vastai SDK directly with the correct runtype string.
- **`Permission denied (publickey)` even with a registered, matching key**
  is the per-host `authorized_keys` cache lag. `actual_status=running` in
  vast.ai's metadata also doesn't mean sshd has bound yet — there's a
  30–90s window where the port refuses TCP. `deploy.py` handles both:
  it `POST`s to `/api/v0/instances/<id>/ssh/` to force-sync the key to
  the host, then runs a synchronous `ssh ... echo ok` probe with up to
  180s of retry tolerating both `Connection refused` and `Permission
  denied`. Full write-up:
  [`docs/architecture-replication.md`](docs/architecture-replication.md)
  — sections "wait_for_ssh returning early" and "Per-host authorized_keys
  cache lag".

## Open follow-ups

- **Pin `TODO_PIN_SHA` entries** in `custom_nodes.json` and `weights.json`.
  GitHub commit SHAs: `git ls-remote <repo> <ref>`. HF revision SHAs:
  `https://huggingface.co/api/models/<repo>/revision/main`.
- **JoyCaption** — pin or patch `1038lab/ComfyUI-JoyCaption` to fix the
  `image.resize(SizeDict, ...)` TypeError, then re-enable the auto-prompt
  branch in the workflow.
- **Workflow handle title for the seed node** — currently using a free-form
  `INPUT_PROMPT` on `Text Multiline`. Confirm `control_after_generate`
  semantics on KSampler when the seed is patched at request time.

For the full lessons-learned writeup (decisions made, dead ends explored,
patterns worth reusing), see
[`docs/architecture-replication.md`](docs/architecture-replication.md).
