# cog-zimage-swap

Replicate-style API for the Z-Image Turbo + SAM3 character headswap workflow,
packaged as a Cog container, deployable to vast.ai cloud GPUs.

See `PLAN.md` for full architecture, phasing, and rationale.
Workflow credit: [Character Head Swap V1 — Civitai](https://civitai.com/models/2478306/character-head-swap-v1-low-vram-or-comfyui-z-image-turbo-sam3).

## Phase 0 status

This is the local-scaffolding milestone — image structure only, no GPU spend.

- `predict.py` — Cog predictor; spawns headless ComfyUI, posts to `/prompt`
- `workflow_patch.py` — pure named-handle substitution (Cog-free, unit-testable)
- `workflows/headswap.json` — **PLACEHOLDER**, see "Workflow re-export needed"
- `custom_nodes.json` — pinned custom-node packs; some entries marked `TODO_PIN_SHA`
- `weights.json` — pinned HF weight URLs; pinned to `main`, marked `TODO_PIN_SHA`
- `cog.yaml` — Cog build config (CUDA 12.1, Python 3.11, ComfyUI from source)
- `tests/test_dry_run.py` — named-handle substitution tests, runnable on Mac
- `test.jpg` — 512×512 placeholder for dry-run validation

## The named-handle contract

The workflow API surface is decoupled from the graph by `_meta.title`:

| Handle title          | Node type                 | Field patched          |
|-----------------------|---------------------------|------------------------|
| `INPUT_TARGET_IMAGE`  | LoadImage                 | `inputs.image`         |
| `INPUT_PROMPT`        | Text Multiline (or any)   | `inputs.text`          |
| `SEED_INPUT`          | KSampler                  | `inputs.seed` (must use `control_after_generate = fixed`) |
| `LORA_LOADER`         | Power Lora Loader (rgthree) | `inputs.lora_*.{on,strength,lora}` (bypass-by-default) |
| `OUTPUT_IMAGE`        | SaveImage                 | sentinel — output is collected from `output/` |

`predict.patch_workflow` rejects any workflow missing one of the four required
handles (`OUTPUT_IMAGE` is required; `LORA_LOADER` is optional). The LLM editing
the workflow can rearrange the graph as long as the titles survive.

## Workflow re-export needed (blocker before Phase 1 inference)

The Civitai source workflow ships in **UI format**, not the **API format**
ComfyUI's `/prompt` endpoint accepts. The source file is **not** redistributed
in this repo — download it yourself, then re-export.

1. Download the workflow from
   [Civitai — Character Head Swap V1](https://civitai.com/models/2478306/character-head-swap-v1-low-vram-or-comfyui-z-image-turbo-sam3).
2. Open it in a local ComfyUI UI.
3. Retitle the parameter-bearing nodes (right-click → Properties → Title):
   - Node 958 "Source Image" → `INPUT_TARGET_IMAGE`
   - Node 1071 "Add important extra info..." → `INPUT_PROMPT`
   - Node 1022 "Head Swap Sampler" → `SEED_INPUT`. Also flip
     `control_after_generate` to `fixed`.
   - Node 1086 "Power Lora Loader (rgthree)" → `LORA_LOADER`
   - Node 984 "Face/Head Swap Output" → `OUTPUT_IMAGE`. Also change the node
     type from `PreviewImage` to `SaveImage`.
4. Click **Save (API Format)**.
5. Replace `workflows/headswap.json` with the export.

(Files matching `Character-Head-Swap*.json` and `characterHeadSwap*` are
gitignored so a local copy won't accidentally land in a commit.)

## Dry-run

Phase 0 exit criterion. With Cog and Docker:

```
COG_SKIP_COMFYUI=1 cog predict -i target_image=@test.jpg -i prompt="young man with glasses" -i dry_run=true
```

Without Cog (Mac with stock Python — recommended for fast iteration):

```
python3 tests/test_dry_run.py
```

Both paths exercise the named-handle patcher against the placeholder workflow.

## Phase 1 — vast.ai deploy + GHA build

### Build (GitHub Actions)

Local `cog build` on Apple Silicon hits a Rosetta 2 + Docker Desktop RAM
ceiling and tends to hang. The image build runs on `ubuntu-latest` (native
amd64) instead.

Trigger:

- **Manual:** Actions tab → "build-and-push" → Run workflow → tag `phase0`
- **On tag:** `git tag v0.1.0 && git push --tags` → builds and pushes
  `ghcr.io/neoevolutions/cog-zimage-swap:0.1.0`

The workflow runs `cog build` and `docker push`. `GITHUB_TOKEN` already has
`write:packages` so no PAT setup needed for CI itself. For local pushes,
you'll need a PAT — see `.env.example`.

### Deploy (vast.ai)

One-time setup:

```
cp .env.example .env             # populate VAST_API_KEY
pip install -r deploy/requirements.txt
python deploy/deploy.py setup    # registers ~/.ssh/id_rsa_vastai.pub with vast.ai
```

Daily loop:

```
python deploy/deploy.py find                # dry-run; lists top 5 matching offers, free
python deploy/deploy.py run                 # provisions, tunnels localhost:5000, arms guardrails
python deploy/predict_client.py --image test.jpg --prompt "young woman with glasses"
# Ctrl-C in the run terminal destroys the instance.
python deploy/deploy.py destroy <id>        # if the run script died but the instance lives
python deploy/deploy.py status              # list active instances
```

Filters and image refs live in `deploy/deploy.yaml`. Cost guardrails (server-
side self-destruct timer, pre-ready health-check, weight-download timeout)
are all wired in `deploy.py`.

## Outstanding follow-ups (still blocking real inference)

- **Re-export the workflow as API format.** The placeholder
  `workflows/headswap.json` has the named handles but no real graph — POSTing
  it to ComfyUI will fail. See "Workflow re-export needed" above.
- **Pin `TODO_PIN_SHA` entries** in `custom_nodes.json` and `weights.json`.
  GitHub commit SHAs: `git ls-remote <repo> <tag>`. HF revision SHAs:
  `https://huggingface.co/api/models/<repo>/refs`.
- **Verify the VAE filename match.** Workflow references
  `z_image_turbo_vae.safetensors`; PLAN.md sources Tongyi-MAI's `ae.safetensors`.
- **Locate SAM3 weight source** — PLAN.md does not specify a HF repo for
  `sam3-fp16.safetensors`.
- **Decide JoyCaption caching strategy.** ~8 GB multi-file bundle.
- **First GHA run** to land an image at `ghcr.io/neoevolutions/cog-zimage-swap:phase0`
  before `deploy.py run` will succeed. After the first push, change the GHCR
  package visibility to public so vast.ai can pull without auth.
