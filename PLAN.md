# Cog + Z-Image Headswap Project Plan

## Project goal

Ship a Replicate-style API for the Z-Image Turbo + SAM3 character headswap workflow,
packaged as a Cog container, deployable to vast.ai cloud GPUs via a YAML-driven
CLI deploy script. ComfyUI runs headless inside the container; callers never see it.

**Reference workflow:** [Character Head Swap V1 — Civitai](https://civitai.com/models/2478306/character-head-swap-v1-low-vram-or-comfyui-z-image-turbo-sam3)
([RunComfy mirror](https://www.runcomfy.com/comfyui-workflows/z-image-turbo-headswap-for-characters-in-comfyui-seamless-face-replacer))

## Scope clarification (important — read first)

This workflow is **prompt-driven head replacement**, not face-A-onto-person-B
swapping. Identity comes from the prompt (auto-augmented by JoyCaption) and an
optional character LoRA. There is **no reference-face image input**. To get
specific-identity output, a LoRA must be trained on that identity (Phase 2).

If you decide later that a face-image-driven approach (PuLID / InstantID) is
what you actually want, the cog-comfyui + vast.ai plumbing built here is
workflow-agnostic and will carry over. Phase 3 includes that off-ramp.

---

## Architecture decisions (locked)

| Decision | Choice | Rationale |
|---|---|---|
| **Engine** | Run ComfyUI headless inside Cog | Re-implementing in raw `diffusers` would mean re-porting `InpaintCropImproved`, `DifferentialDiffusion`, SAM3 nodes, etc. Headless ComfyUI is the same outcome at 5% the cost. |
| **Base repo** | Fork [`replicate/cog-comfyui`](https://github.com/replicate/cog-comfyui) | Already solves: ComfyUI lifecycle, weight staging via `pget`, custom-node install at build time, output collection. Saves ~1 week of plumbing. |
| **API shape** | `target_image: File`, `prompt: str`, `seed: int` | Matches what the workflow actually accepts. JoyCaption auto-augments the prompt. |
| **Workflow JSON** | API-format, baked in at build, optional `workflow_override` input at request time | LLM-editable. Hybrid approach allows iteration without image rebuild. |
| **Parameter injection** | Named-handle convention via `_meta.title` | Decouples graph structure from API contract. The LLM can rearrange the graph as long as named handles survive. |
| **Weights** | Runtime download from Hugging Face, cached on host disk | Image stays small (~3 GB instead of 20 GB). First boot ~5 min; subsequent boots on same host are instant. Migrate to R2 mirror if HF turns flaky. |
| **Hosting** | vast.ai, Mode 2 (spin-up per session, on-demand, not interruptible) | Pay only for usage. SSH tunnel for API access. |
| **API exposure** | SSH tunnel only (no public port) | No auth needed; nothing public-facing. |
| **GPU** | RTX 3090 / 4090, 24 GB VRAM | 8 GB "minimum" claim is misleading once full stack is loaded. 24 GB gives headroom for Phase 2 LoRAs and ControlNets. |
| **Output** | PNG, single composite, sync return as `cog.Path` | Lossless for testing; streams over SSH tunnel in <1 s. |
| **Dev loop** | Strategy 1: dry-run locally on Mac, real inference on a single warm vast.ai instance | Zero new infra; forces a useful `--dry-run` mode. |
| **Registry** | GitHub Container Registry (ghcr.io) | Free for public; no Docker Hub rate limits. |

---

## Weight inventory (~17 GB total)

Download at runtime via `pget` from Hugging Face. **Pin every URL to a specific
commit SHA** (not `/resolve/main/`) — Z-Image and its tooling are still moving
weekly.

| Model | File | Source | Approx size |
|---|---|---|---|
| Z-Image Turbo (4-bit GGUF) | `z_image_turbo-Q5_K_M.gguf` | `jayn7/Z-Image-Turbo-GGUF` | 4.5 GB |
| Qwen3-4B text encoder (GGUF) | `Qwen3-4B-UD-Q5_K_XL.gguf` | `unsloth/Qwen3-4B-GGUF` | 3 GB |
| Z-Image VAE | `ae.safetensors` | `Tongyi-MAI/Z-Image-Turbo` | 300 MB |
| SAM3 segmentation | `sam3-fp16.safetensors` | (per Civitai workflow) | 1.5 GB |
| JoyCaption auto-prompt | (multi-file) | `1038lab/ComfyUI-JoyCaption` deps | ~8 GB |
| Upscaler | `2xLexicaRRDBNet_Sharp.pth` | `Thelocallab` HF repo | 70 MB |

## Custom node inventory

Pin every entry in `custom_nodes.json` to a **commit SHA**, not `main`. The
workflow author already noted SAM3 nodes needed patching; assume churn.

| Node pack | Repo | Used for |
|---|---|---|
| ComfyUI-GGUF | `city96/ComfyUI-GGUF` | Loading Z-Image and Qwen3 in GGUF format |
| comfyui-easy-sam3 | `yolain/ComfyUI-Easy-Sam3` | Head/face segmentation |
| comfyui-joycaption | `1038lab/ComfyUI-JoyCaption` | Auto-caption of input image |
| was-node-suite-comfyui | `WASasquatch/was-node-suite-comfyui` | Mask utilities |
| rgthree-comfy | `rgthree/rgthree-comfy` | Power Lora Loader |

---

## Phase 0 — Local scaffolding (no GPU spend)

**Goal:** Cog project that builds locally on Mac. No inference yet.

### Tasks

1. Fork `replicate/cog-comfyui` into this directory:
   ```
   git clone https://github.com/replicate/cog-comfyui.git .
   ```
   Strip Replicate-specific defaults but keep the bones (`predict.py` skeleton,
   `cog.yaml`, `custom_nodes.json`, weights manifest helpers, `pget` machinery).

2. **One-time workflow conversion:**
   - Install ComfyUI locally.
   - Open the Civitai workflow.
   - Click "Save (API Format)".
   - In the UI, edit each parameter-bearing node's title to add a named handle:
     - LoadImage for the target → title `INPUT_TARGET_IMAGE`
     - Prompt CLIPTextEncode (or equivalent) → title `INPUT_PROMPT`
     - KSampler → title `SEED_INPUT` (and set `control_after_generate = fixed`)
     - LoRA loader → title `LORA_LOADER` (used by Phase 2)
   - Re-export API JSON.
   - Commit to `workflows/headswap.json`.

3. **Implement `predict.py`:**
   - `setup()`: spawn ComfyUI subprocess, wait for `/system_stats` to return 200.
   - `predict(target_image: Path, prompt: str = "", seed: int = -1, workflow_override: dict | None = None) -> Path`:
     - Load workflow JSON (override or default).
     - Find named-handle nodes by `_meta.title`, patch their `inputs`.
     - Copy `target_image` into ComfyUI's `input/` directory.
     - POST to `localhost:8188/prompt`, poll `/history/{prompt_id}` until done.
     - Read output PNG from ComfyUI's `output/` directory.
     - Return `Path(output_png)`.
   - Add `--dry-run` flag (env var or input) that does substitution + validation
     and prints the patched workflow JSON without calling ComfyUI.

4. **Pin all dependencies:**
   - `custom_nodes.json` → all entries pinned to commit SHAs.
   - `weights.json` (or whatever `cog-comfyui` calls it) → all URLs pinned to
     HF revision SHAs: `https://huggingface.co/<repo>/resolve/<sha>/<file>`.

5. **Validation:**
   - `cog build` succeeds on Mac (Docker Desktop required).
   - `cog predict --dry-run -i target_image=@test.jpg -i prompt="young man with glasses"`
     prints valid patched JSON with all named handles substituted.

### Exit criteria

- Image builds locally.
- Dry-run validates a sample input.
- Zero vast.ai spend.

### Estimated effort

- 1–2 evenings, mostly spent on the workflow → API conversion and naming nodes.

---

## Phase 1 — First successful inference on vast.ai

**Goal:** One image generated, end-to-end, via SSH tunnel.

### Tasks

1. **Push Cog image to GHCR:**
   ```
   echo $GH_TOKEN | docker login ghcr.io -u <user> --password-stdin
   cog push ghcr.io/<user>/cog-zimage:phase1
   ```

2. **Write `deploy.py` (or `deploy.sh`):**

   Reads `deploy.yaml`:
   ```yaml
   gpu_name: ["RTX_3090", "RTX_4090"]
   min_vram_gb: 24
   min_inet_down_mbps: 500
   min_reliability: 0.98
   max_dph: 0.50
   cuda_min: 12.0
   disk_gb: 60
   geolocation: ["US", "CA", "EU"]
   verified: true
   interruptible: false
   image: ghcr.io/<user>/cog-zimage:phase1
   max_session_hours: 4
   weight_download_timeout_min: 10
   ```

   Steps:
   1. `vastai search offers` with the filters above; pick cheapest match.
   2. `vastai create instance <id> --image <image> --disk <gb>`.
   3. Poll until SSH is ready.
   4. Open SSH tunnel: `ssh -L 5000:localhost:5000 -N <instance> &`.
   5. Health-check ComfyUI through the tunnel: `curl localhost:5000/system_stats`,
      with retry budget = `weight_download_timeout_min`. On timeout: destroy
      and try the next match.
   6. Schedule **server-side self-destruct timer** on the instance:
      `nohup sh -c "sleep $((max_session_hours*3600)) && vastai destroy instance $ID" &`
      (using a vast.ai API key cached on the host). This fires even if the local
      script dies.
   7. Register `atexit` handler that destroys the instance on local exit.
   8. Print the tunnel address and exit (or stay attached, depending on flag).

3. **Write `predict_client.py`:**
   - POSTs `target_image` + `prompt` + `seed` to `localhost:5000/predictions`.
   - Saves the returned PNG.

4. **Run end-to-end:**
   ```
   python deploy.py --config deploy.yaml
   python predict_client.py --image test.jpg --prompt "young man with brown hair"
   ```
   Inspect output. Iterate.

### Exit criteria

- One PNG headswap output successfully generated via the full pipeline.
- Cost guardrails verified: the self-destruct timer actually fires; the health
  check actually rejects bad hosts.
- Total spend to date: $2–5.

### Estimated effort

- 2–3 evenings, of which ~50% will be spent debugging first-run failures
  (rate-limited HF downloads, custom-node imports, SAM3 model load mismatches,
  vast.ai hosts with bad networks). This is normal.

---

## Phase 2 — LoRA support (specific-identity replacement)

**Goal:** Pass a LoRA URL and get specific-identity output.

### Tasks

1. **Extend `predict()`:**
   ```python
   def predict(
       target_image: Path,
       prompt: str = "",
       seed: int = -1,
       lora_url: str = "",
       lora_scale: float = 1.0,
       workflow_override: dict | None = None,
   ) -> Path: ...
   ```

2. **LoRA fetch & cache:**
   - Hash `lora_url` → cache filename in `loras/`.
   - If not cached, `pget` it.
   - Patch the workflow's `LORA_LOADER` named-handle node:
     `inputs.lora_name = <filename>`, `inputs.strength_model = lora_scale`.
   - If `lora_url` is empty, set the LoRA loader to bypass / strength 0.

3. **Integration test suite:**
   - 3–5 fixed input images + LoRAs + seeds.
   - Save expected outputs as PNGs in `tests/golden/`.
   - Compare new outputs by perceptual hash (not byte-equality — diffusion is
     non-deterministic across CUDA versions).

4. **(Optional) Train a test LoRA:**
   - Use [Ostris AI Toolkit](https://huggingface.co/blog/content-and-code/training-a-lora-for-z-image-turbo)
     on the same vast.ai instance.
   - ~30 min on a 4090 with ~20 reference images of one identity.
   - Validates the end-to-end Phase 2 loop on a real identity.

### Exit criteria

- A LoRA URL produces visibly identity-controlled output.
- Integration tests pass on a clean instance.

---

## Phase 3 — ControlNet + workflow override

**Goal:** Pose/lighting control and the ability to swap workflows without rebuilding.

### Tasks

1. **Wire `workflow_override`:**
   - Already plumbed in Phase 0; verify it works end-to-end.
   - Document the named-handle contract in `README.md` so an LLM editing the
     workflow knows what to preserve.

2. **Add ControlNet inputs:**
   ```python
   controlnet_image: Path | None = None
   controlnet_type: str = "depth"  # or "pose", "canny"
   controlnet_strength: float = 0.7
   ```
   - Add corresponding ControlNet model(s) to weights manifest.
   - Add a `CONTROLNET_*` named-handle nodes to the workflow.
   - When `controlnet_image is None`, skip the ControlNet branch via node bypass.

3. **Decision point — identity approach:**
   By Phase 3 you'll have data on whether LoRA-driven identity is operationally
   acceptable. If it isn't, evaluate swapping the base workflow to a
   PuLID-Flux or InstantID-class graph. The cog-comfyui plumbing, vast.ai
   deploy script, named-handle convention, and weight-fetch machinery all
   carry over unchanged. Only `workflows/*.json` and the weights manifest change.

### Exit criteria

- ControlNet-conditioned headswap works.
- A second workflow JSON (or the override input) produces correct output without
  rebuilding the image.

---

## Operational details

### Local dev loop (Phase 1+)

- **Fast iteration:** edit `predict.py` and workflow JSON locally → `cog build` →
  `cog push`. On vast.ai, restart the Cog server inside the existing instance
  rather than re-provisioning (saves the cold-start tax during a single session).
- **Dry-run first:** every change should pass `cog predict --dry-run` before
  hitting vast.ai.
- **Single warm instance per session:** keep one vast.ai box live for the
  duration of a coding session; tear down at the end. Expect ~$2–3 per session.

### Cost guardrails (all three required for Phase 1)

1. **Server-side self-destruct timer** (`max_session_hours`). Survives local
   script death, network drops, laptop closing.
2. **Health check before "ready"** — confirms ComfyUI booted; otherwise
   destroy and try a different host.
3. **Weight-download timeout** — abort and retry on a different host if
   downloads stall (catches 100 Mbps hosts that slip through filters).

### Failure modes to expect (don't panic when these happen)

- **HF rate-limiting** on the first cold start of a fresh week. Mitigate by
  retrying or by mirroring weights to R2.
- **Custom-node Python import errors** when ComfyUI's main branch has moved
  faster than a pinned node. Mitigation: re-pin to a newer compatible SHA.
- **SAM3 model-load mismatches** — the workflow author already had to patch
  this. If it breaks, look for the newest version of `comfyui-easy-sam3`.
- **Vast.ai hosts with low real-world bandwidth** despite passing filters.
  Trust the weight-download timeout to handle this.
- **ComfyUI workflow validation errors** if a custom-node update changed an
  input name. Inspect the patched JSON via dry-run, fix names.

---

## Sharp edges / risks

- **Z-Image is new (released late 2025).** Tooling, GGUF quantizations, and
  custom-node compatibility shift weekly. Pin everything.
- **Licensing.** Z-Image (Tongyi-MAI/Alibaba) commercial terms vary. SAM3
  (Meta) may be research-only. Verify before any commercial deployment.
- **Civitai workflow attribution.** Credit the original author in the README;
  do not redistribute their workflow JSON without attribution.
- **Cold-start tax.** ~5–10 min from `vastai create` to first inference, every
  session. Acceptable for testing; intolerable for production. If you ever
  ship to real users, graduate to Mode 1 (always-on) or a serverless platform
  with warm pools (Replicate, RunPod, Modal).
- **Identity fidelity.** Without a LoRA, the workflow gives you "a generic head
  matching the prompt" — not the specific person you imagined. Set expectations
  for Phase 1 outputs accordingly.
- **Seed reproducibility.** Verify the `SEED_INPUT` named handle points to a
  KSampler with `control_after_generate = fixed`, otherwise `seed=42` is silently
  ignored.

---

## Repository structure (target)

```
cog-zimage-project/
├── PLAN.md                         # this file
├── README.md                       # quick-start, named-handle contract
├── cog.yaml                        # Cog build config
├── predict.py                      # Cog predict() entry point
├── workflows/
│   └── headswap.json               # API-format ComfyUI workflow with named handles
├── custom_nodes.json               # commit-SHA pins for custom node packs
├── weights.json                    # commit-SHA-pinned HF download URLs
├── deploy/
│   ├── deploy.py                   # vast.ai provision + tunnel + guardrails
│   ├── deploy.yaml                 # vast.ai search filters
│   └── predict_client.py           # local client that hits the tunneled API
├── tests/
│   ├── test_dry_run.py             # named-handle substitution validation
│   ├── test_integration.py         # golden-image perceptual-hash compare (Phase 2)
│   └── golden/                     # expected outputs for integration tests
└── .github/workflows/
    └── build.yml                   # CI: cog build + dry-run on PR
```

---

## Open questions deferred to implementation

These are decisions where the right answer becomes obvious once you have
something running. Don't pre-decide.

- Whether to mirror weights from HF to R2 (decide after Phase 1 cold-start data).
- Whether to graduate to Mode 1 always-on (decide after a few weeks of usage).
- Whether to swap to PuLID/InstantID for identity (decide at Phase 3 based on
  LoRA training friction).
- Whether to expose intermediate outputs (mask, cropped face) in the API
  (decide when you actually need them for debugging).
