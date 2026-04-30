"""Cog predictor for the Z-Image Turbo + SAM3 character headswap workflow.

Hosts a headless ComfyUI subprocess and exposes a named-handle-based API.
Workflow nodes carrying parameters are identified by `_meta.title`:

    INPUT_TARGET_IMAGE   LoadImage          inputs.image
    INPUT_PROMPT         Text Multiline     inputs.text
    SEED_INPUT           KSampler           inputs.seed
    LORA_LOADER          Power Lora Loader  inputs.lora_*.{on,strength}
    OUTPUT_IMAGE         SaveImage          inputs.filename_prefix (sentinel)

This decouples the API surface from graph structure: the workflow author can
rearrange the graph as long as the named handles survive.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path as P

from cog import BasePredictor, Input, Path

from workflow_patch import patch_workflow, random_seed

WORKFLOW_PATH = P(__file__).parent / "workflows" / "headswap.json"
COMFYUI_HOST = os.environ.get("COMFYUI_HOST", "127.0.0.1")
COMFYUI_PORT = int(os.environ.get("COMFYUI_PORT", "8188"))
COMFYUI_DIR = P(os.environ.get("COMFYUI_DIR", "/opt/comfyui"))
COMFYUI_INPUT_DIR = COMFYUI_DIR / "input"
COMFYUI_OUTPUT_DIR = COMFYUI_DIR / "output"
COMFYUI_BOOT_TIMEOUT = int(os.environ.get("COMFYUI_BOOT_TIMEOUT", "120"))
COMFYUI_PREDICT_TIMEOUT = int(os.environ.get("COMFYUI_PREDICT_TIMEOUT", "300"))


SENTINEL_URLS = ("TODO_FIND_HF_URL", "MULTI_FILE_BUNDLE", "TODO_")


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.comfyui_proc: subprocess.Popen | None = None
        if os.environ.get("COG_SKIP_COMFYUI") == "1":
            return
        if not COMFYUI_DIR.exists():
            return
        # Weights aren't baked into the image (see cog.yaml comment) — fetch them
        # from weights.json before ComfyUI starts. Idempotent: skips files
        # already on disk. Raises on missing/truncated files so cog never
        # reports ready with broken state.
        print("[setup] starting weight download", flush=True)
        self._download_weights()
        self._verify_weights()
        print("[setup] starting ComfyUI", flush=True)
        self.comfyui_proc = self._start_comfyui()
        self._wait_for_comfyui(timeout=COMFYUI_BOOT_TIMEOUT)
        print("[setup] ComfyUI ready", flush=True)

    def _download_weights(self) -> None:
        script = P(__file__).parent / "scripts" / "download_weights.py"
        if not script.exists():
            # Image-build bug, not transient — fail loudly so the cog server
            # never reaches ready and deploy.py's health check catches it.
            raise RuntimeError(f"download_weights.py missing at {script}")
        # Run with the project root as cwd so weights.json resolves correctly.
        result = subprocess.run(
            ["python", str(script)], cwd=str(P(__file__).parent),
            capture_output=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"download_weights.py exited {result.returncode}; "
                f"see [download_weights] lines in /var/log/cog.log"
            )

    def _verify_weights(self) -> None:
        """Verify every non-sentinel, expected weights.json entry has its dest
        file with plausible size. Civitai-gated entries are skipped if
        CIVITAI_TOKEN is missing (matches download_weights.py behavior)."""
        weights_json = P(__file__).parent / "weights.json"
        if not weights_json.exists():
            raise RuntimeError(f"weights.json missing at {weights_json}")
        cfg = json.loads(weights_json.read_text())
        missing: list[str] = []
        total_gb = 0.0
        n_files = 0
        for entry in cfg.get("weights") or []:
            url = (entry.get("url") or "").strip()
            if any(s in url for s in SENTINEL_URLS):
                continue
            if entry.get("auth") == "civitai" and not os.environ.get("CIVITAI_TOKEN", "").strip():
                continue
            dest = P(entry.get("dest") or "")
            expected_gb = entry.get("approx_size_gb") or 0
            min_bytes = int(expected_gb * 0.5 * (1024 ** 3))
            if not dest.exists() or dest.stat().st_size < min_bytes:
                missing.append(f"{entry.get('name')!r} ({dest})")
                continue
            total_gb += dest.stat().st_size / (1024 ** 3)
            n_files += 1
        if missing:
            raise RuntimeError(
                f"weights verification FAILED: {len(missing)} missing or "
                f"truncated file(s): {missing}"
            )
        print(f"[setup] downloads complete: {n_files} files, {total_gb:.1f} GB", flush=True)

    def predict(
        self,
        target_image: Path = Input(
            description="Image whose head/face will be replaced.",
        ),
        prompt: str = Input(
            description="Free-form description of the desired head. Auto-augmented by JoyCaption.",
            default="",
        ),
        seed: int = Input(
            description="Sampler seed; -1 picks a random seed.",
            default=-1,
            ge=-1,
        ),
        workflow_override: str = Input(
            description="API-format workflow JSON string. Overrides the baked-in workflow.",
            default="",
        ),
        dry_run: bool = Input(
            description="Substitute named handles and return the patched workflow JSON without running inference.",
            default=False,
        ),
    ) -> Path:
        workflow = self._load_workflow(workflow_override)
        actual_seed = seed if seed >= 0 else random_seed()
        target_filename = self._stage_target_image(target_image, dry_run=dry_run)
        patched = patch_workflow(
            workflow,
            target_filename=target_filename,
            prompt=prompt,
            seed=actual_seed,
        )

        if dry_run:
            out = P("/tmp") / f"dry_run_{uuid.uuid4().hex}.json"
            out.write_text(json.dumps(patched, indent=2))
            return Path(out)

        prompt_id = self._submit(patched)
        output_filename = self._wait_for_output(prompt_id, timeout=COMFYUI_PREDICT_TIMEOUT)
        return Path(COMFYUI_OUTPUT_DIR / output_filename)

    def _load_workflow(self, override: str) -> dict:
        if override.strip():
            return json.loads(override)
        return json.loads(WORKFLOW_PATH.read_text())

    def _stage_target_image(self, target_image: Path, dry_run: bool) -> str:
        target_filename = f"input_{uuid.uuid4().hex}.png"
        if dry_run:
            return target_filename
        COMFYUI_INPUT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(target_image), COMFYUI_INPUT_DIR / target_filename)
        return target_filename

    def _start_comfyui(self) -> subprocess.Popen:
        return subprocess.Popen(
            [
                "python", "main.py",
                "--listen", COMFYUI_HOST,
                "--port", str(COMFYUI_PORT),
                "--disable-auto-launch",
            ],
            cwd=str(COMFYUI_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _wait_for_comfyui(self, timeout: int) -> None:
        url = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}/system_stats"
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:
                time.sleep(1)
        raise TimeoutError(f"ComfyUI did not become ready at {url} within {timeout}s")

    def _submit(self, workflow: dict) -> str:
        url = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}/prompt"
        body = json.dumps({"prompt": workflow}).encode()
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())["prompt_id"]

    def _wait_for_output(self, prompt_id: str, timeout: int) -> str:
        url = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}/history/{prompt_id}"
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    data = json.loads(r.read())
                if prompt_id in data:
                    outputs = data[prompt_id].get("outputs", {})
                    for node_output in outputs.values():
                        for img in node_output.get("images") or []:
                            return img["filename"]
            except Exception:
                pass
            time.sleep(1)
        raise TimeoutError(f"ComfyUI did not produce output for prompt {prompt_id}")
