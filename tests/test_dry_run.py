"""Phase 0 dry-run validation.

These tests run without Cog or ComfyUI installed — they exercise the pure
named-handle substitution logic against the placeholder workflow JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from workflow_patch import (  # noqa: E402
    HANDLE_LORA,
    HANDLE_OUTPUT,
    HANDLE_PROMPT,
    HANDLE_SEED,
    HANDLE_TARGET_IMAGE,
    REQUIRED_HANDLES,
    patch_workflow,
)

WORKFLOW = json.loads((ROOT / "workflows" / "headswap.json").read_text())


def _fresh_workflow() -> dict:
    return json.loads(json.dumps(WORKFLOW))


def test_workflow_contains_all_required_handles() -> None:
    titles = {
        (n.get("_meta") or {}).get("title")
        for n in WORKFLOW.values()
        if isinstance(n, dict)
    }
    for handle in REQUIRED_HANDLES:
        assert handle in titles, f"placeholder workflow is missing {handle}"


def test_patch_substitutes_target_image() -> None:
    wf = _fresh_workflow()
    patched = patch_workflow(
        wf, target_filename="abc.png", prompt="a man", seed=42
    )
    target_nodes = [
        n for n in patched.values()
        if isinstance(n, dict) and (n.get("_meta") or {}).get("title") == HANDLE_TARGET_IMAGE
    ]
    assert target_nodes
    for node in target_nodes:
        assert node["inputs"]["image"] == "abc.png"


def test_patch_substitutes_prompt() -> None:
    wf = _fresh_workflow()
    patched = patch_workflow(
        wf, target_filename="abc.png", prompt="a young woman", seed=42
    )
    prompt_nodes = [
        n for n in patched.values()
        if isinstance(n, dict) and (n.get("_meta") or {}).get("title") == HANDLE_PROMPT
    ]
    assert prompt_nodes
    for node in prompt_nodes:
        assert node["inputs"]["text"] == "a young woman"


def test_patch_substitutes_seed() -> None:
    wf = _fresh_workflow()
    patched = patch_workflow(
        wf, target_filename="abc.png", prompt="x", seed=12345
    )
    seed_nodes = [
        n for n in patched.values()
        if isinstance(n, dict) and (n.get("_meta") or {}).get("title") == HANDLE_SEED
    ]
    assert seed_nodes
    for node in seed_nodes:
        assert node["inputs"]["seed"] == 12345


def test_patch_bypasses_lora_when_unset() -> None:
    wf = _fresh_workflow()
    patched = patch_workflow(
        wf, target_filename="x.png", prompt="x", seed=0
    )
    lora_nodes = [
        n for n in patched.values()
        if isinstance(n, dict) and (n.get("_meta") or {}).get("title") == HANDLE_LORA
    ]
    assert lora_nodes
    for node in lora_nodes:
        for k, v in node["inputs"].items():
            if k.startswith("lora_") and isinstance(v, dict):
                assert v["on"] is False
                assert v["strength"] == 0.0


def test_patch_enables_lora_when_provided() -> None:
    wf = _fresh_workflow()
    patched = patch_workflow(
        wf,
        target_filename="x.png",
        prompt="x",
        seed=0,
        lora_name="rachel.safetensors",
        lora_strength=0.85,
    )
    lora_nodes = [
        n for n in patched.values()
        if isinstance(n, dict) and (n.get("_meta") or {}).get("title") == HANDLE_LORA
    ]
    for node in lora_nodes:
        for k, v in node["inputs"].items():
            if k.startswith("lora_") and isinstance(v, dict):
                assert v["on"] is True
                assert v["lora"] == "rachel.safetensors"
                assert v["strength"] == 0.85


def test_patch_raises_on_missing_handle() -> None:
    bad_workflow = {
        "1": {
            "class_type": "LoadImage",
            "_meta": {"title": HANDLE_TARGET_IMAGE},
            "inputs": {"image": ""},
        },
        "2": {
            "class_type": "SaveImage",
            "_meta": {"title": HANDLE_OUTPUT},
            "inputs": {"filename_prefix": "x"},
        },
    }
    try:
        patch_workflow(
            bad_workflow, target_filename="x.png", prompt="x", seed=0
        )
    except ValueError as e:
        assert HANDLE_PROMPT in str(e)
        assert HANDLE_SEED in str(e)
        return
    raise AssertionError("expected ValueError for missing required handles")


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
