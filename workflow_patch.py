"""Pure named-handle workflow patcher. No Cog/ComfyUI imports.

Kept separate from predict.py so it can be unit-tested on a developer Mac
without the Cog runtime installed.
"""
from __future__ import annotations

import os

HANDLE_TARGET_IMAGE = "INPUT_TARGET_IMAGE"
HANDLE_PROMPT = "INPUT_PROMPT"
HANDLE_SEED = "SEED_INPUT"
HANDLE_LORA = "LORA_LOADER"
HANDLE_OUTPUT = "OUTPUT_IMAGE"

REQUIRED_HANDLES = (HANDLE_TARGET_IMAGE, HANDLE_PROMPT, HANDLE_SEED, HANDLE_OUTPUT)


def patch_workflow(
    workflow: dict,
    *,
    target_filename: str,
    prompt: str,
    seed: int,
    lora_name: str | None = None,
    lora_strength: float = 0.0,
) -> dict:
    """Substitute named-handle nodes in an API-format workflow.

    Mutates and returns the workflow dict so callers can chain or inspect.
    Skips dict entries whose key starts with ``_`` so meta keys like
    ``_comment`` in workflow JSON don't trip class_type lookups.
    """
    nodes_by_handle: dict[str, list[tuple[str, dict]]] = {}
    for node_id, node in workflow.items():
        if node_id.startswith("_") or not isinstance(node, dict):
            continue
        title = (node.get("_meta") or {}).get("title")
        if title:
            nodes_by_handle.setdefault(title, []).append((node_id, node))

    missing = [h for h in REQUIRED_HANDLES if h not in nodes_by_handle]
    if missing:
        raise ValueError(f"Workflow missing required named handles: {missing}")

    for _, node in nodes_by_handle[HANDLE_TARGET_IMAGE]:
        node.setdefault("inputs", {})["image"] = target_filename

    if prompt:
        for _, node in nodes_by_handle[HANDLE_PROMPT]:
            node.setdefault("inputs", {})["text"] = prompt

    for _, node in nodes_by_handle[HANDLE_SEED]:
        node.setdefault("inputs", {})["seed"] = seed

    for _, node in nodes_by_handle.get(HANDLE_LORA, []):
        for key, value in list(node.get("inputs", {}).items()):
            if key.startswith("lora_") and isinstance(value, dict):
                if lora_name:
                    value["on"] = True
                    value["lora"] = lora_name
                    value["strength"] = lora_strength
                else:
                    value["on"] = False
                    value["strength"] = 0.0

    return workflow


def random_seed() -> int:
    return int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF
