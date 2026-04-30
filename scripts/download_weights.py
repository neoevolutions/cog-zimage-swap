#!/usr/bin/env python3
"""Fetch model weights declared in weights.json into ComfyUI's model dirs.

Run at instance startup (from predict.py setup()), not at image build time —
cog COPYs /src into the container *after* run: steps complete, so weights.json
isn't accessible during build. Runtime download also keeps the image small
(~2GB instead of ~12GB) and lets us update weights without rebuilding.

Each entry in weights.json has the shape:
    {"name": "...", "url": "...", "dest": "/abs/path/file", "auth": "civitai" | null}

Skips entries with sentinel URLs (TODO_FIND_HF_URL, MULTI_FILE_BUNDLE) or that
declare auth="civitai" when CIVITAI_TOKEN is missing — never silently fails the
build, just logs and moves on. Files already at the dest path are skipped.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_JSON = ROOT / "weights.json"

SENTINEL_URLS = ("TODO_FIND_HF_URL", "MULTI_FILE_BUNDLE", "TODO_")


def _log(msg: str) -> None:
    print(f"[download_weights] {msg}", flush=True)


def _have_pget() -> bool:
    return shutil.which("pget") is not None


def _file_ok(dest: Path, expected_gb: float | None) -> bool:
    """Treat as already-downloaded if file exists and is plausibly large."""
    if not dest.exists():
        return False
    size_gb = dest.stat().st_size / (1024 ** 3)
    if expected_gb and size_gb < expected_gb * 0.5:
        # Truncated/aborted previous download — refetch.
        _log(f"  partial file at {dest} ({size_gb:.2f}GB < expected {expected_gb}GB); refetching")
        dest.unlink()
        return False
    return True


def _download(url: str, dest: Path, headers: dict[str, str] | None = None) -> bool:
    """Use pget if available (fast parallel) and no custom headers, else curl.
    pget v0.8.2 doesn't support a --header flag — passing it makes pget print
    its help text and exit non-zero. Falling back to curl whenever headers
    are present keeps auth-bearing downloads (e.g. Civitai Bearer) working.
    """
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        # Running outside the container (e.g. local dev smoke test on macOS)
        # where /opt/comfyui isn't writable. Treat as a soft failure.
        _log(f"  cannot create {dest.parent} ({e}); skipping")
        return False
    use_pget = _have_pget() and not headers
    if use_pget:
        cmd = ["pget", url, str(dest)]
        # Don't echo query strings — the source URL may contain credentials.
        safe_url = url.split("?", 1)[0] + ("?…" if "?" in url else "")
        _log(f"  pget {safe_url} -> {dest}")
    else:
        cmd = ["curl", "-fsSL", "-o", str(dest)]
        for k, v in (headers or {}).items():
            cmd.extend(["-H", f"{k}: {v}"])
        cmd.append(url)
        safe_url = url.split("?", 1)[0]
        tool = "curl" if not _have_pget() else "curl (auth)"
        _log(f"  {tool} {safe_url} -> {dest}")
    started = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - started
    if result.returncode != 0:
        _log(f"  FAILED in {elapsed:.0f}s: {result.stderr.strip()[-200:]}")
        if dest.exists():
            dest.unlink()
        return False
    size_mb = dest.stat().st_size / (1024 ** 2) if dest.exists() else 0
    _log(f"  done in {elapsed:.0f}s ({size_mb:.0f}MB)")
    return True


def _resolve_headers(entry: dict) -> dict[str, str] | None:
    auth = entry.get("auth")
    if auth == "civitai":
        token = os.environ.get("CIVITAI_TOKEN", "").strip()
        if not token:
            return None
        # Authorization header (not query string) so the token doesn't leak
        # into HTTP redirect chains, server logs, or pget output.
        return {"Authorization": f"Bearer {token}"}
    if auth in (None, "anonymous"):
        return {}
    raise ValueError(f"unknown auth scheme: {auth!r} on entry {entry.get('name')!r}")


def main() -> int:
    if not WEIGHTS_JSON.exists():
        _log(f"FATAL: {WEIGHTS_JSON} not found")
        return 2
    cfg = json.loads(WEIGHTS_JSON.read_text())
    entries = cfg.get("weights") or []
    _log(f"weights.json: {len(entries)} entries")

    failures: list[str] = []
    for entry in entries:
        name = entry.get("name") or "?"
        url = (entry.get("url") or "").strip()
        dest = entry.get("dest")
        if not url or not dest:
            _log(f"SKIP {name}: missing url/dest")
            continue
        if any(s in url for s in SENTINEL_URLS):
            _log(f"SKIP {name}: sentinel URL ({url})")
            continue

        dest_path = Path(dest)
        if _file_ok(dest_path, entry.get("approx_size_gb")):
            _log(f"OK   {name}: present at {dest_path}")
            continue

        headers = _resolve_headers(entry)
        if headers is None:
            _log(f"SKIP {name}: auth=civitai but CIVITAI_TOKEN not set")
            continue

        _log(f"GET  {name}")
        if not _download(url, dest_path, headers=headers):
            failures.append(name)

    if failures:
        _log(f"completed with failures: {failures}")
        return 1
    _log("all weights present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
