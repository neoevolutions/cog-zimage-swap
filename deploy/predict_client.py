#!/usr/bin/env python3
"""Local client for the cog-zimage tunneled API.

Posts target_image+prompt+seed to localhost:5000/predictions and writes the
returned PNG to predictions/. Cog's HTTP API takes inputs as base64-encoded
data URIs in the JSON body.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
import time
import urllib.request
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"


def encode_data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "application/octet-stream"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def decode_output(output: str | list, dest: Path) -> Path:
    """Cog returns Path outputs as data URIs. Decode and write to disk."""
    payload = output[0] if isinstance(output, list) else output
    if not isinstance(payload, str):
        raise ValueError(f"unexpected output type: {type(payload)}")
    if payload.startswith("data:"):
        _, b64 = payload.split(",", 1)
        dest.write_bytes(base64.b64decode(b64))
    elif payload.startswith("http"):
        with urllib.request.urlopen(payload) as r:
            dest.write_bytes(r.read())
    else:
        raise ValueError(f"can't decode output: {payload[:80]}...")
    return dest


def predict(host: str, port: int, target_image: Path, prompt: str,
            seed: int, dry_run: bool, timeout_s: int) -> dict:
    body = {
        "input": {
            "target_image": encode_data_uri(target_image),
            "prompt": prompt,
            "seed": seed,
            "dry_run": dry_run,
        }
    }
    req = urllib.request.Request(
        f"http://{host}:{port}/predictions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    print(f">>> POST http://{host}:{port}/predictions  (timeout {timeout_s}s)")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        result = json.loads(r.read())
    print(f">>> {result.get('status', '?')} in {time.time() - t0:.1f}s")
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True, type=Path,
                   help="Local target image to send.")
    p.add_argument("--prompt", default="",
                   help="Free-form prompt for the head.")
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--out", type=Path, default=None,
                   help="Output PNG path (default: predictions/<uuid>.png)")
    p.add_argument("--timeout", type=int, default=300,
                   help="Request timeout seconds.")
    p.add_argument("--dry-run", action="store_true",
                   help="Exercise the patcher only; returns the patched workflow JSON.")
    args = p.parse_args()

    if not args.image.exists():
        sys.exit(f"image not found: {args.image}")

    result = predict(
        args.host, args.port, args.image, args.prompt, args.seed,
        args.dry_run, args.timeout,
    )

    status = result.get("status")
    if status not in ("succeeded", "processing"):
        print(json.dumps(result, indent=2))
        sys.exit(f"prediction failed (status={status})")

    output = result.get("output")
    if output is None:
        sys.exit("no output in response")

    PREDICTIONS_DIR.mkdir(exist_ok=True)
    suffix = ".json" if args.dry_run else ".png"
    dest = args.out or PREDICTIONS_DIR / f"{uuid.uuid4().hex[:12]}{suffix}"
    decode_output(output, dest)
    print(f">>> saved {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
