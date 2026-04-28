#!/usr/bin/env python3
"""Clone custom-node packs at pinned SHAs into ComfyUI's custom_nodes/.

Invoked from cog.yaml `run:` during image build.
Usage: install_custom_nodes.py <custom_nodes.json> <target_dir>
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    manifest_path = Path(argv[1])
    target_dir = Path(argv[2])
    target_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text())
    nodes = manifest["nodes"]
    failures: list[str] = []

    for entry in nodes:
        name = entry["name"]
        repo = entry["repo"]
        commit = entry["commit"]
        dest = target_dir / name

        if commit == "TODO_PIN_SHA":
            print(f"[install_custom_nodes] {name}: commit is TODO_PIN_SHA — falling back to {entry.get('version_hint', 'main')}")
            ref = entry.get("version_hint", "main")
        else:
            ref = commit

        if dest.exists():
            print(f"[install_custom_nodes] {name}: already present, skipping")
            continue

        print(f"[install_custom_nodes] cloning {name} @ {ref}")
        try:
            subprocess.run(["git", "clone", repo, str(dest)], check=True)
            subprocess.run(["git", "-C", str(dest), "checkout", ref], check=True)
        except subprocess.CalledProcessError as e:
            failures.append(f"{name}: {e}")
            continue

        reqs = dest / "requirements.txt"
        if reqs.exists():
            print(f"[install_custom_nodes] installing requirements for {name}")
            subprocess.run(
                ["pip", "install", "--no-cache-dir", "-r", str(reqs)],
                check=False,
            )

    if failures:
        print("[install_custom_nodes] FAILURES:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
