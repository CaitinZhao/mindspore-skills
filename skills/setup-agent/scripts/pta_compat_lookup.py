#!/usr/bin/env python3
"""Resolve PTA compatibility from local reference data, with optional remote fallback."""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_REFERENCE = ROOT / "references" / "ascend-compat.md"
REMOTE_README = "https://raw.githubusercontent.com/Ascend/pytorch/master/README.md"


def normalize_torch(version: str | None) -> str | None:
    if not version:
        return None
    version = version.strip()
    if version.startswith("v"):
        version = version[1:]
    return version.split("+", 1)[0]


def normalize_torch_npu(version: str | None) -> str | None:
    if not version:
        return None
    version = version.strip()
    if version.startswith("v"):
        version = version[1:]
    return version


def parse_local_table() -> list[dict[str, str]]:
    text = LOCAL_REFERENCE.read_text(encoding="utf-8")
    rows: list[dict[str, str]] = []
    in_table = False
    for line in text.splitlines():
        if line.strip() == "### Local PTA Compatibility Table":
            in_table = True
            continue
        if in_table and line.startswith("## "):
            break
        if not in_table:
            continue
        if not line.startswith("|"):
            continue
        cols = [part.strip() for part in line.strip().strip("|").split("|")]
        if cols[:2] == ["CANN", "torch"] or cols[0].startswith("------"):
            continue
        if len(cols) != 6:
            continue
        rows.append(
            {
                "source": "local",
                "cann": cols[0],
                "torch": normalize_torch(cols[1]) or "",
                "torch_npu": normalize_torch_npu(cols[2]) or "",
                "python": cols[3],
                "branch": cols[4],
                "note": cols[5],
            }
        )
    return rows


def parse_remote_table() -> list[dict[str, str]]:
    with urllib.request.urlopen(REMOTE_README, timeout=20) as resp:
        text = resp.read().decode("utf-8")

    rows: list[dict[str, str]] = []
    in_table = False
    current_cann = ""
    for line in text.splitlines():
        if line.strip() == "## Ascend Auxiliary Software":
            in_table = True
            continue
        if in_table and line.startswith("## "):
            break
        if not in_table or not line.startswith("|"):
            continue
        cols = [part.strip() for part in line.strip().strip("|").split("|")]
        if cols[:2] == ["CANN Version", "Supported PyTorch Version"] or cols[0].startswith("---"):
            continue
        if len(cols) != 4:
            continue
        if cols[0]:
            current_cann = cols[0].replace("CANN ", "", 1).strip()
        if not current_cann:
            continue
        rows.append(
            {
                "source": "remote",
                "cann": current_cann,
                "torch": normalize_torch(cols[1]) or "",
                "torch_npu": normalize_torch_npu(cols[2]) or "",
                "python": "",
                "branch": cols[3],
                "note": "remote fallback from Ascend/pytorch README",
            }
        )
    return rows


def python_matches(spec: str, version: str | None) -> bool:
    if not version or not spec or spec.startswith("verify upstream PTA Python table"):
        return True
    match = re.match(r"^(\d+)\.(\d+)", version)
    if not match:
        return False
    current = (int(match.group(1)), int(match.group(2)))
    if "-" in spec:
        start, end = spec.split("-", 1)
        low = tuple(map(int, start.split(".")))
        high = tuple(map(int, end.split(".")))
        return low <= current <= high
    return version.startswith(spec)


def filter_rows(rows: list[dict[str, str]], cann: str, torch: str | None, torch_npu: str | None, py: str | None) -> list[dict[str, str]]:
    out = []
    norm_torch = normalize_torch(torch)
    norm_torch_npu = normalize_torch_npu(torch_npu)
    for row in rows:
        if row["cann"] != cann:
            continue
        if norm_torch and row["torch"] != norm_torch:
            continue
        if norm_torch_npu and row["torch_npu"] != norm_torch_npu:
            continue
        if not python_matches(row["python"], py):
            continue
        out.append(row)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cann", required=True)
    parser.add_argument("--torch")
    parser.add_argument("--torch-npu")
    parser.add_argument("--python")
    parser.add_argument("--remote-fallback", action="store_true")
    args = parser.parse_args()

    local_rows = parse_local_table()
    matches = filter_rows(local_rows, args.cann, args.torch, args.torch_npu, args.python)
    source = "local"

    if not matches and args.remote_fallback:
        remote_rows = parse_remote_table()
        matches = filter_rows(remote_rows, args.cann, args.torch, args.torch_npu, args.python)
        source = "remote" if matches else "unresolved"

    result = {
        "source": source,
        "query": {
            "cann": args.cann,
            "torch": normalize_torch(args.torch),
            "torch_npu": normalize_torch_npu(args.torch_npu),
            "python": args.python,
        },
        "matches": matches,
    }
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
