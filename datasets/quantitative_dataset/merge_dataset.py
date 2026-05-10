#!/usr/bin/env python3
"""
Concatenate split quantitative_dataset.sqlite shards back into one SQLite file.

Shards are expected beside this script and named:
    quantitative_dataset.sqlite.part001
    quantitative_dataset.sqlite.part002
    ...
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

PART_PATTERN = re.compile(r"^quantitative_dataset\.sqlite\.part(\d+)$")


def discover_parts(directory: Path) -> list[Path]:
    candidates = []
    for entry in directory.iterdir():
        if not entry.is_file():
            continue
        m = PART_PATTERN.match(entry.name)
        if not m:
            continue
        candidates.append((int(m.group(1)), entry))
    candidates.sort(key=lambda t: t[0])
    if not candidates:
        raise FileNotFoundError(
            f"No shards matching quantitative_dataset.sqlite.partNNN under {directory}"
        )
    nums = [n for n, _ in candidates]
    expected = list(range(1, len(nums) + 1))
    if nums != expected:
        missing = set(expected) - set(nums)
        raise ValueError(
            "Shard sequence is not contiguous starting at part001. "
            f"Found indexes {nums[:5]}… (total {len(nums)}). Missing: {sorted(missing)!r}"
        )
    return [path for _, path in candidates]


def merge_parts(parts: list[Path], destination: Path, chunk_bytes: int = 8 * 1024 * 1024) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".partial")
    try:
        if tmp.exists():
            tmp.unlink()
        with open(tmp, "wb") as out:
            for shard in parts:
                with open(shard, "rb") as src:
                    while True:
                        block = src.read(chunk_bytes)
                        if not block:
                            break
                        out.write(block)
        os.replace(tmp, destination)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-d",
        "--shards-dir",
        type=Path,
        default=None,
        help="Directory containing part files (default: this script's folder).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("quantitative_dataset.sqlite"),
        help="Merged SQLite path (default: ./quantitative_dataset.sqlite).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = (args.shards_dir or Path(__file__).resolve().parent).resolve()
    try:
        parts = discover_parts(base_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Merging {len(parts)} shard(s) from {base_dir} -> {args.output.resolve()}")
    merge_parts(parts, args.output.resolve())
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
