#!/usr/bin/env python
"""Generate shard manifest files for point cloud (.ply) processing.

Example:
  python make_shards.py --ply-root /data/plys --num-workers 8 --prefix shards --relative

Produces:
  shards_shard_000_of_8.txt
  ...
  shards_shard_007_of_8.txt
  shards_meta.json

Each shard text file lists one relative (or absolute) PLY path per line. Empty shards
are still created unless --skip-empty is used. Deterministic ordering unless --shuffle.
"""
from __future__ import annotations

import os
import sys
import json
import math
import random
import datetime as dt
from pathlib import Path
from typing import List

import click


def find_plys(root: Path, pattern: str) -> List[Path]:
    # Using glob pattern; '**/*.ply' by default. Ensure consistent sorting.
    paths = sorted(root.glob(pattern))
    return [p for p in paths if p.is_file() and p.suffix.lower() == '.ply']


def partition(items: List[Path], num_workers: int) -> List[List[Path]]:
    n = len(items)
    if num_workers <= 0:
        raise ValueError("num_workers must be > 0")
    base = n // num_workers
    rem = n % num_workers
    shards: List[List[Path]] = []
    start = 0
    for i in range(num_workers):
        size = base + (1 if i < rem else 0)
        end = start + size
        shards.append(items[start:end])
        start = end
    return shards


def write_shard_file(path: Path, entries: List[Path], root: Path, relative: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for p in entries:
            f.write((str(p.relative_to(root)) if relative else str(p)) + '\n')


def write_meta(path: Path, meta: dict) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)


@click.command()
@click.option('--ply-root', type=click.Path(exists=True, file_okay=False, path_type=Path), required=True, help='Root directory containing .ply files')
@click.option('--num-workers', type=int, required=True, help='Number of shards to create')
@click.option('--prefix', default='shards', show_default=True, help='Output file prefix')
@click.option('--out-root', type=click.Path(file_okay=False, path_type=Path), help='Pipeline output root (required if --skip-completed) to detect finished PLYs')
@click.option('--pattern', default='**/*.ply', show_default=True, help='Glob pattern relative to ply-root')
@click.option('--out-dir', type=click.Path(file_okay=False, path_type=Path), default=Path('.'), show_default=True, help='Directory to place shard files')
@click.option('--relative/--absolute', default=True, show_default=True, help='Store relative paths in shard files')
@click.option('--shuffle', is_flag=True, help='Shuffle PLY list before partitioning')
@click.option('--seed', type=int, default=None, help='Random seed (used only if --shuffle)')
@click.option('--skip-empty', is_flag=True, help='Do not write empty shard files')
@click.option('--skip-completed', is_flag=True, help='Exclude PLYs that already have completed outputs (annotations.json + summary .txt) in out-root')
def main(ply_root: Path, num_workers: int, prefix: str, pattern: str, out_dir: Path, relative: bool, shuffle: bool, seed: int | None, skip_empty: bool, out_root: Path | None, skip_completed: bool):
    ply_root = ply_root.resolve()
    out_dir = out_dir.resolve()

    plys = find_plys(ply_root, pattern)
    if not plys:
        click.echo("No PLY files found; exiting.", err=True)
        raise SystemExit(1)

    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(plys)

    skipped_completed = 0
    if skip_completed:
        if out_root is None:
            click.echo("--skip-completed requires --out-root", err=True)
            raise SystemExit(2)
        out_root = out_root.resolve()
        filtered = []
        for p in plys:
            rel = p.relative_to(ply_root)
            ply_stem = p.stem
            # Reconstruct pipeline per-PLY output directory: out_root/<relative_dirs>/<stem>/
            ply_dir = out_root / rel.parent / ply_stem
            ann_path = ply_dir / 'annotations.json'
            summary_txt = ply_dir / f'{ply_stem}.txt'
            if ann_path.is_file() and summary_txt.is_file():
                skipped_completed += 1
                continue
            filtered.append(p)
        plys = filtered

    shards = partition(plys, num_workers)

    shard_files = []
    for i, shard in enumerate(shards):
        filename = f"{prefix}_shard_{i:03d}_of_{num_workers}.txt"
        shard_path = out_dir / filename
        if shard or not skip_empty:
            write_shard_file(shard_path, shard, ply_root, relative)
            shard_files.append(str(shard_path))

    meta = {
        "timestamp": dt.datetime.utcnow().isoformat() + 'Z',
        "ply_root": str(ply_root),
        "num_workers": num_workers,
        "total_plys": len(plys),
        "pattern": pattern,
        "relative_paths": relative,
        "shuffled": shuffle,
        "seed": seed,
        "skip_empty": skip_empty,
        "skip_completed": skip_completed,
        "skipped_completed_count": skipped_completed,
        "shard_sizes": [len(s) for s in shards],
        "shard_files": shard_files,
        "command": ' '.join(sys.argv),
    }
    write_meta(out_dir / f"{prefix}_meta.json", meta)

    # Summary to stdout
    click.echo(f"Total PLYs: {len(plys)}")
    click.echo(f"Workers: {num_workers}")
    click.echo(f"Shard sizes: {meta['shard_sizes']}")
    if skip_completed:
        click.echo(f"Skipped already completed: {skipped_completed}")
    click.echo("Shard files written:")
    for sf in shard_files:
        click.echo(f"  {sf}")


if __name__ == '__main__':  # pragma: no cover (testing not requested)
    main()
