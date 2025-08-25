#!/usr/bin/env python
"""
End-to-end pipeline to:
  1) Render multi-view images from PLY point clouds
  2) Caption each image via OpenAI chat.completions (vision)
  3) Summarize per-view captions into a single caption per point cloud

Outputs:
  - For each PLY: <out>/<stem>/annotations.json with per-image captions and summary
  - Root JSONL: <out>/dataset_annotations.jsonl (one record per PLY)
"""
from __future__ import annotations

import os
import sys
import json
import time
import glob
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import click
import orjson
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from openai import OpenAI

import ply_pictures as pp


console = Console()


@dataclass
class CaptionResult:
    image: str
    caption: str


@dataclass
class PointCloudAnnotation:
    ply_path: str
    render_dir: str
    per_view: List[CaptionResult]
    summary: str


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, data) -> None:
    with open(path, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))


def read_json(path: str):
    with open(path, "rb") as f:
        return orjson.loads(f.read())


def detect_images(render_dir: str) -> List[str]:
    pats = ["*.png", "*.jpg", "*.jpeg"]
    files: List[str] = []
    for p in pats:
        files.extend(sorted(glob.glob(os.path.join(render_dir, p))))
    return files


def render_views_for_ply(ply_path: str, out_root: str, width: int, height: int, point_size: float,
                         backend: str, max_points: int, radius_scale: float, tight: bool,
                         margin: int, pad_frac: float) -> str:
    pcd = pp.load_ply_as_o3d(ply_path)
    stem = os.path.splitext(os.path.basename(ply_path))[0]
    # mimic ply_pictures main naming for discoverability
    def fmt(x):
        return ("%g" % x)
    render_dir = os.path.join(out_root, f"renders_{stem}_{width}_{height}_{fmt(point_size)}_{max_points}_{fmt(radius_scale)}_{tight}_{margin}")
    pp.render_views(pcd, render_dir, w=width, h=height, point_size=point_size, backend=backend,
                    max_points=max_points, radius_scale=radius_scale, tight=tight, margin=margin, pad_frac=pad_frac)
    return render_dir


def openai_client() -> OpenAI:
    # Relies on OPENAI_API_KEY in environment
    return OpenAI()


def caption_image(client: OpenAI, image_path: str, model: str, prompt: str) -> str:
    # Lightweight re-use of logic from image_caption_api.py without base64 helper duplication
    import base64, mimetypes
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    mt, _ = mimetypes.guess_type(image_path)
    if mt is None:
        mt = "image/png"
    comp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an image captioning assistant. Strictly omit any mention of the background or backdrop, "
                    "describe the main object(s) and its visual attributes."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mt};base64,{b64}"}},
                ],
            },
        ],
    )
    return (comp.choices[0].message.content or "").strip()


def summarize_captions(client: OpenAI, captions: List[str], model: str, extra_instruction: Optional[str] = None) -> str:
    parts = "\n".join(f"- {c}" for c in captions if c)
    prompt = (
        "You are given multiple captions of the same 3D object from different views.\n"
        "Summarize them into a single, concise, self-contained caption (1-3 sentences) that captures the object's identity, shape, parts, color/material, and any notable features.\n"
        "Ignore any descriptions of the background or backdrop."
    )
    if extra_instruction:
        prompt += f"\nConstraints: {extra_instruction}"
    prompt += f"\n\nCaptions:\n{parts}"
    comp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return (comp.choices[0].message.content or "").strip()


def maybe_load_cached(ann_path: str) -> Optional[PointCloudAnnotation]:
    if os.path.isfile(ann_path):
        data = read_json(ann_path)
        try:
            per_view = [CaptionResult(**pv) for pv in data.get("per_view", [])]
            return PointCloudAnnotation(
                ply_path=data["ply_path"],
                render_dir=data["render_dir"],
                per_view=per_view,
                summary=data.get("summary", ""),
            )
        except Exception:
            return None
    return None


def write_jsonl(path: str, record: dict) -> None:
    with open(path, "ab") as f:
        f.write(orjson.dumps(record) + b"\n")


def jsonl_record_from_annotation(ann: PointCloudAnnotation) -> dict:
    """Build the compact JSONL record without per-view image/caption pairs."""
    return {
        "ply_path": ann.ply_path,
        "render_dir": ann.render_dir,
        "summary": ann.summary,
    }


@click.command()
@click.argument("ply_root", type=click.Path(exists=True, path_type=Path))
@click.option("--out", "out_root", type=click.Path(path_type=Path), default=Path("output"), show_default=True,
              help="Output directory for renders and annotations")
@click.option("--pattern", default="**/*.ply", show_default=True, help="Glob to find PLYs under ply_root")
@click.option("--width", default=1280, show_default=True, help="Render width")
@click.option("--height", default=960, show_default=True, help="Render height")
@click.option("--point-size", default=2.5, show_default=True, help="Point size for rendering")
@click.option("--backend", type=click.Choice(["auto","offscreen","legacy","mpl"]), default="mpl", show_default=True)
@click.option("--max-points", default=200000, show_default=True, help="Max points when using mpl fallback")
@click.option("--radius-scale", default=1.2, show_default=True)
@click.option("--tight/--no-tight", default=False, show_default=True)
@click.option("--margin", default=4, show_default=True)
@click.option("--pad-frac", default=0.05, show_default=True)
@click.option("--caption-model", default="gpt-4o-mini", show_default=True)
@click.option("--caption-prompt", default="Describe only the main object(s). Omit any background or backdrop information.", show_default=True)
@click.option("--summary-model", default="gpt-4o-mini", show_default=True)
@click.option("--summary-extra", default=None, help="Extra instruction/constraints for summarization")
@click.option("--resume/--no-resume", default=True, show_default=True, help="Reuse cached annotations if present")
@click.option("--dry-run", is_flag=True, help="Skip API calls; only render and emit stubs")
def main(ply_root: Path, out_root: Path, pattern: str, width: int, height: int, point_size: float,
         backend: str, max_points: int, radius_scale: float, tight: bool, margin: int, pad_frac: float,
         caption_model: str, caption_prompt: str, summary_model: str, summary_extra: Optional[str], resume: bool,
         dry_run: bool):
    """Process a dataset of PLYs and generate captions + summaries."""
    ensure_dir(str(out_root))
    jsonl_path = out_root / "dataset_annotations.jsonl"

    ply_paths = [str(p) for p in sorted(ply_root.glob(pattern)) if p.is_file()]
    if not ply_paths:
        console.print("[yellow]No PLY files found.[/yellow]")
        raise SystemExit(1)

    client = None if dry_run else openai_client()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing PLYs", total=len(ply_paths))
        for ply_path in ply_paths:
            # Preserve input directory structure under the output root
            rel_ply_path = os.path.relpath(ply_path, start=str(ply_root))
            rel_dir = os.path.dirname(rel_ply_path)
            stem = os.path.splitext(os.path.basename(ply_path))[0]
            ply_out_dir = out_root / rel_dir / stem if rel_dir else out_root / stem
            ensure_dir(str(ply_out_dir))
            ann_path = ply_out_dir / "annotations.json"

            cached = maybe_load_cached(str(ann_path)) if resume else None
            if cached and cached.summary:
                console.print(f"[green]Cached[/green] {stem}")
                write_jsonl(str(jsonl_path), jsonl_record_from_annotation(cached))
                progress.advance(task)
                continue

            render_dir = render_views_for_ply(
                ply_path, str(ply_out_dir), width, height, point_size, backend, max_points, radius_scale, tight, margin, pad_frac
            )

            images = detect_images(render_dir)
            if not images:
                console.print(f"[yellow]No images rendered for {stem}[/yellow]")
                progress.advance(task)
                continue

            per_view: List[CaptionResult] = []
            if dry_run:
                for img in images:
                    per_view.append(CaptionResult(image=os.path.relpath(img, start=str(ply_out_dir)), caption=""))
                summary = ""
            else:
                # Caption images
                for img in images:
                    try:
                        cap = caption_image(client, img, model=caption_model, prompt=caption_prompt)
                    except Exception as e:
                        cap = f"[ERROR during captioning: {e}]"
                    per_view.append(CaptionResult(image=os.path.relpath(img, start=str(ply_out_dir)), caption=cap))
                # Summarize
                try:
                    summary = summarize_captions(client, [pv.caption for pv in per_view], model=summary_model, extra_instruction=summary_extra)
                except Exception as e:
                    summary = f"[ERROR during summarization: {e}]"

            ann = PointCloudAnnotation(
                ply_path=rel_ply_path,
                render_dir=os.path.relpath(render_dir, start=str(ply_out_dir)),
                per_view=per_view,
                summary=summary,
            )

            write_json(str(ann_path), asdict(ann))
            write_jsonl(str(jsonl_path), jsonl_record_from_annotation(ann))
            console.print(f"[cyan]Done[/cyan] {stem}")
            progress.advance(task)

if __name__ == "__main__":
    main()