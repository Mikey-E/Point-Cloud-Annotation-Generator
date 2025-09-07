#!/usr/bin/env python
"""
Caption preexisting images and write per-image .txt plus a per-folder summary.

Behavior:
  - Recursively walk the input path.
  - For each folder that contains at least one image (*.png, *.jpg, *.jpeg):
      * Mirror the directory under the output root (omit dirs with no images).
      * For every image in that folder (non-recursive per folder):
          - Create <image_stem>.txt with the caption.
      * Create <folder_name>_summary.txt in that folder:
          - If 1 image, copy that caption verbatim.
          - If >1 image, summarize using same method as pipeline.py.

Notes:
  - Requires OPENAI_API_KEY in environment.
  - Defaults mirror pipeline.py model/prompt choices; can be overridden via CLI.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from openai import OpenAI


console = Console()


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_text(path: str, content: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        f.write(content or "")


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def is_image(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def openai_client() -> OpenAI:
    return OpenAI()


def caption_image(client: OpenAI, image_path: str, model: str, prompt: str) -> str:
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
        "You are given multiple captions of the same object from different images.\n"
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


def iter_image_folders(root: Path):
    """Yield (dirpath, [image_paths]) for folders under root that contain images directly."""
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        images = [str(Path(dirpath) / fn) for fn in filenames if is_image(fn)]
        if images:
            yield Path(dirpath), sorted(images)


@click.command()
@click.argument("input_root", type=click.Path(exists=True, path_type=Path))
@click.option("--out", "out_root", type=click.Path(path_type=Path), default=Path("output_images"), show_default=True,
              help="Output directory root for captions and summaries (mirrors input structure)")
@click.option("--caption-model", default="gpt-4o-mini", show_default=True)
@click.option("--caption-prompt", default="Describe only the main object(s). Omit any background or backdrop information.", show_default=True)
@click.option("--summary-model", default="gpt-4o-mini", show_default=True)
@click.option("--summary-extra", default=None, help="Extra instruction/constraints for summarization")
@click.option("--resume/--no-resume", default=True, show_default=True, help="Reuse existing per-image .txt captions if present")
@click.option("--dry-run", is_flag=True, help="Skip API calls; create directories and empty files only")
def main(input_root: Path, out_root: Path, caption_model: str, caption_prompt: str,
         summary_model: str, summary_extra: Optional[str], resume: bool, dry_run: bool):
    """Caption existing images under INPUT_ROOT and write per-image .txt plus per-folder summary."""
    input_root = input_root.resolve()
    ensure_dir(str(out_root))

    client = None if dry_run else openai_client()

    # Count folders with images for progress
    folders = list(iter_image_folders(input_root))
    if not folders:
        console.print("[yellow]No image-containing folders found.[/yellow]")
        raise SystemExit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing folders", total=len(folders))
        for dirpath, image_paths in folders:
            rel_dir = dirpath.relative_to(input_root)
            out_dir = out_root / rel_dir
            ensure_dir(str(out_dir))

            captions: List[str] = []
            for img in image_paths:
                stem = Path(img).stem
                out_txt = out_dir / f"{stem}.txt"
                if resume and out_txt.exists():
                    try:
                        cap = read_text(str(out_txt)).strip()
                        captions.append(cap)
                        continue
                    except Exception:
                        # Fall through to regenerate
                        pass
                if dry_run:
                    cap = ""
                else:
                    try:
                        cap = caption_image(client, img, model=caption_model, prompt=caption_prompt)
                    except Exception as e:
                        cap = f"[ERROR during captioning: {e}]"
                captions.append(cap)
                try:
                    write_text(str(out_txt), cap)
                except Exception as e:
                    console.print(f"[yellow]Failed to write caption for {img}: {e}[/yellow]")

            # Per-folder summary file goes one level up from the image folder
            folder_name = dirpath.name or "folder"
            parent_out = out_dir.parent
            summary_path = parent_out / f"{folder_name}_summary.txt"
            if len(captions) == 1:
                summary = captions[0]
            else:
                if dry_run:
                    summary = ""
                else:
                    try:
                        summary = summarize_captions(client, captions, model=summary_model, extra_instruction=summary_extra)
                    except Exception as e:
                        summary = f"[ERROR during summarization: {e}]"
            try:
                write_text(str(summary_path), summary)
            except Exception as e:
                console.print(f"[yellow]Failed to write summary for {dirpath}: {e}[/yellow]")

            progress.advance(task)


if __name__ == "__main__":
    main()
