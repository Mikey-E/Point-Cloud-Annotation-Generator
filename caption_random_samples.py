#!/usr/bin/env python
"""
Caption N random images from each scene's image folder and generate a scene summary.

Behavior:
  - For each scene directory under INPUT_ROOT:
      * Find the IMAGE_FOLDER (e.g., "images")
      * Sample N random images (default 6)
      * Create IMAGE_FOLDER_captions/ at the same level
      * Caption each sampled image → <stem>.txt in IMAGE_FOLDER_captions/
      * Summarize captions → <scene_name>_caption_summary.txt at scene root
  - Resume-friendly: skips scenes with existing summaries; reuses existing per-image captions

Notes:
  - Requires OPENAI_API_KEY in environment.
  - Random seed defaults to 0 for reproducibility.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Optional

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
        "You are given multiple captions of the same object/scene from different images.\n"
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


def find_scenes(root: Path, image_folder_name: str) -> List[Path]:
    """Find all scene directories that contain an image folder."""
    scenes = []
    for item in root.iterdir():
        if item.is_dir():
            img_folder = item / image_folder_name
            if img_folder.is_dir() and img_folder.exists():
                scenes.append(item)
    return sorted(scenes)


def sample_images(image_folder: Path, num_samples: int, seed: int) -> List[Path]:
    """Sample N random images from the folder."""
    all_images = [p for p in image_folder.iterdir() if p.is_file() and is_image(str(p))]
    if len(all_images) <= num_samples:
        return sorted(all_images)
    
    random.seed(seed)
    sampled = random.sample(all_images, num_samples)
    return sorted(sampled)


@click.command()
@click.argument("input_root", type=click.Path(exists=True, path_type=Path))
@click.option("--image-folder", default="images", show_default=True,
              help="Name of the image folder within each scene (e.g., 'images')")
@click.option("--num-samples", "-n", default=6, show_default=True,
              help="Number of random images to sample per scene")
@click.option("--seed", default=0, show_default=True,
              help="Random seed for reproducibility")
@click.option("--caption-model", default="gpt-4o-mini", show_default=True)
@click.option("--caption-prompt", default="Describe only the main object(s). Omit any background or backdrop information.", show_default=True)
@click.option("--summary-model", default="gpt-4o-mini", show_default=True)
@click.option("--summary-extra", default=None, help="Extra instruction/constraints for summarization")
@click.option("--resume/--no-resume", default=True, show_default=True,
              help="Skip scenes with existing summaries; reuse existing per-image captions")
@click.option("--dry-run", is_flag=True, help="Skip API calls; create directories and empty files only")
@click.option("--limit", type=int, default=None, help="Process only the first N scenes (for testing)")
def main(input_root: Path, image_folder: str, num_samples: int, seed: int,
         caption_model: str, caption_prompt: str, summary_model: str, summary_extra: Optional[str],
         resume: bool, dry_run: bool, limit: Optional[int]):
    """Caption N random images from each scene and generate scene summaries.
    
    For each scene directory under INPUT_ROOT containing IMAGE_FOLDER:
    - Sample N random images
    - Create IMAGE_FOLDER_captions/ folder
    - Write per-image captions as <stem>.txt
    - Write scene summary as <scene_name>_caption_summary.txt at scene root
    """
    input_root = input_root.resolve()
    
    client = None if dry_run else openai_client()
    
    # Find all scenes
    scenes = find_scenes(input_root, image_folder)
    if not scenes:
        console.print(f"[yellow]No scenes with '{image_folder}' folder found.[/yellow]")
        raise SystemExit(1)
    
    if limit:
        scenes = scenes[:limit]
        console.print(f"[cyan]Processing first {limit} scenes (--limit)[/cyan]")
    
    console.print(f"[cyan]Found {len(scenes)} scene(s) to process[/cyan]")
    
    skipped = 0
    processed = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing scenes", total=len(scenes))
        
        for scene_dir in scenes:
            scene_name = scene_dir.name
            img_folder = scene_dir / image_folder
            captions_folder_name = f"{image_folder}_captions"
            captions_folder = scene_dir / captions_folder_name
            summary_path = scene_dir / f"{scene_name}_caption_summary.txt"
            
            # Check if summary already exists (resume mode)
            if resume and summary_path.exists():
                skipped += 1
                progress.advance(task)
                continue
            
            # Ensure captions folder exists
            ensure_dir(str(captions_folder))
            
            # Find which images to process
            # If we already have some captions, reuse those same images
            existing_captions = list(captions_folder.glob("*.txt"))
            if existing_captions and len(existing_captions) >= num_samples:
                # Use existing sampled images
                sampled_stems = [p.stem for p in existing_captions[:num_samples]]
                sampled_images = [img_folder / f"{stem}{ext}" 
                                for stem in sampled_stems 
                                for ext in IMAGE_EXTS 
                                if (img_folder / f"{stem}{ext}").exists()]
                sampled_images = sampled_images[:num_samples]
            else:
                # Sample new random images
                sampled_images = sample_images(img_folder, num_samples, seed)
            
            if not sampled_images:
                console.print(f"[yellow]Skipping {scene_name}: no images found[/yellow]")
                progress.advance(task)
                continue
            
            # Caption each sampled image
            captions: List[str] = []
            for img_path in sampled_images:
                stem = img_path.stem
                caption_path = captions_folder / f"{stem}.txt"
                
                if resume and caption_path.exists():
                    try:
                        cap = read_text(str(caption_path)).strip()
                        captions.append(cap)
                        continue
                    except Exception:
                        pass
                
                if dry_run:
                    cap = f"[DRY RUN] Caption for {img_path.name}"
                else:
                    try:
                        cap = caption_image(client, str(img_path), model=caption_model, prompt=caption_prompt)
                    except Exception as e:
                        cap = f"[ERROR during captioning: {e}]"
                        console.print(f"[yellow]Failed to caption {img_path.name}: {e}[/yellow]")
                
                captions.append(cap)
                try:
                    write_text(str(caption_path), cap)
                except Exception as e:
                    console.print(f"[yellow]Failed to write caption for {img_path.name}: {e}[/yellow]")
            
            # Generate scene summary
            if len(captions) == 1:
                summary = captions[0]
            else:
                if dry_run:
                    summary = f"[DRY RUN] Summary for {scene_name}"
                else:
                    try:
                        summary = summarize_captions(client, captions, model=summary_model, extra_instruction=summary_extra)
                    except Exception as e:
                        summary = f"[ERROR during summarization: {e}]"
                        console.print(f"[yellow]Failed to summarize {scene_name}: {e}[/yellow]")
            
            try:
                write_text(str(summary_path), summary)
                processed += 1
            except Exception as e:
                console.print(f"[yellow]Failed to write summary for {scene_name}: {e}[/yellow]")
            
            progress.advance(task)
    
    console.print(f"[green]✓ Processed: {processed} scenes[/green]")
    if skipped:
        console.print(f"[cyan]⊘ Skipped (already complete): {skipped} scenes[/cyan]")


if __name__ == "__main__":
    main()
