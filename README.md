# Point-Cloud Annotation Generator

Generate multi-view images from point cloud (PLY) files, caption those images with a Vision-LMM API, and summarize per-view captions into a single caption per point cloud.

## Features

- PLY loader with color normalization (Open3D)
- Rendering backends: matplotlib (default), Open3D offscreen, or Open3D legacy visualizer
- Image captioning via OpenAI (You need an API key and funds)
- Per-point-cloud caption summarization
- Simple caching/resume and structured outputs (JSON + JSONL)

## Quick start

1) Create and activate the conda environment (Python 3.13 required)

```bash
conda env create -f environment.yml
conda activate pc_anno_gen
# optional: pip install -r requirements.txt  # only if you prefer pip inside conda
```

2) Set your OpenAI API key (required for online captioning/summarization)

```bash
export OPENAI_API_KEY=sk-...
```

3) Run the end-to-end pipeline on a folder of PLYs

```bash
python pipeline.py /path/to/ply_dataset --out /path/to/output \
  --width 1280 --height 960 \
  --caption-model gpt-4o-mini --summary-model gpt-4o-mini
```

Outputs:
- The output directory mirrors the input folder structure. For each input PLY, outputs are written under: `<out>/<relative_input_dirs>/<ply_stem>/`
- Inside each PLY folder: a descriptive render directory, an `annotations.json` (per-image captions + summary), and a `<ply_stem>.txt` file containing only the summary text
- At the output root: `dataset_annotations.jsonl` (one record per PLY; includes summary and paths only — per-view captions are stored in each PLY's annotations.json)

## Repository layout

- `ply_pictures.py` – point cloud loading and rendering utilities
- `image_caption_api.py` – minimal image caption example via OpenAI
- `pipeline.py` – end-to-end CLI to render, caption, and summarize
- `caption_preexisting_images.py` – caption existing images in a folder tree and write per-image captions plus per-folder summaries

## Notes on rendering in headless Linux

Default backend is matplotlib, which works in most headless environments. For higher-fidelity renders, you can choose Open3D backends with `--backend offscreen|legacy`. The offscreen backend requires proper EGL/OSMesa setup; if unavailable, use `legacy` or stay on `mpl`. (Untested)

## Caption preexisting images

If you already have image folders (e.g., renders from another pipeline) and just want captions:

```bash
python caption_preexisting_images.py /path/to/images_root \
  --out /path/to/output_images \
  --caption-model gpt-4o-mini \
  --summary-model gpt-4o-mini
```

Behavior:
- Recursively mirrors the input directory tree under `--out`
- For each image: writes `<image_stem>.txt` containing its caption
- For each folder with images: writes `<folder_name>_summary.txt` that summarizes captions in that folder
- Requires `OPENAI_API_KEY` set in the environment

## License

MIT