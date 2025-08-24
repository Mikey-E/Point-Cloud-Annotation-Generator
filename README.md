# Point-Cloud Annotation Generator

Generate multi-view images from point cloud (PLY) files, caption those images with a Vision-LMM API, and summarize per-view captions into a single caption per point cloud.

## Features

- Robust PLY loader with color normalization (Open3D)
- Multi-backend rendering (Open3D offscreen, legacy visualizer, matplotlib fallback)
- Image captioning via OpenAI
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
  --width 1280 --height 960 --backend auto \
  --caption-model gpt-4o-mini --summary-model gpt-4o-mini
```

Outputs:
- For each PLY: rendered images are stored under a descriptive render directory
- For each PLY: an `annotations.json` with per-image captions and a `summary`
- At the output root: `dataset_annotations.jsonl` one record per PLY

## Repository layout

- `ply_pictures.py` – point cloud loading and rendering utilities
- `image_caption_api.py` – minimal image caption example via OpenAI
- `pipeline.py` – end-to-end CLI to render, caption, and summarize

## Notes on rendering in headless Linux

Open3D OffscreenRenderer requires proper EGL/OSMesa setup. This project falls back to the legacy visualizer and finally to a matplotlib-based renderer if offscreen rendering is unavailable. You can force a backend with `--backend offscreen|legacy|mpl`.

## Development

- Format/lint: suggested tools include black/ruff/mypy (add as desired)
- Tests: prefer pytest with responses/vcrpy to mock API calls
- CI: add a GitHub Actions workflow to run lint and tests on PRs

## License

MIT