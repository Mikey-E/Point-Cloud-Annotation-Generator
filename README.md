# Point-Cloud Annotation Generator

Generate multi-view images from point cloud (PLY) files, caption those images with a Vision-LMM API, and summarize per-view captions into a single caption per point cloud.

## Features

- PLY loader with color normalization (Open3D)
- Rendering backends: matplotlib (default), Open3D offscreen, or Open3D legacy visualizer
- Image captioning via OpenAI (You need an API key and funds)
- Per-point-cloud caption summarization
- Simple caching/resume and structured outputs (JSON + JSONL)
- Shard manifest based parallel processing (HPC / SLURM friendly)

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
- At the output root: either a single `dataset_annotations.jsonl` (no sharding) or multiple `dataset_annotations__<shard_stem>.jsonl` files (when using `--shard-manifest`). Each JSONL has one record per processed PLY.

## Repository layout

- `ply_pictures.py` – point cloud loading and rendering utilities
- `image_caption_api.py` – minimal image caption example via OpenAI
- `pipeline.py` – end-to-end CLI to render, caption, and summarize
- `make_shards.py` – generate shard manifest text files for parallel runs
- `caption_preexisting_images.py` – caption existing images in a folder tree and write per-image captions plus per-folder summaries
- `slurm_pipeline.sh` – process a single shard (index-based) on an HPC cluster
- `multi-shard_slurm_pipeline.sh` – convenience loop to submit all shard jobs via `sbatch`

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

## Sharding & Parallelization

Large datasets can be split into shards so multiple jobs (e.g. SLURM array or sequential submissions) can process disjoint subsets safely.

### 1. Generate shard manifests

Use `make_shards.py` to create N shard text files plus a meta JSON:

```bash
python make_shards.py \
  --ply-root /path/to/ply_dataset \
  --num-workers 16 \
  --prefix shards \
  --out-dir /path/to/shards_dir \
  --relative \
  --pattern '**/*.ply'
```

This produces files like:
```
shards_shard_000_of_16.txt
...
shards_shard_015_of_16.txt
shards_meta.json
```
Each shard file lists one PLY path per line (relative to `--ply-root` when `--relative` used). Comments (`# ...`) and blank lines are ignored by the pipeline.

Optional flags:
- `--shuffle` (optionally with `--seed`) to randomize distribution
- `--skip-empty` to omit writing empty shard files
- `--skip-completed --out-root <existing_output_dir>` to exclude PLYs whose outputs already exist (both `annotations.json` and `<stem>.txt`)

### 2. Run the pipeline on a shard

Provide the shard manifest via `--shard-manifest`:
```bash
python pipeline.py /path/to/ply_dataset \
  --out /path/to/output_root \
  --shard-manifest /path/to/shards_dir/shards_shard_003_of_16.txt \
  --caption-model gpt-4o-mini --summary-model gpt-4o-mini
```

The pipeline writes a per-shard JSONL named:
```
dataset_annotations__shards_shard_003_of_16.jsonl
```

### 3. Merging shard outputs (concept)

All per-shard JSONLs are line-oriented with one record per PLY. A future helper script can concatenate & de-duplicate on `ply_path`. (Not yet included.) A simple manual merge:
```bash
cat /path/to/output_root/dataset_annotations__shards_shard_*.jsonl > merged_dataset_annotations.jsonl
```
Ensure you do not process the same shard twice or you'll introduce duplicate records.

### 4. SLURM single-shard script

`slurm_pipeline.sh` activates the conda env, then processes exactly one shard. Positional arguments (all optional except shard index unless using an array job):
```
sbatch slurm_pipeline.sh [shard_index] [total_shards] [shard_dir] [data_root] [out_root]
```
Defaults are embedded in the script. Example running shard 2 of 16:
```bash
sbatch slurm_pipeline.sh 2 16 /path/to/shards_dir /path/to/ply_dataset /path/to/output_root
```
Array job usage (add `#SBATCH --array=0-15` to the script header or pass with sbatch) then omit the first arg:
```bash
sbatch slurm_pipeline.sh '' 16 /path/to/shards_dir /path/to/ply_dataset /path/to/output_root
```

### 5. Batch submit helper

`multi-shard_slurm_pipeline.sh` loops and submits one job per shard index:
```bash
./multi-shard_slurm_pipeline.sh 16
```
Add a delay (seconds) between submissions:
```bash
./multi-shard_slurm_pipeline.sh 16 2
```
To override directories, edit the `sbatch` command to append extra args matching those of `slurm_pipeline.sh` (or adjust the script to forward them).

### 6. Resume / caching

The pipeline skips re-captioning a PLY if its `annotations.json` already contains a summary (unless `--no-resume`). Per-view images are re-used; a missing summary `.txt` is regenerated.

### 7. Safety: path normalization

The pipeline resolves roots and rejects any shard entries that lie outside `ply_root` or normalize to paths beginning with `..`, preventing accidental writes outside the intended output tree.

### 8. Cleaning / ignoring outputs

Add (or confirm) ignore rules for large artifacts in `.gitignore` (already ignoring image renders and base JSONL). If using many shards, consider also ignoring `dataset_annotations__*.jsonl`. But it is expected that you'll probably want to write your new dataset outside this working directory anyway.

```gitignore
dataset_annotations__*.jsonl
```

---
For future enhancements (merge utility, retry/backoff, async captioning), open an issue or extend the scripts as needed.