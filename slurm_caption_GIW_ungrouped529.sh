#!/bin/bash
#SBATCH --account=3dllms
#SBATCH --partition=mb
#SBATCH --job-name=caption_GIW_ungrouped529
#SBATCH --output=./slurm_logs/caption_GIW_ungrouped529_%j.out
#SBATCH --error=./slurm_logs/caption_GIW_ungrouped529_%j.out
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00

set -euo pipefail

# Dataset location
DATA_ROOT=/project/3dllms/melgin/datasets/GIW/ungrouped529

# Flags to avoid hard exits
MISSING_API=0
SKIP_CONDA=0

mkdir -p ./slurm_logs

# Ensure OPENAI_API_KEY is set; if missing, try sourcing export script
if [ -z "${OPENAI_API_KEY:-}" ]; then
	if [ -f ./export_openai_api_key.sh ]; then
		echo "OPENAI_API_KEY not set; sourcing ./export_openai_api_key.sh"
		# shellcheck source=export_openai_api_key.sh
		. ./export_openai_api_key.sh || true
	elif [ -f "$HOME/export_openai_api_key.sh" ]; then
		echo "OPENAI_API_KEY not set; sourcing $HOME/export_openai_api_key.sh"
		# shellcheck source=$HOME/export_openai_api_key.sh
		. "$HOME/export_openai_api_key.sh" || true
	fi
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
	echo "WARNING: OPENAI_API_KEY is not set. Skipping processing." >&2
	MISSING_API=1
fi

# Make 'conda activate' work in non-interactive shells
if [ -n "${CONDA_INSTALL_PATH:-}" ]; then
	CONDA_SH=$CONDA_INSTALL_PATH/etc/profile.d/conda.sh
	if [ ! -e "$CONDA_SH" ]; then
		echo "WARNING: $CONDA_SH does not exist; skipping conda activation." >&2
		SKIP_CONDA=1
	else
		# shellcheck disable=SC1090
		source "$CONDA_SH" || true
	fi
else
	CONDA_SH=/project/3dllms/melgin/conda/etc/profile.d/conda.sh
	echo "WARNING: CONDA_INSTALL_PATH is not set. Trying $CONDA_SH"
	if [ ! -e "$CONDA_SH" ]; then
		echo "WARNING: $CONDA_SH does not exist; skipping conda activation." >&2
		SKIP_CONDA=1
	else
		# shellcheck disable=SC1090
		source "$CONDA_SH" || true
	fi
fi

if [ "$SKIP_CONDA" -eq 0 ]; then
	conda activate pc_anno_gen || echo "[WARN] conda activate pc_anno_gen failed; proceeding without it"
fi

if [ "$MISSING_API" -eq 1 ]; then
	echo "Skipping all processing due to missing OPENAI_API_KEY."
	exit 1
else
	echo "Processing all scenes in $DATA_ROOT"
	echo "Started at: $(date)"
	
	python caption_random_samples.py "$DATA_ROOT" \
		--caption-model gpt-5-nano \
		--summary-model gpt-5-nano \
		--num-samples 6 \
		--seed 0
	
	echo "Completed at: $(date)"
	echo "All scenes processed."
fi
