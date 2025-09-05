#!/bin/bash
#SBATCH --account=3dllms
#SBATCH --partition=non-investor
#SBATCH --job-name=long_folders_pipeline
#SBATCH --output=./slurm_logs/long_folders_%j.out
#SBATCH --error=./slurm_logs/long_folders_%j.out
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00

set -euo pipefail

# Base directory that contains the long folders to process
# Each folder below contains an "images" subdirectory with inputs
BASE_DIR=/project/3dllms/DATASETS/CONVERTED/normal/100percent/ungrouped

# Output root where annotations + renders will be written (per-folder subdir)
OUT_ROOT=${OUT_ROOT:-/gscratch/melgin/long_folders_caption_test}

# Folders to process (names as they appear under $BASE_DIR)
FOLDERS=(
  03_04_2024_S_L_ApplesLong_F_1
  03_04_2024_S_L_FruitsLong2_F_1
  03_04_2024_S_L_FruitsLong3_F_1
  03_04_2024_S_L_FruitsLong_F_1
  03_07_2024_S_L_OrangesLong1_F_1
  04_04_2024_S_C_ApplesLong1_F_1
  04_04_2024_S_C_OrangesLong1_F_1
  04_04_2024_S_C_OrangesLong2_F_1
  04_04_2024_W2_C_ApplesLong1_F_1
  04_04_2024_W2_C_ApplesLong2_F_1
  04_04_2024_W2_C_AvacadosLong1_F_1
  04_04_2024_W2_C_BananasLong1_F_1
  04_04_2024_W2_C_FruitsLong1_F_1
  04_04_2024_W2_C_PotatoesLong1_V_1
  04_04_2024_W2_C_PotatoesLong2_V_1
  04_04_2024_W_C_FruitsLong1_F_1
)

mkdir -p ./slurm_logs
mkdir -p "$OUT_ROOT"

# Flags to avoid hard exits
MISSING_API=0
SKIP_CONDA=0

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
else
	echo "Processing folders under $BASE_DIR"
	for folder in "${FOLDERS[@]}"; do
		IN_DIR="$BASE_DIR/$folder/images"
		if [ ! -d "$IN_DIR" ]; then
			echo "[WARN] Skipping missing directory: $IN_DIR"
			continue
		fi
		OUT_DIR="$OUT_ROOT/$folder"
		echo "[RUN] caption_preexisting_images.py: in=$IN_DIR out=$OUT_DIR"
		python caption_preexisting_images.py "$IN_DIR" \
			--out "$OUT_DIR" \
			--caption-model gpt-5-nano \
			--summary-model gpt-5-nano
	done
fi

echo "All requested folders processed. Outputs at $OUT_ROOT"