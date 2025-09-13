#!/bin/bash
#SBATCH --account=3dllms
#SBATCH --partition=non-investor
#SBATCH --job-name=pipeline
#SBATCH --output=./slurm_logs/pipeline_%j.out
#SBATCH --error=./slurm_logs/pipeline_%j.out
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00

# Ensure OPENAI_API_KEY is set; if missing, try sourcing export script
if [ -z "$OPENAI_API_KEY" ]; then
    if [ -f ./export_openai_api_key.sh ]; then
        echo "OPENAI_API_KEY not set; sourcing ./export_openai_api_key.sh"
        # shellcheck source=export_openai_api_key.sh
        . ./export_openai_api_key.sh
    elif [ -f "$HOME/export_openai_api_key.sh" ]; then
        echo "OPENAI_API_KEY not set; sourcing $HOME/export_openai_api_key.sh"
        # shellcheck source=$HOME/export_openai_api_key.sh
        . "$HOME/export_openai_api_key.sh"
    fi
fi
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable is not set."
    exit 1
fi

#This ensures "conda activate <env>" works in non-interactive shells.
#(running "conda init" every time won't work.)
if [ -n "$CONDA_INSTALL_PATH" ]; then
    CONDA_SH=$CONDA_INSTALL_PATH/etc/profile.d/conda.sh
    if [ ! -e "$CONDA_SH" ]; then
        echo "ERROR: $CONDA_SH does not exist."
        exit 1
    fi
    source "$CONDA_SH"
else
    CONDA_SH=/project/3dllms/melgin/conda/etc/profile.d/conda.sh
    echo "WARNING: CONDA_INSTALL_PATH is not set. Trying $CONDA_SH"
    if [ ! -e "$CONDA_SH" ]; then
        echo "ERROR: $CONDA_SH does not exist."
        exit 1
    fi
    source "$CONDA_SH"
fi
# Now the activation should work
conda activate pc_anno_gen

# legacy run commands below
# python pipeline.py /gscratch/melgin/CEA/Crops3D_debug40/ --out ./output_test/ --caption-model gpt-5-nano --summary-model gpt-5-nano
# python pipeline.py /project/3dllms/melgin/datasets/3d-grand_unzipped \
#     --out /project/3dllms/melgin/datasets/3d-grand_unzipped_gpt-5-nano \
#     --caption-model gpt-5-nano \
#     --summary-model gpt-5-nano

#!/bin/bash
# Single-shard processing.
# Usage standalone:  sbatch slurm_pipeline.sh [shard_index] [total_shards] [shard_dir] [data_root] [out_root]
# Array job:         add '#SBATCH --array=0-(N-1)' and omit shard_index argument; optionally pass remaining args.

DEFAULT_SHARD_DIR=/project/3dllms/melgin/Point-Cloud-Annotation-Generator/shards_3d-front_triage_out_dir
DEFAULT_DATA_ROOT=/project/3dllms/melgin/datasets/3d-grand_unzipped
DEFAULT_OUT_ROOT=/project/3dllms/melgin/datasets/3d-grand_unzipped_gpt-5-nano

# Positional arguments (all optional except shard index unless using array):
# $1 shard_index (or SLURM_ARRAY_TASK_ID)
# $2 total_shards
# $3 shard_dir
# $4 data_root
# $5 out_root
SHARD_INDEX="${1:-${SLURM_ARRAY_TASK_ID:-}}"
TOTAL_SHARDS="${2:-1}"
SHARD_DIR="${3:-$DEFAULT_SHARD_DIR}"
DATA_ROOT="${4:-$DEFAULT_DATA_ROOT}"
OUT_ROOT="${5:-$DEFAULT_OUT_ROOT}"
if ! [[ "$TOTAL_SHARDS" =~ ^[0-9]+$ ]] || [ "$TOTAL_SHARDS" -le 0 ]; then
    echo "ERROR: total_shards must be positive integer (got '$TOTAL_SHARDS')." >&2
    exit 2
fi
if [ -z "$SHARD_INDEX" ]; then
    echo "ERROR: No shard index provided and SLURM_ARRAY_TASK_ID not set." >&2
    exit 2
fi
if ! [[ "$SHARD_INDEX" =~ ^[0-9]+$ ]]; then
    echo "ERROR: shard index must be integer (got '$SHARD_INDEX')." >&2
    exit 2
fi
if [ "$SHARD_INDEX" -ge "$TOTAL_SHARDS" ]; then
    echo "ERROR: shard index $SHARD_INDEX out of range (0..$((TOTAL_SHARDS-1)))." >&2
    exit 2
fi

printf -v SHARD_FILE "%s/shards_shard_%03d_of_%d.txt" "$SHARD_DIR" "$SHARD_INDEX" "$TOTAL_SHARDS"
if [ ! -f "$SHARD_FILE" ]; then
    echo "ERROR: Shard file not found: $SHARD_FILE" >&2
    exit 3
fi

echo "=== Processing shard index $SHARD_INDEX of $TOTAL_SHARDS ==="
echo "Shard file : $SHARD_FILE"
echo "Shard dir  : $SHARD_DIR"
echo "Data root  : $DATA_ROOT"
echo "Out root   : $OUT_ROOT"
python pipeline.py "$DATA_ROOT" \
  --out "$OUT_ROOT" \
  --shard-manifest "$SHARD_FILE" \
  --caption-model gpt-5-nano \
  --summary-model gpt-5-nano
STATUS=$?
if [ $STATUS -ne 0 ]; then
  echo "ERROR: Shard $SHARD_INDEX failed with exit code $STATUS" >&2
  exit $STATUS
fi
echo "Shard $SHARD_INDEX of $TOTAL_SHARDS completed successfully."