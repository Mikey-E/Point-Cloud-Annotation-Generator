#!/bin/bash
# Submit multiple shard jobs sequentially via sbatch.
# Usage: ./multi-shard_slurm_pipeline.sh <total_shards> [delay_seconds]
# Example: ./multi-shard_slurm_pipeline.sh 16 2
# This will sbatch 16 jobs (indices 0..15) with a 2s pause between submissions.

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <total_shards> [delay_seconds]" >&2
  exit 1
fi
TOTAL_SHARDS=$1
DELAY=${2:-0}
if ! [[ "$TOTAL_SHARDS" =~ ^[0-9]+$ ]] || [ "$TOTAL_SHARDS" -le 0 ]; then
  echo "ERROR: total_shards must be positive integer" >&2
  exit 2
fi
if ! [[ "$DELAY" =~ ^[0-9]+$ ]]; then
  echo "ERROR: delay_seconds must be non-negative integer" >&2
  exit 2
fi

SCRIPT=slurm_pipeline.sh
if [ ! -f "$SCRIPT" ]; then
  echo "ERROR: $SCRIPT not found in current directory." >&2
  exit 3
fi

echo "Submitting $TOTAL_SHARDS shard jobs using $SCRIPT"
for (( i=0; i < TOTAL_SHARDS; i++ )); do
  echo "Submitting shard $i of $TOTAL_SHARDS"
  sbatch "$SCRIPT" "$i" "$TOTAL_SHARDS"
  if [ "$DELAY" -gt 0 ]; then
    sleep "$DELAY"
  fi
done

echo "All shard jobs submitted."
