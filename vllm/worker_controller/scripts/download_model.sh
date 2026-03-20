#!/bin/bash
#PBS -q normal
#PBS -l select=1
#PBS -l walltime=2:00:00
#PBS -P personal
#PBS -N download-model
#PBS -j oe

cd "$PBS_O_WORKDIR" || exit $?

MODEL="${MODEL:-Qwen/Qwen3-8B}"
PERSISTENT_CACHE="$HOME/.cache/huggingface"

# PBS copies the script to a spool dir, so use PBS_O_WORKDIR for paths
REPO_ROOT="$(cd "$PBS_O_WORKDIR/../../.." && pwd)"
SIF_PATH="$REPO_ROOT/vllm-openai_latest.sif"

module load singularity

if [ ! -f "$SIF_PATH" ]; then
    echo "Pulling vllm container..."
    singularity pull "$SIF_PATH" docker://vllm/vllm-openai:latest
    if [ ! -f "$SIF_PATH" ]; then
        echo "ERROR: Container pull failed. Exiting."
        exit 1
    fi
fi

echo "Downloading $MODEL to $PERSISTENT_CACHE ..."

singularity exec \
    --bind "$PERSISTENT_CACHE:$PERSISTENT_CACHE" \
    --env HF_HOME="$PERSISTENT_CACHE" \
    --env HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}" \
    "$SIF_PATH" \
    huggingface-cli download "$MODEL"

echo "Download complete: $PERSISTENT_CACHE"
