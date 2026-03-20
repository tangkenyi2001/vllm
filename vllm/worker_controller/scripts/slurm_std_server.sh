#!/bin/bash
#PBS -q normal
#PBS -l select=1:ngpus=1
#PBS -l walltime=4:00:00
#PBS -P personal
#PBS -N vllm-std-server
#PBS -j oe

cd "$PBS_O_WORKDIR" || exit $?

REPO_ROOT="$(cd "$PBS_O_WORKDIR/../../.." && pwd)"
LOG_DIR="$PBS_O_WORKDIR/../logs"
SIF_PATH="$REPO_ROOT/vllm-openai_latest.sif"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
PERSISTENT_CACHE="$HOME/.cache/huggingface"
HF_CACHE="/dev/shm/models"

mkdir -p "$LOG_DIR"

module load singularity

# Copy model from persistent storage into RAM-backed /dev/shm for fast loading
echo "Copying model to /dev/shm/models..."
mkdir -p "$HF_CACHE"
rsync -a "$PERSISTENT_CACHE/" "$HF_CACHE/"
echo "Copy complete."

if [ ! -f "$SIF_PATH" ]; then
    echo "Pulling vllm container..."
    singularity pull "$SIF_PATH" docker://vllm/vllm-openai:latest
    if [ ! -f "$SIF_PATH" ]; then
        echo "ERROR: Container pull failed. Exiting."
        exit 1
    fi
fi

echo "Starting standard vLLM API server..."
echo "Model: $MODEL"

singularity run --nv \
    --bind "$REPO_ROOT:/workspace" \
    --bind "$HF_CACHE:$HF_CACHE" \
    --env HF_HOME="$HF_CACHE" \
    --env HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}" \
    "$SIF_PATH" \
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --port 8000 \
    >> "$LOG_DIR/std_server.$PBS_JOBID" 2>&1
