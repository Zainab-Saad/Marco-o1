#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1                 
#SBATCH --cpus-per-task=24
#SBATCH --gpus=h100:1
#SBATCH --mem=64G
#SBATCH --time=0-11:58:00                    
#SBATCH --job-name=marco_mcts_runner-run 
#SBATCH --output=/home/zainab14/links/projects/def-sdrew/zainab14/Marco-o1/logs/%x-%j.out 
#SBATCH --error=/home/zainab14/links/projects/def-sdrew/zainab14/Marco-o1/logs/%x-%j.err  

set -euo pipefail

echo "Job started on $(hostname) at: $(date)"
echo "Job ID: $SLURM_JOB_ID"

SCRATCH_DIR=/home/zainab14/links/scratch
PROJECT_DIR=/home/zainab14/links/projects/def-sdrew/zainab14/Marco-o1

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

echo "Working directory: $(pwd)"

echo "--- Loading modules ---"
module purge
module load python/3.13

echo "--- GPU info ---"
nvidia-smi || echo "nvidia-smi failed"

echo "--- Activating uv-created virtual environment ---"
source .venv/bin/activate

echo "--- Python info ---"
which python
python --version

# prevent vllm from using flashinfer
export VLLM_USE_FLASHINFER_SAMPLER=0

export HF_HOME=$SCRATCH_DIR/hf_cache
export XDG_CACHE_HOME=$SCRATCH_DIR/.cache
export TRITON_CACHE_DIR=$SCRATCH_DIR/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=$SCRATCH_DIR/.inductor_cache
export VLLM_CACHE_ROOT=$SCRATCH_DIR/.vllm_cache
export XDG_CONFIG_HOME=$SCRATCH_DIR/.config
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$XDG_CONFIG_HOME"

export MPLCONFIGDIR=$SCRATCH_DIR/.mplconfig
export FLASHINFER_WORKSPACE_BASE=$SCRATCH_DIR
export VLLM_CONFIG_ROOT=$SCRATCH_DIR/.vllm_config
export DO_NOT_TRACK=1
export VLLM_NO_USAGE_STATS=1
mkdir -p "$MPLCONFIGDIR" "$FLASHINFER_WORKSPACE_BASE" "$VLLM_CONFIG_ROOT"
# export VLLM_ATTENTION_BACKEND=FLASH_ATTN


export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "--- Running jobs ---"

LEVEL="${LEVEL:?set LEVEL=1..5}"
START="${START:?set START}"
END="${END:?set END}"
MAX_RUNTIME_MIN=${MAX_RUNTIME_MIN:-630}
WORKERS=${WORKERS:-16}
PORT=$((40000 + SLURM_JOB_ID % 10000))
TEACHER=Qwen/Qwen3-32B
CONFIG=${CONFIG:-./tree_search/configs/math_config.json}

vllm serve "$TEACHER" --dtype bfloat16 --max-model-len 20000 --gpu-memory-utilization 0.95 \
    --seed 0 --host 127.0.0.1 --port $PORT > "$PROJECT_DIR/logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do
    kill -0 $VLLM_PID 2>/dev/null || { echo "vllm serve die see logs/vllm-$SLURM_JOB_ID.log"; exit 1; }
    sleep 15
done

echo "vllm server up just logging the date $(date)"

echo "level=$LEVEL shard=[$START,$END) workers=$WORKERS deadline=${MAX_RUNTIME_MIN}min"
cd "$PROJECT_DIR/src/v2/src"
set +e

python main.py --config "$CONFIG" \
    --level "$LEVEL" --start "$START" --end "$END" \
    --workers "$WORKERS" \
    --server-url "http://127.0.0.1:$PORT/v1/completions" \
    --max-runtime-min "$MAX_RUNTIME_MIN"
EXIT=$?
set -e

echo "main.py exited with code $EXIT at: $(date)"
if [ "$EXIT" -eq 99 ]; then 
    echo "INCOMPLETE: level $LEVEL shard [$START,$END). RESUBMIT"; 
    exit 99
elif [ "$EXIT" -eq 0 ]; then 
    echo "COMPLETE: level $LEVEL shard [$START,$END)"
else 
    echo "FAILED (exit $EXIT)"; exit "$EXIT"; fi
echo "Job finished at: $(date)"