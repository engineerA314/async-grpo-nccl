#!/bin/bash
# Remote Run Benchmark: Run both async-grpo variants for each model size
# This script runs INSIDE the GCP instance.
#
# Critical fixes from GCP debugging:
#   - hard_cleanup(): pkill -9 all ray/training processes, stop ray from both venvs, rm /tmp/ray
#   - GPU isolation for Original: Ray manages only gen+logprob GPUs, trainer on separate GPUs
#   - VLLM_USE_V1=1 for NCCL (AsyncLLM), VLLM_USE_V1=0 for Original (vllm 0.7.3)
#   - runtime_env propagation for NCCL (env_vars must include VLLM_USE_V1)
#   - Model-size-dependent wait times for worker startup
#
# Usage:
#   bash remote_run_benchmark.sh 0.5B 3B 7B 14B
#   bash remote_run_benchmark.sh 0.5B          # single model

set -e
set -o pipefail

MODELS="${@:-0.5B 3B 7B}"
LOG_DIR=~/benchmark_results
NCCL_DIR=~/async-grpo-nccl
ORIG_DIR=~/async-grpo-original
NUM_BATCHES=13
PROGRESS_LOG=$LOG_DIR/master_progress.log

# Model -> HuggingFace ID, trainer GPU count, wait times, NCCL timeout
declare -A MODEL_HF_IDS=(
    ["0.5B"]="Qwen/Qwen2.5-0.5B"
    ["3B"]="Qwen/Qwen2.5-3B"
    ["7B"]="Qwen/Qwen2.5-7B"
    ["14B"]="Qwen/Qwen2.5-14B"
)
declare -A TRAINER_GPUS=(
    ["0.5B"]=1
    ["3B"]=1
    ["7B"]=2
    ["14B"]=4
)
# Wait times for generation worker startup (larger models need more time)
declare -A WAIT_GEN=(
    ["0.5B"]=120
    ["3B"]=150
    ["7B"]=180
    ["14B"]=240
)
# Wait times for logprob worker startup
declare -A WAIT_LOGPROB=(
    ["0.5B"]=90
    ["3B"]=120
    ["7B"]=150
    ["14B"]=180
)
# Timeout for NCCL orchestrator runs
declare -A NCCL_TIMEOUT=(
    ["0.5B"]=1200
    ["3B"]=1200
    ["7B"]=1800
    ["14B"]=2400
)

mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$PROGRESS_LOG"
}

# ========================================
# Hard cleanup: aggressively kill all training-related processes
# This is critical when switching between venvs (Ray 2.40 vs 2.47)
# ========================================
hard_cleanup() {
    log "  [hard_cleanup] Killing all Ray and training processes..."

    # Kill all Ray-related processes by name pattern
    pkill -9 -f "raylet" 2>/dev/null || true
    pkill -9 -f "gcs_server" 2>/dev/null || true
    pkill -9 -f "ray::IDLE" 2>/dev/null || true
    pkill -9 -f "ray::Orchestrator" 2>/dev/null || true
    pkill -9 -f "ray::TrainerGPU" 2>/dev/null || true
    pkill -9 -f "ray::LogProb" 2>/dev/null || true
    pkill -9 -f "ray::Generation" 2>/dev/null || true
    pkill -9 -f "worker_dispatcher.py" 2>/dev/null || true
    pkill -9 -f "trainer_core.py" 2>/dev/null || true
    pkill -9 -f "orchestrator" 2>/dev/null || true
    pkill -9 -f "ray_worker" 2>/dev/null || true
    pkill -9 -f "dashboard" 2>/dev/null || true

    # Stop Ray from BOTH venvs (critical for version mismatch prevention)
    ~/venv_nccl/bin/ray stop --force 2>/dev/null || true
    ~/venv_original/bin/ray stop --force 2>/dev/null || true

    # Remove Ray temporary state
    rm -rf /tmp/ray 2>/dev/null || true

    sleep 8
    log "  [hard_cleanup] Done"
}

# ========================================
# Run original async-grpo for one model
# Key: GPU isolation - Ray only manages gen+logprob GPUs, trainer uses separate GPUs
# ========================================
run_original() {
    local SIZE=$1
    local MODEL_ID=${MODEL_HF_IDS[$SIZE]}
    local N_TRAINER=${TRAINER_GPUS[$SIZE]}
    local LOG_FILE="$LOG_DIR/original_${SIZE}.log"
    local OUTPUT_DIR="/tmp/bench_original_${SIZE}"

    local TOTAL_GPUS=8
    local RAY_GPUS=$((TOTAL_GPUS - N_TRAINER))

    # Build Ray GPU list: 0,1,...,(RAY_GPUS-1)
    local RAY_GPU_LIST=""
    for i in $(seq 0 $((RAY_GPUS - 1))); do
        if [ -z "$RAY_GPU_LIST" ]; then
            RAY_GPU_LIST="$i"
        else
            RAY_GPU_LIST="$RAY_GPU_LIST,$i"
        fi
    done

    # Build Trainer GPU list: RAY_GPUS,...,7
    local TRAINER_GPU_LIST=""
    for i in $(seq $RAY_GPUS $((TOTAL_GPUS - 1))); do
        if [ -z "$TRAINER_GPU_LIST" ]; then
            TRAINER_GPU_LIST="$i"
        else
            TRAINER_GPU_LIST="$TRAINER_GPU_LIST,$i"
        fi
    done

    log "=== Original $SIZE START ==="
    log "  Model: $MODEL_ID"
    log "  Ray GPUs: $RAY_GPU_LIST ($RAY_GPUS GPUs for gen+logprob)"
    log "  Trainer GPUs: $TRAINER_GPU_LIST ($N_TRAINER GPUs)"
    hard_cleanup

    source ~/venv_original/bin/activate
    export VLLM_USE_V1=0

    # Start Ray only on gen+logprob GPUs (NOT trainer GPUs)
    CUDA_VISIBLE_DEVICES=$RAY_GPU_LIST ray start --head --num-gpus=$RAY_GPUS --port=9339 --dashboard-host=0.0.0.0 2>&1
    sleep 5

    cd "$ORIG_DIR"

    # Generation worker (gets 1 GPU from Ray pool)
    log "  Starting generation worker..."
    python worker_dispatcher.py \
        --model_path "$MODEL_ID" \
        --mode generation \
        --tensor_parallel_size 1 \
        --max_num_seqs 64 \
        --overhead_seqs 16 \
        --reward_fns countdown \
        --global_num_verifiers 50 \
        > "$LOG_DIR/original_${SIZE}_gen.log" 2>&1 &
    GEN_PID=$!

    local GEN_WAIT=${WAIT_GEN[$SIZE]}
    log "  Waiting ${GEN_WAIT}s for generation worker (PID: $GEN_PID)..."
    sleep $GEN_WAIT

    if ! kill -0 $GEN_PID 2>/dev/null; then
        log "  ERROR: Generation worker died!"
        tail -30 "$LOG_DIR/original_${SIZE}_gen.log" >> "$PROGRESS_LOG"
        deactivate 2>/dev/null || true
        hard_cleanup
        return 1
    fi

    # Logprob worker (gets 1 GPU from Ray pool)
    log "  Starting logprob worker..."
    torchrun \
        --nproc_per_node=1 \
        --master_port=12341 \
        worker_dispatcher.py \
        --model_path "$MODEL_ID" \
        --mode logprob \
        --temperature 1.0 \
        --max_tokens_per_gpu 100000 \
        > "$LOG_DIR/original_${SIZE}_logprob.log" 2>&1 &
    LOGPROB_PID=$!

    local LOGPROB_WAIT=${WAIT_LOGPROB[$SIZE]}
    log "  Waiting ${LOGPROB_WAIT}s for logprob worker (PID: $LOGPROB_PID)..."
    sleep $LOGPROB_WAIT

    if ! kill -0 $LOGPROB_PID 2>/dev/null; then
        log "  ERROR: LogProb worker died!"
        tail -30 "$LOG_DIR/original_${SIZE}_logprob.log" >> "$PROGRESS_LOG"
        kill $GEN_PID 2>/dev/null || true
        deactivate 2>/dev/null || true
        hard_cleanup
        return 1
    fi

    # Trainer on separate GPUs (NOT managed by Ray)
    log "  Starting trainer on GPUs $TRAINER_GPU_LIST ($N_TRAINER GPUs)..."
    mkdir -p "$OUTPUT_DIR"

    local RDZV_PORT=$((54367 + RANDOM % 1000))

    CUDA_VISIBLE_DEVICES=$TRAINER_GPU_LIST torchrun \
        --nnodes=1 \
        --node_rank=0 \
        --nproc_per_node=$N_TRAINER \
        --rdzv_id=$((100 + RANDOM)) \
        --rdzv_endpoint=localhost:$RDZV_PORT \
        trainer_core.py \
        --model-name-or-path "$MODEL_ID" \
        --data-path sample-data/countdown_bench.jsonl \
        --output-dir "$OUTPUT_DIR" \
        --num-training-batches $NUM_BATCHES \
        --batch-size 16 \
        --samples-per-question 4 \
        --max-generation-tokens 128 \
        --train-minibatch-size 32 \
        --max-tokens-per-gpu 4000 \
        --learning-rate 1e-6 \
        --lr-scheduler constant \
        --num-warmup-steps 0 \
        --temperature 1.0 \
        --min-samples-per-checkpoint 999999 \
        --logging-level INFO \
        2>&1 | tee "$LOG_FILE"

    local SYNC_COUNT=$(grep -c 'Updated weights on' "$LOG_FILE" 2>/dev/null || echo "0")
    log "=== Original $SIZE DONE === (weight syncs: $SYNC_COUNT)"

    kill $GEN_PID 2>/dev/null || true
    kill $LOGPROB_PID 2>/dev/null || true
    deactivate 2>/dev/null || true
    hard_cleanup
}

# ========================================
# Run async-grpo-nccl for one model
# Key: VLLM_USE_V1=1 must be propagated via runtime_env
# ========================================
run_nccl() {
    local SIZE=$1
    local MODEL_ID=${MODEL_HF_IDS[$SIZE]}
    local LOG_FILE="$LOG_DIR/nccl_${SIZE}.log"
    local CONFIG_FILE="$NCCL_DIR/benchmark/configs/nccl_$(echo $SIZE | tr '[:upper:]' '[:lower:]').json"
    local TIMEOUT=${NCCL_TIMEOUT[$SIZE]}

    log "=== NCCL $SIZE START ==="
    log "  Model: $MODEL_ID"
    log "  Config: $CONFIG_FILE"
    log "  Timeout: ${TIMEOUT}s"
    hard_cleanup

    source ~/venv_nccl/bin/activate
    export VLLM_USE_V1=1

    # Start Ray with all 8 GPUs (NCCL manages all workers via Ray)
    ray start --head --num-gpus=8 --port=9339 --dashboard-host=0.0.0.0 2>&1
    sleep 5

    cd "$NCCL_DIR"

    if [ ! -f "$CONFIG_FILE" ]; then
        log "  ERROR: Config not found: $CONFIG_FILE"
        deactivate 2>/dev/null || true
        hard_cleanup
        return 1
    fi

    # Run orchestrator with runtime_env propagating VLLM_USE_V1=1
    log "  Starting orchestrator..."
    timeout $TIMEOUT python -c "
import json, ray, sys, time, os
from orchestrator import Orchestrator

with open('$CONFIG_FILE') as f:
    cfg = json.load(f)

ray.init(address='auto', namespace='async-grpo', runtime_env={'env_vars': {'VLLM_USE_V1': '1'}})
orch = Orchestrator.options(
    name='orchestrator',
    namespace='async-grpo',
    lifetime='detached',
    runtime_env={'env_vars': {'VLLM_USE_V1': '1'}},
).remote(config_dict=cfg)

try:
    ray.get(orch.run.remote())
except Exception as e:
    print(f'Orchestrator finished or errored: {e}')
" 2>&1 | tee "$LOG_FILE" || true

    local SYNC_COUNT=$(grep -c '\[WEIGHT_SYNC_TIME\]' "$LOG_FILE" 2>/dev/null || echo "0")
    log "=== NCCL $SIZE DONE === (weight syncs: $SYNC_COUNT)"

    deactivate 2>/dev/null || true
    hard_cleanup
}

# ========================================
# Main: Run all models
# ========================================
log "=============================================="
log "Weight Sync Benchmark"
log "Models: $MODELS"
log "Batches per model: $NUM_BATCHES"
log "Results: $LOG_DIR"
log "=============================================="

for SIZE in $MODELS; do
    if [ -z "${MODEL_HF_IDS[$SIZE]}" ]; then
        log "ERROR: Unknown model size: $SIZE (valid: 0.5B 3B 7B 14B)"
        continue
    fi

    log ""
    log "########################################################"
    log "  MODEL SIZE: $SIZE"
    log "########################################################"

    # Phase 1: async-grpo-nccl (NCCL broadcast)
    run_nccl "$SIZE" || log "WARN: NCCL $SIZE failed, continuing..."

    # Phase 2: Original async-grpo (Ray Object Store)
    run_original "$SIZE" || log "WARN: Original $SIZE failed, continuing..."
done

# ========================================
# Summary
# ========================================
log ""
log "=============================================="
log "ALL BENCHMARKS COMPLETE"
log "=============================================="

log "Results summary:"
for SIZE in $MODELS; do
    NCCL_COUNT=$(grep -c '\[WEIGHT_SYNC_TIME\]' $LOG_DIR/nccl_${SIZE}.log 2>/dev/null || echo "0")
    ORIG_COUNT=$(grep -c 'Updated weights on' $LOG_DIR/original_${SIZE}.log 2>/dev/null || echo "0")
    log "  $SIZE: NCCL=$NCCL_COUNT syncs, Original=$ORIG_COUNT syncs"
done

log ""
log "To view weight sync times:"
log "  grep 'WEIGHT_SYNC_TIME' ~/benchmark_results/nccl_*.log"
log "  grep 'Updated weights on' ~/benchmark_results/original_*.log"
log ""
log "DONE at $(date)"
