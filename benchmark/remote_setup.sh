#!/bin/bash
# Remote Setup: Create 2 separate venvs for original async-grpo and async-grpo-nccl
# This script runs INSIDE the GCP instance (Deep Learning VM with Python 3.11+).
#
# Key requirements discovered during debugging:
#   - Two separate venvs with different Ray versions (2.40.0 vs 2.47.1)
#   - Original async-grpo needs patches for runtime_env and ZeroDivisionError
#   - flash-attn must be built from source, not installed via pip in runtime_env
#   - countdown_bench.jsonl (200 lines) used instead of full dataset for faster runs
set -e

echo "=============================================="
echo "Setting up benchmark environment"
echo "=============================================="

# ========================================
# 0. Check Python and GPU status
# ========================================
echo ""
echo "=== System Check ==="
python3 --version
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "Total GPUs: $NUM_GPUS"
if [ "$NUM_GPUS" -lt 3 ]; then
    echo "ERROR: Need at least 3 GPUs, found $NUM_GPUS" >&2
    exit 1
fi

# ========================================
# 1. Clone original async-grpo
# ========================================
echo ""
echo "=== Cloning original async-grpo ==="
if [ -d ~/async-grpo-original ]; then
    echo "Already exists, pulling latest..."
    cd ~/async-grpo-original && git pull
else
    git clone https://github.com/Red-Hat-AI-Innovation-Team/async-grpo.git ~/async-grpo-original
fi

# ========================================
# 2. Create venv for async-grpo-nccl (Ray 2.47.1, vllm 0.8.5+)
# ========================================
echo ""
echo "=== Creating venv: venv_nccl ==="
if [ ! -d ~/venv_nccl ]; then
    python3 -m venv ~/venv_nccl
fi
source ~/venv_nccl/bin/activate

echo "Installing async-grpo-nccl dependencies..."
cd ~/async-grpo-nccl
pip install --upgrade pip
pip install -r requirements_base.txt 2>&1 | tail -5
pip install -r requirements_fsdp.txt 2>&1 | tail -5
pip install -r requirements_vllm.txt 2>&1 | tail -5

# Extra packages needed at runtime
pip install peft omegaconf trl 2>&1 | tail -3

echo "venv_nccl key packages:"
python3 -c "import ray; print(f'  ray: {ray.__version__}')"
python3 -c "import vllm; print(f'  vllm: {vllm.__version__}')"
python3 -c "import torch; print(f'  torch: {torch.__version__}')"

deactivate

# ========================================
# 3. Create venv for original async-grpo (Ray 2.40.0, vllm 0.7.3)
# ========================================
echo ""
echo "=== Creating venv: venv_original ==="
if [ ! -d ~/venv_original ]; then
    python3 -m venv ~/venv_original
fi
source ~/venv_original/bin/activate

echo "Installing original async-grpo dependencies..."
cd ~/async-grpo-original

pip install --upgrade pip

# Install base deps first
pip install -r requirements_base.txt 2>&1 | tail -5

# FSDP deps - pin transformers to avoid compatibility issues
# Comment out flash-attn from requirements_fsdp.txt (build from source below)
sed -i 's/^flash-attn/#flash-attn/' requirements_fsdp.txt 2>/dev/null || true
pip install -r requirements_fsdp.txt 2>&1 | tail -5
pip install transformers==4.48.3 2>&1 | tail -3

# vLLM deps - pin Ray to 2.40.0 (must match original project's expectation)
sed -i 's/^ray\[default\].*/ray[default]==2.40.0/' requirements_vllm.txt 2>/dev/null || true
pip install -r requirements_vllm.txt 2>&1 | tail -5
pip install ray[default]==2.40.0 2>&1 | tail -3

# Build flash-attn from source (slow but avoids runtime_env pip install issues)
echo "Building flash-attn from source (this takes ~10 minutes)..."
pip install flash-attn --no-build-isolation 2>&1 | tail -5

# Extra packages
pip install peft omegaconf trl 2>&1 | tail -3

echo "venv_original key packages:"
python3 -c "import ray; print(f'  ray: {ray.__version__}')"
python3 -c "import vllm; print(f'  vllm: {vllm.__version__}')"
python3 -c "import torch; print(f'  torch: {torch.__version__}')"

deactivate

# ========================================
# 4. Patch original async-grpo for benchmark compatibility
# ========================================
echo ""
echo "=== Patching original async-grpo ==="
cd ~/async-grpo-original

# Patch 1: Remove pip installs from runtime_env in worker_dispatcher.py
# The original code tries to pip install packages inside Ray runtime_env,
# which causes issues with flash-attn builds. Since we pre-installed everything,
# remove the pip section from runtime_env.
if grep -q '"pip"' worker_dispatcher.py 2>/dev/null; then
    echo "Patching worker_dispatcher.py: removing runtime_env pip installs..."
    python3 -c "
import re
with open('worker_dispatcher.py', 'r') as f:
    content = f.read()
# Remove pip list from runtime_env dict
content = re.sub(r'\"pip\":\s*\[.*?\],?', '', content, flags=re.DOTALL)
with open('worker_dispatcher.py', 'w') as f:
    f.write(content)
print('  Patched successfully')
"
fi

# Patch 2: Fix ZeroDivisionError in trainer_core.py
# When non_masked_output_tokens is 0, division fails. Wrap with max(..., 1).
if grep -q 'non_masked_output_tokens' trainer_core.py 2>/dev/null; then
    echo "Patching trainer_core.py: fixing ZeroDivisionError..."
    sed -i 's|bm\["non_masked_output_tokens"\]|max(bm["non_masked_output_tokens"], 1)|g' trainer_core.py 2>/dev/null || true
    echo "  Patched successfully"
fi

# ========================================
# 5. Prepare benchmark data
# ========================================
echo ""
echo "=== Preparing benchmark data ==="

# Create countdown_bench.jsonl (200 lines subset for faster benchmark runs)
NCCL_DATA=~/async-grpo-nccl/sample-data/countdown.jsonl
BENCH_DATA=~/async-grpo-nccl/sample-data/countdown_bench.jsonl

if [ -f "$NCCL_DATA" ]; then
    TOTAL_LINES=$(wc -l < "$NCCL_DATA")
    echo "Source data: $NCCL_DATA ($TOTAL_LINES lines)"

    if [ ! -f "$BENCH_DATA" ]; then
        head -200 "$NCCL_DATA" > "$BENCH_DATA"
        echo "Created: $BENCH_DATA (200 lines)"
    else
        echo "Already exists: $BENCH_DATA ($(wc -l < "$BENCH_DATA") lines)"
    fi
else
    echo "WARN: $NCCL_DATA not found. You may need to generate it first."
fi

# Copy benchmark data to original project
ORIG_DATA_DIR=~/async-grpo-original/sample-data
mkdir -p "$ORIG_DATA_DIR"
if [ -f "$BENCH_DATA" ]; then
    cp "$BENCH_DATA" "$ORIG_DATA_DIR/countdown_bench.jsonl"
    echo "Copied benchmark data to original project"
fi

# ========================================
# 6. Create results directory
# ========================================
mkdir -p ~/benchmark_results

echo ""
echo "=============================================="
echo "Setup complete!"
echo "  venv_nccl:     ~/venv_nccl (Ray $(~/venv_nccl/bin/python3 -c 'import ray; print(ray.__version__)'))"
echo "  venv_original: ~/venv_original (Ray $(~/venv_original/bin/python3 -c 'import ray; print(ray.__version__)'))"
echo "  original code: ~/async-grpo-original (patched)"
echo "  nccl code:     ~/async-grpo-nccl"
echo "  bench data:    $BENCH_DATA"
echo "  results:       ~/benchmark_results"
echo "=============================================="
