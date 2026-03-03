#!/bin/bash
# Launch Weight Sync Benchmark on GCP A100 x8
#
# This script:
# 1. Creates a new A100 x8 instance (or reuses existing one)
# 2. Uploads async-grpo-nccl code
# 3. Sets up environment (2 venvs: original + nccl)
# 4. Runs benchmark for all model sizes
# 5. Downloads results
#
# Usage:
#   bash benchmark/launch_benchmark.sh
#   bash benchmark/launch_benchmark.sh --reuse_instance my-instance
#   bash benchmark/launch_benchmark.sh --reuse_instance my-instance --skip_setup
#   bash benchmark/launch_benchmark.sh --models "0.5B 3B"

set -e
set -o pipefail

# ========================================
# Configuration
# ========================================
REUSE_INSTANCE=""
SKIP_SETUP="0"
DETACH="1"
MODELS="0.5B 3B 7B"

while [ $# -gt 0 ]; do
  case "$1" in
    --reuse_instance) REUSE_INSTANCE="$2"; shift 2;;
    --skip_setup) SKIP_SETUP="1"; shift 1;;
    --attach) DETACH="0"; shift 1;;
    --models) MODELS="$2"; shift 2;;
    -h|--help)
      echo "Usage: bash benchmark/launch_benchmark.sh [options]"
      echo ""
      echo "Options:"
      echo "  --reuse_instance <name>  Use existing instance"
      echo "  --skip_setup             Skip environment setup"
      echo "  --attach                 Stream logs (do not detach)"
      echo "  --models \"0.5B 3B ...\"   Model sizes to benchmark (default: all)"
      echo ""
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

INSTANCE_NAME="${REUSE_INSTANCE:-weight-sync-bench-$(date +%Y%m%d-%H%M%S)}"
MACHINE_TYPE="a2-ultragpu-8g"
PROJECT_ID=$(gcloud config get-value project)

A100_ZONES=(
    "us-central1-a"
    "us-central1-b"
    "us-central1-c"
    "us-east4-a"
    "us-east4-b"
    "us-east4-c"
    "us-east5-a"
    "us-east5-b"
    "europe-west4-a"
    "europe-west4-b"
    "europe-west4-c"
    "asia-southeast1-a"
    "asia-southeast1-c"
)

ZONE=""
INSTANCE_CREATED=false

echo "=============================================="
echo "Weight Sync Benchmark: async-grpo vs async-grpo-nccl"
echo "=============================================="
echo "Instance: $INSTANCE_NAME"
echo "Machine: $MACHINE_TYPE (8x A100 80GB)"
echo "Project: $PROJECT_ID"
echo "Models: $MODELS"
if [ -n "$REUSE_INSTANCE" ]; then
    echo "Reusing instance: $REUSE_INSTANCE"
fi
if [ "$SKIP_SETUP" = "1" ]; then
    echo "Skip setup: YES"
fi
echo "=============================================="
echo ""
echo "WARN: Instance will be KEPT after completion."
echo "WARN: Manual deletion required: gcloud compute instances delete $INSTANCE_NAME --zone=<zone>"
echo ""

# ========================================
# Step 1: Create or reuse instance
# ========================================
if [ -n "$REUSE_INSTANCE" ]; then
    echo "=== Step 1: Using existing instance: $INSTANCE_NAME ==="
    ZONE=$(gcloud compute instances list --filter="name=$INSTANCE_NAME" --format="value(zone)" 2>/dev/null | head -n1)
    if [ -z "$ZONE" ]; then
        echo "ERROR: Instance $INSTANCE_NAME not found" >&2
        exit 1
    fi
    echo "OK: Found instance in zone: $ZONE"
else
    echo "=== Step 1: Creating A100 x8 instance ==="
    for try_zone in "${A100_ZONES[@]}"; do
        echo "Trying zone: $try_zone..."
        if gcloud compute instances create "$INSTANCE_NAME" \
            --zone="$try_zone" \
            --machine-type="$MACHINE_TYPE" \
            --image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570 \
            --image-project=deeplearning-platform-release \
            --boot-disk-size=500GB \
            --boot-disk-type=pd-ssd \
            --maintenance-policy=TERMINATE \
            --metadata="install-nvidia-driver=True" \
            --scopes=https://www.googleapis.com/auth/cloud-platform 2>&1 | tee /tmp/gcp_create.log; then
            ZONE="$try_zone"
            INSTANCE_CREATED=true
            echo "OK: Instance created in zone: $ZONE"
            break
        else
            if grep -q "does not have enough resources\|ZONE_RESOURCE_POOL_EXHAUSTED\|quota" /tmp/gcp_create.log 2>/dev/null; then
                echo "WARN: Resource exhausted, trying next zone..."
                continue
            else
                echo "WARN: Unexpected error, trying next zone..."
                continue
            fi
        fi
    done
    if [ "$INSTANCE_CREATED" = false ]; then
        echo "ERROR: Failed to create instance in any zone!" >&2
        exit 1
    fi
    echo "Waiting for instance to be ready (90 seconds)..."
    sleep 90
fi

# ========================================
# Step 2: Upload code
# ========================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ "$SKIP_SETUP" != "1" ]; then
    echo ""
    echo "=== Step 2: Uploading async-grpo-nccl code ==="
    cd "$(dirname "$PROJECT_DIR")"
    tar czf /tmp/async-grpo-nccl.tar.gz \
        --exclude='.git' \
        --exclude='*.pth' \
        --exclude='*.bin' \
        --exclude='venv/*' \
        --exclude='.venv/*' \
        --exclude='wandb/*' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        "$(basename "$PROJECT_DIR")"
    gcloud compute scp --zone="$ZONE" --compress /tmp/async-grpo-nccl.tar.gz "$INSTANCE_NAME":~/
    gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command="cd ~ && tar xzf async-grpo-nccl.tar.gz"
    rm /tmp/async-grpo-nccl.tar.gz
    echo "OK: Code uploaded!"
else
    echo ""
    echo "=== Step 2: Skipping code upload (--skip_setup) ==="
fi

# ========================================
# Step 3: Setup environment
# ========================================
if [ "$SKIP_SETUP" != "1" ]; then
    echo ""
    echo "=== Step 3: Setting up environment (2 venvs) ==="
    gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command="bash ~/async-grpo-nccl/benchmark/remote_setup.sh"
    echo "OK: Environment setup complete!"
else
    echo ""
    echo "=== Step 3: Skipping environment setup (--skip_setup) ==="
fi

# ========================================
# Step 4: Run benchmark
# ========================================
echo ""
echo "=== Step 4: Running benchmark ==="

REMOTE_CMD="bash ~/async-grpo-nccl/benchmark/remote_run_benchmark.sh $MODELS"

if [ "$DETACH" = "1" ]; then
    echo "Running in detached mode (nohup). Check logs with:"
    echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command='tail -f ~/benchmark_results/benchmark.log'"
    gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command="nohup $REMOTE_CMD > ~/benchmark_results/benchmark.log 2>&1 &"
    echo ""
    echo "Benchmark is running in background."
    echo ""
    echo "To download results when done:"
    echo "  gcloud compute scp --zone=$ZONE --recurse $INSTANCE_NAME:~/benchmark_results/ ./benchmark/results/"
    echo ""
    echo "To generate plots:"
    echo "  python benchmark/collect_and_plot.py ./benchmark/results/"
    echo ""
    echo "To delete instance when done:"
    echo "  gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
else
    gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command="$REMOTE_CMD"

    echo ""
    echo "=== Step 5: Downloading results ==="
    mkdir -p "$PROJECT_DIR/benchmark/results"
    gcloud compute scp --zone="$ZONE" --recurse "$INSTANCE_NAME":~/benchmark_results/ "$PROJECT_DIR/benchmark/results/"
    echo "OK: Results downloaded to benchmark/results/"

    echo ""
    echo "=== Step 6: Generating plots ==="
    python "$PROJECT_DIR/benchmark/collect_and_plot.py" "$PROJECT_DIR/benchmark/results/"

    echo ""
    echo "Done! Delete instance when no longer needed:"
    echo "  gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
fi
