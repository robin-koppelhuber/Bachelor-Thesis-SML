#!/bin/bash
# ============================================================
# One-time setup script for the Euler cluster.
# Run this from the LOGIN NODE (not inside a job).
#
# What it does:
#   1. Creates the $SCRATCH directory structure
#   2. Syncs the uv venv (CUDA + cluster dependency group)
#   3. Downloads all datasets needed across all benchmarks
#   4. Downloads all fine-tuned models needed across all benchmarks
#
# Prerequisites:
#   - $HOME/Bachelor-Thesis-SML is a symlink to the repo in $SCRATCH
#     (or a git clone directly in $SCRATCH)
#   - uv is available on PATH (https://docs.astral.sh/uv/getting-started/installation/)
#   - .env file exists in the repo root with WANDB_API_KEY and HF_TOKEN
# ============================================================

set -euo pipefail

REPO_DIR="${HOME}/Bachelor-Thesis-SML"
SCRATCH_DIR="${SCRATCH}/bachelor-thesis-sml"

echo "======================================================"
echo "Euler cluster setup"
echo "SCRATCH_DIR: ${SCRATCH_DIR}"
echo "REPO_DIR:    ${REPO_DIR}"
echo "======================================================"

# ---- 1. Directory structure ----
echo ""
echo "[1/4] Creating $SCRATCH directory structure..."
mkdir -p "${SCRATCH_DIR}/data/raw"
mkdir -p "${SCRATCH_DIR}/checkpoints/base"
mkdir -p "${SCRATCH_DIR}/checkpoints/finetuned"
mkdir -p "${SCRATCH_DIR}/checkpoints/trained"
mkdir -p "${SCRATCH_DIR}/checkpoints/eval_cache"
mkdir -p "${SCRATCH_DIR}/outputs"
echo "      Done."

# ---- 2. Python environment ----
echo ""
echo "[2/4] Syncing uv environment (cuda + cluster group)..."
cd "${REPO_DIR}"
export UV_CACHE_DIR="${SCRATCH}/.cache/uv"

# Load CUDA so torch[cuda] installs correctly
module load cuda 2>/dev/null || echo "      (no CUDA module — torch will install but check CUDA availability)"

uv sync --extra cuda --group cluster
echo "      Done."

# ---- 3. Load .env ----
if [ -f ".env" ]; then
    set -a; source .env; set +a
    echo "      Loaded .env"
else
    echo "WARNING: .env not found. HF_TOKEN and WANDB_API_KEY may be missing."
fi

# ---- 4. Download datasets ----
echo ""
echo "[3/4] Downloading all datasets..."
uv run python scripts/setup_data.py --all-benchmarks cluster=euler
echo "      Done."

# ---- 5. Download models ----
echo ""
echo "[4/4] Downloading all fine-tuned models..."
uv run python scripts/setup_models.py --all-benchmarks cluster=euler
echo "      Done."

echo ""
echo "======================================================"
echo "Setup complete! You can now submit jobs:"
echo ""
echo "  sbatch --export=BENCHMARK=glue-2-label,METHOD=chebyshev \\"
echo "    scripts/cluster/run_benchmark.slurm"
echo ""
echo "  sbatch --export=BENCHMARK=poc,METHOD=epo \\"
echo "    scripts/cluster/run_benchmark.slurm"
echo "======================================================"
