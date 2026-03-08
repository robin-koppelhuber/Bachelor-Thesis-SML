#!/bin/bash
# ============================================================
# SLURM job script for running a single benchmark×method pair
#
# Required exports:
#   BENCHMARK  — benchmark config name
#   METHOD     — method config name
#
# Optional exports:
#   EXTRA_ARGS — additional Hydra overrides appended verbatim to the command
#
# Usage (from the repo root on the Euler login node):
#   sbatch --export=BENCHMARK=glue-2-label,METHOD=chebyshev \
#     scripts/cluster/run_benchmark.slurm
#
#   # With extra Hydra overrides (seed, num_samples, wandb group, etc.):
#   sbatch --export=BENCHMARK=glue-2-label,METHOD=chebyshev,\
#     EXTRA_ARGS="seed=123 benchmark.evaluation.num_samples=500 wandb.group=sweep1" \
#     scripts/cluster/run_benchmark.slurm
#
# Available benchmarks: poc, glue-2-label, recovery-cola, recovery-mrpc,
#                        recovery-qnli, recovery-sst2
# Available methods:    averaging, ties, chebyshev, epo, self_position,
#                        chebyshev_recovery, epo_recovery
# ============================================================

#SBATCH --job-name=sml-bench
#SBATCH --output=%x_%j.out          # stdout: <jobname>_<jobid>.out
#SBATCH --error=%x_%j.err           # stderr: <jobname>_<jobid>.err
#SBATCH --time=4:00:00              # wall time (adjust per method)
#SBATCH --mem-per-cpu=4G             # EPO offloads ~2 GB of gradients to CPU; 4×4=16 GB total
#SBATCH --cpus-per-task=4           # 1-2 dataloader workers per task; 4 is comfortable for EPO
#SBATCH --gpus=1
#SBATCH --gres=gpumem:10g           # ~2x safety margin over RoBERTa-base worst-case (~4.5 GB)

# ---- Guard: require BENCHMARK and METHOD to be set ----
if [ -z "${BENCHMARK}" ] || [ -z "${METHOD}" ]; then
    echo "ERROR: BENCHMARK and METHOD must be set via --export."
    echo "Example: sbatch --export=BENCHMARK=glue-2-label,METHOD=chebyshev $0"
    exit 1
fi

echo "======================================================"
echo "Job:        ${SLURM_JOB_ID}"
echo "Node:       ${SLURM_NODELIST}"
echo "Benchmark:  ${BENCHMARK}"
echo "Method:     ${METHOD}"
echo "Extra args: ${EXTRA_ARGS:-<none>}"
echo "Started:    $(date)"
echo "======================================================"

# ---- Environment ----
# PyTorch cu128 wheels bundle their own CUDA runtime — no module load needed.
# Only the GPU driver on the compute node is required (always present on Euler).

# Change to the repo (symlink from $HOME points to the repo in $SCRATCH)
REPO_DIR="${HOME}/Bachelor-Thesis-SML"
cd "${REPO_DIR}" || { echo "ERROR: ${REPO_DIR} not found"; exit 1; }

# Load .env for WANDB_API_KEY, HF_TOKEN, etc.
# uv inherits the shell environment, so exporting here is sufficient.
if [ -f ".env" ]; then
    set -a; source .env; set +a
fi

# ---- Run ----
CMD="uv run python main.py cluster=euler benchmark=${BENCHMARK} method=${METHOD} device=cuda ${EXTRA_ARGS:-}"
echo "Running: ${CMD}"

# shellcheck disable=SC2086  # EXTRA_ARGS intentionally word-splits here
uv run python main.py \
    cluster=euler \
    benchmark="${BENCHMARK}" \
    method="${METHOD}" \
    device=cuda \
    ${EXTRA_ARGS:-}

EXIT_CODE=$?
echo "======================================================"
echo "Finished: $(date)  |  Exit code: ${EXIT_CODE}"
echo "======================================================"
exit ${EXIT_CODE}
