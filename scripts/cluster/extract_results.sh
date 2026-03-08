#!/bin/bash
# ============================================================
# Pull results from Euler $SCRATCH to your local machine.
# Run this from your LOCAL machine (not on Euler).
#
# Usage:
#   bash scripts/cluster/extract_results.sh [euler_username]
#
# What it syncs (read-only files only, no checkpoints):
#   - All .png  plot files
#   - All .log  log files
#   - All .yaml config snapshots (.hydra/)
#   - All .md   report templates
#
# Large model checkpoints are intentionally excluded.
# For full artifacts, use W&B artifact download instead.
# ============================================================

set -euo pipefail

EULER_USER="${1:-${USER}}"
EULER_HOST="euler.ethz.ch"
EULER_SCRATCH_PATH="/cluster/scratch/${EULER_USER}/bachelor-thesis-sml/outputs"
LOCAL_DEST="./outputs_euler"

echo "Syncing from ${EULER_USER}@${EULER_HOST}:${EULER_SCRATCH_PATH}/"
echo "         to ${LOCAL_DEST}/"
echo ""

rsync -avz --progress \
    --include="*/" \
    --include="*.png" \
    --include="*.log" \
    --include="*.yaml" \
    --include="*.md" \
    --exclude="*" \
    "${EULER_USER}@${EULER_HOST}:${EULER_SCRATCH_PATH}/" \
    "${LOCAL_DEST}/"

echo ""
echo "Done. Results saved to: ${LOCAL_DEST}/"
