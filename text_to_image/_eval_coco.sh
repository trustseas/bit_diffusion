#!/bin/bash
# The same entrypoint evaluates original flow, noisy flow, and SDE checkpoints.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
: "${CHECKPOINT:?Set CHECKPOINT}"
: "${DATA_ROOT:?Set DATA_ROOT to training manifests and the token vocabulary table}"
: "${COCO_ROOT:?Set COCO_ROOT}"
: "${OUT_DIR:?Set OUT_DIR to a new output directory}"
task=${1:?Usage: bash _eval_coco.sh t2i|i2t [extra generation flags]}
shift
export PYTHONPATH=".:..${PYTHONPATH:+:$PYTHONPATH}"
generate=(torchrun --standalone --nproc-per-node="${NPROC:-8}" coco_eval.py generate
  --checkpoint "$CHECKPOINT" --data-root "$DATA_ROOT" --coco-root "$COCO_ROOT"
  --task "$task" --output "$OUT_DIR" --steps "${EVAL_STEPS:-500}"
  --cfg-scale "${CFG_SCALE:-0}" --seed "${SEED:-0}" "$@")
score=(python coco_eval.py score --coco-root "$COCO_ROOT" --output "$OUT_DIR"
  --fid-mode "${FID_MODE:-clean}")
if [[ "${DRYRUN:-0}" == 1 ]]; then
  printf '%q ' "${generate[@]}"; printf '\n'
  printf '%q ' "${score[@]}"; printf '\n'
else
  "${generate[@]}"
  "${score[@]}"
fi
