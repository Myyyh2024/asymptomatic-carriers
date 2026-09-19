#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 TEACHER_CHECKPOINT [CONFIG_PATH]" >&2
  exit 2
fi

CHECKPOINT_PATH="$1"
CONFIG_PATH="${2:-configs/cifar10.yaml}"
NUM_PROCESSES="${NPROC_PER_NODE:-2}"

torchrun --standalone --nproc_per_node="${NUM_PROCESSES}" \
  main.py --mode distill --cfg "${CONFIG_PATH}" --resume "${CHECKPOINT_PATH}"
