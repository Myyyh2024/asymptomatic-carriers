#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/cifar10.yaml}"
NUM_PROCESSES="${NPROC_PER_NODE:-2}"

torchrun --standalone --nproc_per_node="${NUM_PROCESSES}" \
  main.py --mode attack --cfg "${CONFIG_PATH}"
