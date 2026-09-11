#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
M_CASES=${M_CASES:-1,16,128}
REFERENCE_MAX_M=${REFERENCE_MAX_M:-128}

export NPU_OPS_TRANSFORMER_OPS_IMPORT_MODE=minimal

torchrun \
  --standalone \
  --nproc-per-node="${NPROC_PER_NODE}" \
  "${SCRIPT_DIR}/validate_welm_megamoe.py" \
  --m "${M_CASES}" \
  --reference-max-m "${REFERENCE_MAX_M}" \
  --csv "${SCRIPT_DIR}/welm_megamoe_op.csv" \
  "$@"
