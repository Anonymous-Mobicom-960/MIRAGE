#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
# Pi5: keep onnxruntime to physical cores; tune if it thermal-throttles.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
python3 mirage_tier1.py "$@"
