#!/usr/bin/env bash
# End-to-end smoke test on a GPU box. ~15 minutes on a T4.
# Run this before spending free GPU hours on the real config.
set -euo pipefail

CONFIG="${1:-configs/smoke-1.5b.yaml}"
DATA_DIR="${DATA_DIR:-data/smoke}"

echo "==> config: $CONFIG"
ladder show-config --config "$CONFIG" >/dev/null

echo "==> build data (cpu)"
ladder build-data --config "$CONFIG" --out "$DATA_DIR"

echo "==> train"
ladder train --config "$CONFIG" --data "$DATA_DIR"

OUT=$(python -c "
import sys; sys.path.insert(0,'src')
from ladder.config import load_config
print(load_config('$CONFIG').train.output_dir)
")

echo "==> eval (executes generated code -- container or disposable VM only)"
ladder eval --config "$CONFIG" --adapter "$OUT"

echo "==> smoke test passed"
