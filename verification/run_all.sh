#!/usr/bin/env bash
# Reproduce the verification for the result-generation fix.
# Everything runs in the caller's temp dir and uses an isolated venv.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"
VENV="${VENV:-/tmp/joblib_hold_venv}"

if [ ! -x "$VENV/bin/python" ]; then
  "$PY" -m venv "$VENV"
  "$VENV/bin/python" -m pip install -q --upgrade pip
  "$VENV/bin/python" -m pip install -q numpy pytest pytest-asyncio loky threadpoolctl ruff
fi
export PYTHONPATH="$ROOT"
export PYTHONDONTWRITEBYTECODE=1

echo "== manual probes"
for f in repro_bug scenarios handoff_windows legacy_cache legacy_repeat_pin replacements; do
  timeout 300 "$VENV/bin/python" "$ROOT/verification/$f.py" || exit 1
done

echo "== targeted regression"
timeout 600 "$VENV/bin/python" -m pytest -q \
  "$ROOT/joblib/test/test_memory.py" \
  "$ROOT/joblib/test/test_store_backends.py" \
  "$ROOT/joblib/test/test_result_holds.py" \
  "$ROOT/joblib/test/test_parallel.py" || exit 1

echo "== full suite"
timeout 900 "$VENV/bin/python" -m pytest -q "$ROOT/joblib/test/" || exit 1

echo "== scale probe"
timeout 900 "$VENV/bin/python" "$ROOT/verification/bench_generation.py"
