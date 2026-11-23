#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> 1/5 Preparing Python virtual environment (.venv)..."
if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install -q --upgrade pip >/dev/null 2>&1 || true
.venv/bin/python -m pip install -q -r requirements.txt

echo "==> 2/5 Regenerating all 6 real-data SVG figures..."
.venv/bin/python scripts/generate_exam_figures.py

echo "==> 3/5 Re-executing all 4 exam Jupyter notebooks..."
.venv/bin/python scripts/build_and_execute_notebooks.py

echo "==> 4/5 Running Python ML & RAG test suite..."
.venv/bin/pytest -q tests_python/

echo "==> 5/5 Running TypeScript Scoring & Hybrid RRF test suite..."
node --test tests/scoring.test.mjs tests/concierge-hybrid-search.test.mjs

echo ""
echo "All figures (6), notebooks (4), Python tests (54), and Node RAG/scoring tests (13) verified and ready for examination."
