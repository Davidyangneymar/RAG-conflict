#!/usr/bin/env bash
# One-command validation of the P2 deliverable.
#
# Runs, in order:
#   1. Adapter contract tests                (fast, no model)
#   2. Fusion + role routing smoke tests     (fast, no model)
#   3. Conflict typing smoke tests           (fast, no model)
#   4. End-to-end AVeriTeC closed-loop       (loads BERT, slower)
#
# Exits non-zero on the first failing stage.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

echo "=========================================================="
echo "[1/4] P1 -> P2 adapter contract test"
echo "=========================================================="
python scripts/test_contract.py scripts/sample_p1_payload_with_roles.json

echo ""
echo "=========================================================="
echo "[2/4] Fusion + role routing smoke tests"
echo "=========================================================="
python scripts/smoke_test_p2_fusion.py

echo ""
echo "=========================================================="
echo "[3/4] Conflict typing smoke tests"
echo "=========================================================="
python scripts/smoke_test_conflict_typing.py

echo ""
echo "=========================================================="
echo "[4/4] End-to-end AVeriTeC closed-loop (loads BERT model)"
echo "=========================================================="
if [[ ! -d "outputs/fnc1_bert_upgrade_full" ]]; then
  echo "SKIP: outputs/fnc1_bert_upgrade_full not found."
  echo "      Stage 4 needs the trained BERT artifacts to run."
  echo "      The first three stages already cover all P2 logic"
  echo "      without the model, so this skip does not hide bugs."
  exit 0
fi
python scripts/run_averitec_pipeline.py scripts/sample_averitec_records.json \
    --out_dir outputs/averitec_demo

echo ""
echo "All P2 checks passed."
