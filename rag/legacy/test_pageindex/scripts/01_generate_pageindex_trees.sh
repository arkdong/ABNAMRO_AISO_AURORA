#!/usr/bin/env bash
set -euo pipefail

# Run this from inside the cloned PageIndex repository.
# Example:
#   git clone https://github.com/VectifyAI/PageIndex.git
#   cd PageIndex
#   python3 -m venv .venv && source .venv/bin/activate
#   pip3 install --upgrade -r requirements.txt
#   # create .env with OPENAI_API_KEY=...
#   bash ../pageindex_local_poc/scripts/01_generate_pageindex_trees.sh

POC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORPUS_DIR="${POC_ROOT}/corpus"
PYTHON_BIN="./.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing PageIndex/.venv interpreter at ${PYTHON_BIN}."
  echo "Run: python3 -m venv .venv && source .venv/bin/activate && python -m pip install -r requirements.txt"
  exit 1
fi

# "${PYTHON_BIN}" run_pageindex.py \
#   --pdf_path "${CORPUS_DIR}/Writing Guide 2026-V1.1.pdf" \
#   --if-add-node-summary yes \
#   --if-add-node-text yes \
#   --if-add-doc-description yes

for f in "${CORPUS_DIR}"/*.md; do
  "${PYTHON_BIN}" run_pageindex.py \
    --md_path "$f" \
    --if-add-node-summary yes \
    --if-add-node-text yes \
    --if-add-doc-description yes
done

echo
echo "Done. PageIndex tree JSON files should now be in:"
echo "  $(pwd)/results"
echo
echo "Next step:"
echo "  python3 ${POC_ROOT}/scripts/select_and_retrieve.py \\"
echo "    --manifest ${POC_ROOT}/corpus_manifest.json \\"
echo "    --pageindex-results $(pwd)/results \\"
echo "    --query \"Refine a draft on AI adoption in European tech companies for ABN AMRO Insights. Keep it concrete, warm and authoritative.\" \\"
echo "    --print-prompts"
