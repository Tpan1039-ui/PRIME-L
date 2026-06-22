#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
conda run -n protein python -m primel.cli prepare
conda run -n protein python -m primel.cli finetune-ala --epochs 1 --train-batch 1 --max-iter 1 --list-size 2 --output outputs/checkpoints/smoke_ala
conda run -n protein python -m primel.cli predict-single --checkpoint outputs/checkpoints/smoke_ala --limit 32 --batch-size 32 --output outputs/predictions/smoke_single.csv --top-k 8
conda run -n protein python -m primel.cli train-combo --checkpoint outputs/checkpoints/smoke_ala --round 2 --max-candidates 16 --epochs 2 --folds 2 --output-dir outputs/checkpoints/smoke_combo --top-k 8
