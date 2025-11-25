#!/bin/bash
# GSM8K Pipeline - Run everything in sequence
# Usage: ./run_pipeline.sh [--parallel]

set -e  # Exit on error

PARALLEL_FLAG=""
if [[ "$1" == "--parallel" ]]; then
    PARALLEL_FLAG="--parallel"
    echo "========================================="
    echo "GSM8K Synthetic CoT Pipeline (PARALLEL MODE)"
    echo "========================================="
else
    echo "========================================="
    echo "GSM8K Synthetic CoT Pipeline"
    echo "========================================="
fi
echo ""

# 1. Generate synthetic data
echo "Step 1/3: Generating synthetic CoT data..."
cd scripts
python3 cot_gen.py
cd ..
echo "Data generation complete"
echo ""

# 2. Fine-tune models
echo "Step 2/3: Fine-tuning models..."
cd scripts
python3 fine_tune.py --teacher all $PARALLEL_FLAG
cd ..
echo "Training complete"
echo ""

# 3. Evaluate models
echo "Step 3/3: Evaluating models..."
cd scripts
python3 evaluate.py --teacher all
cd ..
echo "Evaluation complete"
echo ""

echo "========================================="
echo "Pipeline finished successfully."
echo "Results saved to results/"
echo "========================================="
