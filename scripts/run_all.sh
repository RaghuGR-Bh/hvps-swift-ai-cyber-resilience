#!/bin/bash
set -e
echo "=== HVPS/SWIFT AI Cyber Resilience — Full Pipeline ==="
echo "Step 1/4: Generating data..."
python3 notebooks/01_generate_data.py
echo "Step 2/4: Training models..."
python3 notebooks/02_train_evaluate.py
echo "Step 3/4: Stress tests..."
python3 notebooks/03_stress_test.py
echo "Step 4/4: Explainability..."
python3 notebooks/04_shap_explainability.py
echo "=== Complete. Outputs in data/ ==="
