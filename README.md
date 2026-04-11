# AI-Augmented Cyber Resilience for HVPS/SWIFT Networks

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19519911.svg)](https://doi.org/10.5281/zenodo.19519911)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code and synthetic data accompanying:

> **Machine-Learning Fraud Detection and Cyber Stress Testing in RTGS and SWIFT Networks: An AI-Augmented Resilience Framework**
> *Journal of Cybersecurity* (Oxford University Press), 2026.

## Repository Structure

```
├── config.yaml                             # All parameters (seed=42, reproducible)
├── src/
│   ├── generator/hvps_transaction_generator.py  # (i)  PaySim-adapted HVPS generator
│   ├── models/
│   │   ├── supervised.py                        # (ii) XGBoost, RF, DNN
│   │   ├── unsupervised.py                      # (ii) Isolation Forest, Autoencoder
│   │   ├── gnn_model.py                         # (ii) GNN + RL-adaptive detection
│   │   └── hybrid_pipeline.py                   # (ii) Ensemble fusion pipeline
│   ├── stress_test/cyber_stress_engine.py       # (iii) 4-scenario HVPS simulation
│   └── evaluation/metrics.py                    # Evaluation utilities
├── notebooks/
│   ├── 01_generate_data.py                      # Generate 500K transactions
│   ├── 02_train_evaluate.py                     # Train all models
│   ├── 03_stress_test.py                        # Run S1–S4 scenarios
│   └── 04_shap_explainability.py                # SHAP analysis
└── scripts/run_all.sh                           # End-to-end pipeline
```

## Quick Start

```bash
pip install -r requirements.txt
bash scripts/run_all.sh
```

## Reproducible Results

| Model / Pipeline             | AUC-ROC | F1    | FPR    |
|------------------------------|---------|-------|--------|
| XGBoost (baseline)           | 0.987   | 0.94  | 0.008  |
| XGBoost (optimised)          | 0.991   | 0.95  | 0.006  |
| Random Forest                | 0.976   | 0.92  | 0.011  |
| DNN                          | 0.982   | 0.93  | 0.009  |
| GNN                          | 0.891   | 0.86  | 0.021  |
| RL-augmented GNN             | 0.912   | 0.924 | 0.017  |
| **Hybrid (XGB+IF fusion)**   | **0.993** | **0.96** | **0.006** |

| Scenario | Attack Type          | Reduction |
|----------|---------------------|-----------|
| S1       | DDoS                | 78%       |
| S2       | Integrity attack    | 61%       |
| S3       | Fraud injection     | 52%       |
| S4       | Coordinated attack  | 34%       |

All results deterministic with `seed=42`.

## Citation

```bibtex
@article{ramakrishna2026hvps,
  title={Machine-Learning Fraud Detection and Cyber Stress Testing in RTGS and SWIFT Networks},
  author={[Authors]},
  journal={Journal of Cybersecurity},
  year={2026},
  publisher={Oxford University Press},
  doi={10.5281/zenodo.19519911}
}
```

## License

MIT License. See [LICENSE](LICENSE).
