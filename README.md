# RUBRIC: A Unified Benchmark and Filtering Toolkit for Imbalanced Classification

## What is RUBRIC?

RUBRIC is a standardized, extensible benchmark and reference implementation for imbalanced classification. It provides:

- A unified training/evaluation pipeline and fair baselines
- A generator-agnostic filtering add-on (RUBRIC) that works with many oversamplers
- Reproducible experiments, consistent metrics (ROC-AUC, PR-AUC, F1-Macro), and comparison reports

## Overview

This repository provides a minimal, unified training entry point and a lightweight RUBRIC filtering add-on. Public release focuses on the training script; data preparation and extended pipelines are intentionally simplified to avoid over-claiming on unstable variants.

---

## Quick Start

### 0) Environment Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate
pip install -r requirements.txt
```

### 1) Data (minimal)

Use standard public datasets (e.g., Credit Card Fraud, NSL-KDD, IEEE-CIS, Santander). Place raw files under `data/` following their respective licenses. Basic normalization is handled internally; users may adapt preprocessing to their environment.

### 2) Train

Credit Card Fraud Detection (example):
```bash
# Baselines (same preprocessing & median-gamma)
python src/train.py --dataset creditcard --augment none --rbf-gamma -1.0 --seed 42
python src/train.py --dataset creditcard --augment smote --target-ratio 0.3 --rbf-gamma -1.0 --seed 42
python src/train.py --dataset creditcard --augment adasyn --target-ratio 0.3 --rbf-gamma -1.0 --seed 42
python src/train.py --dataset creditcard --augment borderline-smote --target-ratio 0.3 --rbf-gamma -1.0 --seed 42
python src/train.py --dataset creditcard --augment svm-smote --target-ratio 0.3 --rbf-gamma -1.0 --seed 42

# RUBRIC add-on (generator-agnostic). Example 1: SMOTE + RUBRIC
python src/train.py --dataset creditcard --augment adv --gen-kind smote --target-ratio 0.3 \
  --keep-frac 0.65 --adv-C 1.0 --rbf-gamma -1.0 --seed 42

# Example 2: Borderline-SMOTE + RUBRIC
python src/train.py --dataset creditcard --augment adv --gen-kind borderline --target-ratio 0.3 \
  --keep-frac 0.65 --adv-C 1.0 --rbf-gamma -1.0 --seed 42

# Example 3: SVM-SMOTE + RUBRIC
python src/train.py --dataset creditcard --augment adv --gen-kind svm --target-ratio 0.3 \
  --keep-frac 0.65 --adv-C 1.0 --rbf-gamma -1.0 --seed 42

# Advanced training with custom parameters
python src/train.py --dataset creditcard --augment adv --rbf-components 400 --rbf-gamma 0.5 --svm-C 1.0 --test-size 0.2 --seed 42
```

IEEE-CIS and Santander usage is analogous; call `src/train.py` directly by analogy.

Key parameters:
- `--dataset`: `creditcard` | `nsl_kdd` | `ieee_fraud_detection` | `santander`
- `--augment`: `none`, `smote`, `adv`, `adasyn`, `borderline-smote`, `svm-smote`, `smote-tomek`, `smote-enn`
- `--gen-kind` (when `adv`): `smote`, `borderline`, `borderline2`, `svm`, `kmeans`, `smote-tomek`, `smote-enn`, `adasyn`
- `--keep-frac` alias of `--keep-top-frac`
- `--rbf-components`: number of RFF components (default: 300)
- `--rbf-gamma`: RBF bandwidth parameter (default: 0.5; `-1.0` means median heuristic)
- `--svm-C`: SVM regularization strength (default: 1.0)
- `--test-size`: test set fraction (default: 0.2)

### 3) Notes on data processing

Preprocessing (standardization/encoding) is intentionally lightweight in this public release. For rigorous evaluation, adapt preprocessing to your deployment setting and follow best practices to avoid leakage.

### 4) Outputs

- Metrics per run: `outputs/<run_name>/metrics.json`
- Plots: ROC/PR curves in `outputs/<run_name>/plots/`
- Augmented datasets: `data/augmented/*_augmentation_*.csv`
- Comparison reports: `outputs/*_comparison_report/`
- IEEE augmentation comparison doc: `data/ieee-fraud-detection/augmentation_comparison.txt`

---

## Reproducibility and Ethics (concise)

- Multi-seed protocol: repeat each configuration over a fixed seed set (e.g., `{42, 52, 62, 72, 82}`) and report mean±std. Thresholds are chosen on validation by maximizing F1 and applied once to test for every method consistently.
- Determinism: scikit-learn seeds are set; minor nondeterminism may remain due to data shuffles or library internals.
- Variance expectation: near-boundary filtering and stochastic candidate generation can yield small run-to-run fluctuations. Claims should be made on aggregated statistics; single-seed results are illustrative only.
- Deployment caution: for fraud/health-like use cases, evaluate operating thresholds, calibration (Brier score, reliability plots), and application-specific costs before deployment.

See `REPRODUCIBILITY.md` for environment and seed notes. `RESULTS.md` is currently a paper-submission placeholder; updates will follow.

---

## Project Structure (public)

```
RUBRIC/
├── data/
│   └── ...                   # user-provided public datasets
├── outputs/
│   └── <run folders containing metrics.json and plots/>
├── src/
│   └── train.py              # public training entry point
├── requirements.txt
├── README.md
├── REPRODUCIBILITY.md        # environment, seeds (concise)
└── RESULTS.md                # paper-submission placeholder
```

---

## Baselines

SMOTE-family baselines can be installed via `imbalanced-learn` if desired: `pip install imbalanced-learn`.

## RUBRIC Filtering

### Algorithm Overview

RUBRIC enhances oversampling by adding a filtering step:

1. Generate synthetic minority samples to a target ratio
2. Train a discriminator (logistic regression) to separate real vs synthetic
3. Keep synthetic samples closest to the decision boundary (hardest to distinguish)
4. Combine majority + minority originals with the filtered synthetic minority

### Generator-agnostic usage (examples)

Apply RUBRIC on top of different generators with a unified CLI:

```bash
# Pattern
python src/train.py --dataset <creditcard|nsl_kdd|ieee_fraud_detection|santander> --augment adv \
  --gen-kind <smote|borderline|borderline2|svm|kmeans|smote-tomek|smote-enn|adasyn> \
  --target-ratio 0.3 --keep-frac 0.65 --adv-C 1.0 --rbf-gamma -1.0 --seed 42

# Examples
python src/train.py --dataset creditcard --augment adv --gen-kind smote --target-ratio 0.3 --keep-frac 0.65 --adv-C 1.0 --rbf-gamma -1.0 --seed 42
python src/train.py --dataset creditcard --augment adv --gen-kind borderline --target-ratio 0.3 --keep-frac 0.65 --adv-C 1.0 --rbf-gamma -1.0 --seed 42
python src/train.py --dataset creditcard --augment adv --gen-kind svm --target-ratio 0.3 --keep-frac 0.65 --adv-C 1.0 --rbf-gamma -1.0 --seed 42
python src/train.py --dataset creditcard --augment adv --gen-kind smote-tomek --target-ratio 0.3 --keep-frac 0.65 --adv-C 1.0 --rbf-gamma -1.0 --seed 42
python src/train.py --dataset creditcard --augment adv --gen-kind smote-enn --target-ratio 0.3 --keep-frac 0.65 --adv-C 1.0 --rbf-gamma -1.0 --seed 42
```

Notes:
- RUBRIC is the same across generators; only `--gen-kind` switches the underlying oversampler.
- On balanced datasets, the training script auto-adjusts an effective target ratio slightly above the current class ratio to keep settings valid.

---

## Results Snapshot (illustrative)

Per-dataset and per-method results vary with seeds. Reference CSVs are included under `outputs/` and mirror paper aggregates (means±std). To regenerate, follow the multi-seed protocol in `REPRODUCIBILITY.md`.

---

## Authors

**Yanxuan Yu**  
Engineering School, Columbia University  
Email: yy3523@columbia.edu  
GitHub: [@zorinayu](https://github.com/zorinayu)

**Dong Liu**  
Department of Computer Science, Yale University  
Email: dong.liu.dl2367@yale.edu

## License

MIT

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{rubric2025,
  title={RUBRIC: A Unified Benchmark and Filtering Toolkit for Imbalanced Classification},
  author={Yanxuan Yu and Dong Liu},
  year={2025},
  url={https://github.com/zorinayu/RUBRIC}
}
```

### Additional References

- `imbalanced-learn` project: https://github.com/scikit-learn-contrib/imbalanced-learn
- `smote-variants` project: https://github.com/analyticalmindsltd/smote_variants