# RUBRIC: A Unified Benchmark and Adversarial Filtering Toolkit for Imbalanced Classification

## What is RUBRIC?

RUBRIC is a standardized, extensible benchmark and reference implementation for imbalanced classification. It provides:

- A unified training/evaluation pipeline and fair baselines
- A generator-agnostic adversarial filtering add-on (ADV) that works with many oversamplers
- Reproducible experiments, consistent metrics (ROC-AUC, PR-AUC, F1-Macro), and comparison reports

## Overview

This repository provides a compact, reproducible pipeline for imbalanced classification across multiple datasets, with consistent preprocessing, feature mapping, models, augmentation plug-ins, metrics, and reports:

1. Up-dimension via RBF feature mapping (Random Fourier Features)
2. Linear SVM classifier with class weighting
3. ADV add-on: adversarial filtering using a logistic regression discriminator
4. Multi-dataset support: Credit Card Fraud, NSL-KDD
5. Optional 2D visualization (UMAP/PCA)
6. Baselines: SMOTE, ADASYN, Borderline-SMOTE, SVM-SMOTE (via imbalanced-learn)

Supported datasets:
- Credit Card Fraud Detection (284,807 transactions; fraud ≈ 0.172%) — extreme imbalance
- NSL-KDD Network Intrusion Detection (148,517 connections; attacks ≈ 48.12%) — roughly balanced

---

## Quick Start

### 0) Environment Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate
pip install -r requirements.txt
```

### 1) Download Data (one-time)

Credit Card Fraud dataset:
- Visit the dataset page: https://www.kaggle.com/mlg-ulb/creditcardfraud
- Download `creditcard.csv` to `data/creditcard.csv`.

NSL-KDD dataset:
- Visit the dataset page: https://www.kaggle.com/datasets/hassan06/nslkdd
- Place files under `data/NSL-KDD dataset/` (e.g., `KDDTrain+.txt`, `KDDTest+.txt`).

### 2) Train + Evaluate

Credit Card Fraud Detection:
```bash
# Baselines (same preprocessing & median-gamma)
python src/train.py --dataset creditcard --augment none --rbf-gamma -1.0 --seed 42
python src/train.py --dataset creditcard --augment smote --target-ratio 0.3 --rbf-gamma -1.0 --seed 42
python src/train.py --dataset creditcard --augment adasyn --target-ratio 0.3 --rbf-gamma -1.0 --seed 42
python src/train.py --dataset creditcard --augment borderline-smote --target-ratio 0.3 --rbf-gamma -1.0 --seed 42
python src/train.py --dataset creditcard --augment svm-smote --target-ratio 0.3 --rbf-gamma -1.0 --seed 42

# ADV add-on (generator-agnostic). Example 1: SMOTE + ADV
python src/train.py --dataset creditcard --augment adv --gen-kind smote --target-ratio 0.3 \
  --keep-frac 0.65 --adv-C 1.0 --rbf-gamma -1.0 --seed 42

# Example 2: Borderline-SMOTE + ADV
python src/train.py --dataset creditcard --augment adv --gen-kind borderline --target-ratio 0.3 \
  --keep-frac 0.65 --adv-C 1.0 --rbf-gamma -1.0 --seed 42

# Example 3: SVM-SMOTE + ADV
python src/train.py --dataset creditcard --augment adv --gen-kind svm --target-ratio 0.3 \
  --keep-frac 0.65 --adv-C 1.0 --rbf-gamma -1.0 --seed 42

# Advanced training with custom parameters
python src/train.py --dataset creditcard --augment adv --rbf-components 400 --rbf-gamma 0.5 --svm-C 1.0 --test-size 0.2 --seed 42
```

NSL-KDD usage is analogous.

Key parameters:
- `--dataset`: `creditcard` or `nsl_kdd`
- `--augment`: `none`, `smote`, `adv`, `adasyn`, `borderline-smote`, `svm-smote`, `smote-tomek`, `smote-enn`
- `--gen-kind` (when `adv`): `smote`, `borderline`, `borderline2`, `svm`, `kmeans`, `smote-tomek`, `smote-enn`, `adasyn`
- `--keep-frac` alias of `--keep-top-frac`
- `--rbf-components`: number of RFF components (default: 300)
- `--rbf-gamma`: RBF bandwidth parameter (default: 0.5; `-1.0` means median heuristic)
- `--svm-C`: SVM regularization strength (default: 1.0)
- `--test-size`: test set fraction (default: 0.2)

### 3) Generate Augmented Datasets (optional)

Credit Card Fraud dataset:
```bash
# Generate dataset with a chosen method (example: kmeans-smote)
python scripts/generate_augmented_datasets.py --method kmeans-smote --dataset creditcard --input data/creditcard.csv --output-dir data/augmented --seed 42
```

NSL-KDD dataset:
```bash
# Generate dataset with a chosen method
python scripts/generate_augmented_datasets.py --method adasyn --dataset nsl_kdd --output-dir data/augmented --seed 42
```

### 4) Run Complete Experiments

Credit Card Fraud Detection:
```bash
python scripts/run_creditcard_experiments.py
```

NSL-KDD experiments are optional:
```bash
python scripts/run_nsl_kdd_experiments.py
```

### 5) Outputs

- Metrics per run: `outputs/<run_name>/metrics.json`
- Plots: ROC/PR curves in `outputs/<run_name>/plots/`
- Augmented datasets: `data/augmented/*_augmentation_*.csv`
- Comparison reports: `outputs/*_comparison_report/`

---

## Project Structure

```
RUBRIC/
├── data/
│   ├── creditcard.csv
│   ├── NSL-KDD dataset/
│   └── augmented/
├── outputs/
│   └── <multiple run folders containing metrics.json and plots/>
├── scripts/
│   ├── generate_augmented_datasets.py
│   ├── run_creditcard_experiments.py
│   └── run_nsl_kdd_experiments.py
├── src/
│   ├── train.py
│   ├── data.py
│   ├── preprocess.py
│   ├── augment.py            # ADV add-on implementation
│   ├── evaluate.py
│   └── models/
│       └── svm_rff.py
├── requirements.txt
└── README.md
```

---

## Baselines and Sources

We compare ADV-augmented methods against widely used baselines implemented in mature libraries:

- SMOTE, Borderline-SMOTE, SVMSMOTE, ADASYN from `imbalanced-learn` ([GitHub](https://github.com/scikit-learn-contrib/imbalanced-learn), [Docs](https://imbalanced-learn.org/stable/over_sampling.html))
  Install: `pip install imbalanced-learn`

## Adversarial Filtering (ADV) in RUBRIC

### Algorithm Overview

ADV enhances oversampling by adding an adversarial filtering step:

1. Generate synthetic minority samples to a target ratio
2. Train a discriminator (logistic regression) to separate real vs synthetic
3. Keep synthetic samples closest to the decision boundary (hardest to distinguish)
4. Combine majority + minority originals with the filtered synthetic minority

### Generator-agnostic usage

Apply ADV on top of different generators with a unified CLI:

```bash
# Pattern
python src/train.py --dataset <creditcard|nsl_kdd> --augment adv \
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
- ADV is the same across generators; only `--gen-kind` switches the underlying oversampler.
- On balanced datasets, the training script auto-adjusts an effective target ratio slightly above the current class ratio to keep settings valid.

### Results (Credit Card Fraud, seed=42, RFF=300, median-gamma)

Paired base vs +ADV (same underlying generator):

| Generator | ROC-AUC (base) | PR-AUC (base) | F1-Macro (base) | ROC-AUC (+ADV) | PR-AUC (+ADV) | F1-Macro (+ADV) | ΔPR-AUC | ΔF1-Macro |
|-----------|-----------------|---------------|------------------|----------------|---------------|------------------|---------|-----------|
| SMOTE | 0.958127 | 0.510969 | 0.549454 | 0.939537 | 0.483470 | 0.546874 | -0.027499 | -0.002580 |
| Borderline-SMOTE | 0.943584 | 0.548030 | 0.667823 | 0.948718 | 0.559856 | 0.676186 | +0.011826 | +0.008363 |
| SVM-SMOTE | 0.9609 | 0.6707 | 0.7024 | 0.9606 | 0.6685 | 0.6955 | -0.0022 | -0.0069 |

Absolute metrics (all methods in this run):

| Method | ROC-AUC | PR-AUC | F1-Weighted | F1-Macro |
|--------|---------|--------|-------------|----------|
| None | 0.9686 | 0.5586 | 0.9985 | 0.7441 |
| SMOTE | 0.9613 | 0.5157 | 0.9921 | 0.5867 |
| ADASYN | 0.911581 | 0.386259 | 0.978275 | 0.521606 |
| Borderline-SMOTE | 0.943584 | 0.548030 | 0.996416 | 0.667823 |
| SVM-SMOTE | 0.9609 | 0.6707 | 0.9969 | 0.7024 |
| SMOTE + ADV | 0.939537 | 0.483470 | 0.987263 | 0.546874 |
| Borderline + ADV | 0.948718 | 0.559856 | 0.996586 | 0.676186 |
| SMOTE-Tomek + ADV | 0.858890 | 0.418473 | 0.989305 | 0.547409 |
| SMOTE-ENN + ADV | 0.881601 | 0.433934 | 0.989316 | 0.553480 |
| SVM-SMOTE + ADV | 0.9606 | 0.6685 | 0.9967 | 0.6955 |

Figures:
- Paired deltas per metric: `outputs/creditcard_comparison_report/paired_deltas.png`
- Win-rate of +ADV over base (PR-AUC): `outputs/creditcard_comparison_report/win_rate.png`
- Overview plots: `outputs/creditcard_comparison_report/creditcard_comparison_plots.png`

Runtime (see `outputs/creditcard_comparison_report/summary.csv` for per-run and averages).

---

### Results (NSL-KDD, seed=42, RFF=300, median-gamma)

Paired base vs +ADV (same underlying generator):

| Generator | ROC-AUC (base) | PR-AUC (base) | F1-Macro (base) | ROC-AUC (+ADV) | PR-AUC (+ADV) | F1-Macro (+ADV) | ΔPR-AUC | ΔF1-Macro |
|-----------|-----------------|---------------|------------------|----------------|---------------|------------------|---------|-----------|
| SMOTE | 0.996364 | 0.996565 | 0.977206 | 0.996367 | 0.996557 | 0.977307 | -0.000008 | +0.000101 |
| Borderline-SMOTE | 0.996617 | 0.996394 | 0.973883 | 0.996606 | 0.996598 | 0.974788 | +0.000204 | +0.000905 |
| SVM-SMOTE | 0.996723 | 0.996045 | 0.976210 | 0.996587 | 0.996169 | 0.975260 | +0.000124 | -0.000950 |
| SMOTE-ENN | 0.995442 | 0.995825 | 0.979166 | 0.996373 | 0.996546 | 0.977106 | +0.000721 | -0.002060 |
| SMOTE-Tomek | 0.996316 | 0.996540 | 0.977544 | 0.996376 | 0.996564 | 0.977174 | +0.000024 | -0.000370 |

Absolute metrics (all methods in this run):

| Method | ROC-AUC | PR-AUC | F1-Weighted | F1-Macro |
|--------|---------|--------|-------------|----------|
| None | 0.996362 | 0.996530 | 0.977274 | 0.977241 |
| SMOTE | 0.996364 | 0.996565 | 0.977240 | 0.977206 |
| ADASYN | 0.996746 | 0.996597 | 0.973813 | 0.973781 |
| Borderline-SMOTE | 0.996617 | 0.996394 | 0.973914 | 0.973883 |
| SVM-SMOTE | 0.996723 | 0.996045 | 0.976237 | 0.976210 |
| SMOTE + ADV | 0.996367 | 0.996557 | 0.977341 | 0.977307 |
| Borderline + ADV | 0.996606 | 0.996598 | 0.974821 | 0.974788 |
| SMOTE-Tomek + ADV | 0.996376 | 0.996564 | 0.977207 | 0.977174 |
| SMOTE-ENN + ADV | 0.996373 | 0.996546 | 0.977140 | 0.977106 |
| SVM-SMOTE + ADV | 0.996587 | 0.996169 | 0.975292 | 0.975260 |

Figures:
- Paired deltas per metric: `outputs/nsl_kdd_comparison_report/paired_deltas.png`
- Win-rate of +ADV over base (PR-AUC): `outputs/nsl_kdd_comparison_report/win_rate.png`
- Overview plots: `outputs/nsl_kdd_comparison_report/nsl_kdd_comparison_plots.png`

---

## Reproducibility

- StandardScaler + median-gamma (`--rbf-gamma -1.0`)
- Seeds via `--seed`, deterministic scikit-learn settings
- All configs/metrics persisted per run

---

## Notes
- This README reflects `outputs/*_comparison_report/summary.csv` where available. Re-run the experiment scripts to regenerate.
- The ADV add-on often improves PR-AUC or F1-Macro, but gains are dataset- and generator-dependent.

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
  title={RUBRIC: A Unified Benchmark and Adversarial Filtering Toolkit for Imbalanced Classification},
  author={Yanxuan Yu and Dong Liu},
  year={2025},
  url={https://github.com/zorinayu/RUBRIC}
}
```

### Additional References

- `imbalanced-learn` project: https://github.com/scikit-learn-contrib/imbalanced-learn
- `smote-variants` project: https://github.com/analyticalmindsltd/smote_variants