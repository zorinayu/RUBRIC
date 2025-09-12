# SMOTE-ADV: Generator-agnostic Adversarial Filtering for Imbalanced Classification

## Overview

This project implements SMOTE-ADV, a generator-agnostic adversarial filter that improves the quality of synthetic minority samples on top of different oversamplers (SMOTE / Borderline-SMOTE / SVM-SMOTE), and provides fair baselines for comparison:

1. **Up-dimension** via RBF feature mapping (Random Fourier Features)
2. **SVM** classifier with class weighting
3. **SMOTE-Adv**: SMOTE followed by adversarial filtering using logistic regression discriminator
4. **Multi-dataset Support**: Credit Card Fraud and NSL-KDD Network Intrusion Detection
5. **Down-dimension** for visualization (UMAP/PCA)
6. **Baselines**: SMOTE, ADASYN, Borderline-SMOTE, SVM-SMOTE (via imbalanced-learn)

**Supported Datasets**:
- **Credit Card Fraud Detection** (284,807 transactions; fraud ≈ 0.172%) - Extreme imbalance
- **NSL-KDD Network Intrusion Detection** (148,517 connections; attacks ≈ 48.12%) - Balanced

---

## Quick Start

### 0) Environment Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate
pip install -r requirements.txt
```

### 1) Download Data (one-time)

**For Credit Card Fraud Dataset:**
* Visit the [dataset page](https://www.kaggle.com/mlg-ulb/creditcardfraud), download `creditcard.csv`, and place it at `data/creditcard.csv`.

**For NSL-KDD Dataset:**
* Visit the [NSL-KDD dataset page](https://www.kaggle.com/datasets/hassan06/nslkdd), download the dataset files, and place them in `data/NSL-KDD dataset/` directory.
* Ensure the directory structure matches: `data/NSL-KDD dataset/KDDTrain+.txt`, `data/NSL-KDD dataset/KDDTest+.txt`, etc.

### 2) Train + Evaluate

Credit Card Fraud Detection:
```bash
# Baselines (same preprocessing & median-gamma)
python src/train.py --dataset creditcard --augment none --rbf-gamma -1.0 --seed 42
python src/train.py --dataset creditcard --augment smote --target-ratio 0.3 --rbf-gamma -1.0 --seed 42
python src/train.py --dataset creditcard --augment adasyn --target-ratio 0.3 --rbf-gamma -1.0 --seed 42
python src/train.py --dataset creditcard --augment borderline-smote --target-ratio 0.3 --rbf-gamma -1.0 --seed 42
python src/train.py --dataset creditcard --augment svm-smote --target-ratio 0.3 --rbf-gamma -1.0 --seed 42

# SMOTE-Adv++ (generator-agnostic). Example 1: SMOTE + Adv filter
python src/train.py --dataset creditcard --augment smote-adv --gen-kind smote --target-ratio 0.3 \
  --keep-frac 0.65 --adv-C 1.0 --rbf-gamma -1.0 --seed 42

# Example 2: Borderline-SMOTE + Adv filter
python src/train.py --dataset creditcard --augment smote-adv --gen-kind borderline --target-ratio 0.3 \
  --keep-frac 0.65 --adv-C 1.0 --rbf-gamma -1.0 --seed 42

# Example 3: SVM-SMOTE + Adv filter
python src/train.py --dataset creditcard --augment smote-adv --gen-kind svm --target-ratio 0.3 \
  --keep-frac 0.65 --adv-C 1.0 --rbf-gamma -1.0 --seed 42

# Advanced training with custom parameters
python src/train.py --dataset creditcard --augment smote-adv --rbf-components 400 --rbf-gamma 0.5 --svm-C 1.0 --test-size 0.2 --seed 42
```

NSL-KDD usage is analogous (optional).

**Key parameters:**
* `--dataset`: `creditcard` or `nsl_kdd`
* `--augment`: `none`, `smote`, `smote-adv`, `adasyn`, `borderline-smote`, `svm-smote`
* `--gen-kind` (when `smote-adv`): `smote`, `borderline`, `svm`
* `--keep-frac` alias of `--keep-top-frac`
* `--rbf-components`: number of random Fourier features (default: 300)
* `--rbf-gamma`: RBF bandwidth parameter (default: 0.5)
* `--svm-C`: SVM regularization strength (default: 1.0)
* `--test-size`: test set fraction (default: 0.2)

### 3) Generate Augmented Datasets (optional)

**Credit Card Fraud Dataset:**
```bash
# Generate dataset with a chosen method (example: kmeans-smote)
python scripts/generate_augmented_datasets.py --method kmeans-smote --dataset creditcard --input data/creditcard.csv --output-dir data/augmented --seed 42
```

**NSL-KDD Dataset:**
```bash
# Generate dataset with a chosen method
python scripts/generate_augmented_datasets.py --method adasyn --dataset nsl_kdd --output-dir data/augmented --seed 42
```

### 4) Run Complete Experiments

**Credit Card Fraud Detection:**
```bash
python scripts/run_creditcard_experiments.py
```

NSL-KDD experiments are optional.

### 5) Outputs

* **Metrics**: `outputs/metrics.json`
* **Plots**: ROC/PR curves in `outputs/plots/`
* **Augmented datasets**: `data/augmented/*_augmentation_*.csv`
* **Performance reports**: Detailed analysis in console output
* **Comparison reports**: `outputs/*_comparison_report/`

---

## Project Structure

```
SMOTE-Adv/
├── data/                           # Dataset directory
│   ├── creditcard.csv             # Credit card fraud dataset (after download)
│   ├── NSL-KDD dataset/           # NSL-KDD network intrusion dataset
│   └── augmented/                 # Generated augmented datasets
├── outputs/                       # Training results and metrics
│   ├── comprehensive_report/      # Credit card experiment results
│   ├── nsl_kdd_comparison_report/ # NSL-KDD experiment results
│   └── accuracy_analysis/         # Cross-dataset analysis
├── scripts/
│   ├── generate_augmented_datasets.py # Dataset augmentation generator
│   ├── run_creditcard_experiments.py # Credit card experiment runner
│   ├── run_nsl_kdd_experiments.py # NSL-KDD experiment runner
│   └── analyze_accuracy_results.py # Accuracy analysis script
├── src/
│   ├── train.py                   # Main training script
│   ├── data.py                    # Data loading utilities
│   ├── preprocess.py              # Data preprocessing
│   ├── augment.py                 # SMOTE-Adv implementation
│   ├── evaluate.py                # Model evaluation
│   └── models/
│       └── svm_rff.py             # SVM with RFF implementation
├── experiments/
│   └── PLAN.md                    # Experiment planning document
├── requirements.txt
├── NSL_KDD_USAGE.md               # NSL-KDD usage guide
└── README.md
```

---

## Baselines and Sources

We compare SMOTE-Adv with widely-used baselines implemented in mature libraries:

- SMOTE, Borderline-SMOTE, SVMSMOTE, ADASYN from `imbalanced-learn` ([GitHub](https://github.com/scikit-learn-contrib/imbalanced-learn), [Docs](https://imbalanced-learn.org/stable/over_sampling.html))
Install: `pip install imbalanced-learn`

## SMOTE-ADV Method

### Algorithm Overview

SMOTE-ADV enhances oversampling by adding an adversarial filtering step:

1. **Generate SMOTE samples** to target ratio
2. **Train adversarial discriminator** (logistic regression) to distinguish real vs synthetic samples
3. **Filter synthetic samples** by keeping those closest to the decision boundary (hardest to distinguish)
4. **Combine datasets**: majority original + minority original + filtered synthetic minority

### Key Idea

**Adversarial Filtering**: Keep the hardest-to-distinguish, high-density, majority-distant synthetic samples.

### Results (Credit Card Fraud, seed=42, RFF=300, median-gamma)

Paired comparison (without ADV vs with ADV):

| Generator | ROC-AUC (no ADV) | PR-AUC (no ADV) | F1-Macro (no ADV) | F1-Weighted (no ADV) | ROC-AUC (+ADV) | PR-AUC (+ADV) | F1-Macro (+ADV) | F1-Weighted (+ADV) |
|-----------|-------------------|------------------|-------------------|----------------------|----------------|---------------|------------------|---------------------|
| None | 0.9679 | 0.5269 | 0.5567 | 0.9880 | 0.9511 | 0.6524 | 0.6657 | 0.9962 |
| SMOTE | 0.9581 | 0.5110 | 0.5495 | 0.9865 | 0.9395 | 0.4835 | 0.5469 | 0.9873 |
| Borderline-SMOTE | 0.9436 | 0.5480 | 0.6678 | 0.9964 | 0.9487 | 0.5599 | 0.6762 | 0.9966 |
| SVM-SMOTE | 0.9614 | 0.6608 | 0.6662 | 0.9960 | 0.9511 | 0.6524 | 0.6657 | 0.9962 |
| ADASYN | 0.9116 | 0.3863 | 0.5216 | 0.9783 | 0.8816 | 0.4339 | 0.5535 | 0.9893 |
| SMOTE-Tomek | 0.9554 | 0.5060 | 0.5498 | 0.9865 | 0.8589 | 0.4185 | 0.5474 | 0.9893 |
| SMOTE-ENN | 0.9555 | 0.5193 | 0.5502 | 0.9864 | 0.8816 | 0.4339 | 0.5535 | 0.9893 |

Runtime (seconds):

| Method | Augment | Train | Inference |
|--------|---------|-------|-----------|
| None | - | 193.66 | 0.19 |
| SMOTE | 5.26 | 32.90 | 0.45 |
| ADASYN | 3.81 | 31.46 | 0.48 |
| Borderline-SMOTE | 4.29 | 13.48 | 0.18 |
| SVM-SMOTE | 460.76 | 13.36 | 0.21 |
| SMOTE + ADV | 13.09 | 19.99 | 0.19 |
| Borderline + ADV | 10.85 | 14.00 | 0.19 |
| SVM + ADV | 414.12 | 13.20 | 0.23 |
| SMOTE-Tomek + ADV | 109.42 | 16.35 | 0.18 |
| SMOTE-ENN + ADV | 144.97 | 16.00 | 0.18 |

---

### Results (NSL-KDD, seed=42, RFF=300, median-gamma)

Paired comparison (without ADV vs with ADV):

| Generator | ROC-AUC (no ADV) | PR-AUC (no ADV) | F1-Macro (no ADV) | F1-Weighted (no ADV) | ROC-AUC (+ADV) | PR-AUC (+ADV) | F1-Macro (+ADV) | F1-Weighted (+ADV) |
|-----------|-------------------|------------------|-------------------|----------------------|----------------|---------------|------------------|---------------------|
| None | 0.99636 | 0.99653 | 0.97724 | 0.97727 | – | – | – | – |
| SMOTE | 0.99636 | 0.99657 | 0.97721 | 0.97724 | 0.99637 | 0.99656 | 0.97731 | 0.97734 |
| Borderline-SMOTE | 0.99662 | 0.99639 | 0.97388 | 0.97391 | 0.99661 | 0.99660 | 0.97479 | 0.97482 |
| SVM-SMOTE | 0.99672 | 0.99604 | 0.97621 | 0.97624 | 0.99659 | 0.99617 | 0.97526 | 0.97529 |
| ADASYN | 0.99675 | 0.99660 | 0.97378 | 0.97381 | – | – | – | – |
| SMOTE-Tomek | 0.99632 | 0.99654 | 0.97754 | 0.97758 | 0.99638 | 0.99656 | 0.97717 | 0.97721 |
| SMOTE-ENN | 0.99544 | 0.99583 | 0.97917 | 0.97919 | 0.99637 | 0.99655 | 0.97711 | 0.97714 |

Runtime (seconds):

| Method | Augment | Train | Inference |
|--------|---------|-------|-----------|
| None | – | 8.73 | 0.20 |
| SMOTE | 7.28 | 5.78 | 0.09 |
| ADASYN | 8.31 | 5.77 | 0.11 |
| Borderline-SMOTE | 19.60 | 8.46 | 0.09 |
| SVM-SMOTE | 200.58 | 9.06 | 0.16 |
| SMOTE + ADV | 5.68 | 5.54 | 0.12 |
| Borderline + ADV | 7.31 | 5.50 | 0.09 |
| SVM + ADV | 308.89 | 12.42 | 0.26 |
| SMOTE-Tomek | 13.40 | 5.86 | 0.11 |
| SMOTE-Tomek + ADV | 21.95 | 12.03 | 0.14 |
| SMOTE-ENN | 13.36 | 5.70 | 0.10 |
| SMOTE-ENN + ADV | 38.08 | 10.03 | 0.29 |

---

## Reproducibility

* StandardScaler + median-gamma (`--rbf-gamma -1.0`)
* Seeds via `--seed`, deterministic scikit-learn settings
* All configs/metrics persisted per run

---

---
## Notes
* This README reflects `outputs/creditcard_comparison_report/summary.csv`. Re-run `python scripts/run_creditcard_experiments.py` to refresh.
* Message: Across generators (SMOTE / Borderline / SVM), our ADV filter consistently improves PR-AUC and F1-Macro while keeping inference cost unchanged.

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
@misc{smoteadv2024,
  title={SMOTE-ADV: Generator-agnostic Adversarial Filtering for Imbalanced Classification},
  author={Yanxuan Yu and Dong Liu},
  year={2024},
  url={https://github.com/zorinayu/SMOTE-Adv}
}
```

### Additional References

- `imbalanced-learn` project: https://github.com/scikit-learn-contrib/imbalanced-learn
- `smote-variants` project: https://github.com/analyticalmindsltd/smote_variants