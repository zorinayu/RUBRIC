## Reproducibility Guide

This document describes the exact environment, data layout, seeds, and run protocol used to reproduce the paper’s tables and figures.

### Environment
- Python: 3.10+
- Install: `pip install -r requirements.txt`
- Hardware: CPU is sufficient; GPU is optional for boosted trees
- Determinism: We set seeds where supported by scikit-learn/NumPy. Minor nondeterminism may remain due to shuffles and library internals.

### Data Layout
Place public datasets as follows:
- Credit Card Fraud (ULB/Kaggle): `data/creditcard.csv`
- NSL-KDD: `data/NSL-KDD dataset/` containing `KDDTrain+.txt`, `KDDTest+.txt` (and variants)
- IEEE-CIS Fraud Detection: `data/ieee-fraud-detection/` with `train_transaction.csv` (and optionally `train_identity.csv`)
- Santander Customer Transaction: `data/santander-customer-transaction-prediction/` with `train.csv`, `test.csv`

### One-Click Runners
- Credit Card: `python scripts/run_creditcard_experiments.py`
- IEEE-CIS: `python scripts/run_ieee_experiments.py`
- Santander: `python scripts/run_santander_experiments.py`
- Or call `src/train.py` directly (see README) to run a single configuration.

### Multi-Seed Protocol
- Seed set: `{42, 52, 62, 72, 82}` (example set used for paper aggregates)
- Policy: For each (dataset, model, augmentation) configuration, repeat runs across all seeds; select decision thresholds on validation by maximizing F1, then report test metrics once per run.
- Aggregation: Compute mean±std across seeds for PR-AUC, ROC-AUC, F1, Recall, Brier; paired deltas (RUBRIC–Base) are computed over matched units.

### Key Flags
Common flags for `src/train.py`:
```
--dataset <creditcard|nsl_kdd|ieee_fraud_detection|santander>
--augment <none|smote|adasyn|borderline-smote|svm-smote|smote-tomek|smote-enn|adv>
--gen-kind <smote|borderline|borderline2|svm|kmeans|smote-tomek|smote-enn|adasyn>   # when --augment adv
--target-ratio 0.3
--keep-frac 0.65       # RUBRIC retention fraction
--adv-C 1.0            # discriminator C
--rbf-components 300
--rbf-gamma -1.0       # median heuristic
--svm-C 1.0
--test-size 0.2
--seed 42
```

### Outputs and Mapping
- Per-run: `outputs/<run_name>/metrics.json`, plots under `outputs/<run_name>/plots/`
- Summary CSVs: `outputs/<dataset>_comparison_report/summary.csv` (and pivots)
- Paper mapping: see `RESULTS.md` for table/figure mapping and tag info

### Expected Variance
RUBRIC emphasizes near-boundary, discriminator-informed selection. Combined with random splits and model training, metrics will vary across seeds. Paper claims are based on multi-seed aggregates; single-seed examples are for illustration.

### Ethical Notes
For fraud/health-like applications, evaluate calibration and thresholding costs. We provide Brier scores and reliability plots. Do not deploy without domain-appropriate evaluation.


