# NSL-KDD Dataset Usage Guide

## Overview
This project now supports both Credit Card Fraud Detection and NSL-KDD Network Intrusion Detection datasets. You can run experiments on either dataset independently.

## NSL-KDD Dataset Results

### Baseline Performance Summary
The following baseline experiments were run on the NSL-KDD dataset:

| Method | PR-AUC | ROC-AUC | F1-Macro | F1-Weighted |
|--------|--------|---------|----------|-------------|
| None (baseline) | 0.9694 | 0.9767 | 0.9311 | 0.9312 |
| SMOTE | 0.9693 | 0.9767 | 0.9311 | 0.9311 |
| SMOTE-Adv | 0.9694 | 0.9768 | 0.9308 | 0.9309 |

### Key Findings
- **Best Performance**: No augmentation (baseline) achieved the highest PR-AUC of 0.9694
- **Dataset Characteristics**: NSL-KDD has a more balanced class distribution (48.12% attacks vs 51.88% normal) compared to credit card fraud
- **Augmentation Impact**: Data augmentation methods show minimal improvement due to the relatively balanced nature of the dataset
- **Consistent Performance**: All methods achieve very high performance (>96% PR-AUC), indicating the dataset is well-suited for classification

## Usage Instructions

### Running NSL-KDD Experiments

1. **Single Experiment**:
```bash
python src/train.py --dataset nsl_kdd --augment none --rbf-components 300 --rbf-gamma 0.5 --test-size 0.2 --seed 42
```

2. **Run All Baselines**:
```bash
python scripts/run_nsl_kdd_experiments.py
```

3. **Generate Augmented Datasets**:
```bash
# Generate no-augmentation dataset
python scripts/generate_augmented_datasets.py --method none --dataset nsl_kdd --output-dir data/augmented --seed 42

# Generate SMOTE augmented dataset
python scripts/generate_augmented_datasets.py --method smote --dataset nsl_kdd --output-dir data/augmented --seed 42

# Generate SMOTE-Adv augmented dataset
python scripts/generate_augmented_datasets.py --method smote-adv --dataset nsl_kdd --output-dir data/augmented --seed 42
```

### Running Credit Card Experiments

1. **Single Experiment**:
```bash
python src/train.py --dataset creditcard --augment none --rbf-components 300 --rbf-gamma 0.5 --test-size 0.2 --seed 42
```

2. **Run All Baselines**:
```bash
python scripts/run_nsl_kdd_experiments.py
```

## Generated Files

### NSL-KDD Augmented Datasets
- `data/augmented/nsl_kdd_augmentation_none.csv` - Original dataset (no augmentation)
- `data/augmented/nsl_kdd_augmentation_smote.csv` - SMOTE augmented dataset
- `data/augmented/nsl_kdd_augmentation_smote-adv.csv` - SMOTE-Adv augmented dataset

### Performance Metrics
- `data/augmented/nsl_kdd_augmentation_*_metrics.json` - Detailed performance metrics for each method

### Experiment Results
- `outputs/nsl_kdd_comparison_report/` - Comprehensive comparison report
- `outputs/nsl_kdd_*/` - Individual experiment results with plots and metrics

## Dataset Selection

The system is designed to work with either dataset independently:

- **Credit Card Fraud**: Highly imbalanced dataset (0.17% fraud rate) where data augmentation shows significant benefits
- **NSL-KDD**: More balanced dataset (48.12% attack rate) where augmentation has minimal impact

Choose the appropriate dataset based on your research needs:
- Use Credit Card dataset for studying extreme class imbalance
- Use NSL-KDD dataset for network security applications or balanced classification studies

## Notes

- All experiments use the same random seed (42) for reproducibility
- The NSL-KDD dataset is automatically loaded from `data/NSL-KDD dataset/` directory
- Results show that for relatively balanced datasets like NSL-KDD, sophisticated augmentation methods may not provide significant benefits over baseline approaches
- The high performance across all methods (>96% PR-AUC) indicates the NSL-KDD dataset is well-suited for binary classification tasks
