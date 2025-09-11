# Experiment Plan for SMOTE-Adv

## Datasets
- Kaggle Credit Card Fraud (tabular, extreme imbalance)
- NSL-KDD (network intrusion)
- MIT-BIH Arrhythmia (ECG)
- Optional: Chest X-ray14 (image; convert to tabular via embeddings or use separate pipeline)

## Baselines
- No-augment (class_weight only)
- SMOTE, ADASYN, Borderline-SMOTE (b1), SVM-SMOTE, KMeans-SMOTE
- SMOTE-GAN / CTGAN (optional if adding torch)
- Our SMOTE-Adv

## Metrics
- PR-AUC (primary), ROC-AUC, F1 (macro/weighted), Recall@Minority, Confusion Matrix

## Ablations
- α ∈ {0.2, 0.4, 0.6, 0.8}
- keep_frac κ ∈ {0.5, 0.7, 0.9}
- rounds R ∈ {1, 2, 3}
- RFF components ∈ {200, 400, 800}

## Reporting
- Mean±std over 5 seeds (42..46)
- Tables + PR/ROC curves

## Commands (example)
```
python src/train.py --augment none
python src/train.py --augment smote
python src.train.py --augment adasyn
python src/train.py --augment borderline-smote
python src/train.py --augment svm-smote
python src/train.py --augment kmeans-smote
python src/train.py --augment smote-adv
```
