#!/usr/bin/env python3
"""
Dataset Generator for Augmented Credit Card Fraud Detection
Generates augmented datasets using the best performing parameters
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import time

from data import load_creditcard_csv, load_nsl_kdd_data
from preprocess import preprocess_df
from augment import (
    smote_oversample,
    smote_then_adversarial_filter,
    adasyn_oversample,
    borderline_smote_oversample,
    svm_smote_oversample,
    kmeans_smote_oversample,
)
from models.svm_rff import SVMWithRFF

# Best parameters from experiments
BEST_PARAMS = {
    'none': {
        'rbf_components': 400,
        'rbf_gamma': 0.5,
        'svm_C': 1.0
    },
    'smote': {
        'rbf_components': 400,
        'rbf_gamma': 0.5,
        'svm_C': 1.0,
        'target_ratio': 1.0
    },
    'smote-adv': {
        'rbf_components': 400,
        'rbf_gamma': 0.5,
        'svm_C': 1.0,
        'target_ratio': 1.0,
        'keep_top_frac': 0.7
    },
    'adasyn': {
        'rbf_components': 400,
        'rbf_gamma': 0.5,
        'svm_C': 1.0,
        'target_ratio': 1.0
    },
    'borderline-smote': {
        'rbf_components': 400,
        'rbf_gamma': 0.5,
        'svm_C': 1.0,
        'target_ratio': 1.0,
        'kind': 'borderline-1'
    },
    'svm-smote': {
        'rbf_components': 400,
        'rbf_gamma': 0.5,
        'svm_C': 1.0,
        'target_ratio': 1.0
    },
    'kmeans-smote': {
        'rbf_components': 400,
        'rbf_gamma': 0.5,
        'svm_C': 1.0,
        'target_ratio': 1.0
    },
}

def generate_augmented_dataset(method, input_file, output_dir, dataset_type='creditcard', test_size=0.2, seed=42):
    """
    Generate augmented dataset using specified method
    
    Args:
        method: 'none', 'smote', or 'smote-adv'
        input_file: Path to input CSV file (for creditcard) or None (for nsl_kdd)
        output_dir: Directory to save augmented dataset
        dataset_type: 'creditcard' or 'nsl_kdd'
        test_size: Fraction of data to use for testing
        seed: Random seed for reproducibility
    
    Returns:
        dict: Performance metrics and dataset info
    """
    print(f"Generating augmented dataset using method: {method.upper()}")
    print(f"Dataset type: {dataset_type}")
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    start_time = time.time()
    
    if dataset_type == 'creditcard':
        df = load_creditcard_csv(input_file)
    elif dataset_type == 'nsl_kdd':
        df = load_nsl_kdd_data()
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    X_df, y = preprocess_df(df, dataset_type=dataset_type)
    X = X_df.values.astype(np.float32)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    
    original_train_size = len(X_train)
    original_minority_count = np.sum(y_train == 1)
    original_majority_count = np.sum(y_train == 0)
    
    print(f"Original training data: {original_train_size} samples")
    print(f"  - Majority class: {original_majority_count} samples")
    print(f"  - Minority class: {original_minority_count} samples")
    print(f"  - Minority ratio: {original_minority_count/original_train_size:.4f}")
    
    # Apply augmentation
    print(f"Applying {method.upper()} augmentation...")
    aug_start_time = time.time()
    
    if method == 'none':
        X_train_aug, y_train_aug = X_train, y_train
        augmentation_info = "No augmentation applied"
        
    elif method == 'smote':
        params = BEST_PARAMS['smote']
        X_train_aug, y_train_aug = smote_oversample(
            X_train, y_train, 
            random_state=seed, 
            ratio=params['target_ratio']
        )
        augmentation_info = f"SMOTE with ratio={params['target_ratio']}"
        
    elif method == 'smote-adv':
        params = BEST_PARAMS['smote-adv']
        X_train_aug, y_train_aug = smote_then_adversarial_filter(
            X_train, y_train, 
            random_state=seed, 
            ratio=params['target_ratio'],
            keep_top_frac=params['keep_top_frac']
        )
        augmentation_info = f"SMOTE-Adv with ratio={params['target_ratio']}, keep_frac={params['keep_top_frac']}"
    elif method == 'adasyn':
        params = BEST_PARAMS['adasyn']
        X_train_aug, y_train_aug = adasyn_oversample(
            X_train, y_train,
            random_state=seed,
            ratio=params['target_ratio']
        )
        augmentation_info = f"ADASYN with ratio={params['target_ratio']}"
    elif method == 'borderline-smote':
        params = BEST_PARAMS['borderline-smote']
        X_train_aug, y_train_aug = borderline_smote_oversample(
            X_train, y_train,
            random_state=seed,
            ratio=params['target_ratio'],
            kind=params['kind']
        )
        augmentation_info = f"Borderline-SMOTE ({params['kind']}) with ratio={params['target_ratio']}"
    elif method == 'svm-smote':
        params = BEST_PARAMS['svm-smote']
        X_train_aug, y_train_aug = svm_smote_oversample(
            X_train, y_train,
            random_state=seed,
            ratio=params['target_ratio']
        )
        augmentation_info = f"SVM-SMOTE with ratio={params['target_ratio']}"
    elif method == 'kmeans-smote':
        params = BEST_PARAMS['kmeans-smote']
        X_train_aug, y_train_aug = kmeans_smote_oversample(
            X_train, y_train,
            random_state=seed,
            ratio=params['target_ratio']
        )
        augmentation_info = f"KMeans-SMOTE with ratio={params['target_ratio']}"
    
    aug_time = time.time() - aug_start_time
    print(f"Augmentation completed in {aug_time:.2f} seconds")
    
    # Calculate augmentation statistics
    augmented_train_size = len(X_train_aug)
    augmented_minority_count = np.sum(y_train_aug == 1)
    augmented_majority_count = np.sum(y_train_aug == 0)
    added_samples = augmented_train_size - original_train_size
    
    print(f"Augmented training data: {augmented_train_size} samples")
    print(f"  - Majority class: {augmented_majority_count} samples")
    print(f"  - Minority class: {augmented_minority_count} samples")
    print(f"  - Minority ratio: {augmented_minority_count/augmented_train_size:.4f}")
    print(f"  - Added samples: {added_samples}")
    print(f"  - Augmentation ratio: {added_samples/original_train_size:.2f}x")
    
    # Train model and evaluate performance
    print("Training model and evaluating performance...")
    model_start_time = time.time()
    
    params = BEST_PARAMS[method]
    model = SVMWithRFF(
        n_components=params['rbf_components'],
        gamma=params['rbf_gamma'],
        C=params['svm_C'],
        random_state=seed
    )
    model.fit(X_train_aug, y_train_aug)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_scores = model.decision_function(X_test)
    
    roc_auc = roc_auc_score(y_test, y_scores)
    pr_auc = average_precision_score(y_test, y_scores)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    model_time = time.time() - model_start_time
    total_time = time.time() - start_time
    
    print(f"Model training and evaluation completed in {model_time:.2f} seconds")
    print(f"Total time: {total_time:.2f} seconds")
    
    # Performance summary
    print("\nPerformance Summary:")
    print(f"  - ROC-AUC: {roc_auc:.4f}")
    print(f"  - PR-AUC: {pr_auc:.4f}")
    print(f"  - F1-Macro: {report['macro avg']['f1-score']:.4f}")
    print(f"  - F1-Weighted: {report['weighted avg']['f1-score']:.4f}")
    print(f"  - Minority Precision: {report['1']['precision']:.4f}")
    print(f"  - Minority Recall: {report['1']['recall']:.4f}")
    print(f"  - Minority F1: {report['1']['f1-score']:.4f}")
    
    # Save augmented dataset
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output filename
    if dataset_type == 'creditcard':
        input_filename = Path(input_file).stem
    else:
        input_filename = dataset_type
    output_filename = f"{input_filename}_augmentation_{method}.csv"
    output_path = output_dir / output_filename
    
    # Combine augmented training data with test data
    X_combined = np.vstack([X_train_aug, X_test])
    y_combined = np.hstack([y_train_aug, y_test])
    
    # Create DataFrame with original column names
    augmented_df = pd.DataFrame(X_combined, columns=X_df.columns)
    augmented_df['Class'] = y_combined
    
    # Save to CSV
    augmented_df.to_csv(output_path, index=False)
    print(f"\nAugmented dataset saved to: {output_path}")
    
    # Save performance metrics
    metrics_path = output_dir / f"{input_filename}_augmentation_{method}_metrics.json"
    import json
    metrics = {
        'method': method,
        'augmentation_info': augmentation_info,
        'original_samples': original_train_size,
        'augmented_samples': augmented_train_size,
        'added_samples': added_samples,
        'augmentation_ratio': added_samples/original_train_size,
        'original_minority_count': int(original_minority_count),
        'augmented_minority_count': int(augmented_minority_count),
        'original_minority_ratio': original_minority_count/original_train_size,
        'augmented_minority_ratio': augmented_minority_count/augmented_train_size,
        'performance': {
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'f1_macro': float(report['macro avg']['f1-score']),
            'f1_weighted': float(report['weighted avg']['f1-score']),
            'minority_precision': float(report['1']['precision']),
            'minority_recall': float(report['1']['recall']),
            'minority_f1': float(report['1']['f1-score'])
        },
        'timing': {
            'augmentation_time': float(aug_time),
            'model_time': float(model_time),
            'total_time': float(total_time)
        },
        'parameters': params
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Performance metrics saved to: {metrics_path}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Generate augmented datasets for fraud/intrusion detection')
    parser.add_argument('--method', type=str, required=True, 
                       choices=['none', 'smote', 'smote-adv', 'adasyn', 'borderline-smote', 'svm-smote', 'kmeans-smote'],
                       help='Augmentation method to use')
    parser.add_argument('--dataset', type=str, default='creditcard',
                       choices=['creditcard', 'nsl_kdd'],
                       help='Dataset type to use')
    parser.add_argument('--input', type=str, default='data/creditcard.csv',
                       help='Input CSV file path (for creditcard dataset)')
    parser.add_argument('--output-dir', type=str, default='data/augmented',
                       help='Output directory for augmented datasets')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Fraction of data to use for testing')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Generate augmented dataset
    metrics = generate_augmented_dataset(
        method=args.method,
        input_file=args.input,
        output_dir=args.output_dir,
        dataset_type=args.dataset,
        test_size=args.test_size,
        seed=args.seed
    )
    
    print(f"\nDataset generation completed successfully!")
    print(f"Method: {metrics['method'].upper()}")
    print(f"Added {metrics['added_samples']} samples ({metrics['augmentation_ratio']:.2f}x increase)")
    print(f"Final dataset size: {metrics['augmented_samples']} samples")
    print(f"Performance: ROC-AUC={metrics['performance']['roc_auc']:.4f}, PR-AUC={metrics['performance']['pr_auc']:.4f}")

if __name__ == '__main__':
    main()
