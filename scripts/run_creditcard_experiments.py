#!/usr/bin/env python3
"""
Credit Card Fraud Detection Experiment Runner
Runs baseline experiments for Credit Card fraud detection dataset
"""

import subprocess
import json
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def run_experiment(cmd, description):
    """Run a single experiment"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("Experiment completed successfully")
            return True
        else:
            print(f"Experiment failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"Experiment failed with exception: {e}")
        return False

def collect_results():
    """Collect results from all Credit Card experiment directories"""
    results = []
    outputs_dir = Path('outputs')
    
    if not outputs_dir.exists():
        print("No outputs directory found")
        return results
    
    for exp_dir in outputs_dir.iterdir():
        if exp_dir.is_dir() and exp_dir.name.startswith('creditcard_'):
            config_file = exp_dir / 'config.json'
            metrics_file = exp_dir / 'metrics.json'
            
            if config_file.exists() and metrics_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    
                    result = {
                        'experiment_dir': exp_dir.name,
                        **config,
                        **metrics
                    }
                    results.append(result)
                except Exception as e:
                    print(f"Error reading {exp_dir}: {e}")
    
    return results

def create_comparison_report(results):
    """Create a comprehensive comparison report for Credit Card experiments"""
    if not results:
        print("No results to compare")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Create comparison report
    report_dir = Path('outputs/creditcard_comparison_report')
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    df.to_csv(report_dir / 'all_results.csv', index=False)
    
    # Create summary table (with generator info when available)
    # Flatten timings if present in config.json
    def get_timing(row, key):
        try:
            return row.get('timings', {}).get(key)
        except Exception:
            return None

    # Unpack timing columns into the dataframe
    df['train_s'] = df.get('timings.train_s') if 'timings.train_s' in df.columns else None
    df['inference_s'] = df.get('timings.inference_s') if 'timings.inference_s' in df.columns else None
    # Attempt to read nested
    for r in range(len(df)):
        if 'timings' in df.columns and isinstance(df.at[r, 'timings'], dict):
            t = df.at[r, 'timings']
            df.at[r, 'augment_total_s'] = t.get('augment_total_s')
            df.at[r, 'smote_generate_s'] = t.get('smote_generate_s')
            df.at[r, 'filter_lr_train_s'] = t.get('filter_lr_train_s')
            df.at[r, 'filter_knn_s'] = t.get('filter_knn_s')
            df.at[r, 'filter_select_s'] = t.get('filter_select_s')
            df.at[r, 'adv_total_s'] = t.get('adv_total_s')
            df.at[r, 'train_s'] = t.get('train_s')
            df.at[r, 'inference_s'] = t.get('inference_s')

    # Extract generator kind for smote-adv runs
    gen_kinds = []
    for i in range(len(df)):
        g = None
        if 'adv_params' in df.columns and isinstance(df.at[i, 'adv_params'], dict):
            g = df.at[i, 'adv_params'].get('gen_kind')
        gen_kinds.append(g)
    df['generator'] = gen_kinds

    summary_cols = ['experiment_dir', 'augment', 'generator', 'rbf_components', 'rbf_gamma',
                   'svm_C', 'roc_auc', 'pr_auc', 'f1_macro', 'f1_weighted',
                   'augment_total_s', 'train_s', 'inference_s']
    summary_df = df[summary_cols].copy()
    summary_df = summary_df.sort_values('pr_auc', ascending=False)
    
    # Save summary
    summary_df.to_csv(report_dir / 'summary.csv', index=False)
    
    # Create comparison plots
    create_comparison_plots(df, report_dir)
    
    # Print summary
    print(f"\n{'='*80}")
    print("CREDIT CARD FRAUD DETECTION EXPERIMENT COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"Total experiments: {len(df)}")
    print(f"Report saved to: {report_dir}")
    
    print(f"\nTOP 5 PERFORMERS (by PR-AUC):")
    print("-" * 80)
    for i, row in summary_df.head().iterrows():
        print(f"{i+1}. {row['experiment_dir']}")
        print(f"   PR-AUC: {row['pr_auc']:.4f}, ROC-AUC: {row['roc_auc']:.4f}, F1-Macro: {row['f1_macro']:.4f}")
        print(f"   Config: {row['augment']}, RFF={row['rbf_components']}, γ={row['rbf_gamma']}, C={row['svm_C']}")
        print()
    
    return summary_df

def create_comparison_plots(df, report_dir):
    """Create comparison plots for Credit Card experiments"""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Credit Card Fraud Detection Experiment Comparison Results', fontsize=16, fontweight='bold')
    
    # PR-AUC by augmentation method
    ax1 = axes[0, 0]
    sns.boxplot(data=df, x='augment', y='pr_auc', ax=ax1)
    ax1.set_title('PR-AUC by Augmentation Method')
    ax1.set_ylabel('PR-AUC')
    ax1.tick_params(axis='x', rotation=45)
    
    # ROC-AUC by augmentation method
    ax2 = axes[0, 1]
    sns.boxplot(data=df, x='augment', y='roc_auc', ax=ax2)
    ax2.set_title('ROC-AUC by Augmentation Method')
    ax2.set_ylabel('ROC-AUC')
    ax2.tick_params(axis='x', rotation=45)
    
    # F1-Macro by augmentation method
    ax3 = axes[1, 0]
    sns.boxplot(data=df, x='augment', y='f1_macro', ax=ax3)
    ax3.set_title('F1-Macro by Augmentation Method')
    ax3.set_ylabel('F1-Macro')
    ax3.tick_params(axis='x', rotation=45)
    
    # PR-AUC vs RFF Components
    ax4 = axes[1, 1]
    for augment in df['augment'].unique():
        subset = df[df['augment'] == augment]
        ax4.scatter(subset['rbf_components'], subset['pr_auc'], 
                   label=augment, alpha=0.7, s=50)
    ax4.set_xlabel('RFF Components')
    ax4.set_ylabel('PR-AUC')
    ax4.set_title('PR-AUC vs RFF Components')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(report_dir / 'creditcard_comparison_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to: {report_dir / 'creditcard_comparison_plots.png'}")

def main():
    """Main Credit Card experiment runner"""
    print("Credit Card Fraud Detection - Experiment Runner")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define Credit Card baseline experiments
    base = 'python src/train.py --dataset creditcard --rbf-components 300 --rbf-gamma -1.0 --test-size 0.2 --seed 42'
    experiments = [
        { 'cmd': f"{base} --augment none", 'desc': 'Baseline: No augmentation (class_weight only)' },
        { 'cmd': f"{base} --augment smote", 'desc': 'SMOTE: Standard SMOTE oversampling' },
        { 'cmd': f"{base} --augment smote-adv --gen-kind smote --target-ratio 0.3 --keep-frac 0.6 --adv-C 2.0 --w-density 0.2", 'desc': 'SMOTE + Adv filter (best from grid)' },
        { 'cmd': f"{base} --augment adasyn", 'desc': 'ADASYN oversampling' },
        { 'cmd': f"{base} --augment borderline-smote", 'desc': 'Borderline-SMOTE (borderline-1)' },
        { 'cmd': f"{base} --augment svm-smote", 'desc': 'SVM-SMOTE oversampling' },
        { 'cmd': f"{base} --augment smote-adv --gen-kind borderline --target-ratio 0.3 --keep-frac 0.6 --adv-C 2.0 --w-density 0.2", 'desc': 'Borderline + Adv filter' },
        { 'cmd': f"{base} --augment smote-adv --gen-kind svm --target-ratio 0.3 --keep-frac 0.6 --adv-C 2.0 --w-density 0.2", 'desc': 'SVM-SMOTE + Adv filter' },
        # { 'cmd': f"{base} --augment kmeans-smote", 'desc': 'KMeans-SMOTE oversampling' },  # optional
    ]
    
    # Run experiments
    successful_experiments = 0
    for exp in experiments:
        if run_experiment(exp['cmd'], exp['desc']):
            successful_experiments += 1
    
    print(f"\nExperiment Summary: {successful_experiments}/{len(experiments)} completed successfully")
    
    # Collect and compare results
    print("\nCollecting and comparing results...")
    results = collect_results()
    if results:
        summary_df = create_comparison_report(results)
        
        # Print recommendations
        print(f"\nRECOMMENDATIONS:")
        print("-" * 40)
        best_exp = summary_df.iloc[0]
        print(f"Best performing configuration: {best_exp['experiment_dir']}")
        print(f"PR-AUC: {best_exp['pr_auc']:.4f}")
        print(f"ROC-AUC: {best_exp['roc_auc']:.4f}")
        
        if best_exp['pr_auc'] < 0.1:
            print("All experiments show low PR-AUC - consider:")
            print("- Trying different hyperparameters")
            print("- Using different augmentation techniques")
            print("- Checking data quality")
        else:
            print("Good performance achieved!")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main()
