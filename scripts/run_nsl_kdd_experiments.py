#!/usr/bin/env python3
"""
NSL-KDD Dataset Experiment Runner
Runs baseline experiments for NSL-KDD network intrusion dataset
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
    """Collect results from all NSL-KDD experiment directories"""
    results = []
    outputs_dir = Path('outputs')
    
    if not outputs_dir.exists():
        print("No outputs directory found")
        return results
    
    for exp_dir in outputs_dir.iterdir():
        if exp_dir.is_dir() and exp_dir.name.startswith('nsl_kdd_'):
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
    """Create a comprehensive comparison report for NSL-KDD experiments"""
    if not results:
        print("No results to compare")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Create comparison report
    report_dir = Path('outputs/nsl_kdd_comparison_report')
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    df.to_csv(report_dir / 'all_results.csv', index=False)
    
    # Create summary table
    summary_cols = ['experiment_dir', 'augment', 'rbf_components', 'rbf_gamma', 
                   'svm_C', 'roc_auc', 'pr_auc', 'f1_macro', 'f1_weighted']
    summary_df = df[summary_cols].copy()
    summary_df = summary_df.sort_values('pr_auc', ascending=False)
    
    # Save summary
    summary_df.to_csv(report_dir / 'summary.csv', index=False)
    
    # Create comparison plots
    create_comparison_plots(df, report_dir)
    
    # Print summary
    print(f"\n{'='*80}")
    print("NSL-KDD EXPERIMENT COMPARISON SUMMARY")
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
    """Create comparison plots for NSL-KDD experiments"""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('NSL-KDD Experiment Comparison Results', fontsize=16, fontweight='bold')
    
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
    plt.savefig(report_dir / 'nsl_kdd_comparison_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to: {report_dir / 'nsl_kdd_comparison_plots.png'}")

def main():
    """Main NSL-KDD experiment runner"""
    print("NSL-KDD Network Intrusion Detection - Experiment Runner")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define NSL-KDD baseline experiments
    experiments = [
        {
            'cmd': 'python src/train.py --dataset nsl_kdd --augment none --rbf-components 300 --rbf-gamma 0.5 --test-size 0.2 --seed 42',
            'desc': 'Baseline: No augmentation (class_weight only)'
        },
        {
            'cmd': 'python src/train.py --dataset nsl_kdd --augment smote --rbf-components 300 --rbf-gamma 0.5 --test-size 0.2 --seed 42',
            'desc': 'SMOTE: Standard SMOTE oversampling'
        },
        {
            'cmd': 'python src/train.py --dataset nsl_kdd --augment smote-adv --rbf-components 300 --rbf-gamma 0.5 --test-size 0.2 --seed 42',
            'desc': 'SMOTE-Adv: Adversarially-filtered SMOTE'
        }
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
