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
import numpy as np

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
    
    # Flatten timings and extract generator when available
    if 'timings' in df.columns:
        for r in range(len(df)):
            t = df.at[r, 'timings'] if isinstance(df.at[r, 'timings'], dict) else {}
            df.at[r, 'augment_total_s'] = t.get('augment_total_s')
            df.at[r, 'train_s'] = t.get('train_s')
            df.at[r, 'inference_s'] = t.get('inference_s')
    gens = []
    for r in range(len(df)):
        g = None
        if 'adv_params' in df.columns and isinstance(df.at[r, 'adv_params'], dict):
            g = df.at[r, 'adv_params'].get('gen_kind')
        gens.append(g)
    df['generator'] = gens

    # Create summary table
    summary_cols = ['experiment_dir', 'augment', 'generator', 'rbf_components', 'rbf_gamma', 
                   'svm_C', 'roc_auc', 'pr_auc', 'f1_macro', 'f1_weighted',
                   'augment_total_s', 'train_s', 'inference_s']
    summary_df = df[summary_cols].copy()
    summary_df = summary_df.sort_values('pr_auc', ascending=False)
    
    # Save summary
    summary_df.to_csv(report_dir / 'summary.csv', index=False)
    
    # Create comparison plots
    create_comparison_plots(df, report_dir)

    # Compute delta stats with bootstrap CI and permutation-test p-values
    compute_and_save_delta_stats(df, report_dir)
    
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

def _paired_delta(df, metric):
    base = df[df['augment'].isin(['smote','svm-smote','borderline-smote','adasyn','smote-tomek','smote-enn'])].copy()
    adv = df[(df['augment']=='smote-adv')].copy()
    adv['generator'] = adv.get('generator', None) if 'generator' in adv.columns else None
    rows = []
    for g in adv['generator'].dropna().unique():
        b = base[base['augment'].isin([g if g in ['smote','adasyn'] else f"{g}"])]
        a = adv[adv['generator']==g]
        if 'seed' in df.columns:
            b = b.set_index(['seed'])
            a = a.set_index(['seed'])
            pair = b[[metric]].join(a[[metric]], lsuffix='_base', rsuffix='_adv', how='inner')
            for seed, r in pair.iterrows():
                rows.append({'generator': g, 'seed': seed, 'delta': r[f'{metric}_adv'] - r[f'{metric}_base']})
        else:
            if len(b)>0 and len(a)>0:
                rows.append({'generator': g, 'seed': None, 'delta': float(a[metric].mean() - b[metric].mean())})
    return pd.DataFrame(rows)


def _bootstrap_ci(x, n_boot=1000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    if len(x) == 0:
        return 0.0, 0.0, 0.0
    means = []
    for _ in range(n_boot):
        s = rng.choice(x, size=len(x), replace=True)
        means.append(np.mean(s))
    m = float(np.mean(x))
    lo = float(np.percentile(means, 100*alpha/2))
    hi = float(np.percentile(means, 100*(1-alpha/2)))
    return m, lo, hi


def create_comparison_plots(df, report_dir):
    """Create comparison plots for NSL-KDD experiments (add paired deltas)."""
    plt.style.use('default')
    plt.style.use('seaborn-whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('NSL-KDD Experiment Comparison Results', fontsize=18, fontweight='bold')
    
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

    # New: PR vs ROC scatter with hue=augment
    ax5 = axes[0, 2]
    sns.scatterplot(data=df, x='roc_auc', y='pr_auc', hue='augment', ax=ax5, s=60, alpha=0.8)
    ax5.set_title('PR-AUC vs ROC-AUC (by method)')
    ax5.set_xlabel('ROC-AUC')
    ax5.set_ylabel('PR-AUC')
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # New: Runtime bars
    ax6 = axes[1, 2]
    rt = df.groupby('augment').agg(train_s=('train_s','mean'), inference_s=('inference_s','mean'), augment_s=('augment_total_s','mean')).reset_index()
    rt.plot(x='augment', y=['augment_s','train_s','inference_s'], kind='bar', ax=ax6)
    ax6.set_title('Runtime (avg seconds)')
    ax6.set_ylabel('Seconds')
    ax6.tick_params(axis='x', rotation=45)
    
    # New: Paired Δ plots
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 9))
    metrics = [
        ('pr_auc', 'ΔPR-AUC'),
        ('recall_at_fpr_1pct', 'ΔRecall@FPR=1%'),
        ('lift_at_5pct', 'ΔLift@5%'),
        ('f1_macro', 'ΔF1-Macro'),
    ]
    for ax, (m, title) in zip(axes2.flatten(), metrics):
        pair = _paired_delta(df, m)
        if len(pair)==0:
            ax.set_title(f"{title} (no pairs)")
            ax.axhline(0, ls='--', alpha=0.4)
            continue
        rows = []
        for g, gdf in pair.groupby('generator'):
            mean, lo, hi = _bootstrap_ci(gdf['delta'].values)
            rows.append({'generator': g, 'mean': mean, 'lo': lo, 'hi': hi})
        dd = pd.DataFrame(rows).sort_values('mean')
        ax.hlines(0, -0.5, len(dd)-0.5, linestyles='dashed', alpha=0.4)
        ax.errorbar(range(len(dd)), dd['mean'], yerr=[dd['mean']-dd['lo'], dd['hi']-dd['mean']], fmt='o')
        ax.set_xticks(range(len(dd)))
        ax.set_xticklabels(dd['generator'], rotation=45, ha='right')
        ax.set_ylabel(title)
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(report_dir / 'paired_deltas.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Win-rate plot (PR-AUC)
    if 'seed' in df.columns:
        adv = df[df['augment']=='smote-adv']
        base = df[df['augment'].isin(['smote','adasyn','svm-smote','borderline-smote','smote-tomek','smote-enn'])]
        wins = []
        for g in adv['generator'].dropna().unique():
            a = adv[adv['generator']==g].set_index('seed')
            b = base[base['augment'].isin([g if g in ['smote','adasyn'] else f"{g}"])].set_index('seed')
            pair = a[['pr_auc']].join(b[['pr_auc']], lsuffix='_adv', rsuffix='_base', how='inner')
            if len(pair)>0:
                win_rate = float(np.mean(pair['pr_auc_adv'] > pair['pr_auc_base']))
                wins.append({'generator': g, 'win_rate': win_rate})
        if wins:
            wr = pd.DataFrame(wins)
            plt.figure(figsize=(6,4))
            sns.barplot(data=wr, x='generator', y='win_rate')
            plt.ylim(0,1)
            plt.title('Win-rate of +ADV over base (PR-AUC)')
            plt.ylabel('Win-rate')
            plt.xlabel('Generator')
            plt.tight_layout()
            plt.savefig(report_dir / 'win_rate.png', dpi=300, bbox_inches='tight')
            plt.close()

    plt.tight_layout()
    plt.savefig(report_dir / 'nsl_kdd_comparison_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to: {report_dir / 'nsl_kdd_comparison_plots.png'}")


def _paired_rows(df, metric):
    adv = df[df['augment']=='smote-adv'].copy()
    base = df[df['augment'].isin(['smote','adasyn','svm-smote','borderline-smote','smote-tomek','smote-enn'])].copy()
    rows = []
    for g in adv['generator'].dropna().unique():
        a = adv[adv['generator']==g]
        base_name = g if g in ['smote','adasyn'] else g if g in base['augment'].unique() else None
        if base_name is None:
            continue
        b = base[base['augment']==base_name]
        if 'seed' in df.columns:
            a = a.set_index('seed')
            b = b.set_index('seed')
            pair = b[[metric]].join(a[[metric]], lsuffix='_base', rsuffix='_adv', how='inner')
            for seed, r in pair.iterrows():
                rows.append({'generator': g, 'seed': seed, 'delta': float(r[f'{metric}_adv'] - r[f'{metric}_base'])})
        else:
            if len(a)>0 and len(b)>0:
                rows.append({'generator': g, 'seed': None, 'delta': float(a[metric].mean() - b[metric].mean())})
    return pd.DataFrame(rows)


def _bootstrap_ci_array(x, n_boot=2000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    if len(x) == 0:
        return 0.0, 0.0, 0.0
    means = []
    for _ in range(n_boot):
        s = rng.choice(x, size=len(x), replace=True)
        means.append(np.mean(s))
    m = float(np.mean(x))
    lo = float(np.percentile(means, 100*alpha/2))
    hi = float(np.percentile(means, 100*(1-alpha/2)))
    return m, lo, hi


def _paired_permutation_pvalue(deltas, n_perm=5000, seed=123):
    rng = np.random.default_rng(seed)
    d = np.asarray(deltas, dtype=float)
    if len(d) == 0:
        return 1.0
    obs = float(np.mean(d))
    cnt = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(d), replace=True)
        mu = float(np.mean(signs * d))
        if abs(mu) >= abs(obs) - 1e-12:
            cnt += 1
    p = (cnt + 1) / (n_perm + 1)
    return float(p)


def compute_and_save_delta_stats(df, report_dir):
    metrics = ['pr_auc', 'recall_at_fpr_1pct', 'lift_at_5pct', 'f1_macro']
    all_rows = []
    for m in metrics:
        rows = []
        pair = _paired_rows(df, m)
        for g, gdf in pair.groupby('generator'):
            d = gdf['delta'].values
            mean, lo, hi = _bootstrap_ci_array(d)
            pval = _paired_permutation_pvalue(d)
            rows.append({'generator': g, 'metric': m, 'mean_delta': mean, 'ci_lo': lo, 'ci_hi': hi, 'p_value': pval, 'n_pairs': int(len(d))})
            all_rows.append({'generator': g, 'metric': m, 'mean_delta': mean, 'ci_lo': lo, 'ci_hi': hi, 'p_value': pval, 'n_pairs': int(len(d))})
        if rows:
            pd.DataFrame(rows).to_csv(report_dir / f'delta_stats_{m}.csv', index=False)
    if all_rows:
        wide = pd.DataFrame(all_rows)
        wide.to_csv(report_dir / 'delta_stats_all_metrics.csv', index=False)
        print("\nPaired delta statistics (mean ± 95% CI, p-value):")
        for g in wide['generator'].unique():
            sub = wide[wide['generator']==g]
            msg = [f"{g}:"]
            for _, r in sub.iterrows():
                msg.append(f"{r['metric']}={r['mean_delta']:.4f} [{r['ci_lo']:.4f},{r['ci_hi']:.4f}], p={r['p_value']:.4f}")
            print("  ".join(msg))

def main():
    """Main NSL-KDD experiment runner"""
    print("NSL-KDD Network Intrusion Detection - Experiment Runner")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Multi-seed experiments
    seeds = [13, 21, 34, 42, 87]
    base = 'python src/train.py --dataset nsl_kdd --rbf-components 300 --rbf-gamma -1.0 --test-size 0.2'
    experiments = []
    for sd in seeds:
        prefix = f"{base} --seed {sd}"
        experiments.extend([
            { 'cmd': f"{prefix} --augment none", 'desc': f'Baseline (seed={sd})' },
            { 'cmd': f"{prefix} --augment smote", 'desc': f'SMOTE (seed={sd})' },
            { 'cmd': f"{prefix} --augment adasyn", 'desc': f'ADASYN (seed={sd})' },
            { 'cmd': f"{prefix} --augment borderline-smote", 'desc': f'Borderline-SMOTE (seed={sd})' },
            { 'cmd': f"{prefix} --augment svm-smote", 'desc': f'SVM-SMOTE (seed={sd})' },
            { 'cmd': f"{prefix} --augment smote-tomek", 'desc': f'SMOTE-Tomek (seed={sd})' },
            { 'cmd': f"{prefix} --augment smote-enn", 'desc': f'SMOTE-ENN (seed={sd})' },
            { 'cmd': f"{prefix} --augment smote-adv --gen-kind smote --target-ratio 0.3 --keep-frac 0.6 --adv-C 2.0 --w-density 0.2 --adapt-keep", 'desc': f'SMOTE + Adv (seed={sd})' },
            { 'cmd': f"{prefix} --augment smote-adv --gen-kind borderline --target-ratio 0.3 --keep-frac 0.65 --adv-C 2.0 --w-density 0.3 --adapt-keep", 'desc': f'Borderline + Adv (seed={sd})' },
            { 'cmd': f"{prefix} --augment smote-adv --gen-kind svm --target-ratio 0.3 --keep-frac 0.55 --adv-C 2.0 --w-density 0.25 --adapt-keep", 'desc': f'SVM-SMOTE + Adv (seed={sd})' },
            { 'cmd': f"{prefix} --augment smote-adv --gen-kind smote-tomek --target-ratio 0.3 --keep-frac 0.6 --adv-C 2.0 --w-density 0.2 --adapt-keep", 'desc': f'SMOTE-Tomek + Adv (seed={sd})' },
            { 'cmd': f"{prefix} --augment smote-adv --gen-kind smote-enn --target-ratio 0.3 --keep-frac 0.6 --adv-C 2.0 --w-density 0.2 --adapt-keep", 'desc': f'SMOTE-ENN + Adv (seed={sd})' },
        ])
    
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
