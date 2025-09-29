#!/usr/bin/env python3
"""
Santander Customer Transaction Prediction Experiment Runner
Runs baseline experiments for Santander Customer Transaction Prediction dataset
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
    """Collect results from all Santander experiment directories"""
    results = []
    outputs_dir = Path('outputs')
    
    if not outputs_dir.exists():
        print("No outputs directory found")
        return results
    
    for exp_dir in outputs_dir.iterdir():
        if exp_dir.is_dir() and exp_dir.name.startswith('santander_'):
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
    """Create a comprehensive comparison report for Santander experiments"""
    if not results:
        print("No results to compare")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Create comparison report
    report_dir = Path('outputs/santander_comparison_report')
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

    # Compute delta stats with bootstrap CI and permutation-test p-values
    compute_and_save_delta_stats(df, report_dir)
    
    # Print summary
    print(f"\n{'='*80}")
    print("SANTANDER CUSTOMER TRANSACTION PREDICTION EXPERIMENT COMPARISON SUMMARY")
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
    # Use generator to pair base vs adv
    base = base.rename(columns={'augment':'base_augment'})
    # Infer base generator for adv rows
    adv['generator'] = adv.get('generator', None) if 'generator' in adv.columns else None
    merged = None
    rows = []
    for g in adv['generator'].dropna().unique():
        b = base[base['base_augment'].isin([g if g in ['smote','adasyn'] else f"{g}"])]
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
    """Create comparison plots for Santander experiments (polished layout + paired deltas)."""
    plt.style.use('seaborn-whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Santander Customer Transaction Prediction Experiment Comparison Results', fontsize=18, fontweight='bold')

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
        ax4.scatter(subset['rbf_components'], subset['pr_auc'], label=augment, alpha=0.7, s=50)
    ax4.set_xlabel('RFF Components')
    ax4.set_ylabel('PR-AUC')
    ax4.set_title('PR-AUC vs RFF Components')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # PR vs ROC scatter with hue=augment
    ax5 = axes[0, 2]
    sns.scatterplot(data=df, x='roc_auc', y='pr_auc', hue='augment', ax=ax5, s=60, alpha=0.8)
    ax5.set_title('PR-AUC vs ROC-AUC (by method)')
    ax5.set_xlabel('ROC-AUC')
    ax5.set_ylabel('PR-AUC')
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Runtime bars if timings available
    ax6 = axes[1, 2]
    if 'train_s' in df.columns or 'timings' in df.columns:
        # Expand timings per row if nested
        for r in range(len(df)):
            if 'timings' in df.columns and isinstance(df.at[r, 'timings'], dict):
                t = df.at[r, 'timings']
                df.at[r, 'augment_total_s'] = t.get('augment_total_s')
                df.at[r, 'train_s'] = t.get('train_s')
                df.at[r, 'inference_s'] = t.get('inference_s')
        rt = df.groupby('augment').agg(train_s=('train_s','mean'), inference_s=('inference_s','mean'), augment_s=('augment_total_s','mean')).reset_index()
        rt.plot(x='augment', y=['augment_s','train_s','inference_s'], kind='bar', ax=ax6)
        ax6.set_title('Runtime (avg seconds)')
        ax6.set_ylabel('Seconds')
        ax6.tick_params(axis='x', rotation=45)
    else:
        ax6.axis('off')

    # New: Paired Δ plots (PR-AUC, Recall@1%, Lift@5%, F1-Macro)
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

    # New: Win-rate bars by generator (PR-AUC)
    if 'seed' in df.columns:
        adv = df[df['augment']=='smote-adv']
        base = df[df['augment'].isin(['smote','adasyn','svm-smote','borderline-smote','smote-tomek','smote-enn'])]
        wins = []
        for g in adv['generator'].dropna().unique():
            a = adv[adv['generator']==g].set_index('seed')
            # map to base augment name
            b = base[base['augment'].isin([g if g in ['smote','adasyn'] else f"{g}-smote".replace('-smote','-smote')])].set_index('seed')
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
    plt.savefig(report_dir / 'santander_comparison_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Comparison plots saved to: {report_dir / 'santander_comparison_plots.png'}")


def _paired_rows(df, metric):
    adv = df[df['augment']=='smote-adv'].copy()
    base = df[df['augment'].isin(['smote','adasyn','svm-smote','borderline-smote','smote-tomek','smote-enn'])].copy()
    rows = []
    # Pair by generator and seed
    for g in adv['generator'].dropna().unique():
        a = adv[adv['generator']==g]
        # map base augment name to generator
        # expect base augment equals g for smote/adasyn; others already have their name
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
    # Paired permutation: randomly flip signs of deltas
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
        # Print concise summary
        print("\nPaired delta statistics (mean ± 95% CI, p-value):")
        for g in wide['generator'].unique():
            sub = wide[wide['generator']==g]
            msg = [f"{g}:"]
            for _, r in sub.iterrows():
                msg.append(f"{r['metric']}={r['mean_delta']:.4f} [{r['ci_lo']:.4f},{r['ci_hi']:.4f}], p={r['p_value']:.4f}")
            print("  ".join(msg))

def main():
    """Main Santander experiment runner"""
    print("Santander Customer Transaction Prediction - Experiment Runner")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define multi-seed runs
    seeds = [13, 21, 34, 42, 87]
    base = 'python src/train.py --dataset santander --rbf-components 300 --rbf-gamma -1.0 --test-size 0.2'
    experiments = []
    for sd in seeds:
        prefix = f"{base} --seed {sd}"
        experiments.extend([
            { 'cmd': f"{prefix} --augment none", 'desc': f'Baseline: No augmentation (seed={sd})' },
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

# ================== Comprehensive multi-model, multi-method pipeline ==================
# The block below extends this runner to execute an in-process comprehensive experiment
# that evaluates many classifiers across multiple augmentation methods (SMOTE, ADASYN,
# Borderline-SMOTE, SVM-SMOTE, SMOTE-Tomek, SMOTE-ENN, and SMOTE-Adv via RUBRIC),
# generates PR and calibration plots, and writes detailed CSVs and a text report under
# the outputs/ directory.

import os as _os
import sys as _sys
import time as _time
import warnings as _warnings
from pathlib import Path as _Path

def _suppress_sklearn_warnings():
    _warnings.filterwarnings('ignore', category=UserWarning)
    _warnings.filterwarnings('ignore', category=FutureWarning)
    _warnings.filterwarnings('ignore', category=DeprecationWarning)
    _warnings.filterwarnings('ignore', message='.*convergence.*')
    _warnings.filterwarnings('ignore', message='.*max_iter.*')
    _warnings.filterwarnings('ignore', message='.*n_iter.*')
    _warnings.filterwarnings('ignore', message='.*verbose.*')
    _warnings.filterwarnings('ignore', message='.*eval_metric.*')
    _warnings.filterwarnings('ignore', message='.*early_stopping.*')
    _warnings.filterwarnings('ignore', message='.*ensure_all_finite.*')
    _warnings.filterwarnings('ignore', message='.*force_all_finite.*')
    _warnings.filterwarnings('ignore', message='.*check_feature_names.*')

def _ensure_src_on_path():
    root = _Path(__file__).resolve().parent.parent
    src_dir = str(root / 'src')
    if src_dir not in _sys.path:
        _sys.path.append(src_dir)

def _load_santander_data():
    import numpy as _np
    import pandas as _pd
    data_path = _Path('data') / 'santander-customer-transaction-prediction' / 'train.csv'
    if not data_path.exists():
        print(f"Error: {data_path} not found!")
        return None, None, None
    print("Loading Santander data...")
    df = _pd.read_csv(data_path)
    print(f"Data loaded: {df.shape}")
    
    # Extract features and target
    feature_cols = [col for col in df.columns if col.startswith('var_')]
    X = df[feature_cols].values
    y = df['target'].values
    return X, y, df

def _apply_augmentation(X, y, method: str):
    # Import augmentation utilities from src/augment.py
    from augment import (
        smote_oversample,
        adasyn_oversample,
        borderline_smote_oversample,
        svm_smote_oversample,
        kmeans_smote_oversample,
        smote_tomek_resample,
        smote_enn_resample,
        smote_then_rubric_filter,
        generate_then_filter,
    )
    method = (method or 'none').lower()
    if method == 'none':
        print('   No augmentation applied')
        return X, y
    if method == 'smote':
        print('   Applying Standard SMOTE...')
        return smote_oversample(X, y, random_state=42, ratio=1.0)
    if method == 'adasyn':
        print('   Applying ADASYN...')
        return adasyn_oversample(X, y, random_state=42, ratio=1.0)
    if method in ('borderline_smote', 'borderline-smote'):
        print('   Applying Borderline-SMOTE...')
        return borderline_smote_oversample(X, y, random_state=42, ratio=1.0)
    if method in ('svm_smote', 'svm-smote'):
        print('   Applying SVM-SMOTE...')
        return svm_smote_oversample(X, y, random_state=42, ratio=1.0)
    if method in ('kmeans_smote', 'kmeans-smote'):
        print('   Applying KMeans-SMOTE...')
        return kmeans_smote_oversample(X, y, random_state=42, ratio=1.0, cluster_balance_threshold=0.05)
    if method in ('smote_tomek', 'tomek'):
        print('   Applying SMOTE-Tomek...')
        return smote_tomek_resample(X, y, random_state=42, ratio=1.0)
    if method in ('smote_enn', 'enn'):
        print('   Applying SMOTE-ENN...')
        X_res, y_res = smote_enn_resample(X, y, random_state=42, ratio=1.0)
        # Check if SMOTE-ENN created a single-class dataset
        import numpy as __np
        if len(__np.unique(y_res)) < 2:
            print('   WARNING: SMOTE-ENN created single-class dataset, falling back to SMOTE')
            return smote_oversample(X, y, random_state=42, ratio=0.5)
        return X_res, y_res
    if method in ('smote_adv', 'smote-adv', 'rubric', 'smote_rubric', 'none_rubric', 'none-rubric'):
        print('   Applying RUBRIC (SVM-SMOTE + Adversarial Filter, final 50:50)...')
        # Generate to 1:1 via SVM-SMOTE
        X_sm, y_sm = svm_smote_oversample(X, y, random_state=42, ratio=1.0)
        import numpy as __np
        n_orig = len(X)
        minority_label = 1
        X_tail, y_tail = X_sm[n_orig:], y_sm[n_orig:]
        X_synth_min = X_tail[y_tail == minority_label]
        X_real_min = X[y == minority_label]
        X_maj = X[y == 0]
        from augment import rubric_filter_synthetic as __rfs
        X_keep = __rfs(
            X_real_min,
            X_synth_min,
            keep_top_frac=0.65,
            max_iter=100,  # Reduced from 500 to speed up
            C=2.0,
            k_density=11,
            k_majority=11,
            X_majority=X_maj,
            weights=(0.4, 0.4, 0.2),
            gate_frac=0.9,
            return_times=None,
        )
        X_min_all = __np.vstack([X_real_min, X_keep])
        y_min_all = __np.ones(len(X_min_all), dtype=int)
        from numpy.random import default_rng as __rng
        R = __rng(42)
        maj_idx = __np.arange(len(X_maj))
        sel = R.choice(maj_idx, size=min(len(X_maj), len(X_min_all)), replace=False)
        X_maj_sel = X_maj[sel]
        y_maj_sel = __np.zeros(len(X_maj_sel), dtype=int)
        X_new = __np.vstack([X_maj_sel, X_min_all])
        y_new = __np.hstack([y_maj_sel, y_min_all])
        return X_new, y_new
    # Generic RUBRIC add-on for other generators
    if method.endswith('_rubric') or method.endswith('-rubric'):
        base_method = method.replace('_rubric', '').replace('-rubric', '')
        print(f'   Applying RUBRIC ({base_method.upper()} + Adversarial Filter)...')
        
        # First generate synthetic samples using the base method
        if base_method == 'smote':
            X_gen, y_gen = smote_oversample(X, y, random_state=42, ratio=1.0)
        elif base_method == 'adasyn':
            X_gen, y_gen = adasyn_oversample(X, y, random_state=42, ratio=1.0)
        elif base_method in ('borderline_smote', 'borderline-smote'):
            X_gen, y_gen = borderline_smote_oversample(X, y, random_state=42, ratio=1.0)
        elif base_method in ('svm_smote', 'svm-smote'):
            X_gen, y_gen = svm_smote_oversample(X, y, random_state=42, ratio=1.0)
        elif base_method in ('kmeans_smote', 'kmeans-smote'):
            X_gen, y_gen = kmeans_smote_oversample(X, y, random_state=42, ratio=1.0, cluster_balance_threshold=0.05)
        elif base_method in ('smote_tomek', 'tomek'):
            X_gen, y_gen = smote_tomek_resample(X, y, random_state=42, ratio=1.0)
        elif base_method in ('smote_enn', 'enn'):
            X_gen, y_gen = smote_enn_resample(X, y, random_state=42, ratio=1.0)
            # Check if SMOTE-ENN created a single-class dataset
            import numpy as __np
            if len(__np.unique(y_gen)) < 2:
                print('   WARNING: SMOTE-ENN created single-class dataset, falling back to SMOTE')
                X_gen, y_gen = smote_oversample(X, y, random_state=42, ratio=0.5)
        else:
            raise ValueError(f"Unknown base method for RUBRIC: {base_method}")
        
        # Apply RUBRIC filtering
        import numpy as __np
        n_orig = len(X)
        minority_label = 1
        X_tail, y_tail = X_gen[n_orig:], y_gen[n_orig:]
        X_synth_min = X_tail[y_tail == minority_label]
        X_real_min = X[y == minority_label]
        X_maj = X[y == 0]
        
        from augment import rubric_filter_synthetic as __rfs
        X_keep = __rfs(
            X_real_min,
            X_synth_min,
            keep_top_frac=0.65,  # Slightly higher keep fraction for better quality
            max_iter=100,
            C=2.0,
            k_density=11,
            k_majority=11,
            X_majority=X_maj,
            weights=(0.4, 0.4, 0.2),
            gate_frac=0.9,
            return_times=None,
        )
        
        # Combine real minority, filtered synthetic minority, and subsample majority to 50:50
        X_min_all = __np.vstack([X_real_min, X_keep])
        y_min_all = __np.ones(len(X_min_all), dtype=int)
        
        from numpy.random import default_rng as __rng
        R = __rng(42)
        maj_idx = __np.arange(len(X_maj))
        sel = R.choice(maj_idx, size=min(len(X_maj), len(X_min_all)), replace=False)
        X_maj_sel = X_maj[sel]
        y_maj_sel = __np.zeros(len(X_maj_sel), dtype=int)
        
        X_new = __np.vstack([X_maj_sel, X_min_all])
        y_new = __np.hstack([y_maj_sel, y_min_all])
        
        print(f'   Final distribution after RUBRIC: {__np.mean(y_new):.4f}')
        return X_new, y_new
        
    raise ValueError(f"Unknown method: {method}")

def _find_best_threshold(y_true, y_pred_proba):
    import numpy as _np
    from sklearn.metrics import precision_recall_curve as _prc
    precision, recall, thresholds = _prc(y_true, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = int(_np.argmax(f1_scores))
    return float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

def _train_and_eval_models(X_train, y_train, X_test, y_test, models: dict):
    import numpy as _np
    from sklearn.preprocessing import MinMaxScaler as _MinMaxScaler
    from sklearn.metrics import (
        precision_score as _precision_score,
        recall_score as _recall_score,
        f1_score as _f1_score,
        roc_auc_score as _roc_auc_score,
        average_precision_score as _average_precision_score,
        brier_score_loss as _brier_score_loss,
        accuracy_score as _accuracy_score,
        balanced_accuracy_score as _balanced_accuracy_score,
    )
    from sklearn.calibration import CalibratedClassifierCV as _CalibratedClassifierCV

    # Check if we have both classes in training data
    unique_train = _np.unique(y_train)
    if len(unique_train) < 2:
        print(f"   WARNING: Training data only contains class {unique_train[0]}. Skipping model training.")
        return {}

    results = {}
    for name, model in models.items():
        print(f"   Training {name}...")
        t0 = _time.time()
        Xtr_used, Xte_used = X_train, X_test
        needs_nonneg = False
        try:
            from sklearn.naive_bayes import ComplementNB as _CNB
            needs_nonneg = isinstance(model, _CNB)
        except Exception:
            needs_nonneg = False
        if needs_nonneg:
            mm = _MinMaxScaler()
            Xtr_used = mm.fit_transform(X_train)
            Xte_used = mm.transform(X_test)
        
        try:
            model.fit(Xtr_used, y_train)
            train_time = _time.time() - t0

            calibrated_model = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_pred_proba = model.predict_proba(Xte_used)
                    # Check if we have probabilities for both classes
                    if y_pred_proba.shape[1] > 1:
                        y_pred_proba = y_pred_proba[:, 1]
                        train_proba = model.predict_proba(Xtr_used)[:, 1]
                    else:
                        # Single class prediction - use the class probability
                        y_pred_proba = y_pred_proba[:, 0]
                        train_proba = model.predict_proba(Xtr_used)[:, 0]
                        print(f"     WARNING: Model only predicts single class")
                except Exception as e:
                    # Try calibration
                    try:
                        calibrated_model = _CalibratedClassifierCV(model, method='sigmoid', cv=3)
                        calibrated_model.fit(Xtr_used, y_train)
                        y_pred_proba = calibrated_model.predict_proba(Xte_used)[:, 1]
                        train_proba = calibrated_model.predict_proba(Xtr_used)[:, 1]
                    except Exception as e2:
                        print(f"     ERROR: Failed to get probabilities: {e2}")
                        continue
            else:
                try:
                    calibrated_model = _CalibratedClassifierCV(model, method='sigmoid', cv=3)
                    calibrated_model.fit(Xtr_used, y_train)
                    y_pred_proba = calibrated_model.predict_proba(Xte_used)[:, 1]
                    train_proba = calibrated_model.predict_proba(Xtr_used)[:, 1]
                except Exception as e:
                    print(f"     ERROR: Failed to calibrate model: {e}")
                    continue

            best_threshold = _find_best_threshold(y_train, train_proba)
            y_pred = (y_pred_proba >= best_threshold).astype(int)

            # Ensure predictions contain both classes for metrics calculation
            if len(_np.unique(y_pred)) == 1:
                print(f"     WARNING: Model predicts only class {_np.unique(y_pred)[0]}")
                precision = 0.0 if _np.unique(y_pred)[0] == 0 else _np.mean(y_test == 1)
                recall = 0.0 if _np.unique(y_pred)[0] == 0 else 1.0
                f1 = 0.0
            else:
                precision = _precision_score(y_test, y_pred)
                recall = _recall_score(y_test, y_pred)
                f1 = _f1_score(y_test, y_pred)
            
            auc = _roc_auc_score(y_test, y_pred_proba)
            ap = _average_precision_score(y_test, y_pred_proba)
            brier = _brier_score_loss(y_test, y_pred_proba)
            acc = _accuracy_score(y_test, y_pred)
            bacc = _balanced_accuracy_score(y_test, y_pred)

            results[name] = {
                'model': model,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'auc': float(auc),
                'ap': float(ap),
                'brier': float(brier),
                'accuracy': float(acc),
                'balanced_accuracy': float(bacc),
                'train_time': float(train_time),
                'best_threshold': float(best_threshold),
                'y_true': y_test,
            }
            print(f"     Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}, Acc: {acc:.4f}, BAcc: {bacc:.4f}, Threshold: {best_threshold:.3f}")
        except Exception as e:
            print(f"     ERROR: Failed to train {name}: {e}")
            continue
    
    return results

def _plot_precision_recall_curves(results_dict: dict, dataset_name: str, output_dir: str):
    import os as __os
    import matplotlib.pyplot as __plt
    from sklearn.metrics import precision_recall_curve as __prc
    __plt.figure(figsize=(12, 8))
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'gray', 'teal']
    for i, (method, results) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        for model_name, model_results in results.items():
            precision, recall, _ = __prc(model_results['y_true'], model_results['y_pred_proba'])
            ap = model_results['ap']
            __plt.plot(recall, precision, color=color, alpha=0.7, label=f"{method.upper()} - {model_name} (AP={ap:.3f})")
    __plt.xlabel('Recall')
    __plt.ylabel('Precision')
    __plt.title(f'Precision-Recall Curves - {dataset_name}')
    __plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    __plt.grid(True, alpha=0.3)
    __plt.tight_layout()
    __plt.savefig(_os.path.join(output_dir, f"pr_curves_{dataset_name.lower().replace(' ', '_')}.png"), dpi=300, bbox_inches='tight')
    __plt.close()

def _plot_calibration_curves(results_dict: dict, dataset_name: str, output_dir: str):
    import os as __os
    import matplotlib.pyplot as __plt
    from sklearn.calibration import calibration_curve as __cal_curve
    fig, axes = __plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    for i, (method, results) in enumerate(results_dict.items()):
        if i >= 4:
            break
        ax = axes[i]
        for model_name, model_results in results.items():
            fraction_of_positives, mean_predicted_value = __cal_curve(model_results['y_true'], model_results['y_pred_proba'], n_bins=10)
            ax.plot(mean_predicted_value, fraction_of_positives, 's-', label=f"{model_name} (Brier={model_results['brier']:.3f})")
        ax.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'Calibration Plot - {method.upper()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    __plt.suptitle(f'Calibration Curves - {dataset_name}', fontsize=16)
    __plt.tight_layout()
    __plt.savefig(_os.path.join(output_dir, f"calibration_curves_{dataset_name.lower().replace(' ', '_')}.png"), dpi=300, bbox_inches='tight')
    __plt.close()

def _save_detailed_results(results_dict: dict, dataset_name: str, output_dir: str):
    import os as __os
    import pandas as __pd
    detailed_dir = _Path(output_dir) / 'detailed_results'
    detailed_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, method_results in results_dict.items():
        for model_name, model_results in method_results.items():
            rows.append({
                'Dataset': dataset_name,
                'Method': method.upper(),
                'Model': model_name,
                'F1_Score': model_results.get('f1'),
                'Precision': model_results.get('precision'),
                'Recall': model_results.get('recall'),
                'AUC': model_results.get('auc'),
                'Average_Precision': model_results.get('ap'),
                'Brier_Score': model_results.get('brier'),
                'Accuracy': model_results.get('accuracy'),
                'Balanced_Accuracy': model_results.get('balanced_accuracy'),
                'Best_Threshold': model_results.get('best_threshold'),
                'Train_Time': model_results.get('train_time'),
                'Test_Samples': len(model_results.get('y_true', [])),
                'Timestamp': _time.strftime('%Y-%m-%d %H:%M:%S'),
            })
    df = __pd.DataFrame(rows)
    csv_path = detailed_dir / f"{dataset_name.lower().replace(' ', '_')}_detailed_results.csv"
    df.to_csv(csv_path, index=False)
    # Summary by method
    sum_rows = []
    for method, method_results in results_dict.items():
        metrics = ['f1', 'precision', 'recall', 'auc', 'ap', 'brier', 'accuracy', 'balanced_accuracy']
        agg = {}
        for m in metrics:
            vals = [r.get(m) for r in method_results.values()]
            vals = [v for v in vals if v is not None]
            if len(vals) == 0:
                agg[f'Avg_{m}'] = None
                agg[f'Std_{m}'] = None
                agg[f'Max_{m}'] = None
                agg[f'Min_{m}'] = None
            else:
                import numpy as __np
                agg[f'Avg_{m}'] = float(__np.nanmean(vals))
                agg[f'Std_{m}'] = float(__np.nanstd(vals))
                agg[f'Max_{m}'] = float(__np.nanmax(vals))
                agg[f'Min_{m}'] = float(__np.nanmin(vals))
        sum_rows.append({'Dataset': dataset_name, 'Method': method.upper(), 'Models_Tested': len(method_results), **agg, 'Timestamp': _time.strftime('%Y-%m-%d %H:%M:%S')})
    summary_df = __pd.DataFrame(sum_rows)
    summary_path = detailed_dir / f"{dataset_name.lower().replace(' ', '_')}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Detailed results saved to: {csv_path}")
    print(f"Summary statistics saved to: {summary_path}")
    return df, summary_df

def _generate_text_report(results_dict: dict, dataset_name: str, output_dir: str):
    report_path = _Path(output_dir) / f"comprehensive_report_{dataset_name.lower().replace(' ', '_')}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Comprehensive Test Report - {dataset_name}\n")
        f.write("=" * 60 + "\n\n")
        methods = list(results_dict.keys())
        f.write("Model Performance Comparison:\n")
        f.write("-" * 30 + "\n")
        model_names = sorted({mn for res in results_dict.values() for mn in res.keys()})
        for model_name in model_names:
            f.write(f"\n{model_name} Results:\n")
            f.write("Method".ljust(12) + "Recall".ljust(10) + "Precision".ljust(12) + "F1".ljust(8) + "AUC".ljust(8) + "AP".ljust(8) + "Brier".ljust(8) + "Acc".ljust(10) + "BAcc".ljust(8) + "\n")
            f.write("-" * 80 + "\n")
            for method in methods:
                if model_name in results_dict[method]:
                    r = results_dict[method][model_name]
                    f.write(f"{method.upper().ljust(12)}{r['recall']:.4f}".ljust(22) + f"{r['precision']:.4f}".ljust(12) + f"{r['f1']:.4f}".ljust(8) + f"{r['auc']:.4f}".ljust(8) + f"{r['ap']:.4f}".ljust(8) + f"{r['brier']:.4f}".ljust(8) + f"{r['accuracy']:.4f}".ljust(10) + f"{r['balanced_accuracy']:.4f}".ljust(8) + "\n")
    print(f"Report saved to: {report_path}")
    return str(report_path)

def _create_comparison_tables(results_dict: dict, dataset_name: str, output_dir: str):
    import pandas as __pd
    comp_dir = _Path(output_dir) / 'comprehensive_comparison'
    comp_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, method_results in results_dict.items():
        for model_name, r in method_results.items():
            rows.append({
                'Dataset': dataset_name,
                'Method': method.upper(),
                'Model': model_name,
                'F1_Score': r.get('f1'),
                'Precision': r.get('precision'),
                'Recall': r.get('recall'),
                'AUC': r.get('auc'),
                'Average_Precision': r.get('ap'),
                'Brier_Score': r.get('brier'),
                'Accuracy': r.get('accuracy'),
                'Balanced_Accuracy': r.get('balanced_accuracy'),
                'Best_Threshold': r.get('best_threshold'),
                'Train_Time_Seconds': r.get('train_time'),
                'Test_Samples': len(r.get('y_true', [])),
                'Method_Model_Combination': f"{method.upper()}_{model_name}",
                'Timestamp': _time.strftime('%Y-%m-%d %H:%M:%S'),
            })
    df = __pd.DataFrame(rows)
    comp_path = comp_dir / f"{dataset_name.lower().replace(' ', '_')}_comprehensive_comparison.csv"
    df.to_csv(comp_path, index=False)
    # pivots
    for metric in ['F1_Score', 'Precision', 'Recall', 'AUC']:
        pv = df.pivot_table(index='Model', columns='Method', values=metric, aggfunc='mean')
        pv.to_csv(comp_dir / f"{dataset_name.lower().replace(' ', '_')}_{metric.lower()}_pivot.csv")
    print(f"Comprehensive comparison table saved to: {comp_path}")
    return df

def _save_augmented_santander(X_aug, y_aug, method: str):
    """Save augmented Santander dataset snapshot under data/augmented/santander.
    Uses original feature names from santander train.csv (excluding ID_code and target).
    """
    import pandas as __pd
    from pathlib import Path as __Path
    data_path = __Path('data') / 'santander-customer-transaction-prediction' / 'train.csv'
    if not data_path.exists():
        # Fallback generic feature names
        feat_cols = [f'var_{i}' for i in range(X_aug.shape[1])]
    else:
        df_src = __pd.read_csv(data_path, nrows=1)
        feat_cols = [c for c in df_src.columns if c.startswith('var_')]
        if len(feat_cols) != X_aug.shape[1]:
            feat_cols = [f'var_{i}' for i in range(X_aug.shape[1])]
    out_dir = __Path('data') / 'augmented' / 'santander'
    out_dir.mkdir(parents=True, exist_ok=True)
    df = __pd.DataFrame(X_aug, columns=feat_cols)
    df['target'] = y_aug
    fp = out_dir / f"santander_{method}_augmented.csv"
    df.to_csv(fp, index=False)
    print(f"   Augmented data saved to: {fp}")

def run_comprehensive_santander_experiments():
    from sklearn.model_selection import train_test_split as _tts
    from sklearn.preprocessing import StandardScaler as _StandardScaler
    # Optional models
    try:
        import lightgbm as _lgb
        _HAS_LGB = True
    except Exception:
        _HAS_LGB = False
    try:
        import xgboost as _xgb
        _HAS_XGB = True
    except Exception:
        _HAS_XGB = False
    # Optional imbalanced-learn ensembles
    try:
        from imblearn.ensemble import BalancedRandomForestClassifier as _BRF
        _HAS_IMB = True
    except Exception:
        _HAS_IMB = False

    _suppress_sklearn_warnings()
    _ensure_src_on_path()

    print("RUBRIC comparison (Santander Customer Transaction Prediction)")
    out_dir = 'outputs/santander-customer-transaction-prediction'
    _Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    X, y, df = _load_santander_data()
    if X is None:
        return

    # Split
    import numpy as _np
    idx = _np.arange(len(X))
    tr_idx, te_idx = _tts(idx, test_size=0.2, random_state=42, stratify=y)
    X_train, X_test = X[tr_idx], X[te_idx]
    y_train, y_test = y[tr_idx], y[te_idx]

    scaler = _StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    print(f"Training set: {X_train_s.shape}, Test set: {X_test_s.shape}")
    print(f"Class distribution - Train: {_np.mean(y_train):.4f}, Test: {_np.mean(y_test):.4f}")

    # Models
    from sklearn.linear_model import LogisticRegression as _LR, SGDClassifier as _SGD, PassiveAggressiveClassifier as _PA, RidgeClassifier as _Ridge
    from sklearn.svm import LinearSVC as _LinearSVC
    from sklearn.neighbors import KNeighborsClassifier as _KNN, NearestCentroid as _NC
    from sklearn.naive_bayes import GaussianNB as _GNB, BernoulliNB as _BNB, ComplementNB as _CNB
    from sklearn.tree import DecisionTreeClassifier as _DT
    from sklearn.ensemble import RandomForestClassifier as _RF, ExtraTreesClassifier as _ET, BaggingClassifier as _Bag
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as _LDA, QuadraticDiscriminantAnalysis as _QDA
    from sklearn.neural_network import MLPClassifier as _MLP
    from sklearn.ensemble import HistGradientBoostingClassifier as _HGB

    models = {
        'RandomForest': _RF(n_estimators=100, random_state=42, n_jobs=-1),
        'LogisticRegression': _LR(max_iter=1000, class_weight='balanced', random_state=42, solver='liblinear'),
        'LinearSVC': _LinearSVC(C=1.0, class_weight='balanced', random_state=42, max_iter=1000),
        'GaussianNB': _GNB(),
        'DecisionTree': _DT(random_state=42, class_weight='balanced'),
        'HistGradientBoosting': _HGB(random_state=42, max_iter=100, learning_rate=0.1),
    }
    if _HAS_LGB:
        import lightgbm as _lgb
        models['LightGBM'] = _lgb.LGBMClassifier(random_state=42, verbose=-1, num_leaves=31, learning_rate=0.1, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=5, min_data_in_leaf=20)
    if _HAS_XGB:
        import xgboost as _xgb
        models['XGBoost'] = _xgb.XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0, tree_method='hist', enable_categorical=False)
    if _HAS_IMB:
        from imblearn.ensemble import BalancedRandomForestClassifier as _BRF
        models['BalancedRandomForest'] = _BRF(n_estimators=300, random_state=42, n_jobs=-1)

    env_models = _os.environ.get('EDA_MODELS')
    if env_models:
        selected = [m.strip() for m in env_models.split(',') if m.strip()]
        models = {k: v for k, v in models.items() if k in selected}

    methods = [
        'none',
        'none_rubric',
        'smote', 
        'adasyn', 
        'borderline_smote', 
        'svm_smote', 
        'kmeans_smote',
        'smote_tomek',
        'smote_enn',
        'smote_rubric',
        'adasyn_rubric',
        'borderline_smote_rubric',
        'svm_smote_rubric',
        'kmeans_smote_rubric',
        'smote_tomek_rubric',
        'smote_enn_rubric',
    ]
    env_methods = _os.environ.get('EDA_METHODS')
    if env_methods:
        selected_methods = [m.strip() for m in env_methods.split(',') if m.strip()]
        methods = [m for m in methods if m in selected_methods]
    
    # Quick mode for faster testing - disable by default
    if _os.environ.get('QUICK_MODE', '').lower() == 'true':
        print("Running in QUICK MODE - testing only basic methods")
        methods = ['none', 'smote', 'svm_smote', 'smote_rubric']
    else:
        print(f"Running full experiment with {len(methods)} augmentation methods")

    all_results = {}
    total_methods = len(methods)
    for idx, method in enumerate(methods, 1):
        print(f"\n{'-'*40}\nTesting {method.upper()} method ({idx}/{total_methods})\n{'-'*40}")
        t0 = _time.time()
        if method in ('none',):
            X_aug, y_aug = X_train_s, y_train
        else:
            try:
                X_aug, y_aug = _apply_augmentation(X_train_s, y_train, method)
            except Exception as e:
                print(f"   [skip] Augmentation {method} failed: {e}")
                print("   Skipping this method and continuing...")
                continue
        print(f"   Augmentation completed in {_time.time()-t0:.2f}s")
        print(f"   Augmented shape: {X_aug.shape}")
        import numpy as __np
        print(f"   New minority ratio: {__np.mean(y_aug):.4f}")

        # Save augmented snapshot for reproducibility
        try:
            _save_augmented_santander(X_aug, y_aug, method)
        except Exception as e:
            print(f"   [warn] Failed to save augmented dataset for {method}: {e}")

        # Train and evaluate
        method_results = _train_and_eval_models(X_aug, y_aug, X_test_s, y_test, models)
        all_results[method] = method_results

    dataset_name = 'Santander Customer Transaction Prediction'
    print(f"\nGenerating plots and report for {dataset_name}...")
    _plot_precision_recall_curves(all_results, dataset_name, out_dir)
    _plot_calibration_curves(all_results, dataset_name, out_dir)
    _generate_text_report(all_results, dataset_name, out_dir)
    _save_detailed_results(all_results, dataset_name, out_dir)
    _create_comparison_tables(all_results, dataset_name, out_dir)
    print(f"[OK] {dataset_name} comprehensive testing completed! Results in: {out_dir}")


def main():
    # Run the comprehensive in-process pipeline instead of delegating to src/train.py
    run_comprehensive_santander_experiments()


if __name__ == '__main__':
    main()
