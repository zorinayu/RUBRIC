#!/usr/bin/env python3
"""
Comprehensive experiments for IEEE-CIS Fraud Detection dataset
 - Multiple augmentation methods (SMOTE family + RUBRIC)
 - Many classifiers with calibration and threshold tuning
 - Saves plots, CSVs, and an augmentation pre/post comparison doc
"""

import os
import sys
import time
import warnings
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    balanced_accuracy_score,
)


def _suppress_warnings():
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', message='.*convergence.*')
    warnings.filterwarnings('ignore', message='.*max_iter.*')
    warnings.filterwarnings('ignore', message='.*n_iter.*')
    warnings.filterwarnings('ignore', message='.*verbose.*')
    warnings.filterwarnings('ignore', message='.*eval_metric.*')
    warnings.filterwarnings('ignore', message='.*early_stopping.*')
    warnings.filterwarnings('ignore', message='.*ensure_all_finite.*')
    warnings.filterwarnings('ignore', message='.*force_all_finite.*')
    warnings.filterwarnings('ignore', message='.*check_feature_names.*')


def _ensure_src_path():
    root = Path(__file__).resolve().parent.parent
    src_dir = str(root / 'src')
    if src_dir not in sys.path:
        sys.path.append(src_dir)


def load_ieee_data():
    """
    Load IEEE-CIS data from data/ieee-fraud-detection/
    - Uses train_transaction.csv (required)
    - Optionally merges train_identity.csv on TransactionID
    - Keeps numeric features only; fills missing with median
    Returns: (X, y, df_numeric, feature_names)
    """
    data_dir = Path('data') / 'ieee-fraud-detection'
    trans = data_dir / 'train_transaction.csv'
    ident = data_dir / 'train_identity.csv'

    if not trans.exists():
        print(f"Error: {trans} not found. Please download from Kaggle.")
        return None, None, None, None

    print('Loading IEEE-CIS Fraud Detection dataset...')
    df_tr = pd.read_csv(trans)
    if ident.exists():
        try:
            df_id = pd.read_csv(ident)
            if 'TransactionID' in df_tr.columns and 'TransactionID' in df_id.columns:
                df_tr = df_tr.merge(df_id, on='TransactionID', how='left')
        except Exception as e:
            print(f"Warning: failed to merge identity file: {e}")

    if 'isFraud' not in df_tr.columns:
        print('Error: isFraud column not found in transaction file')
        return None, None, None, None

    # Keep numeric features
    df_num = df_tr.select_dtypes(include=[np.number]).copy()
    # Ensure label is present
    if 'isFraud' not in df_num.columns:
        df_num['isFraud'] = df_tr['isFraud'].astype(int)
    # Fill missing with median
    med = df_num.median(numeric_only=True)
    df_num = df_num.fillna(med)

    feature_cols = [c for c in df_num.columns if c != 'isFraud']
    X = df_num[feature_cols].values
    y = df_num['isFraud'].astype(int).values

    print(f"Dataset shape: {X.shape}, positive ratio: {np.mean(y):.4f}")
    return X, y, df_num, feature_cols


def _predict_proba_chunked(model, X, batch_size=20000):
    """Memory-safe predict_proba with chunking. Returns probabilities for class 1.
    Falls back to decision_function min-max scaled if predict_proba fails.
    """
    import numpy as _np
    n = len(X)
    out = _np.empty(n, dtype=float)
    # Try predict_proba in chunks
    if hasattr(model, 'predict_proba'):
        try:
            for i in range(0, n, batch_size):
                sl = slice(i, min(n, i + batch_size))
                out[sl] = model.predict_proba(X[sl])[:, 1]
            return _np.clip(out, 0.0, 1.0)
        except Exception:
            pass
    # Try decision_function in chunks
    if hasattr(model, 'decision_function'):
        vals = _np.empty(n, dtype=float)
        for i in range(0, n, batch_size):
            sl = slice(i, min(n, i + batch_size))
            vals[sl] = model.decision_function(X[sl])
        vmin, vmax = float(_np.min(vals)), float(_np.max(vals))
        rng = vmax - vmin if vmax > vmin else 1e-9
        out = (vals - vmin) / rng
        return _np.clip(out, 0.0, 1.0)
    # Last resort: predict labels as probs
    preds = model.predict(X)
    return _np.clip(preds.astype(float), 0.0, 1.0)


def apply_augmentation(X, y, method: str):
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
    m = (method or 'none').lower()
    if m == 'none':
        print('   No augmentation applied')
        return X, y
    if m == 'smote':
        print('   Applying SMOTE...')
        return smote_oversample(X, y, random_state=42, ratio=1.0)
    if m == 'adasyn':
        print('   Applying ADASYN...')
        return adasyn_oversample(X, y, random_state=42, ratio=1.0)
    if m in ('borderline_smote', 'borderline-smote'):
        print('   Applying Borderline-SMOTE...')
        return borderline_smote_oversample(X, y, random_state=42, ratio=1.0)
    if m in ('svm_smote', 'svm-smote'):
        print('   Applying SVM-SMOTE...')
        return svm_smote_oversample(X, y, random_state=42, ratio=1.0)
    if m in ('kmeans_smote', 'kmeans-smote'):
        print('   Applying KMeans-SMOTE...')
        # Slightly looser balance threshold for robustness on highly imbalanced data
        return kmeans_smote_oversample(X, y, random_state=42, ratio=1.0, cluster_balance_threshold=0.05)
    if m in ('smote_tomek', 'tomek'):
        print('   Applying SMOTE-Tomek...')
        return smote_tomek_resample(X, y, random_state=42, ratio=1.0)
    if m in ('smote_enn', 'enn'):
        print('   Applying SMOTE-ENN...')
        return smote_enn_resample(X, y, random_state=42, ratio=1.0)
    if m in ('smote_adv', 'smote-adv', 'rubric', 'smote_rubric'):
        print('   Applying SMOTE-Adv (RUBRIC)...')
        return smote_then_rubric_filter(
            X, y,
            random_state=42,
            ratio=1.0,
            keep_top_frac=0.65,
            C=2.0,
            k_density=11,
            k_majority=11,
            weights=(0.4, 0.4, 0.2),
            gate_frac=0.9,
        )
    # Generic RUBRIC add-on for other generators
    if m.endswith('_rubric') or m.endswith('-rubric'):
        base = m.replace('-rubric', '').replace('_rubric', '')
        if base in ('none',):
            print('   NONE+RUBRIC: no generator; returning original dataset')
            return X, y
        gen_map = {
            'adasyn': 'adasyn',
            'borderline_smote': 'borderline',
            'svm_smote': 'svm',
            'kmeans_smote': 'kmeans',
            'smote_tomek': 'smote-tomek',
            'smote_enn': 'smote-enn',
            'smote': 'smote',
        }
        generator = gen_map.get(base)
        if generator is None:
            raise ValueError(f"Unknown RUBRIC base generator: {base}")
        print(f"   Applying {generator.upper()} + RUBRIC...")
        return generate_then_filter(
            X,
            y,
            generator=generator,
            ratio=1.0,
            random_state=42,
            k_neighbors=5,
            keep_top_frac=0.65,
            C=2.0,
            k_density=11,
            k_majority=11,
            weights=(0.4, 0.4, 0.2),
            gate_frac=0.9,
            return_times=None,
        )
    raise ValueError(f"Unknown method: {method}")


def find_best_threshold(y_true, y_pred_proba):
    p, r, thr = precision_recall_curve(y_true, y_pred_proba)
    f1 = 2 * (p * r) / (p + r + 1e-8)
    bi = int(np.argmax(f1))
    return float(thr[bi]) if bi < len(thr) else 0.5


def train_and_evaluate_models(X_train, y_train, X_test, y_test, models: dict):
    results = {}
    for name, model in models.items():
        print(f"   Training {name}...")
        t0 = time.time()
        Xtr_used, Xte_used = X_train, X_test
        y_train_used, y_test_used = y_train, y_test
        needs_nonneg = False
        try:
            from sklearn.naive_bayes import ComplementNB as _CNB
            needs_nonneg = isinstance(model, _CNB)
        except Exception:
            needs_nonneg = False
        if needs_nonneg:
            mm = MinMaxScaler()
            Xtr_used = mm.fit_transform(X_train)
            Xte_used = mm.transform(X_test)

        # Extra downsampling for very slow models (e.g., KNN) to keep runtime <~1min
        try:
            from sklearn.neighbors import KNeighborsClassifier as _KNN
            if isinstance(model, _KNN):
                from sklearn.model_selection import train_test_split as _tts
                max_knn_train = min(5000, len(Xtr_used))
                max_knn_test = min(20000, len(Xte_used))
                Xtr_used, _, y_train_used, _ = _tts(Xtr_used, y_train_used, train_size=max_knn_train, stratify=y_train_used, random_state=42)
                Xte_used, _, y_test_used, _ = _tts(Xte_used, y_test_used, train_size=max_knn_test, stratify=y_test_used, random_state=42)
        except Exception:
            pass

        try:
            model.fit(Xtr_used, y_train_used)
            train_time = time.time() - t0

            # Fast probability scoring (memory-safe)
            train_proba = _predict_proba_chunked(model, Xtr_used, batch_size=20000)
            y_pred_proba = _predict_proba_chunked(model, Xte_used, batch_size=20000)

            thr = find_best_threshold(y_train_used, train_proba)
            y_pred = (y_pred_proba >= thr).astype(int)

            precision = precision_score(y_test_used, y_pred)
            recall = recall_score(y_test_used, y_pred)
            f1 = f1_score(y_test_used, y_pred)
            auc = roc_auc_score(y_test_used, y_pred_proba)
            ap = average_precision_score(y_test_used, y_pred_proba)
            brier = brier_score_loss(y_test_used, y_pred_proba)
            acc = accuracy_score(y_test_used, y_pred)
            bacc = balanced_accuracy_score(y_test_used, y_pred)

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
                'best_threshold': float(thr),
                'y_true': y_test_used,
                'model_status': 'ok',
                'model_error': None,
            }
            print(f"     Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}, Acc: {acc:.4f}, BAcc: {bacc:.4f}, Threshold: {thr:.3f}")
        except Exception as me:
            err_msg = f"{type(me).__name__}: {me}"
            print(f"     [model-failed] {name}: {err_msg}")
            results[name] = {
                'model': None,
                'y_pred': None,
                'y_pred_proba': None,
                'precision': None,
                'recall': None,
                'f1': None,
                'auc': None,
                'ap': None,
                'brier': None,
                'accuracy': None,
                'balanced_accuracy': None,
                'train_time': float(time.time() - t0),
                'best_threshold': None,
                'y_true': y_test_used,
                'model_status': 'failed',
                'model_error': err_msg,
            }
    return results


def plot_precision_recall_curves(results_dict, dataset_name, output_dir):
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'gray', 'teal']
    for i, (method, results) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        for model_name, model_results in results.items():
            precision, recall, _ = precision_recall_curve(
                model_results['y_true'], model_results['y_pred_proba']
            )
            ap = model_results['ap']
            plt.plot(recall, precision, color=color, alpha=0.7,
                     label=f"{method.upper()} - {model_name} (AP={ap:.3f})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curves - {dataset_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(output_dir) / f"pr_curves_{dataset_name.lower().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_calibration_curves(results_dict, dataset_name, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    for i, (method, results) in enumerate(results_dict.items()):
        if i >= 4:
            break
        ax = axes[i]
        for model_name, model_results in results.items():
            fraction_of_positives, mean_predicted_value = calibration_curve(
                model_results['y_true'], model_results['y_pred_proba'], n_bins=10
            )
            ax.plot(mean_predicted_value, fraction_of_positives, 's-',
                    label=f"{model_name} (Brier={model_results['brier']:.3f})")
        ax.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'Calibration Plot - {method.upper()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle(f'Calibration Curves - {dataset_name}', fontsize=16)
    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(output_dir) / f"calibration_curves_{dataset_name.lower().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close()


def save_detailed_results_to_csv(results_dict, dataset_name, output_dir, method_info_map=None):
    detailed_dir = Path(output_dir) / 'detailed_results'
    detailed_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, method_results in results_dict.items():
        minfo = (method_info_map or {}).get(method, {})
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
                'Train_Time': r.get('train_time'),
                'Test_Samples': len(r.get('y_true', [])),
                'Model_Status': r.get('model_status'),
                'Model_Error': r.get('model_error'),
                'Augmentation_Failed': bool(minfo.get('failed', False)),
                'Augmentation_Error': minfo.get('error'),
                'Augmented_Shape': minfo.get('shape'),
                'Augmented_Pos_Ratio': minfo.get('pos_ratio'),
                'Aug_Time_Sec': minfo.get('aug_time'),
                'Augmented_CSV_Path': minfo.get('csv_path'),
                'Timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            })
    df = pd.DataFrame(rows)
    csv_path = detailed_dir / f"{dataset_name.lower().replace(' ', '_')}_detailed_results.csv"
    df.to_csv(csv_path, index=False)

    # Summary per method
    sum_rows = []
    for method, method_results in results_dict.items():
        metrics = ['f1', 'precision', 'recall', 'auc', 'ap', 'brier', 'accuracy', 'balanced_accuracy']
        agg = {}
        vals_dict = {m: [r.get(m) for r in method_results.values()] for m in metrics}
        for m, vals in vals_dict.items():
            v = [x for x in vals if x is not None]
            if len(v) == 0:
                agg[f'Avg_{m}'] = None
                agg[f'Std_{m}'] = None
                agg[f'Max_{m}'] = None
                agg[f'Min_{m}'] = None
            else:
                agg[f'Avg_{m}'] = float(np.nanmean(v))
                agg[f'Std_{m}'] = float(np.nanstd(v))
                agg[f'Max_{m}'] = float(np.nanmax(v))
                agg[f'Min_{m}'] = float(np.nanmin(v))
        sum_rows.append({'Dataset': dataset_name, 'Method': method.upper(), 'Models_Tested': len(method_results), **agg, 'Timestamp': time.strftime('%Y-%m-%d %H:%M:%S')})
    summary_df = pd.DataFrame(sum_rows)
    summary_path = detailed_dir / f"{dataset_name.lower().replace(' ', '_')}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Detailed results saved to: {csv_path}")
    print(f"Summary statistics saved to: {summary_path}")
    return df, summary_df


def create_comparison_table(results_dict, dataset_name, output_dir, method_info_map=None):
    comp_dir = Path(output_dir) / 'comprehensive_comparison'
    comp_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, method_results in results_dict.items():
        minfo = (method_info_map or {}).get(method, {})
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
                'Model_Status': r.get('model_status'),
                'Model_Error': r.get('model_error'),
                'Augmentation_Failed': bool(minfo.get('failed', False)),
                'Augmentation_Error': minfo.get('error'),
                'Augmented_Shape': minfo.get('shape'),
                'Augmented_Pos_Ratio': minfo.get('pos_ratio'),
                'Aug_Time_Sec': minfo.get('aug_time'),
                'Augmented_CSV_Path': minfo.get('csv_path'),
                'Timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            })
    df = pd.DataFrame(rows)
    comp_path = comp_dir / f"{dataset_name.lower().replace(' ', '_')}_comprehensive_comparison.csv"
    df.to_csv(comp_path, index=False)
    for metric in ['F1_Score', 'Precision', 'Recall', 'AUC']:
        pv = df.pivot_table(index='Model', columns='Method', values=metric, aggfunc='mean')
        pv.to_csv(comp_dir / f"{dataset_name.lower().replace(' ', '_')}_{metric.lower()}_pivot.csv")
    print(f"Comprehensive comparison table saved to: {comp_path}")
    return df


def save_augmented_dataset(X_aug, y_aug, feature_names, method):
    out_dir = Path('data') / 'augmented' / 'ieee_fraud_detection'
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(X_aug, columns=feature_names)
    df['isFraud'] = y_aug
    fname = f"ieee_fraud_detection_{method}_augmented.csv"
    fpath = out_dir / fname
    df.to_csv(fpath, index=False)
    print(f"   Augmented data saved: {fpath} | shape={X_aug.shape} | pos_ratio={np.mean(y_aug):.4f}")
    return str(fpath)


def write_augmentation_comparison_doc(baseline_results, all_results, y_train, doc_path):
    lines = []
    lines.append('IEEE-CIS Fraud Detection - Augmentation Pre/Post Comparison')
    lines.append('=' * 70)
    lines.append('Baseline (NONE) vs augmented methods. Metrics on test set.')
    lines.append('')
    # Baseline per model
    base = baseline_results
    lines.append('Baseline (NONE) metrics per model:')
    for mn, r in base.items():
        lines.append(f"- {mn}: F1={r['f1']:.4f}, P={r['precision']:.4f}, R={r['recall']:.4f}, AUC={r['auc']:.4f}, AP={r['ap']:.4f}, Brier={r['brier']:.4f}")
    lines.append('')
    # Deltas vs baseline
    for method, res in all_results.items():
        if method == 'none':
            continue
        lines.append(f"Method: {method.upper()}")
        for mn, r in res.items():
            if mn in base:
                lines.append(
                    f"  {mn}: ΔF1={r['f1']-base[mn]['f1']:.4f}, ΔAP={r['ap']-base[mn]['ap']:.4f}, ΔRecall={r['recall']-base[mn]['recall']:.4f}, ΔPrecision={r['precision']-base[mn]['precision']:.4f}"
                )
        lines.append('')
    Path(doc_path).parent.mkdir(parents=True, exist_ok=True)
    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Augmentation comparison doc written to: {doc_path}")


def main():
    _suppress_warnings()
    _ensure_src_path()

    # Output directory for IEEE
    outputs_dir = Path('outputs') / 'ieee_fraud_detection'
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X, y, df_num, feature_names = load_ieee_data()
    if X is None:
        return

    # Split and scale
    idx = np.arange(len(X))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    X_train, X_test = X[tr_idx], X[te_idx]
    y_train, y_test = y[tr_idx], y[te_idx]

    # FAST mode: subsample train/test for speed
    FAST_MODE = os.environ.get('EDA_FAST', '1') != '0'
    if FAST_MODE:
        max_train = int(os.environ.get('IEEE_MAX_TRAIN', '50000'))
        max_test = int(os.environ.get('IEEE_MAX_TEST', '50000'))
        if len(X_train) > max_train:
            X_train, _, y_train, _ = train_test_split(
                X_train, y_train, train_size=max_train, stratify=y_train, random_state=42
            )
            print(f"FAST mode: subsampled training to {len(X_train)} rows")
        if len(X_test) > max_test:
            X_test, _, y_test, _ = train_test_split(
                X_test, y_test, train_size=max_test, stratify=y_test, random_state=42
            )
            print(f"FAST mode: subsampled test to {len(X_test)} rows")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    print(f"Training set: {X_train_s.shape}, Test set: {X_test_s.shape}")
    print(f"Training positive ratio: {np.mean(y_train):.4f}")

    # Models
    try:
        import lightgbm as lgb
        HAS_LGB = True
    except Exception:
        HAS_LGB = False
    try:
        import xgboost as xgb
        HAS_XGB = True
    except Exception:
        HAS_XGB = False
    try:
        from imblearn.ensemble import BalancedRandomForestClassifier as BRF, BalancedBaggingClassifier as BB
        HAS_IMB = True
    except Exception:
        HAS_IMB = False

    from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier
    from sklearn.svm import LinearSVC
    from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
    from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, HistGradientBoostingClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.neural_network import MLPClassifier

    # All models; in FAST_MODE we tune hyperparams to be quicker
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100 if FAST_MODE else 300, random_state=42, n_jobs=-1),
        'LogisticRegression': LogisticRegression(max_iter=500 if FAST_MODE else 2000, class_weight='balanced', random_state=42, solver=('saga' if FAST_MODE else 'liblinear'), n_jobs=-1 if FAST_MODE else None),
        'LinearSVC': LinearSVC(C=1.0, class_weight='balanced', random_state=42, max_iter=1000 if FAST_MODE else 2000),
        'SGD_Logistic': SGDClassifier(loss='log_loss', class_weight='balanced', random_state=42, max_iter=1000 if FAST_MODE else 2000),
        'KNN_15': KNeighborsClassifier(n_neighbors=15, weights='distance', n_jobs=-1),
        'GaussianNB': GaussianNB(),
        'BernoulliNB': BernoulliNB(),
        'ComplementNB': ComplementNB(),
        'DecisionTree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=200 if FAST_MODE else 300, random_state=42, n_jobs=-1, class_weight='balanced'),
        'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42, max_iter=100 if FAST_MODE else 200, learning_rate=0.1),
        'Bagging': BaggingClassifier(
            estimator=DecisionTreeClassifier(class_weight='balanced', random_state=42),
            n_estimators=25 if FAST_MODE else 100,
            max_samples=0.5,
            max_features=0.5,
            n_jobs=1,
            random_state=42,
        ),
        'LDA': LinearDiscriminantAnalysis(),
        'QDA': QuadraticDiscriminantAnalysis(reg_param=0.1 if FAST_MODE else 0.0),
        'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=100 if FAST_MODE else 200, random_state=42, early_stopping=True, validation_fraction=0.1),
        'PassiveAggressive': PassiveAggressiveClassifier(random_state=42, class_weight='balanced', max_iter=1000 if FAST_MODE else 2000),
        'RidgeClassifier': RidgeClassifier(random_state=42, class_weight='balanced', alpha=1.0),
        'NearestCentroid': NearestCentroid(),
    }
    if HAS_LGB:
        models['LightGBM'] = lgb.LGBMClassifier(random_state=42, verbose=-1, num_leaves=31, learning_rate=0.1, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=5, min_data_in_leaf=20, n_estimators=200 if FAST_MODE else 300)
    if HAS_XGB:
        models['XGBoost'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0, tree_method='hist', enable_categorical=False, n_estimators=200 if FAST_MODE else 400, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8)

    env_models = os.environ.get('EDA_MODELS')
    if env_models:
        selected = [m.strip() for m in env_models.split(',') if m.strip()]
        models = {k: v for k, v in models.items() if k in selected}

    methods = [
        'none', 'none_rubric',
        'smote', 'smote_rubric',
        'adasyn', 'adasyn_rubric',
        'borderline_smote', 'borderline_smote_rubric',
        'svm_smote', 'svm_smote_rubric',
        'kmeans_smote', 'kmeans_smote_rubric',
        'smote_tomek', 'smote_tomek_rubric',
        'smote_enn', 'smote_enn_rubric',
    ]
    env_methods = os.environ.get('EDA_METHODS')
    if env_methods:
        selected_methods = [m.strip() for m in env_methods.split(',') if m.strip()]
        methods = [m for m in methods if m in selected_methods]

    all_results = {}
    method_info_map = {}
    for method in methods:
        print(f"\n{'-'*40}\nTesting {method.upper()} method\n{'-'*40}")
        t0 = time.time()
        aug_failed = False
        aug_error = None
        csv_path = None
        try:
            if method == 'none':
                X_aug, y_aug = X_train_s, y_train
            else:
                X_aug, y_aug = apply_augmentation(X_train_s, y_train, method)
        except Exception as e:
            aug_failed = True
            aug_error = f"{type(e).__name__}: {e}"
            print(f"   [augmentation-failed] {method}: {aug_error}")
            # Fallback: use original training set so we still evaluate models and record failure
            X_aug, y_aug = X_train_s, y_train
        aug_time = time.time()-t0
        print(f"   Augmentation completed in {aug_time:.2f}s")
        print(f"   Augmented shape: {X_aug.shape}")
        print(f"   New positive ratio: {np.mean(y_aug):.4f}")

        # Save augmented dataset snapshot when augmentation succeeded or method is 'none'
        try:
            csv_path = save_augmented_dataset(X_aug, y_aug, feature_names, method)
        except Exception:
            csv_path = None

        # Record method-level info
        method_info_map[method] = {
            'failed': aug_failed,
            'error': aug_error,
            'aug_time': float(aug_time),
            'shape': tuple(X_aug.shape) if hasattr(X_aug, 'shape') else None,
            'pos_ratio': float(np.mean(y_aug)) if y_aug is not None else None,
            'csv_path': csv_path,
        }

        # Train and evaluate
        method_results = train_and_evaluate_models(X_aug, y_aug, X_test_s, y_test, models)
        all_results[method] = method_results

    dataset_name = 'IEEE Fraud Detection'
    print(f"\nGenerating plots and report for {dataset_name}...")
    plot_precision_recall_curves(all_results, dataset_name, outputs_dir)
    plot_calibration_curves(all_results, dataset_name, outputs_dir)
    save_detailed_results_to_csv(all_results, dataset_name, outputs_dir, method_info_map)
    create_comparison_table(all_results, dataset_name, outputs_dir, method_info_map)

    # Write augmentation pre/post comparison document
    if 'none' in all_results:
        doc_path = Path('data') / 'ieee-fraud-detection' / 'augmentation_comparison.txt'
        write_augmentation_comparison_doc(all_results['none'], all_results, y_train, doc_path)

    print(f"[OK] IEEE-CIS comprehensive testing completed! Results in: {outputs_dir}")


if __name__ == '__main__':
    main()


