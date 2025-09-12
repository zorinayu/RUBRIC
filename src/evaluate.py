from __future__ import annotations
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter errors
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    classification_report, confusion_matrix, precision_recall_curve, roc_curve
)

def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)

def _recall_at_fpr(y_true, scores, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    # Interpolate TPR at or below target FPR (use the max TPR where FPR <= target)
    mask = fpr <= target_fpr
    if not np.any(mask):
        return 0.0
    return float(np.max(tpr[mask]))


def _lift_at_k(y_true, scores, k_frac: float = 0.05) -> float:
    n = len(scores)
    if n == 0:
        return 0.0
    k = max(1, int(n * k_frac))
    order = np.argsort(-np.asarray(scores))
    top_k = order[:k]
    precision_at_k = float(np.mean(np.asarray(y_true)[top_k] == 1))
    base_rate = float(np.mean(y_true))
    if base_rate <= 0:
        return 0.0
    return float(precision_at_k / (base_rate + 1e-12))


def evaluate_and_plot(y_true, scores, y_pred, out_dir='outputs'):
    ensure_dir(out_dir)
    ensure_dir(Path(out_dir) / 'plots')

    metrics = {}
    # Scores: decision_function outputs (can be any real). For ROC/PR we need positive scores for class 1
    roc_auc = roc_auc_score(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)
    f1w = f1_score(y_true, y_pred, average='weighted')
    f1m = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, digits=4, output_dict=True)

    # Low-FPR and lift metrics
    recall_fpr_1pct = _recall_at_fpr(y_true, scores, target_fpr=0.01)
    recall_fpr_0_5pct = _recall_at_fpr(y_true, scores, target_fpr=0.005)
    lift_at_5pct = _lift_at_k(y_true, scores, k_frac=0.05)

    metrics.update({
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'f1_weighted': float(f1w),
        'f1_macro': float(f1m),
        'recall_at_fpr_1pct': float(recall_fpr_1pct),
        'recall_at_fpr_0_5pct': float(recall_fpr_0_5pct),
        'lift_at_5pct': float(lift_at_5pct),
        'confusion_matrix': cm,
        'classification_report': report
    })

    # Save metrics
    with open(Path(out_dir)/'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, scores)
    plt.figure()
    plt.plot(recall, precision, label=f'PR-AUC={pr_auc:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, ls='--', alpha=0.4)
    plt.savefig(Path(out_dir)/'plots/pr_curve.png', dpi=180, bbox_inches='tight')
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC-AUC={roc_auc:.3f}')
    plt.plot([0,1],[0,1],'--', alpha=0.5)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, ls='--', alpha=0.4)
    plt.savefig(Path(out_dir)/'plots/roc_curve.png', dpi=180, bbox_inches='tight')
    plt.close()

    return metrics
