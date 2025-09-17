from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN
from typing import Tuple
import time

def smote_oversample(X: np.ndarray, y: np.ndarray, random_state=42, k_neighbors=5, ratio=1.0):
    """
    Standard SMOTE oversampling.
    
    Args:
        X: Feature matrix
        y: Labels
        random_state: Random seed
        k_neighbors: Number of nearest neighbors for SMOTE
        ratio: Sampling ratio (1.0 = balance classes)
        
    Returns:
        Tuple of (augmented_X, augmented_y)
    """
    print(f"   Generating SMOTE samples...")
    sm = SMOTE(sampling_strategy=ratio, k_neighbors=k_neighbors, random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res

def adasyn_oversample(X: np.ndarray, y: np.ndarray, random_state=42, n_neighbors=5, ratio=1.0):
    """
    ADASYN oversampling using imbalanced-learn.
    """
    print(f"   Generating ADASYN samples...")
    ada = ADASYN(sampling_strategy=ratio, n_neighbors=n_neighbors, random_state=random_state)
    X_res, y_res = ada.fit_resample(X, y)
    return X_res, y_res

def borderline_smote_oversample(X: np.ndarray, y: np.ndarray, random_state=42, k_neighbors=5, ratio=1.0, kind="borderline-1"):
    """
    Borderline-SMOTE oversampling using imbalanced-learn.
    kind: "borderline-1" or "borderline-2"
    """
    print(f"   Generating Borderline-SMOTE ({kind}) samples...")
    bsm = BorderlineSMOTE(sampling_strategy=ratio, k_neighbors=k_neighbors, random_state=random_state, kind=kind)
    X_res, y_res = bsm.fit_resample(X, y)
    return X_res, y_res

def svm_smote_oversample(X: np.ndarray, y: np.ndarray, random_state=42, k_neighbors=5, ratio=1.0, m_neighbors=10):
    """
    SVM-SMOTE oversampling using imbalanced-learn.
    """
    print(f"   Generating SVM-SMOTE samples...")
    svm_sm = SVMSMOTE(sampling_strategy=ratio, k_neighbors=k_neighbors, m_neighbors=m_neighbors, random_state=random_state)
    X_res, y_res = svm_sm.fit_resample(X, y)
    return X_res, y_res

def kmeans_smote_oversample(X: np.ndarray, y: np.ndarray, random_state=42, k_neighbors=5, ratio=1.0, kmeans_estimator=None, cluster_balance_threshold=0.1):
    """
    KMeans-SMOTE oversampling using imbalanced-learn.
    """
    print(f"   Generating KMeans-SMOTE samples...")
    kms = KMeansSMOTE(
        sampling_strategy=ratio,
        k_neighbors=k_neighbors,
        random_state=random_state,
        kmeans_estimator=kmeans_estimator,
        cluster_balance_threshold=cluster_balance_threshold
    )
    X_res, y_res = kms.fit_resample(X, y)
    return X_res, y_res

def smote_tomek_resample(X: np.ndarray, y: np.ndarray, random_state=42, ratio=1.0, k_neighbors=5):
    """
    SMOTE + Tomek links cleaning combined resampling.
    """
    print(f"   Generating SMOTE-Tomek samples...")
    st = SMOTETomek(sampling_strategy=ratio, random_state=random_state, smote=SMOTE(k_neighbors=k_neighbors, random_state=random_state))
    X_res, y_res = st.fit_resample(X, y)
    return X_res, y_res

def smote_enn_resample(X: np.ndarray, y: np.ndarray, random_state=42, ratio=1.0, k_neighbors=5):
    """
    SMOTE + ENN cleaning combined resampling.
    """
    print(f"   Generating SMOTE-ENN samples...")
    se = SMOTEENN(sampling_strategy=ratio, random_state=random_state, smote=SMOTE(k_neighbors=k_neighbors, random_state=random_state))
    X_res, y_res = se.fit_resample(X, y)
    return X_res, y_res

# -------------------- Generator-agnostic factory + Adv filter wrapper --------------------

def make_oversampler(kind: str, ratio: float, random_state: int, k_neighbors: int = 5):
    kind_l = (kind or 'smote').lower()
    if kind_l == 'smote':
        return SMOTE(sampling_strategy=ratio, k_neighbors=k_neighbors, random_state=random_state)
    elif kind_l in ('borderline', 'borderline-smote', 'bsmote'):
        return BorderlineSMOTE(sampling_strategy=ratio, k_neighbors=k_neighbors, random_state=random_state, kind='borderline-1')
    elif kind_l in ('borderline2', 'borderline-2'):
        return BorderlineSMOTE(sampling_strategy=ratio, k_neighbors=k_neighbors, random_state=random_state, kind='borderline-2')
    elif kind_l in ('svm', 'svm-smote'):
        return SVMSMOTE(sampling_strategy=ratio, k_neighbors=k_neighbors, m_neighbors=10, random_state=random_state)
    elif kind_l in ('kmeans', 'kmeans-smote'):
        return KMeansSMOTE(sampling_strategy=ratio, k_neighbors=k_neighbors, random_state=random_state)
    elif kind_l in ('smote-tomek', 'tomek'):
        return SMOTETomek(sampling_strategy=ratio, random_state=random_state, smote=SMOTE(k_neighbors=k_neighbors, random_state=random_state))
    elif kind_l in ('smote-enn', 'enn'):
        return SMOTEENN(sampling_strategy=ratio, random_state=random_state, smote=SMOTE(k_neighbors=k_neighbors, random_state=random_state))
    elif kind_l in ('adasyn',):
        return ADASYN(sampling_strategy=ratio, n_neighbors=k_neighbors, random_state=random_state)
    else:
        raise ValueError(f"Unknown generator: {kind}")

def generate_then_filter(
    X: np.ndarray,
    y: np.ndarray,
    generator: str = 'smote',
    ratio: float = 0.3,
    random_state: int = 42,
    k_neighbors: int = 5,
    keep_top_frac: float = 0.65,
    max_iter: int = 500,
    C: float = 1.0,
    k_density: int = 10,
    k_majority: int = 10,
    weights: tuple[float, float, float] = (0.6, 0.3, 0.1),
    gate_frac: float = 0.9,
    return_times: dict | None = None,
):
    print(f"   Generating {generator.upper()} samples...")
    t0 = time.time()
    osr = make_oversampler(generator, ratio, random_state, k_neighbors=k_neighbors)
    X_res, y_res = osr.fit_resample(X, y)
    t_gen = time.time() - t0

    minority_label = 1
    X_real_min = X[y == minority_label]
    X_maj = X[y == 0]

    n_orig = len(X)
    X_tail, y_tail = X_res[n_orig:], y_res[n_orig:]
    X_synth_min = X_tail[y_tail == minority_label]

    X_keep = adversarial_filter_synthetic(
        X_real_min,
        X_synth_min,
        keep_top_frac=keep_top_frac,
        max_iter=max_iter,
        C=C,
        k_density=k_density,
        k_majority=k_majority,
        X_majority=X_maj,
        weights=weights,
        gate_frac=gate_frac,
        return_times=return_times,
    )

    X_new = np.vstack([X_maj, X_real_min, X_keep])
    y_new = np.hstack([
        np.zeros(len(X_maj), dtype=int),
        np.ones(len(X_real_min), dtype=int),
        np.ones(len(X_keep), dtype=int)
    ])

    if return_times is not None:
        return_times['generator_total_s'] = float(t_gen)

    return X_new, y_new

from sklearn.neighbors import NearestNeighbors

# Optional: XGBoost for proxy model if available
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

def adversarial_filter_synthetic(
    X_real_min,
    X_synth_min,
    keep_top_frac=0.7,
    max_iter=500,
    C=1.0,
    k_density: int = 10,
    k_majority: int = 10,
    X_majority: np.ndarray | None = None,
    weights: tuple[float, float, float] = (0.6, 0.3, 0.1),
    gate_frac: float = 0.9,
    return_times: dict | None = None,
):
    """
    SMOTE-Adv: Adversarial filtering of synthetic samples.
    
    This method trains a logistic regression discriminator to distinguish between
    real and synthetic minority samples, then keeps the synthetic samples that
    are hardest to distinguish (closest to decision boundary).
    
    Args:
        X_real_min: Real minority samples
        X_synth_min: Synthetic minority samples
        keep_top_frac: Fraction of synthetic samples to keep
        max_iter: Maximum iterations for logistic regression
        C: Regularization strength for logistic regression
        
    Returns:
        Filtered synthetic samples
    """
    print(f"   Training adversarial discriminator...")
    
    # Prepare training data for discriminator
    X_train = np.vstack([X_real_min, X_synth_min])
    y_train = np.hstack([np.ones(len(X_real_min)), np.zeros(len(X_synth_min))])
    
    # Train discriminator
    t0 = time.time()
    clf = LogisticRegression(max_iter=max_iter, C=C, n_jobs=None, random_state=42)
    clf.fit(X_train, y_train)
    t_lr = time.time() - t0
    
    # Get discriminator scores for synthetic samples
    synth_scores = clf.predict_proba(X_synth_min)[:, 1]
    
    # Calculate closeness to decision boundary (0.5)
    # Samples closer to 0.5 are harder to distinguish and thus more realistic
    # Realism score (1 near boundary, 0 far)
    realism = 1.0 - np.minimum(1.0, np.abs(synth_scores - 0.5) * 2.0)

    # Optional sparse gate by realism (keep top gate_frac by realism)
    n_gate = max(1, int(len(realism) * gate_frac))
    gate_idx = np.argsort(-realism)[:n_gate]

    # Density score among minority (higher is denser)
    t1 = time.time()
    nn_min = NearestNeighbors(n_neighbors=min(k_density, max(1, len(X_real_min)-1))).fit(X_real_min)
    dists_min, _ = nn_min.kneighbors(X_synth_min[gate_idx], return_distance=True)
    # exclude the first neighbor if it can be self; here synth vs real -> no need
    dens = 1.0 / (np.mean(dists_min, axis=1) + 1e-8)
    # Normalize to 0..1
    dens = (dens - dens.min()) / (dens.max() - dens.min() + 1e-12)

    # Majority proximity penalty (higher when closer to majority)
    if X_majority is not None and len(X_majority) > 0:
        nn_maj = NearestNeighbors(n_neighbors=min(k_majority, len(X_majority))).fit(X_majority)
        dists_maj, _ = nn_maj.kneighbors(X_synth_min[gate_idx], return_distance=True)
        maj_close = 1.0 / (np.mean(dists_maj, axis=1) + 1e-8)
        maj_close = (maj_close - maj_close.min()) / (maj_close.max() - maj_close.min() + 1e-12)
    else:
        maj_close = np.zeros_like(dens)
    t_knn = time.time() - t1

    # Combine scores
    w_realism, w_density, w_majority = weights
    realism_sel = realism[gate_idx]
    # Normalize realism over the gated set
    realism_sel = (realism_sel - realism_sel.min()) / (realism_sel.max() - realism_sel.min() + 1e-12)
    combined = w_realism * realism_sel + w_density * dens - w_majority * maj_close

    # Keep top fraction by combined score
    keep_n = max(1, int(len(gate_idx) * keep_top_frac))
    sel_order = np.argsort(-combined)
    keep_gate_idx = sel_order[:keep_n]
    keep_idx = gate_idx[keep_gate_idx]

    if return_times is not None:
        return_times['filter_lr_train_s'] = float(t_lr)
        return_times['filter_knn_s'] = float(t_knn)
        return_times['filter_select_s'] = float(max(0.0, time.time() - t0 - t_lr - t_knn))
        return_times['filter_total_s'] = float(time.time() - t0)

    print(f"   Kept {len(keep_idx)}/{len(X_synth_min)} synthetic samples after SMOTE-ADV filtering")
    return X_synth_min[keep_idx]

# -------------------- PR-AUC-aligned ADV selection (soft/hard) --------------------

import numpy as np

def _knn_density(X_minority: np.ndarray, k: int = 11) -> np.ndarray:
    k = min(k, max(2, len(X_minority) - 1))
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(X_minority)
    dists, _ = nbrs.kneighbors(X_minority)
    if dists.shape[1] > 1 and np.allclose(dists[:, 0], 0):
        dists = dists[:, 1:]
    inv_mean = 1.0 / (1e-9 + dists.mean(axis=1))
    return inv_mean

def _rank_norm(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(v), dtype=float)
    return ranks / max(1, len(v) - 1)

def _minmax(v: np.ndarray) -> np.ndarray:
    v = v.astype(float)
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi <= lo + 1e-12:
        return np.zeros_like(v, dtype=float)
    return (v - lo) / (hi - lo)

def _normalize(v: np.ndarray, mode: str) -> np.ndarray:
    if mode == "rank":
        return _rank_norm(v)
    if mode == "minmax":
        return _minmax(v)
    return v.astype(float)

def _train_calibrated_discriminator(X_real_min: np.ndarray, X_synth_min: np.ndarray, C: float = 2.0):
    X = np.vstack([X_real_min, X_synth_min])
    y = np.hstack([np.ones(len(X_real_min), dtype=int), np.zeros(len(X_synth_min), dtype=int)])
    clf = LogisticRegression(C=C, penalty="l2", solver="liblinear", max_iter=200)
    cal = CalibratedClassifierCV(clf, method="sigmoid", cv=3)
    cal.fit(X, y)
    return cal

def _proxy_model(name: str, random_state: int = 42):
    if name == "xgb" and _HAS_XGB:
        return XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            eval_metric="aucpr",
            random_state=random_state,
            n_jobs=-1,
        )
    return LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=300)

def _proxy_pr_auc(X_tr, y_tr, X_val, y_val, sample_weight=None, proxy_name="logreg", random_state=42) -> float:
    proxy = _proxy_model(proxy_name, random_state)
    try:
        proxy.fit(X_tr, y_tr, sample_weight=sample_weight)
    except TypeError:
        proxy.fit(X_tr, y_tr)
    try:
        p = proxy.predict_proba(X_val)[:, 1]
    except Exception:
        s = proxy.decision_function(X_val)
        p = _minmax(s)
    return average_precision_score(y_val, p)

def _utility_score(boundary_model, X_synth_min: np.ndarray, norm: str = "rank") -> np.ndarray:
    try:
        p = boundary_model.predict_proba(X_synth_min)[:, 1]
    except Exception:
        s = boundary_model.decision_function(X_synth_min)
        p = _minmax(s)
    u = 1.0 - np.abs(p - 0.5) * 2.0
    return _normalize(u, norm)

def _fit_boundary_model_for_utility(X_train, y_train):
    return LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=300).fit(X_train, y_train)

def _decile_bins(values: np.ndarray, n_bins: int = 10):
    qs = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    qs = np.unique(qs)
    bins = np.digitize(values, qs[1:-1], right=True)
    return bins, len(qs) - 1

def adv_select_synthetics(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_min_real: np.ndarray,
    X_min_synth: np.ndarray,
    args,
    rng=np.random,
):
    disc = _train_calibrated_discriminator(
        X_min_real, X_min_synth, C=args.adv_C if hasattr(args, "adv_C") else 2.0
    )
    R = disc.predict_proba(X_min_synth)[:, 1]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=getattr(args, 'adv_val_frac', 0.2), stratify=y_train, random_state=42
    )
    boundary = _fit_boundary_model_for_utility(X_tr, y_tr)
    U = _utility_score(boundary, X_min_synth, norm=getattr(args, 'adv_norm', 'rank'))

    dens_real = _knn_density(X_min_real, k=getattr(args, 'adv_density_k', 11))
    nn = NearestNeighbors(n_neighbors=1).fit(X_min_real)
    _, idx = nn.kneighbors(X_min_synth, n_neighbors=1)
    D = dens_real[idx[:, 0]]

    w_density_final = args.w_density2 if getattr(args, 'w_density2', None) is not None else getattr(args, 'w_density', 0.3)
    Rn = _normalize(R, getattr(args, 'adv_norm', 'rank'))
    Un = _normalize(U, getattr(args, 'adv_norm', 'rank'))
    Dn = _normalize(D, getattr(args, 'adv_norm', 'rank'))

    alpha = getattr(args, 'w_realism', 0.4)
    beta = getattr(args, 'w_utility', 0.4)
    gamma = w_density_final
    S = alpha * Rn + beta * Un + gamma * Dn

    dec, _ = _decile_bins(Dn, n_bins=10)
    keep_floor = getattr(args, 'adv_min_keep_per_decile', 5)

    if getattr(args, 'adv_mode', 'soft') == 'soft':
        w = _normalize(S, getattr(args, 'adv_norm', 'rank'))
        temps = [0.5, 1.0, 2.0]
        best_ap, best_w = -1.0, w
        for t in temps:
            wt = np.power(w, 1.0 / t)
            ap = _proxy_pr_auc(X_tr, y_tr, X_val, y_val, sample_weight=None, proxy_name=getattr(args, 'adv_proxy', 'logreg'))
            if ap > best_ap:
                best_ap, best_w = ap, wt
        w_final = best_w.copy()
        for b in np.unique(dec):
            idx_b = np.where(dec == b)[0]
            if len(idx_b) == 0:
                continue
            if (w_final[idx_b] > 1e-8).sum() < keep_floor:
                topk = np.argsort(w_final[idx_b])[::-1][:min(keep_floor, len(idx_b))]
                w_final[idx_b[topk]] = np.maximum(w_final[idx_b[topk]], 1e-3)
        return w_final

    # hard mode
    if getattr(args, 'adv_thresh_mode', 'val-pr-auc') == 'quantile':
        q = float(getattr(args, 'adv_keep_quantile', 0.6))
        thr = np.quantile(S, 1.0 - q)
        keep = S >= thr
    else:
        quantiles = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        best_ap, best_keep = -1.0, None
        for q in quantiles:
            thr = np.quantile(S, 1.0 - q)
            mask = S >= thr
            for b in np.unique(dec):
                idx_b = np.where(dec == b)[0]
                m_b = np.where(mask[idx_b])[0]
                if len(m_b) < keep_floor and len(idx_b) > 0:
                    topk = np.argsort(S[idx_b])[::-1][:min(keep_floor, len(idx_b))]
                    mask[idx_b[topk]] = True
                max_cap = getattr(args, 'adv_max_keep_per_decile', 1000000)
                if mask[idx_b].sum() > max_cap:
                    topk = np.argsort(S[idx_b])[::-1][:max_cap]
                    tmp = np.zeros_like(mask[idx_b], dtype=bool)
                    tmp[topk] = True
                    mask[idx_b] = tmp
            X_aug = np.vstack([X_tr, X_min_synth[mask]])
            y_aug = np.hstack([y_tr, np.ones(mask.sum(), dtype=int)])
            ap = _proxy_pr_auc(X_aug, y_aug, X_val, y_val, sample_weight=None, proxy_name=getattr(args, 'adv_proxy', 'logreg'))
            if ap > best_ap:
                best_ap, best_keep = ap, mask
        keep = best_keep

    return keep.astype(bool)

def smote_then_adversarial_filter(
    X,
    y,
    random_state=42,
    ratio=1.0,
    keep_top_frac=0.65,
    max_iter=500,
    C=1.0,
    k_density: int = 10,
    k_majority: int = 10,
    weights: tuple[float, float, float] = (0.6, 0.3, 0.1),
    gate_frac: float = 0.9,
    return_times: dict | None = None,
):
    """
    SMOTE-Adv: SMOTE followed by adversarial filtering.
    
    This is the main SMOTE-Adv implementation that:
    1. Generates synthetic samples using SMOTE
    2. Filters synthetic samples using adversarial discriminator
    3. Combines real majority, real minority, and filtered synthetic minority
    
    Args:
        X: Feature matrix
        y: Labels
        random_state: Random seed
        ratio: Sampling ratio (1.0 = balance classes)
        keep_top_frac: Fraction of synthetic samples to keep after filtering
        max_iter: Maximum iterations for discriminator
        C: Regularization strength for discriminator
        
    Returns:
        Tuple of (augmented_X, augmented_y)
    """
    start_time = time.time()
    print("   Starting SMOTE-Adv...")
    
    # Step 1: Generate SMOTE samples
    t0 = time.time()
    X_sm, y_sm = smote_oversample(X, y, random_state=random_state, ratio=ratio)
    t_smote = time.time() - t0
    
    # Step 2: Extract minority samples from original and synthetic sets
    minority_label = 1
    X_real_min = X[y == minority_label]
    X_maj = X[y == 0]
    # Assume SMOTE appends synthetic samples after originals
    n_orig = len(X)
    X_tail = X_sm[n_orig:]
    y_tail = y_sm[n_orig:]
    X_synth_min = X_tail[y_tail == minority_label]
    
    # Step 3: Apply adversarial filtering
    X_synth_keep = adversarial_filter_synthetic(
        X_real_min,
        X_synth_min,
        keep_top_frac=keep_top_frac,
        max_iter=max_iter,
        C=C,
        k_density=k_density,
        k_majority=k_majority,
        X_majority=X_maj,
        weights=weights,
        gate_frac=gate_frac,
        return_times=return_times,
    )
    
    # Step 4: Rebuild dataset
    X_maj = X[y == 0]
    y_maj = np.zeros(len(X_maj), dtype=int)
    y_min_real = np.ones(len(X_real_min), dtype=int)
    y_min_syn = np.ones(len(X_synth_keep), dtype=int)
    
    X_new = np.vstack([X_maj, X_real_min, X_synth_keep])
    y_new = np.hstack([y_maj, y_min_real, y_min_syn])
    
    elapsed_time = time.time() - start_time
    if return_times is not None:
        return_times['smote_generate_s'] = float(t_smote)
        return_times['adv_total_s'] = float(elapsed_time)
    print(f"   SMOTE-ADV completed in {elapsed_time:.2f}s")
    print(f"   Final class distribution: majority={np.sum(y_new==0)}, minority={np.sum(y_new==1)}")
    
    return X_new, y_new

def analyze_smote_adv_performance(X_original, y_original, X_smote_adv, y_smote_adv, 
                                 X_smote, y_smote):
    """
    Analyze SMOTE-Adv performance compared to original SMOTE.
    
    Args:
        X_original: Original feature matrix
        y_original: Original labels
        X_smote_adv: SMOTE-Adv augmented features
        y_smote_adv: SMOTE-Adv augmented labels
        X_smote: Standard SMOTE augmented features
        y_smote: Standard SMOTE augmented labels
    """
    print("\n" + "="*60)
    print("SMOTE-Adv Performance Analysis Report")
    print("="*60)
    
    # Data distribution analysis
    print("\nData Distribution Comparison:")
    print(f"   Original: majority={np.sum(y_original==0)}, minority={np.sum(y_original==1)}")
    print(f"   After SMOTE: majority={np.sum(y_smote==0)}, minority={np.sum(y_smote==1)}")
    print(f"   After SMOTE-Adv: majority={np.sum(y_smote_adv==0)}, minority={np.sum(y_smote_adv==1)}")
    
    # Filtering effect analysis
    original_minority = np.sum(y_original == 1)
    smote_minority = np.sum(y_smote == 1)
    smote_adv_minority = np.sum(y_smote_adv == 1)
    
    smote_generated = smote_minority - original_minority
    smote_adv_generated = smote_adv_minority - original_minority
    filtering_ratio = smote_adv_generated / smote_generated if smote_generated > 0 else 0
    
    print(f"\nFiltering Effect:")
    print(f"   SMOTE generated: {smote_generated}")
    print(f"   SMOTE-Adv kept: {smote_adv_generated}")
    print(f"   Filtering ratio: {(1-filtering_ratio)*100:.1f}%")
    
    # Class balance evaluation
    balance_ratio_original = np.sum(y_original == 1) / np.sum(y_original == 0)
    balance_ratio_smote = np.sum(y_smote == 1) / np.sum(y_smote == 0)
    balance_ratio_smote_adv = np.sum(y_smote_adv == 1) / np.sum(y_smote_adv == 0)
    
    print(f"\nClass Balance Ratios:")
    print(f"   Original minority ratio: {balance_ratio_original*100:.3f}%")
    print(f"   After SMOTE: {balance_ratio_smote*100:.3f}%")
    print(f"   After SMOTE-Adv: {balance_ratio_smote_adv*100:.3f}%")
    
    print("\n" + "="*60)
    print("Analysis Complete")
    print("="*60)


def quick_smoke_test(X, y, rng=42):
    """
    Train/val/test split once, run 4 conditions quickly:
    None, SMOTE, SMOTE+ADV(hard), SMOTE+ADV(soft)
    Print PR-AUC deltas to sanity-check selection logic quickly.
    """
    from sklearn.model_selection import train_test_split as _tts
    from imblearn.over_sampling import SMOTE as _SMOTE

    X_tr, X_te, y_tr, y_te = _tts(X, y, test_size=0.2, stratify=y, random_state=rng)
    base = _proxy_model("logreg", rng).fit(X_tr, y_tr)
    p_base = base.predict_proba(X_te)[:, 1]
    ap_base = average_precision_score(y_te, p_base)

    Xs, ys = _SMOTE(random_state=rng, k_neighbors=5).fit_resample(X_tr, y_tr)
    base_sm = _proxy_model("logreg", rng).fit(Xs, ys)
    ap_sm = average_precision_score(y_te, base_sm.predict_proba(X_te)[:, 1])

    y_tr_arr = np.asarray(y_tr)
    X_min_real = X_tr[y_tr_arr == 1]
    Xs2, ys2 = _SMOTE(random_state=rng, k_neighbors=5).fit_resample(X_tr, y_tr)
    synth_idx = np.where((ys2 == 1))[0][len(np.where(y_tr == 1)[0]):]
    X_min_synth = Xs2[synth_idx]

    class _Args: pass
    args = _Args()
    args.adv_C = 2.0
    args.w_density = 0.2
    args.w_density2 = None
    args.w_realism = 0.4
    args.w_utility = 0.4
    args.adv_density_k = 11
    args.adv_norm = "rank"
    args.adv_proxy = "logreg"
    args.adv_val_frac = 0.2
    args.adv_min_keep_per_decile = 5
    args.adv_max_keep_per_decile = 10**9
    args.adv_mode = "hard"
    args.adv_thresh_mode = "val-pr-auc"
    args.adv_keep_quantile = 0.6

    keep = adv_select_synthetics(X_tr, y_tr, X_min_real, X_min_synth, args)
    X_aug_h = np.vstack([X_tr, X_min_synth[keep]])
    y_aug_h = np.hstack([y_tr, np.ones(keep.sum(), dtype=int)])
    ap_hard = average_precision_score(y_te, _proxy_model("logreg", rng).fit(X_aug_h, y_aug_h).predict_proba(X_te)[:, 1])

    args.adv_mode = "soft"
    w = adv_select_synthetics(X_tr, y_tr, X_min_real, X_min_synth, args)
    X_aug_s = np.vstack([X_tr, X_min_synth])
    y_aug_s = np.hstack([y_tr, np.ones(len(X_min_synth), dtype=int)])
    sw = np.hstack([np.ones(len(X_tr)), 1.0 + 1.0 * w])
    ap_soft = average_precision_score(y_te, _proxy_model("logreg", rng).fit(X_aug_s, y_aug_s, sample_weight=sw).predict_proba(X_te)[:, 1])

    print(f"[QuickSmoke] PR-AUC base={ap_base:.3f} | SMOTE={ap_sm:.3f} | ADV-hard={ap_hard:.3f} | ADV-soft={ap_soft:.3f}")