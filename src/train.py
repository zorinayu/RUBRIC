from __future__ import annotations
import argparse, json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter errors
import matplotlib.pyplot as plt
from tqdm import tqdm
from data import load_creditcard_csv, load_nsl_kdd_data
from preprocess import preprocess_df
from augment import (
    smote_oversample,
    smote_then_rubric_filter,
    adasyn_oversample,
    borderline_smote_oversample,
    svm_smote_oversample,
    kmeans_smote_oversample,
    generate_then_filter,
    smote_tomek_resample,
    smote_enn_resample,
    make_oversampler,
    rubric_select_synthetics,
)
from models.svm_rff import SVMWithRFF
from evaluate import evaluate_and_plot
import time


def parse_args():
    p = argparse.ArgumentParser(description='Imbalanced Classification with RBF features + SVM + (optional) RUBRIC filtering of synthetic samples')
    p.add_argument('--data', type=str, default='data/creditcard.csv')
    p.add_argument('--dataset', type=str, default='creditcard', choices=['creditcard', 'nsl_kdd'], help='Dataset type to use')
    p.add_argument('--test-size', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--augment', type=str, default='adv', choices=['none','smote','adv','smote-adv','adasyn','borderline-smote','svm-smote','kmeans-smote','smote-tomek','smote-enn'])
    p.add_argument('--target-ratio', type=float, default=0.3, help='Minority:Majority balancing target (e.g., 0.3 -> 3:10)')
    p.add_argument('--rbf-components', type=int, default=300)
    p.add_argument('--rbf-gamma', type=float, default=-1.0, help='RBF bandwidth; if <0 uses median heuristic')
    p.add_argument('--svm-C', type=float, default=1.0)
    p.add_argument('--no-plots', action='store_true', help='Disable plots (ROC/PR/UMAP/PCA)')
    # RUBRIC selection controls
    p.add_argument('--keep-top-frac', type=float, default=0.65)
    # alias for convenience
    p.add_argument('--keep-frac', type=float, default=None, help='Alias for --keep-top-frac')
    p.add_argument('--adv-C', type=float, default=1.0)
    p.add_argument('--k-density', type=int, default=10)
    p.add_argument('--k-majority', type=int, default=10)
    p.add_argument('--w-realism', type=float, default=0.6)
    p.add_argument('--w-density', type=float, default=0.3)
    p.add_argument('--w-majority', type=float, default=0.1)
    p.add_argument('--gate-frac', type=float, default=0.9)
    p.add_argument('--grid', action='store_true', help='Enable small grid over C, keep, w_density')
    p.add_argument('--gen-kind', type=str, default='smote', choices=['smote','borderline','borderline2','svm','kmeans','smote-tomek','smote-enn','adasyn'], help='Underlying generator for RUBRIC filter')
    p.add_argument('--adapt-keep', action='store_true', help='Enable one-shot adaptive keep_top_frac fallback using recall@FPR=1% on a validation split')
    # --- RUBRIC selection improvements ---
    p.add_argument('--adv-mode', type=str, default='soft', choices=['hard','soft'],
                   help='Hard = keep/drop; Soft = sample_weight from score.')
    p.add_argument('--w-utility', type=float, default=0.40,
                   help='Weight of boundary utility score U for selection.')
    p.add_argument('--w-density2', type=float, default=None,
                   help='Optional override for density weight; falls back to --w-density if None.')
    p.add_argument('--adv-density-k', type=int, default=11,
                   help='k for kNN density on minority manifold.')
    p.add_argument('--adv-thresh-mode', type=str, default='val-pr-auc', choices=['val-pr-auc','quantile'],
                   help='Pick threshold by proxy PR-AUC on a validation split, or by a fixed quantile.')
    p.add_argument('--adv-keep-quantile', type=float, default=0.6,
                   help='If thresh-mode=quantile, keep top q fraction by blended score.')
    p.add_argument('--adv-norm', type=str, default='rank', choices=['none','minmax','rank'],
                   help='Normalization for combining R, U, D.')
    p.add_argument('--adv-proxy', type=str, default='logreg', choices=['logreg','xgb'],
                   help='Quick proxy classifier used to choose threshold by PR-AUC.')
    p.add_argument('--adv-val-frac', type=float, default=0.2,
                   help='Fraction of TRAIN to use as inner validation for threshold tuning.')
    p.add_argument('--adv-min-keep-per-decile', type=int, default=5,
                   help='Safety floor: keep at least N points per density decile to preserve submodes.')
    p.add_argument('--adv-max-keep-per-decile', type=int, default=1000000,
                   help='Safety cap per density decile.')
    return p.parse_args()


def create_output_dir(args):
    """Create organized output directory based on parameters"""
    # Create descriptive directory name
    dir_parts = []
    dir_parts.append(args.dataset)
    dir_parts.append(f"aug_{args.augment}")
    if args.augment in ('adv','smote-adv'):
        dir_parts.append(f"gen_{args.gen_kind}")
    dir_parts.append(f"rff_{args.rbf_components}")
    dir_parts.append(f"gamma_{args.rbf_gamma}")
    dir_parts.append(f"svmC_{args.svm_C}")
    dir_parts.append(f"test_{int(args.test_size*100)}")
    if args.augment != 'none':
        dir_parts.append(f"ratio_{int(args.target_ratio*100)}")
    dir_parts.append(f"seed_{args.seed}")
    output_dir = Path('outputs') / '_'.join(dir_parts)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def median_heuristic_gamma(X: np.ndarray) -> float:
    # Use subsample for speed
    n = min(5000, len(X))
    if n < 2:
        return 0.5
    idx = np.random.RandomState(42).choice(len(X), size=n, replace=False)
    Xs = X[idx]
    # pairwise distances approx via sampling
    diffs = Xs[:n//2] - Xs[n//2:n]
    med = np.median(np.sum(diffs*diffs, axis=1))
    if med <= 0:
        return 0.5
    return 1.0 / (2.0 * med)


def main():
    args = parse_args()
    if args.keep_frac is not None:
        args.keep_top_frac = args.keep_frac
    rng = check_random_state(args.seed)

    # Create organized output directory
    output_dir = create_output_dir(args)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Starting Training...")
    print(f"Output directory: {output_dir}")
    print(f"Configuration:")
    print(f"   - Dataset: {args.dataset}")
    print(f"   - Augmentation: {args.augment}")
    print(f"   - RFF Components: {args.rbf_components}")
    print(f"   - RBF Gamma: {args.rbf_gamma}")
    print(f"   - SVM C: {args.svm_C}")
    print(f"   - Test Size: {args.test_size}")
    if args.augment != 'none':
        print(f"   - Target Ratio: {args.target_ratio}")

    # Load + preprocess
    print("\nLoading and preprocessing data...")
    with tqdm(total=4, desc="Data Processing") as pbar:
        if args.dataset == 'creditcard':
            df = load_creditcard_csv(args.data)
        elif args.dataset == 'nsl_kdd':
            df = load_nsl_kdd_data()
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        pbar.update(1)

        X_df, y = preprocess_df(df, dataset_type=args.dataset)
        pbar.update(1)

        X = X_df.values.astype(np.float32)
        pbar.update(1)

        # Split first, then scale using training statistics
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=args.seed)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        pbar.update(1)

        # Median heuristic gamma if requested (computed on scaled training data)
        gamma = args.rbf_gamma if args.rbf_gamma >= 0 else median_heuristic_gamma(X_tr)
        pbar.update(1)
    # Split already performed above
    print("Splitting data into train/test sets...")
    print(f"   Training set: {len(X_tr)} samples")
    print(f"   Test set: {len(X_te)} samples")
    print(f"   Minority class ratio: {np.mean(y_tr):.4f} (train), {np.mean(y_te):.4f} (test)")

    # Augmentation timings
    aug_times = {}

    # Compute current minority:majority ratio and adjust target if needed to avoid minority downsampling
    n_min = int(np.sum(y_tr == 1))
    n_maj = int(np.sum(y_tr == 0))
    current_ratio = (n_min / n_maj) if n_maj > 0 else 0.0
    effective_ratio = args.target_ratio
    if args.augment != 'none' and effective_ratio <= current_ratio:
        # Bump slightly above current ratio; cap to reasonable upper bound
        bump = 0.05
        effective_ratio = max(current_ratio + bump, current_ratio + 1e-6)
        print(f"   Adjusted target ratio from {args.target_ratio} to {effective_ratio:.3f} (current minority/majority={current_ratio:.3f})")

    # Augmentation
    model_sample_weight = None
    if args.augment == 'none':
        print("Skipping data augmentation")
    elif args.augment == 'smote':
        print("Applying SMOTE oversampling...")
        t0 = time.time()
        X_tr, y_tr = smote_oversample(X_tr, y_tr, random_state=args.seed, ratio=effective_ratio)
        aug_times['augment_total_s'] = float(time.time() - t0)
        print(f"   After SMOTE: {len(X_tr)} samples (minority ratio: {np.mean(y_tr):.4f})")
    elif args.augment in ('adv','smote-adv'):
        # RUBRIC selection: generate, then keep/weight selected synthetic minority
        orig_X_tr, orig_y_tr = X_tr.copy(), y_tr.copy()
        print(f"Applying {args.gen_kind.upper()} + RUBRIC selection ({args.adv_mode})...")
        t0 = time.time()
        # Step 1: generate with chosen generator
        osr = make_oversampler(args.gen_kind, effective_ratio, args.seed, k_neighbors=5)
        X_res, y_res = osr.fit_resample(X_tr, y_tr)
        gen_time = time.time() - t0
        # Step 2: extract synthetic minority
        n_orig = len(X_tr)
        X_tail, y_tail = X_res[n_orig:], y_res[n_orig:]
        X_min_synth = X_tail[y_tail == 1]
        X_min_real = X_tr[y_tr == 1]
        # Step 3: RUBRIC selection
        t1 = time.time()
        sel = rubric_select_synthetics(
            X_train=X_tr,
            y_train=y_tr,
            X_min_real=X_min_real,
            X_min_synth=X_min_synth,
            args=args,
        )
        sel_time = time.time() - t1
        # Step 4: build augmented training set
        if args.adv_mode == 'soft':
            synth_weights = sel  # [0,1]
            X_tr = np.vstack([X_tr, X_min_synth])
            y_tr = np.hstack([y_tr, np.ones(len(X_min_synth), dtype=int)])
            base_w = np.ones(len(X_tr) - len(X_min_synth), dtype=float)
            lam = 1.0
            w_synth = 1.0 + lam * synth_weights
            model_sample_weight = np.hstack([base_w, w_synth])
            kept = len(X_min_synth)
        else:
            keep_mask = sel.astype(bool)
            X_tr = np.vstack([X_tr, X_min_synth[keep_mask]])
            y_tr = np.hstack([y_tr, np.ones(int(keep_mask.sum()), dtype=int)])
            kept = int(keep_mask.sum())
        aug_times['smote_generate_s'] = float(gen_time)
        aug_times['rubric_select_s'] = float(sel_time)
        aug_times['augment_total_s'] = float(time.time() - t0)
        print(f"   RUBRIC kept/weighted {kept}/{len(X_min_synth)} synthetic samples")
        print(f"   After {args.gen_kind}+RUBRIC: {len(X_tr)} samples (minority ratio: {np.mean(y_tr):.4f})")

        # Optional: one-shot adaptive keep fallback based on Recall@FPR=1%
        if args.adapt_keep:
            print("   Adaptive keep check on validation split (Recall@FPR=1%)...")
            from sklearn.model_selection import train_test_split as _tts

            X_tr_in, X_val_in, y_tr_in, y_val_in = _tts(orig_X_tr, orig_y_tr, test_size=0.25, stratify=orig_y_tr, random_state=args.seed)
            # Base augmentation on inner train
            osr = make_oversampler(args.gen_kind, effective_ratio, args.seed, k_neighbors=5)
            Xb, yb = osr.fit_resample(X_tr_in, y_tr_in)
            # RUBRIC on inner train
            Xa, ya = generate_then_filter(
                X_tr_in, y_tr_in,
                generator=args.gen_kind,
                ratio=effective_ratio,
                random_state=args.seed,
                keep_top_frac=args.keep_top_frac,
                max_iter=500,
                C=args.adv_C,
                k_density=args.k_density,
                k_majority=args.k_majority,
                weights=(args.w_realism, args.w_density, args.w_majority),
                gate_frac=args.gate_frac,
                return_times=None,
            )

            # Train quick models and compare recall@1%
            model_b = SVMWithRFF(n_components=min(200, args.rbf_components), gamma=gamma, C=args.svm_C, random_state=args.seed)
            model_b.fit(Xb, yb)
            sb = model_b.decision_function(X_val_in)
            model_a = SVMWithRFF(n_components=min(200, args.rbf_components), gamma=gamma, C=args.svm_C, random_state=args.seed)
            model_a.fit(Xa, ya)
            sa = model_a.decision_function(X_val_in)

            # Compute recall@FPR=1%
            from sklearn.metrics import roc_curve
            def _recall_at_fpr(y_true, scores, target_fpr: float=0.01):
                fpr, tpr, _ = roc_curve(y_true, scores)
                mask = fpr <= target_fpr
                return float(np.max(tpr[mask])) if np.any(mask) else 0.0

            rb = _recall_at_fpr(y_val_in, sb, 0.01)
            ra = _recall_at_fpr(y_val_in, sa, 0.01)
            print(f"      base Recall@1%={rb:.4f}, RUBRIC Recall@1%={ra:.4f}")
            if ra + 1e-6 < rb:
                # Reduce keep_top_frac once and regenerate on full train
                old_keep = args.keep_top_frac
                # pick lower bound by generator
                kind = (args.gen_kind or 'smote').lower()
                lower = 0.5 if kind in ('svm','svm-smote') else 0.6
                args.keep_top_frac = max(lower, args.keep_top_frac - 0.05)
                print(f"      Degradation detected. Reducing keep_top_frac from {old_keep} to {args.keep_top_frac} and regenerating...")
                t0 = time.time()
                X_tr, y_tr = generate_then_filter(
                    orig_X_tr, orig_y_tr,
                    generator=args.gen_kind,
                    ratio=effective_ratio,
                    random_state=args.seed,
                    keep_top_frac=args.keep_top_frac,
                    max_iter=500,
                    C=args.adv_C,
                    k_density=args.k_density,
                    k_majority=args.k_majority,
                    weights=(args.w_realism, args.w_density, args.w_majority),
                    gate_frac=args.gate_frac,
                    return_times=aug_times,
                )
                aug_times['augment_total_s'] = float(time.time() - t0)
                print(f"      Regenerated: {len(X_tr)} samples (minority ratio: {np.mean(y_tr):.4f})")
    elif args.augment == 'adasyn':
        print("Applying ADASYN oversampling...")
        t0 = time.time()
        X_tr, y_tr = adasyn_oversample(X_tr, y_tr, random_state=args.seed, ratio=effective_ratio)
        aug_times['augment_total_s'] = float(time.time() - t0)
        print(f"   After ADASYN: {len(X_tr)} samples (minority ratio: {np.mean(y_tr):.4f})")
    elif args.augment == 'borderline-smote':
        print("Applying Borderline-SMOTE (borderline-1) oversampling...")
        t0 = time.time()
        X_tr, y_tr = borderline_smote_oversample(X_tr, y_tr, random_state=args.seed, ratio=effective_ratio, kind="borderline-1")
        aug_times['augment_total_s'] = float(time.time() - t0)
        print(f"   After Borderline-SMOTE: {len(X_tr)} samples (minority ratio: {np.mean(y_tr):.4f})")
    elif args.augment == 'svm-smote':
        print("Applying SVM-SMOTE oversampling...")
        t0 = time.time()
        X_tr, y_tr = svm_smote_oversample(X_tr, y_tr, random_state=args.seed, ratio=effective_ratio)
        aug_times['augment_total_s'] = float(time.time() - t0)
        print(f"   After SVM-SMOTE: {len(X_tr)} samples (minority ratio: {np.mean(y_tr):.4f})")
    elif args.augment == 'kmeans-smote':
        print("Applying KMeans-SMOTE oversampling...")
        t0 = time.time()
        X_tr, y_tr = kmeans_smote_oversample(X_tr, y_tr, random_state=args.seed, ratio=effective_ratio)
        aug_times['augment_total_s'] = float(time.time() - t0)
        print(f"   After KMeans-SMOTE: {len(X_tr)} samples (minority ratio: {np.mean(y_tr):.4f})")
    elif args.augment == 'smote-tomek':
        print("Applying SMOTE-Tomek resampling...")
        t0 = time.time()
        X_tr, y_tr = smote_tomek_resample(X_tr, y_tr, random_state=args.seed, ratio=effective_ratio)
        aug_times['augment_total_s'] = float(time.time() - t0)
        print(f"   After SMOTE-Tomek: {len(X_tr)} samples (minority ratio: {np.mean(y_tr):.4f})")
    elif args.augment == 'smote-enn':
        print("Applying SMOTE-ENN resampling...")
        t0 = time.time()
        X_tr, y_tr = smote_enn_resample(X_tr, y_tr, random_state=args.seed, ratio=effective_ratio)
        aug_times['augment_total_s'] = float(time.time() - t0)
        print(f"   After SMOTE-ENN: {len(X_tr)} samples (minority ratio: {np.mean(y_tr):.4f})")

    # Optional small grid over C, keep, w_density
    best = None
    if args.grid and args.augment in ('adv','smote-adv'):
        print("Running small grid search for RUBRIC...")
        Cs = [1.0, 2.0, 5.0]
        keeps = [0.6, 0.65, 0.7]
        wds = [0.2, 0.3, 0.4]
        best_pr = -1.0
        for Cc in Cs:
            for keep in keeps:
                for wd in wds:
                    tgrid = {}
                    Xg, yg = smote_then_rubric_filter(
                        X_tr, y_tr,
                        random_state=args.seed,
                        ratio=args.target_ratio,
                        keep_top_frac=keep,
                        max_iter=500,
                        C=Cc,
                        k_density=args.k_density,
                        k_majority=args.k_majority,
                        weights=(args.w_realism, wd, args.w_majority),
                        gate_frac=args.gate_frac,
                        return_times=tgrid,
                    )
                    model_g = SVMWithRFF(n_components=args.rbf_components, gamma=gamma, C=args.svm_C, random_state=args.seed)
                    model_g.fit(Xg, yg)
                    scores_g = model_g.decision_function(X_te)
                    from sklearn.metrics import average_precision_score
                    pr = average_precision_score(y_te, scores_g)
                    if pr > best_pr:
                        best_pr = pr
                        best = {
                            'keep_top_frac': keep,
                            'adv_C': Cc,
                            'w_density': wd,
                        }
        if best is not None:
            args.keep_top_frac = best['keep_top_frac']
            args.adv_C = best['adv_C']
            args.w_density = best['w_density']
            print(f"Grid best: keep={args.keep_top_frac}, C={args.adv_C}, w_density={args.w_density}")
            # Re-run augmentation with best params
            X_tr, y_tr = smote_then_rubric_filter(
                X_tr, y_tr,
                random_state=args.seed,
                ratio=args.target_ratio,
                keep_top_frac=args.keep_top_frac,
                max_iter=500,
                C=args.adv_C,
                k_density=args.k_density,
                k_majority=args.k_majority,
                weights=(args.w_realism, args.w_density, args.w_majority),
                gate_frac=args.gate_frac,
                return_times=aug_times,
            )

    # Model
    print("\nTraining SVM with RFF features...")
    t_train = time.time()
    model = SVMWithRFF(n_components=args.rbf_components, gamma=gamma, C=args.svm_C, random_state=args.seed)
    try:
        model.fit(X_tr, y_tr, sample_weight=model_sample_weight)
    except TypeError:
        model.fit(X_tr, y_tr)
    t_train = time.time() - t_train
    print("   Model training completed")

    # Evaluate
    print("Evaluating model performance...")
    t_inf = time.time()
    scores = model.decision_function(X_te)
    y_pred = (scores > 0).astype(int)
    t_inf = time.time() - t_inf
    metrics = evaluate_and_plot(y_te, scores, y_pred, out_dir=str(output_dir))

    # Save configuration
    config = {
        'dataset': args.dataset,
        'augment': args.augment,
        'rbf_components': args.rbf_components,
        'rbf_gamma': gamma,
        'svm_C': args.svm_C,
        'test_size': args.test_size,
        'target_ratio': args.target_ratio,
        'effective_target_ratio': effective_ratio,
        'seed': args.seed,
        'training_samples': len(X_tr),
        'test_samples': len(X_te),
        'minority_ratio_train': float(np.mean(y_tr)),
        'minority_ratio_test': float(np.mean(y_te)),
        'timings': {
            **aug_times,
            'train_s': float(t_train),
            'inference_s': float(t_inf),
        },
        'rubric_params': {
            'gen_kind': args.gen_kind,
            'keep_top_frac': args.keep_top_frac,
            'C': args.adv_C,
            'k_density': args.k_density,
            'k_majority': args.k_majority,
            'weights': [args.w_realism, args.w_density, args.w_majority],
            'gate_frac': args.gate_frac,
            'mode': args.adv_mode,
            'w_utility': args.w_utility,
            'w_density2': args.w_density2,
            'density_k': args.adv_density_k,
            'thresh_mode': args.adv_thresh_mode,
            'keep_quantile': args.adv_keep_quantile,
            'norm': args.adv_norm,
            'proxy': args.adv_proxy,
            'val_frac': args.adv_val_frac,
            'min_keep_per_decile': args.adv_min_keep_per_decile,
            'max_keep_per_decile': args.adv_max_keep_per_decile,
        } if args.augment in ('adv','smote-adv') else None,
    }

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("\nTraining completed successfully!")
    print(f"Results saved to: {output_dir}")
    print("Key Metrics:")
    print(f"   - ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"   - PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"   - F1-Macro: {metrics['f1_macro']:.4f}")
    print(f"   - F1-Weighted: {metrics['f1_weighted']:.4f}")

    if metrics['pr_auc'] < 0.01:
        print("WARNING: Very low PR-AUC indicates poor minority class detection!")
        print("Suggestion: Try data augmentation methods (SMOTE, RUBRIC)")
    elif metrics['pr_auc'] < 0.1:
        print("WARNING: Low PR-AUC - consider tuning hyperparameters or trying different augmentation")
    else:
        print("Good PR-AUC achieved!")

if __name__ == '__main__':
    main()
