"""
Model utilities for training and cross-validation
"""

import numpy as np
from collections import defaultdict

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone

from lightgbm import LGBMClassifier, early_stopping, log_evaluation

from config import (
    RANDOM_STATE, LGBM_PARAMS, RF_PARAMS, ETC_PARAMS,
    SVD_PARAMS, LR_PARAMS, META_LR_PARAMS
)


def scaffold_stratified_kfold_indices(scaffolds, y, n_splits=5, seed=42):
    """
    Greedy scaffold-based stratified K-fold split
    
    Ensures no scaffold overlap between folds while maintaining class balance
    
    Args:
        scaffolds: Array of scaffold identifiers
        y: Array of labels
        n_splits: Number of folds
        seed: Random seed
        
    Yields:
        Tuples of (train_indices, test_indices)
    """
    np.random.seed(seed)
    
    # Group indices by scaffold
    scaffold_to_indices = defaultdict(list)
    for idx, scaf in enumerate(scaffolds):
        scaffold_to_indices[scaf].append(idx)
    
    # Calculate scaffold sizes and labels
    scaffold_list = []
    for scaf, indices in scaffold_to_indices.items():
        labels = y[indices]
        scaffold_list.append({
            'scaffold': scaf,
            'indices': indices,
            'size': len(indices),
            'n_active': int(labels.sum()),
            'n_inactive': int(len(labels) - labels.sum())
        })
    
    # Sort by size (descending) for greedy assignment
    scaffold_list.sort(key=lambda x: x['size'], reverse=True)
    
    # Initialize folds
    folds = [{'indices': [], 'n_active': 0, 'n_inactive': 0} for _ in range(n_splits)]
    
    # Greedy assignment
    for scaf_info in scaffold_list:
        # Find fold with minimum total compounds (or balance if needed)
        fold_sizes = [len(f['indices']) for f in folds]
        min_fold_idx = np.argmin(fold_sizes)
        
        # Add to smallest fold
        folds[min_fold_idx]['indices'].extend(scaf_info['indices'])
        folds[min_fold_idx]['n_active'] += scaf_info['n_active']
        folds[min_fold_idx]['n_inactive'] += scaf_info['n_inactive']
    
    # Print fold statistics
    print("\nScaffold-stratified fold statistics:")
    for i, fold in enumerate(folds):
        total = len(fold['indices'])
        active_pct = 100 * fold['n_active'] / total if total > 0 else 0
        print(f"  Fold {i+1}: {total} samples "
              f"({fold['n_active']} active [{active_pct:.1f}%], "
              f"{fold['n_inactive']} inactive)")
    
    # Generate train/test splits
    all_indices = np.arange(len(y))
    for i in range(n_splits):
        test_indices = np.array(folds[i]['indices'])
        train_indices = np.array([idx for j in range(n_splits) 
                                 if j != i for idx in folds[j]['indices']])
        yield train_indices, test_indices


def get_base_models(scale_pos_weight):
    """
    Initialize base models with proper class weight
    
    Args:
        scale_pos_weight: Weight for positive class (for LGBM)
        
    Returns:
        List of (name, model) tuples
    """
    lgbm_params = LGBM_PARAMS.copy()
    lgbm_params['scale_pos_weight'] = scale_pos_weight
    
    base_models = [
        ("lgbm", LGBMClassifier(**lgbm_params)),
        ("rf", RandomForestClassifier(**RF_PARAMS)),
        ("etc", ExtraTreesClassifier(**ETC_PARAMS)),
        ("svd_lr", Pipeline([
            ("svd", TruncatedSVD(**SVD_PARAMS)),
            ("lr", LogisticRegression(**LR_PARAMS)),
        ])),
    ]
    
    return base_models


def get_meta_model():
    """
    Initialize meta-learner (stacking)
    
    Returns:
        Meta-model estimator
    """
    return LogisticRegression(**META_LR_PARAMS)


def _fit_with_early_stopping_if_lgbm(est, X_tr, y_tr, X_va, y_va):
    """
    Fit model with early stopping for LGBM, regular fit for others
    """
    if isinstance(est, LGBMClassifier):
        est.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            callbacks=[
                early_stopping(stopping_rounds=300, verbose=False),
                log_evaluation(period=0)
            ],
        )
        return est
    
    est.fit(X_tr, y_tr)
    return est


def _fit_full_with_internal_val_if_lgbm(est, X, y, seed=42):
    """
    Fit on full data with internal validation for LGBM
    """
    if isinstance(est, LGBMClassifier):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.12, random_state=seed)
        tr_i, va_i = next(sss.split(X, y))
        est.fit(
            X[tr_i], y[tr_i],
            eval_set=[(X[va_i], y[va_i])],
            eval_metric="auc",
            callbacks=[
                early_stopping(stopping_rounds=300, verbose=False),
                log_evaluation(period=0)
            ],
        )
        return est
    
    est.fit(X, y)
    return est


def get_base_oof_and_test(models, X_tr, y_tr, X_te, n_splits=5, seed=42):
    """
    Generate out-of-fold predictions for training and test predictions
    
    Args:
        models: List of (name, estimator) tuples
        X_tr: Training features
        y_tr: Training labels
        X_te: Test features
        n_splits: Number of CV folds
        seed: Random seed
        
    Returns:
        Tuple of (oof_predictions, test_predictions)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n_train, n_test = X_tr.shape[0], X_te.shape[0]
    oof = np.zeros((n_train, len(models)), dtype=np.float32)
    te = np.zeros((n_test, len(models)), dtype=np.float32)
    
    for j, (name, est) in enumerate(models):
        print(f"  Training base model {j+1}/{len(models)}: {name}")
        
        # Generate OOF predictions
        oof_j = np.zeros(n_train, dtype=np.float32)
        for fold_idx, (tr_i, va_i) in enumerate(skf.split(X_tr, y_tr)):
            est_fold = clone(est)
            est_fold = _fit_with_early_stopping_if_lgbm(
                est_fold, X_tr[tr_i], y_tr[tr_i], X_tr[va_i], y_tr[va_i])
            oof_j[va_i] = est_fold.predict_proba(X_tr[va_i])[:, 1]
        
        oof[:, j] = oof_j
        
        # Fit on full training data and predict test
        est_full = clone(est)
        est_full = _fit_full_with_internal_val_if_lgbm(
            est_full, X_tr, y_tr, seed=seed + 1000 + j)
        te[:, j] = est_full.predict_proba(X_te)[:, 1]
    
    return oof, te


def best_threshold_from_inner_cv(Z, y, calibrator_cv=3, inner_folds=3, 
                                 metric="balanced_accuracy", seed=42):
    """
    Find optimal threshold using inner cross-validation
    
    Args:
        Z: Meta-features (base model predictions)
        y: Labels
        calibrator_cv: CV folds for calibration
        inner_folds: CV folds for threshold tuning
        metric: Metric to optimize ('balanced_accuracy' or 'accuracy')
        seed: Random seed
        
    Returns:
        Tuple of (best_threshold, best_value, oof_probabilities)
    """
    skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    oof_prob = np.zeros(len(y), dtype=np.float32)
    meta_base = get_meta_model()
    
    for tr_i, va_i in skf.split(Z, y):
        base = clone(meta_base)
        cal = CalibratedClassifierCV(base, method="sigmoid", cv=calibrator_cv)
        cal.fit(Z[tr_i], y[tr_i])
        oof_prob[va_i] = cal.predict_proba(Z[va_i])[:, 1]
    
    # Search for best threshold
    ts = np.linspace(0.05, 0.95, 181)
    best_t, best_val = 0.5, -1.0
    
    for t in ts:
        yhat = (oof_prob >= t).astype(int)
        if metric == "accuracy":
            from sklearn.metrics import accuracy_score
            val = accuracy_score(y, yhat)
        else:
            from sklearn.metrics import balanced_accuracy_score
            val = balanced_accuracy_score(y, yhat)
        
        if val > best_val:
            best_val, best_t = float(val), float(t)
    
    return best_t, best_val, oof_prob


def train_full_base_models(models, X, y, seed=42):
    """
    Train all base models on full dataset
    
    Args:
        models: List of (name, estimator) tuples
        X: Features
        y: Labels
        seed: Random seed
        
    Returns:
        List of (name, fitted_model) tuples
    """
    trained_models = []
    
    for j, (name, est) in enumerate(models):
        print(f"  Training full base model {j+1}/{len(models)}: {name}")
        est_full = clone(est)
        est_full = _fit_full_with_internal_val_if_lgbm(
            est_full, X, y, seed=seed + 9000 + j)
        trained_models.append((name, est_full))
    
    return trained_models
