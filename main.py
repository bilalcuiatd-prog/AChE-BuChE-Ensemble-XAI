"""
Main script for Acetylcholinesterase Inhibitor Prediction Pipeline
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import RDLogger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_curve, precision_recall_curve, accuracy_score, 
    balanced_accuracy_score, matthews_corrcoef
)

# Import custom modules
from config import *
from data_utils import (
    load_and_prepare_train_data, load_and_prepare_external_data,
    assert_no_inchikey_overlap, assert_no_scaffold_overlap
)
from feature_utils import build_features_and_bitinfo, get_feature_names
from model_utils import (
    scaffold_stratified_kfold_indices, get_base_models, get_meta_model,
    get_base_oof_and_test, best_threshold_from_inner_cv, train_full_base_models
)
from evaluation import (
    report_metrics, plot_fold_summary, plot_roc_curves, plot_pr_curves,
    plot_external_roc, plot_external_pr, plot_similarity_distribution,
    compute_applicability_domain, save_predictions
)


def setup_environment():
    """Setup environment and suppress warnings"""
    np.random.seed(RANDOM_STATE)
    RDLogger.DisableLog("rdApp.*")
    warnings.filterwarnings("ignore")
    
    plt.style.use(PLOT_STYLE)
    plt.rcParams.update(PLOT_RC_PARAMS)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_nested_cv(X_all, y_all, scaffolds, inchikeys):
    """
    Run nested scaffold-based cross-validation
    
    Args:
        X_all: Feature matrix
        y_all: Labels
        scaffolds: Scaffold identifiers
        inchikeys: InChIKey identifiers
        
    Returns:
        Dictionary with CV results
    """
    print("\n" + "="*70)
    print("NESTED SCAFFOLD CROSS-VALIDATION")
    print("="*70)
    
    # Calculate class weight
    pos = int(y_all.sum())
    neg = int(len(y_all) - pos)
    scale_pos_weight = neg / max(pos, 1)
    
    # Initialize models
    base_models = get_base_models(scale_pos_weight)
    model_names = [n for n, _ in base_models]
    
    # Storage for results
    fold_rows = []
    roc_curves = []
    pr_curves = []
    
    # Outer CV loop
    for fold_id, (idx_tr, idx_te) in enumerate(
        scaffold_stratified_kfold_indices(scaffolds, y_all, 
                                         n_splits=OUTER_FOLDS, 
                                         seed=RANDOM_STATE),
        start=1
    ):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_id}/{OUTER_FOLDS}")
        print(f"{'='*70}")
        
        # Check for leakage
        assert_no_inchikey_overlap(idx_tr, idx_te, inchikeys)
        assert_no_scaffold_overlap(idx_tr, idx_te, scaffolds)
        
        X_tr, y_tr = X_all[idx_tr], y_all[idx_tr]
        X_te, y_te = X_all[idx_te], y_all[idx_te]
        
        # Generate base model OOF predictions
        print("\nGenerating base model predictions...")
        oof_base, te_base = get_base_oof_and_test(
            base_models, X_tr, y_tr, X_te, 
            n_splits=INNER_FOLDS_OOF, 
            seed=RANDOM_STATE + fold_id
        )
        
        Z_tr = oof_base.astype(np.float32)
        Z_te = te_base.astype(np.float32)
        
        # Tune thresholds
        print("\nTuning classification thresholds...")
        thr_bacc, _, _ = best_threshold_from_inner_cv(
            Z_tr, y_tr, 
            calibrator_cv=CALIBRATION_CV, 
            inner_folds=INNER_FOLDS_THR,
            metric="balanced_accuracy", 
            seed=RANDOM_STATE + 100 + fold_id
        )
        
        thr_acc, _, _ = best_threshold_from_inner_cv(
            Z_tr, y_tr, 
            calibrator_cv=CALIBRATION_CV, 
            inner_folds=INNER_FOLDS_THR,
            metric="accuracy", 
            seed=RANDOM_STATE + 200 + fold_id
        )
        
        # Train calibrated meta-model
        print("\nTraining calibrated meta-model...")
        meta_base = get_meta_model()
        cal_final = CalibratedClassifierCV(meta_base, method="sigmoid", 
                                          cv=CALIBRATION_CV)
        cal_final.fit(Z_tr, y_tr)
        p_te = cal_final.predict_proba(Z_te)[:, 1]
        
        # Calculate metrics
        yhat_b = (p_te >= thr_bacc).astype(int)
        from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
        
        fold_rows.append({
            "fold": fold_id,
            "roc_auc": roc_auc_score(y_te, p_te),
            "pr_auc": average_precision_score(y_te, p_te),
            "brier": brier_score_loss(y_te, p_te),
            "acc@thr_bacc": accuracy_score(y_te, yhat_b),
            "bacc@thr_bacc": balanced_accuracy_score(y_te, yhat_b),
            "mcc@thr_bacc": matthews_corrcoef(y_te, yhat_b),
            "thr_bacc": thr_bacc,
            "thr_acc": thr_acc,
        })
        
        # Store curves
        fpr, tpr, _ = roc_curve(y_te, p_te)
        prec, rec, _ = precision_recall_curve(y_te, p_te)
        roc_curves.append((fpr, tpr))
        pr_curves.append((rec, prec))
        
        print(f"\nFold {fold_id} Results:")
        print(f"  ROC-AUC: {fold_rows[-1]['roc_auc']:.4f}")
        print(f"  PR-AUC: {fold_rows[-1]['pr_auc']:.4f}")
        print(f"  BACC: {fold_rows[-1]['bacc@thr_bacc']:.4f}")
    
    # Summarize results
    res = pd.DataFrame(fold_rows)
    summary_cols = ["roc_auc", "pr_auc", "brier", 
                   "acc@thr_bacc", "bacc@thr_bacc", "mcc@thr_bacc"]
    
    summary_df = pd.DataFrame([{
        "metric": c,
        "mean": float(res[c].mean()),
        "std": float(res[c].std(ddof=1)) if len(res) > 1 else 0.0
    } for c in summary_cols])
    
    print("\n" + "="*70)
    print("CROSS-VALIDATION SUMMARY")
    print("="*70)
    print("\nFold-wise metrics:")
    print(res.to_string(index=False))
    print("\nSummary (mean ± std):")
    print(summary_df.to_string(index=False))
    
    # Save results
    res.to_csv(os.path.join(OUTPUT_DIR, "cv_fold_metrics.csv"), index=False)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "cv_summary.csv"), index=False)
    
    # Plot results
    plot_fold_summary(summary_df, 
                     os.path.join(OUTPUT_DIR, "cv_summary.png"))
    plot_roc_curves(roc_curves, 
                   os.path.join(OUTPUT_DIR, "cv_roc_curves.png"))
    plot_pr_curves(pr_curves, 
                  os.path.join(OUTPUT_DIR, "cv_pr_curves.png"))
    
    return {
        'fold_results': res,
        'summary': summary_df,
        'roc_curves': roc_curves,
        'pr_curves': pr_curves
    }


def train_final_model(X_all, y_all, scaffolds, inchikeys):
    """
    Train final model on full dataset
    
    Args:
        X_all: Feature matrix
        y_all: Labels
        scaffolds: Scaffold identifiers
        inchikeys: InChIKey identifiers
        
    Returns:
        Dictionary with trained models and thresholds
    """
    print("\n" + "="*70)
    print("TRAINING FINAL MODEL ON FULL DATASET")
    print("="*70)
    
    # Calculate class weight
    pos = int(y_all.sum())
    neg = int(len(y_all) - pos)
    scale_pos_weight = neg / max(pos, 1)
    
    # Initialize models
    base_models = get_base_models(scale_pos_weight)
    n_models = len(base_models)
    
    # Generate full-data OOF for threshold tuning
    print("\nGenerating full-data OOF predictions...")
    Z_oof_full = np.zeros((len(y_all), n_models), dtype=np.float32)
    
    from sklearn.base import clone
    from model_utils import _fit_with_early_stopping_if_lgbm
    
    for j, (name, est) in enumerate(base_models):
        print(f"  Base model {j+1}/{n_models}: {name}")
        oof_j = np.zeros(len(y_all), dtype=np.float32)
        
        for idx_tr, idx_va in scaffold_stratified_kfold_indices(
            scaffolds, y_all, n_splits=OUTER_FOLDS, seed=RANDOM_STATE + 500 + j
        ):
            assert_no_inchikey_overlap(idx_tr, idx_va, inchikeys)
            assert_no_scaffold_overlap(idx_tr, idx_va, scaffolds)
            
            est_fold = clone(est)
            est_fold = _fit_with_early_stopping_if_lgbm(
                est_fold, X_all[idx_tr], y_all[idx_tr], 
                X_all[idx_va], y_all[idx_va]
            )
            oof_j[idx_va] = est_fold.predict_proba(X_all[idx_va])[:, 1]
        
        Z_oof_full[:, j] = oof_j
    
    # Tune thresholds on full OOF
    print("\nTuning final thresholds...")
    thr_bacc_full, _, _ = best_threshold_from_inner_cv(
        Z_oof_full, y_all, 
        calibrator_cv=CALIBRATION_CV, 
        inner_folds=INNER_FOLDS_THR,
        metric="balanced_accuracy", 
        seed=RANDOM_STATE + 700
    )
    
    thr_acc_full, _, _ = best_threshold_from_inner_cv(
        Z_oof_full, y_all, 
        calibrator_cv=CALIBRATION_CV, 
        inner_folds=INNER_FOLDS_THR,
        metric="accuracy", 
        seed=RANDOM_STATE + 701
    )
    
    print(f"\nFinal thresholds:")
    print(f"  BACC-optimized: {thr_bacc_full:.3f}")
    print(f"  ACC-optimized: {thr_acc_full:.3f}")
    
    # Train calibrated meta-model on full OOF
    meta_base = get_meta_model()
    meta_cal_full = CalibratedClassifierCV(meta_base, method="sigmoid", 
                                          cv=CALIBRATION_CV)
    meta_cal_full.fit(Z_oof_full, y_all)
    
    # Train full base models
    print("\nTraining final base models on full dataset...")
    trained_base_models = train_full_base_models(base_models, X_all, y_all, 
                                                 seed=RANDOM_STATE)
    
    return {
        'base_models': trained_base_models,
        'meta_model': meta_cal_full,
        'thr_bacc': thr_bacc_full,
        'thr_acc': thr_acc_full,
        'model_names': [n for n, _ in trained_base_models]
    }


def evaluate_external(final_model, X_ext, y_ext, ext_df, train_smiles):
    """
    Evaluate model on external validation set
    
    Args:
        final_model: Dictionary with trained models
        X_ext: External feature matrix
        y_ext: External labels
        ext_df: External DataFrame
        train_smiles: Training SMILES for applicability domain
        
    Returns:
        Dictionary with external evaluation results
    """
    print("\n" + "="*70)
    print("EXTERNAL VALIDATION")
    print("="*70)
    
    # Get predictions from base models
    n_models = len(final_model['base_models'])
    P_ext = np.zeros((len(ext_df), n_models), dtype=np.float32)
    
    for j, (name, est) in enumerate(final_model['base_models']):
        P_ext[:, j] = est.predict_proba(X_ext)[:, 1]
    
    # Get stacked predictions
    p_ext = final_model['meta_model'].predict_proba(P_ext)[:, 1]
    
    # Report metrics
    results = {}
    results['bacc_thr'] = report_metrics(
        "EXTERNAL - Stacked (BACC threshold)", 
        y_ext, p_ext, final_model['thr_bacc']
    )
    
    results['acc_thr'] = report_metrics(
        "EXTERNAL - Stacked (ACC threshold)", 
        y_ext, p_ext, final_model['thr_acc']
    )
    
    # Plot curves
    plot_external_roc(y_ext, p_ext, 
                     title="External ROC (BindingDB)",
                     output_path=os.path.join(OUTPUT_DIR, "external_roc.png"))
    
    plot_external_pr(y_ext, p_ext, 
                    title="External PR (BindingDB)",
                    output_path=os.path.join(OUTPUT_DIR, "external_pr.png"))
    
    # Applicability domain analysis
    if USE_APPLICABILITY_DOMAIN_REPORT:
        max_sims, in_domain = compute_applicability_domain(
            train_smiles, ext_df["smiles_std"].values, 
            threshold=AD_SIM_THRESHOLD
        )
        
        ext_df["max_tanimoto_to_train"] = max_sims
        
        # Report in-domain performance
        if in_domain.sum() > 0:
            results['in_domain'] = report_metrics(
                f"EXTERNAL In-Domain (Tanimoto >= {AD_SIM_THRESHOLD:.2f}) - ACC threshold",
                y_ext[in_domain], p_ext[in_domain], final_model['thr_acc']
            )
        
        # Plot similarity distribution
        plot_similarity_distribution(
            max_sims, AD_SIM_THRESHOLD,
            output_path=os.path.join(OUTPUT_DIR, "ad_similarity_dist.png")
        )
    
    # Save predictions
    pred_dict = {name.upper(): P_ext[:, j] 
                for j, name in enumerate(final_model['model_names'])}
    pred_dict['STACK_CAL'] = p_ext
    
    save_predictions(
        ext_df, pred_dict,
        os.path.join(OUTPUT_DIR, "external_predictions.csv")
    )
    
    return results


def main():
    """Main execution function"""
    print("="*70)
    print("ACETYLCHOLINESTERASE INHIBITOR PREDICTION PIPELINE")
    print("="*70)
    
    # Setup
    setup_environment()
    
    # Load and prepare training data
    print("\n" + "="*70)
    print("DATA LOADING AND PREPARATION")
    print("="*70)
    
    df_train = load_and_prepare_train_data(TRAIN_CSV_PATH)
    
    y_all = df_train["is_active"].values.astype(int)
    smiles_list = df_train["smiles_std"].values
    inchikeys = df_train["inchikey"].values
    scaffolds = df_train["scaffold"].values
    
    # Build features
    print("\nBuilding molecular features...")
    X_all, bitinfos = build_features_and_bitinfo(smiles_list)
    print(f"Feature matrix shape: {X_all.shape}")
    
    # Run nested CV
    cv_results = run_nested_cv(X_all, y_all, scaffolds, inchikeys)
    
    # Train final model
    final_model = train_final_model(X_all, y_all, scaffolds, inchikeys)
    
    # External validation
    if os.path.exists(BINDINGDB_TSV_PATH):
        ext_df = load_and_prepare_external_data(
            BINDINGDB_TSV_PATH,
            KI_ACTIVE_MAX, KI_INACTIVE_MIN,
            set(inchikeys.tolist()), set(scaffolds.tolist()),
            EXTERNAL_SAMPLE_ACTIVE, EXTERNAL_SAMPLE_INACTIVE,
            RANDOM_STATE
        )
        
        X_ext, _ = build_features_and_bitinfo(ext_df["smiles_std"].values)
        y_ext = ext_df["y_ext"].values.astype(int)
        
        ext_results = evaluate_external(final_model, X_ext, y_ext, 
                                       ext_df, smiles_list)
    else:
        print(f"\nWarning: External data file not found: {BINDINGDB_TSV_PATH}")
        print("Skipping external validation.")
    
    # Optional SHAP analysis
    if RUN_SHAP:
        print("\n" + "="*70)
        print("SHAP ANALYSIS")
        print("="*70)
        
        import shap
        from feature_utils import get_feature_names
        
        # Get LGBM model
        lgbm_model = final_model['base_models'][0][1]  # Assuming first is LGBM
        
        # Sample data for SHAP
        rng = np.random.default_rng(RANDOM_STATE)
        bg_n = min(800, len(y_all))
        ex_n = min(800, len(y_all))
        bg_idx = rng.choice(len(y_all), size=bg_n, replace=False)
        ex_idx = rng.choice(len(y_all), size=ex_n, replace=False)
        
        X_bg = X_all[bg_idx]
        X_ex = X_all[ex_idx]
        
        # Compute SHAP values
        explainer = shap.TreeExplainer(
            lgbm_model, data=X_bg, 
            feature_perturbation="interventional"
        )
        shap_values = explainer.shap_values(X_ex, check_additivity=False)
        
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
        
        # Create DataFrame with feature names
        feature_names = get_feature_names()
        X_ex_df = pd.DataFrame(X_ex, columns=feature_names)
        
        # Plot SHAP summary
        shap.summary_plot(shap_values, X_ex_df, plot_type="bar", 
                         max_display=30, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "shap_importance.png"), 
                   dpi=140, bbox_inches='tight')
        plt.show()
        
        shap.summary_plot(shap_values, X_ex_df, max_display=30, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), 
                   dpi=140, bbox_inches='tight')
        plt.show()
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("  - cv_fold_metrics.csv")
    print("  - cv_summary.csv")
    print("  - cv_summary.png")
    print("  - cv_roc_curves.png")
    print("  - cv_pr_curves.png")
    if os.path.exists(BINDINGDB_TSV_PATH):
        print("  - external_predictions.csv")
        print("  - external_roc.png")
        print("  - external_pr.png")
        if USE_APPLICABILITY_DOMAIN_REPORT:
            print("  - ad_similarity_dist.png")
    if RUN_SHAP:
        print("  - shap_importance.png")
        print("  - shap_summary.png")


if __name__ == "__main__":
    main()
