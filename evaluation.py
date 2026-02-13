"""
Evaluation utilities for metrics, reporting, and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, matthews_corrcoef,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, roc_curve, precision_recall_curve
)

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from config import RADIUS, N_BITS


def report_metrics(title, y_true, p, thr):
    """
    Print comprehensive metrics for predictions
    
    Args:
        title: Report title
        y_true: True labels
        p: Predicted probabilities
        thr: Classification threshold
    """
    yhat = (p >= thr).astype(int)
    
    acc = accuracy_score(y_true, yhat)
    bacc = balanced_accuracy_score(y_true, yhat)
    mcc = matthews_corrcoef(y_true, yhat)
    
    if len(np.unique(y_true)) == 2:
        roc = roc_auc_score(y_true, p)
        pr = average_precision_score(y_true, p)
    else:
        roc = np.nan
        pr = np.nan
    
    brier = brier_score_loss(y_true, p)
    cm = confusion_matrix(y_true, yhat)
    
    print(f"\n[{title}] thr={thr:.3f}")
    print(f"  ACC={acc:.4f}  BACC={bacc:.4f}  MCC={mcc:.4f}  "
          f"ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}  Brier={brier:.4f}")
    print("  Confusion matrix [TN FP; FN TP]:")
    print(cm)
    
    return {
        'accuracy': acc,
        'balanced_accuracy': bacc,
        'mcc': mcc,
        'roc_auc': roc,
        'pr_auc': pr,
        'brier': brier,
        'confusion_matrix': cm
    }


def plot_fold_summary(summary_df, output_path=None):
    """
    Plot summary of fold-wise metrics
    
    Args:
        summary_df: DataFrame with 'metric', 'mean', 'std' columns
        output_path: Optional path to save figure
    """
    plt.figure(figsize=(8, 4))
    x = np.arange(len(summary_df))
    plt.errorbar(x, summary_df["mean"], yerr=summary_df["std"], 
                fmt="o", capsize=5, markersize=8)
    plt.xticks(x, summary_df["metric"], rotation=20, ha="right")
    plt.ylabel("Value")
    plt.title("Nested Scaffold CV: Mean ± Std")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=140, bbox_inches='tight')
    plt.show()


def plot_roc_curves(roc_curves, output_path=None):
    """
    Plot ROC curves for all folds
    
    Args:
        roc_curves: List of (fpr, tpr) tuples
        output_path: Optional path to save figure
    """
    plt.figure(figsize=(6, 5))
    
    for i, (fpr, tpr) in enumerate(roc_curves, start=1):
        plt.plot(fpr, tpr, label=f"Fold {i}", alpha=0.7)
    
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (Outer Folds)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=140, bbox_inches='tight')
    plt.show()


def plot_pr_curves(pr_curves, output_path=None):
    """
    Plot Precision-Recall curves for all folds
    
    Args:
        pr_curves: List of (recall, precision) tuples
        output_path: Optional path to save figure
    """
    plt.figure(figsize=(6, 5))
    
    for i, (rec, prec) in enumerate(pr_curves, start=1):
        plt.plot(rec, prec, label=f"Fold {i}", alpha=0.7)
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (Outer Folds)")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=140, bbox_inches='tight')
    plt.show()


def plot_external_roc(y_true, y_prob, title="External ROC", output_path=None):
    """
    Plot ROC curve for external validation
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        title: Plot title
        output_path: Optional path to save figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=140, bbox_inches='tight')
    plt.show()


def plot_external_pr(y_true, y_prob, title="External PR", output_path=None):
    """
    Plot Precision-Recall curve for external validation
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        title: Plot title
        output_path: Optional path to save figure
    """
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label=f"AUPRC = {auprc:.3f}", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=140, bbox_inches='tight')
    plt.show()


def plot_similarity_distribution(similarities, threshold, 
                                 title="Similarity to Training Set",
                                 output_path=None):
    """
    Plot distribution of Tanimoto similarities
    
    Args:
        similarities: Array of similarity scores
        threshold: Applicability domain threshold
        title: Plot title
        output_path: Optional path to save figure
    """
    plt.figure(figsize=(6, 3))
    plt.hist(similarities, bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(threshold, linestyle="--", color="red", 
               label=f"Threshold = {threshold:.2f}")
    plt.xlabel("Max Tanimoto Similarity to Training Set")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=140, bbox_inches='tight')
    plt.show()


def compute_applicability_domain(train_smiles, test_smiles, threshold=0.35,
                                 radius=RADIUS, n_bits=N_BITS):
    """
    Compute applicability domain using Tanimoto similarity
    
    Args:
        train_smiles: List of training SMILES
        test_smiles: List of test SMILES
        threshold: Similarity threshold for in-domain
        radius: Morgan fingerprint radius
        n_bits: Number of bits
        
    Returns:
        Tuple of (max_similarities, in_domain_mask)
    """
    print("\nComputing applicability domain (max Tanimoto to training set)...")
    
    # Generate training fingerprints
    train_fps = []
    for smi in train_smiles:
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
            train_fps.append(fp)
    
    # Compute max similarity for each test compound
    max_sims = []
    for i, smi in enumerate(test_smiles):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(test_smiles)} compounds")
        
        m = Chem.MolFromSmiles(smi)
        if m is None or len(train_fps) == 0:
            max_sims.append(0.0)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
            sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
            max_sims.append(float(max(sims)) if sims else 0.0)
    
    max_sims = np.array(max_sims)
    in_domain = max_sims >= threshold
    
    print(f"Applicability domain threshold: {threshold:.2f}")
    print(f"In-domain: {int(in_domain.sum())}/{len(test_smiles)} "
          f"({100 * in_domain.sum() / len(test_smiles):.1f}%)")
    
    return max_sims, in_domain


def save_predictions(df, predictions_dict, output_path):
    """
    Save predictions to CSV file
    
    Args:
        df: DataFrame with compound info
        predictions_dict: Dictionary of {model_name: predictions}
        output_path: Path to save CSV
    """
    df_out = df.copy()
    
    for model_name, preds in predictions_dict.items():
        df_out[f"Prob_{model_name}"] = preds
    
    df_out.to_csv(output_path, index=False)
    print(f"\nSaved predictions to: {output_path}")
    
    return df_out
