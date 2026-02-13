"""
Example script: Making predictions on new compounds
"""

import pickle
import numpy as np
import pandas as pd
from data_utils import sanitize_smiles
from feature_utils import build_features_and_bitinfo


def save_trained_model(final_model, filepath="trained_model.pkl"):
    """
    Save trained model to disk
    
    Args:
        final_model: Dictionary with trained models
        filepath: Path to save pickle file
    """
    with open(filepath, 'wb') as f:
        pickle.dump(final_model, f)
    print(f"Model saved to: {filepath}")


def load_trained_model(filepath="trained_model.pkl"):
    """
    Load trained model from disk
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Dictionary with trained models
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from: {filepath}")
    return model


def predict_new_compounds(smiles_list, model_path="trained_model.pkl"):
    """
    Predict activity for new compounds
    
    Args:
        smiles_list: List of SMILES strings
        model_path: Path to trained model pickle file
        
    Returns:
        DataFrame with predictions
    """
    print(f"\nPredicting activity for {len(smiles_list)} compounds...")
    
    # Load model
    final_model = load_trained_model(model_path)
    
    # Standardize SMILES
    print("Standardizing SMILES...")
    smiles_std = [sanitize_smiles(smi) for smi in smiles_list]
    
    # Filter out invalid SMILES
    valid_idx = [i for i, smi in enumerate(smiles_std) if smi is not None]
    valid_smiles = [smiles_std[i] for i in valid_idx]
    
    if len(valid_smiles) == 0:
        print("Error: No valid SMILES found!")
        return None
    
    print(f"Valid SMILES: {len(valid_smiles)}/{len(smiles_list)}")
    
    # Build features
    print("Building molecular features...")
    X, _ = build_features_and_bitinfo(valid_smiles)
    
    # Get base model predictions
    n_models = len(final_model['base_models'])
    P = np.zeros((len(valid_smiles), n_models), dtype=np.float32)
    
    for j, (name, est) in enumerate(final_model['base_models']):
        P[:, j] = est.predict_proba(X)[:, 1]
    
    # Get stacked predictions
    p_stack = final_model['meta_model'].predict_proba(P)[:, 1]
    
    # Apply thresholds
    pred_bacc = (p_stack >= final_model['thr_bacc']).astype(int)
    pred_acc = (p_stack >= final_model['thr_acc']).astype(int)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'original_smiles': [smiles_list[i] for i in valid_idx],
        'standardized_smiles': valid_smiles,
        'probability': p_stack,
        'prediction_bacc': pred_bacc,
        'prediction_acc': pred_acc,
        'bacc_threshold': final_model['thr_bacc'],
        'acc_threshold': final_model['thr_acc'],
    })
    
    # Add individual model probabilities
    for j, name in enumerate(final_model['model_names']):
        results[f'prob_{name}'] = P[:, j]
    
    print("\nPrediction complete!")
    print(f"Active (BACC threshold): {pred_bacc.sum()}/{len(pred_bacc)}")
    print(f"Active (ACC threshold): {pred_acc.sum()}/{len(pred_acc)}")
    
    return results


def main():
    """Example usage"""
    
    # Example: New compounds to predict
    new_smiles = [
        "CC(C)Cc1ccc(cc1)C(C)C(O)=O",  # Ibuprofen
        "CC(=O)Oc1ccccc1C(=O)O",        # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", # Caffeine
        "invalid_smiles",                # Invalid SMILES
    ]
    
    print("="*70)
    print("EXAMPLE: Predicting Activity for New Compounds")
    print("="*70)
    
    # Note: You need to run main.py first to generate the trained model
    # This example assumes you've modified main.py to save the model:
    #
    # In main.py, after training final_model, add:
    #   from example_predict import save_trained_model
    #   save_trained_model(final_model, "trained_model.pkl")
    
    try:
        results = predict_new_compounds(new_smiles, model_path="trained_model.pkl")
        
        if results is not None:
            print("\nResults:")
            print(results.to_string(index=False))
            
            # Save results
            results.to_csv("new_predictions.csv", index=False)
            print("\nResults saved to: new_predictions.csv")
            
    except FileNotFoundError:
        print("\nError: trained_model.pkl not found!")
        print("Please run main.py first to train the model.")
        print("Then modify main.py to save the model by adding:")
        print("  from example_predict import save_trained_model")
        print("  save_trained_model(final_model, 'trained_model.pkl')")


if __name__ == "__main__":
    main()
