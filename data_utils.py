"""
Data utilities for loading, cleaning, and standardizing molecular data
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdinchi import MolToInchiKey
from rdkit.Chem.Scaffolds import MurckoScaffold


def _std_mol(m):
    """
    Standardize molecule: largest fragment, uncharge, canonicalize tautomer
    """
    if m is None:
        return None
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize
        m = rdMolStandardize.LargestFragmentChooser().choose(m)
        m = rdMolStandardize.Uncharger().uncharge(m)
        te = rdMolStandardize.TautomerEnumerator()
        m = te.Canonicalize(m)
        Chem.SanitizeMol(m)
        return m
    except Exception:
        try:
            Chem.SanitizeMol(m)
            return m
        except Exception:
            return None


def sanitize_smiles(smi: str):
    """
    Convert SMILES to standardized canonical SMILES
    """
    try:
        m = Chem.MolFromSmiles(str(smi))
        m = _std_mol(m)
        if m is None:
            return None
        return Chem.MolToSmiles(m, canonical=True, isomericSmiles=False)
    except Exception:
        return None


def to_inchikey(smiles_std: str):
    """
    Convert standardized SMILES to InChIKey
    """
    try:
        m = Chem.MolFromSmiles(smiles_std)
        return MolToInchiKey(m) if m is not None else None
    except Exception:
        return None


def get_scaffold(smiles_std: str) -> str:
    """
    Extract Murcko scaffold from standardized SMILES
    """
    m = Chem.MolFromSmiles(smiles_std)
    if m is None:
        return "NONE"
    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
        return scaf if scaf else "NONE"
    except Exception:
        return "NONE"


def map_class(x: str) -> int:
    """
    Map class labels to binary (1=active, 0=inactive)
    """
    s = str(x).strip().lower()
    return 1 if s in {"active", "1", "true", "yes", "y"} else 0


def load_and_prepare_train_data(csv_path: str):
    """
    Load and prepare training data with standardization and deduplication
    
    Args:
        csv_path: Path to training CSV file
        
    Returns:
        DataFrame with standardized SMILES, labels, InChIKeys, and scaffolds
    """
    print(f"\nLoading training data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Check required columns
    required_cols = ["canonical_smiles", "class"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found in dataset.")
    
    # Map class labels
    df["is_active"] = df["class"].map(map_class).astype(int)
    
    print("Training label balance:")
    print(df["is_active"].value_counts())
    
    # Standardize SMILES
    print("\nStandardizing SMILES...")
    df["smiles_std"] = df["canonical_smiles"].map(sanitize_smiles)
    df = df.dropna(subset=["smiles_std"]).reset_index(drop=True)
    
    # Generate InChIKeys
    df["inchikey"] = df["smiles_std"].map(to_inchikey)
    df = df.dropna(subset=["inchikey"]).reset_index(drop=True)
    
    # Deduplicate by InChIKey
    df = df.drop_duplicates(subset=["inchikey"]).reset_index(drop=True)
    
    # Extract scaffolds
    df["scaffold"] = df["smiles_std"].map(get_scaffold)
    
    print(f"Training data after standardization and deduplication: {len(df)} compounds")
    
    return df


def load_and_prepare_external_data(tsv_path: str, ki_active_max: float, 
                                   ki_inactive_min: float, train_inchikeys: set,
                                   train_scaffolds: set, sample_active: int = 100,
                                   sample_inactive: int = 75, random_state: int = 42):
    """
    Load and prepare external BindingDB data
    
    Args:
        tsv_path: Path to BindingDB TSV file
        ki_active_max: Maximum Ki (nM) for active label
        ki_inactive_min: Minimum Ki (nM) for inactive label
        train_inchikeys: Set of InChIKeys from training data (to avoid overlap)
        train_scaffolds: Set of scaffolds from training data (to avoid overlap)
        sample_active: Number of active compounds to sample
        sample_inactive: Number of inactive compounds to sample
        random_state: Random seed
        
    Returns:
        DataFrame with external validation data
    """
    print(f"\nLoading BindingDB data from: {tsv_path}")
    
    def _to_float(x):
        try:
            s = str(x).strip()
            if s == "" or s.lower() in {"nan", "none"}:
                return np.nan
            return float(s)
        except Exception:
            return np.nan
    
    ext = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    
    # Filter for acetylcholinesterase
    if "Target Name" in ext.columns:
        ext = ext[ext["Target Name"].astype(str).str.lower().str.contains(
            "acetylcholinesterase", na=False)].copy()
    print(f"BindingDB rows after target filter: {len(ext)}")
    
    # Check required columns
    if "Ligand SMILES" not in ext.columns or "Ki (nM)" not in ext.columns:
        raise ValueError("BindingDB file must contain 'Ligand SMILES' and 'Ki (nM)' columns.")
    
    # Parse Ki values
    ext["Ki_nM"] = ext["Ki (nM)"].map(_to_float)
    ext = ext.dropna(subset=["Ligand SMILES", "Ki_nM"]).copy()
    
    # Label based on Ki values
    ext["y_ext"] = np.where(ext["Ki_nM"] <= ki_active_max, 1,
                     np.where(ext["Ki_nM"] >= ki_inactive_min, 0, np.nan))
    ext = ext.dropna(subset=["y_ext"]).copy()
    ext["y_ext"] = ext["y_ext"].astype(int)
    
    print(f"\nExternal labeled rows (Ki <= {ki_active_max} active, "
          f"Ki >= {ki_inactive_min} inactive): {len(ext)}")
    print("External label balance:")
    print(ext["y_ext"].value_counts())
    
    # Standardize SMILES
    ext["smiles_std"] = ext["Ligand SMILES"].map(sanitize_smiles)
    ext = ext.dropna(subset=["smiles_std"]).copy()
    ext["inchikey"] = ext["smiles_std"].map(to_inchikey)
    ext = ext.dropna(subset=["inchikey"]).copy()
    ext["scaffold"] = ext["smiles_std"].map(get_scaffold)
    
    # Remove overlaps with training data
    before = len(ext)
    ext = ext[~ext["inchikey"].isin(train_inchikeys)].copy()
    after_inchi = len(ext)
    ext = ext[~ext["scaffold"].isin(train_scaffolds)].copy()
    after_scaf = len(ext)
    
    print(f"\nExternal after removing training overlaps: {after_scaf} | "
          f"overlaps removed: {before - after_scaf} "
          f"(InChIKey: {before - after_inchi}, Scaffold: {after_inchi - after_scaf})")
    
    # Sample balanced subset
    act = ext[ext["y_ext"] == 1].copy()
    ina = ext[ext["y_ext"] == 0].copy()
    n_a = min(sample_active, len(act))
    n_i = min(sample_inactive, len(ina))
    
    act_s = act.sample(n=n_a, random_state=random_state) if n_a > 0 else act
    ina_s = ina.sample(n=n_i, random_state=random_state) if n_i > 0 else ina
    ext_s = pd.concat([act_s, ina_s], axis=0).sample(
        frac=1.0, random_state=random_state).reset_index(drop=True)
    
    print(f"\nExternal sampled set: active={len(act_s)} "
          f"inactive={len(ina_s)} total={len(ext_s)}")
    
    return ext_s


def assert_no_inchikey_overlap(train_idx, test_idx, inchikeys):
    """
    Check for InChIKey leakage between train and test sets
    """
    tr = set(inchikeys[train_idx])
    te = set(inchikeys[test_idx])
    inter = tr.intersection(te)
    if inter:
        raise RuntimeError(f"Leakage: {len(inter)} InChIKey overlaps in train/test!")


def assert_no_scaffold_overlap(train_idx, test_idx, scaffolds):
    """
    Check for scaffold leakage between train and test sets
    """
    tr = set(scaffolds[train_idx])
    te = set(scaffolds[test_idx])
    inter = tr.intersection(te)
    if inter:
        raise RuntimeError(f"Leakage: {len(inter)} scaffold overlaps in train/test!")
