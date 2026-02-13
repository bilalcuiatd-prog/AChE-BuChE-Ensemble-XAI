"""
Feature engineering utilities for molecular fingerprints and descriptors
"""

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, MACCSkeys

from config import (
    N_BITS, RADIUS, USE_MACCS, USE_RDKitFP, USE_MORGAN_COUNTS,
    MORGAN_COUNT_BITS, DESC_NAMES
)


def _bv_to_np(bv):
    """
    Convert RDKit bit vector to numpy array
    """
    arr = np.zeros((bv.GetNumBits(),), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def morgan_bits(smiles_std: str, radius=RADIUS, n_bits=N_BITS):
    """
    Generate Morgan fingerprint as bit vector
    
    Args:
        smiles_std: Standardized SMILES string
        radius: Morgan fingerprint radius
        n_bits: Number of bits
        
    Returns:
        Tuple of (numpy array of bits, bitInfo dict)
    """
    m = Chem.MolFromSmiles(smiles_std)
    if m is None:
        return np.zeros(n_bits, dtype=np.float32), {}
    
    bitInfo = {}
    bv = AllChem.GetMorganFingerprintAsBitVect(
        m, radius, nBits=n_bits, bitInfo=bitInfo)
    return _bv_to_np(bv), bitInfo


def morgan_counts(smiles_std: str, radius=RADIUS, n_bits=MORGAN_COUNT_BITS):
    """
    Generate Morgan fingerprint as count vector (log1p transformed)
    
    Args:
        smiles_std: Standardized SMILES string
        radius: Morgan fingerprint radius
        n_bits: Number of bits
        
    Returns:
        Numpy array of log1p transformed counts
    """
    m = Chem.MolFromSmiles(smiles_std)
    if m is None:
        return np.zeros(n_bits, dtype=np.float32)
    
    fp = AllChem.GetHashedMorganFingerprint(m, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    for k, v in fp.GetNonzeroElements().items():
        arr[int(k)] = float(v)
    
    # log1p to stabilize counts
    return np.log1p(arr).astype(np.float32)


def rdkitfp_bits(smiles_std: str, n_bits=2048):
    """
    Generate RDKit fingerprint as bit vector
    
    Args:
        smiles_std: Standardized SMILES string
        n_bits: Number of bits
        
    Returns:
        Numpy array of bits
    """
    m = Chem.MolFromSmiles(smiles_std)
    if m is None:
        return np.zeros(n_bits, dtype=np.float32)
    
    bv = Chem.RDKFingerprint(m, fpSize=n_bits)
    return _bv_to_np(bv)


def maccs_bits(smiles_std: str):
    """
    Generate MACCS keys fingerprint
    
    Args:
        smiles_std: Standardized SMILES string
        
    Returns:
        Numpy array of 167 MACCS keys
    """
    m = Chem.MolFromSmiles(smiles_std)
    if m is None:
        return np.zeros(167, dtype=np.float32)
    
    bv = MACCSkeys.GenMACCSKeys(m)
    return _bv_to_np(bv)


def physchem_desc_13(smiles_std: str):
    """
    Calculate 13 physicochemical descriptors
    
    Args:
        smiles_std: Standardized SMILES string
        
    Returns:
        Numpy array of 13 descriptors
    """
    m = Chem.MolFromSmiles(smiles_std)
    if m is None:
        return np.zeros(13, dtype=np.float32)
    
    return np.array([
        Descriptors.MolWt(m),
        Descriptors.TPSA(m),
        Descriptors.MolLogP(m),
        Descriptors.NumHDonors(m),
        Descriptors.NumHAcceptors(m),
        Descriptors.NumRotatableBonds(m),
        Descriptors.HeavyAtomCount(m),
        Descriptors.RingCount(m),
        Descriptors.NumAromaticRings(m),
        Descriptors.NumSaturatedRings(m),
        Descriptors.NumAliphaticRings(m),
        Descriptors.FractionCSP3(m),
        Descriptors.HallKierAlpha(m),
    ], dtype=np.float32)


def build_features_and_bitinfo(smiles_std_list):
    """
    Build complete feature matrix from SMILES list
    
    Combines:
    - Morgan fingerprint (bits)
    - Morgan fingerprint (counts) - optional
    - RDKit fingerprint - optional
    - MACCS keys - optional
    - Physicochemical descriptors (13)
    
    Args:
        smiles_std_list: List of standardized SMILES strings
        
    Returns:
        Tuple of (feature matrix, list of bitInfo dicts)
    """
    fps_bits, bitinfos = [], []
    fps_counts, fps_rd, fps_maccs, descs = [], [], [], []
    
    print(f"Building features for {len(smiles_std_list)} compounds...")
    
    for i, smi in enumerate(smiles_std_list):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(smiles_std_list)} compounds")
        
        # Morgan bits (always included)
        b, bi = morgan_bits(smi)
        fps_bits.append(b)
        bitinfos.append(bi)
        
        # Optional features
        if USE_MORGAN_COUNTS:
            fps_counts.append(morgan_counts(smi))
        if USE_RDKitFP:
            fps_rd.append(rdkitfp_bits(smi))
        if USE_MACCS:
            fps_maccs.append(maccs_bits(smi))
        
        # Physicochemical descriptors (always included)
        descs.append(physchem_desc_13(smi))
    
    # Concatenate all features
    X = [np.stack(fps_bits).astype(np.float32)]
    
    if USE_MORGAN_COUNTS:
        X.append(np.stack(fps_counts).astype(np.float32))
    if USE_RDKitFP:
        X.append(np.stack(fps_rd).astype(np.float32))
    if USE_MACCS:
        X.append(np.stack(fps_maccs).astype(np.float32))
    
    X.append(np.stack(descs).astype(np.float32))
    
    feature_matrix = np.concatenate(X, axis=1).astype(np.float32)
    print(f"Final feature matrix shape: {feature_matrix.shape}")
    
    return feature_matrix, bitinfos


def get_feature_names():
    """
    Generate feature names for all features
    
    Returns:
        List of feature names
    """
    names = [f"ECFP6_bit_{i}" for i in range(N_BITS)]
    
    if USE_MORGAN_COUNTS:
        names += [f"ECFP6_count_{i}" for i in range(MORGAN_COUNT_BITS)]
    if USE_RDKitFP:
        names += [f"RDKFP_{i}" for i in range(2048)]
    if USE_MACCS:
        names += [f"MACCS_{i}" for i in range(167)]
    
    names += DESC_NAMES
    
    return names
