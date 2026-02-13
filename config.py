"""
Configuration file for Acetylcholinesterase Inhibitor Prediction Pipeline
"""

import os

# ==================== PATHS ====================
# Update these paths to match your local setup
TRAIN_CSV_PATH = "data/Acetylcholinesterase_5.csv"
BINDINGDB_TSV_PATH = "data/bilalcuiatd_gmail.com9289.tsv"
OUTPUT_DIR = "outputs"

# ==================== RANDOM STATE ====================
RANDOM_STATE = 42

# ==================== CROSS-VALIDATION PARAMETERS ====================
OUTER_FOLDS = 5
INNER_FOLDS_OOF = 5
INNER_FOLDS_THR = 3
CALIBRATION_CV = 3

# ==================== FEATURE PARAMETERS ====================
# Morgan Fingerprint
N_BITS = 4096
RADIUS = 3

# Feature flags
USE_MACCS = True
USE_RDKitFP = True
USE_MORGAN_COUNTS = True
MORGAN_COUNT_BITS = 2048

# ==================== EXTERNAL LABELING ====================
# Choose STRICT for easier, cleaner external validation
STRICT_EXTERNAL = True

if STRICT_EXTERNAL:
    KI_ACTIVE_MAX = 100.0       # <=100 nM is active
    KI_INACTIVE_MIN = 100000.0  # >=100,000 nM is inactive
else:
    KI_ACTIVE_MAX = 1000.0
    KI_INACTIVE_MIN = 10000.0

# Sampling parameters for external set
EXTERNAL_SAMPLE_ACTIVE = 100
EXTERNAL_SAMPLE_INACTIVE = 75

# ==================== APPLICABILITY DOMAIN ====================
USE_APPLICABILITY_DOMAIN_REPORT = True
AD_SIM_THRESHOLD = 0.35

# ==================== XAI (SHAP) ====================
RUN_SHAP = False

# ==================== DESCRIPTORS ====================
DESC_NAMES = [
    "MolWt", "TPSA", "MolLogP", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "HeavyAtomCount", "RingCount",
    "NumAromaticRings", "NumSaturatedRings", "NumAliphaticRings",
    "FractionCSP3", "HallKierAlpha"
]

# ==================== MODEL PARAMETERS ====================
# These will be set dynamically based on class balance
LGBM_PARAMS = {
    'n_estimators': 20000,
    'learning_rate': 0.01,
    'num_leaves': 127,
    'max_depth': 10,
    'min_child_samples': 60,
    'min_split_gain': 0.05,
    'reg_alpha': 1.5,
    'reg_lambda': 10.0,
    'colsample_bytree': 0.75,
    'subsample': 0.8,
    'subsample_freq': 1,
    'n_jobs': -1,
    'random_state': RANDOM_STATE,
}

RF_PARAMS = {
    'n_estimators': 1600,
    'max_features': 'sqrt',
    'min_samples_leaf': 2,
    'class_weight': 'balanced',
    'n_jobs': -1,
    'random_state': RANDOM_STATE,
}

ETC_PARAMS = {
    'n_estimators': 2200,
    'max_features': 'sqrt',
    'min_samples_leaf': 2,
    'class_weight': 'balanced',
    'n_jobs': -1,
    'random_state': RANDOM_STATE,
}

SVD_PARAMS = {
    'n_components': 384,
    'random_state': RANDOM_STATE,
}

LR_PARAMS = {
    'C': 0.15,
    'class_weight': 'balanced',
    'max_iter': 8000,
    'random_state': RANDOM_STATE,
}

META_LR_PARAMS = {
    'C': 0.05,
    'class_weight': 'balanced',
    'max_iter': 12000,
    'random_state': RANDOM_STATE,
}

# ==================== PLOTTING STYLE ====================
PLOT_STYLE = "seaborn-whitegrid"
PLOT_RC_PARAMS = {
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.figsize": (7, 4),
    "axes.spines.right": False,
    "axes.spines.top": False,
}
