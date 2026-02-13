# Acetylcholinesterase Inhibitor Prediction Pipeline

A robust machine learning pipeline for predicting acetylcholinesterase (AChE) inhibitors using ensemble learning and scaffold-based cross-validation.

## Features

- **Comprehensive Molecular Featurization**:
  - Morgan (ECFP6) fingerprints (bits and counts)
  - RDKit fingerprints
  - MACCS keys
  - 13 physicochemical descriptors

- **Rigorous Validation**:
  - Nested scaffold-based cross-validation
  - Prevents data leakage via InChIKey and scaffold overlap checks
  - External validation on BindingDB data

- **Ensemble Learning**:
  - LightGBM with early stopping
  - Random Forest
  - Extra Trees
  - SVD + Logistic Regression
  - Meta-learner (calibrated stacking)

- **Advanced Features**:
  - Automatic threshold optimization (balanced accuracy & accuracy)
  - Applicability domain assessment
  - Optional SHAP explainability
  - Tautomer standardization and molecular cleaning

## Project Structure

```
.
├── main.py                 # Main execution script
├── config.py              # Configuration and hyperparameters
├── data_utils.py          # Data loading and preprocessing
├── feature_utils.py       # Molecular feature engineering
├── model_utils.py         # Model training and CV utilities
├── evaluation.py          # Metrics and visualization
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── data/                 # Data directory (create this)
│   ├── Acetylcholinesterase_5.csv
│   └── bilalcuiatd_gmail.com9289.tsv
└── outputs/              # Results directory (auto-created)
    ├── cv_fold_metrics.csv
    ├── cv_summary.csv
    ├── external_predictions.csv
    └── *.png (plots)
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Setup

1. Clone this repository:

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install RDKit (if not included via pip):
```bash
conda install -c conda-forge rdkit  # Using conda
# OR
pip install rdkit-pypi  # Using pip
```

## Usage

### Data Preparation

1. Create a `data/` directory in the project root
2. Place your training data as `data/Acetylcholinesterase_5.csv` with columns:
   - `canonical_smiles`: SMILES strings
   - `class`: Activity labels (active/inactive or 1/0)

3. (Optional) Place BindingDB external validation data as `data/bilalcuiatd_gmail.com9289.tsv`

### Configuration

Edit `config.py` to customize:

- **Paths**: Update `TRAIN_CSV_PATH` and `BINDINGDB_TSV_PATH`
- **Cross-validation**: Adjust `OUTER_FOLDS`, `INNER_FOLDS_OOF`, etc.
- **Features**: Toggle `USE_MACCS`, `USE_RDKitFP`, `USE_MORGAN_COUNTS`
- **External labeling**: Set `KI_ACTIVE_MAX` and `KI_INACTIVE_MIN`
- **Applicability domain**: Adjust `AD_SIM_THRESHOLD`
- **SHAP**: Set `RUN_SHAP = True` for explainability analysis

### Running the Pipeline

Execute the main script:
```bash
python main.py
```

The pipeline will:
1. Load and standardize molecular data
2. Generate molecular features
3. Run nested scaffold-based cross-validation
4. Train final model on full dataset
5. Evaluate on external BindingDB data (if available)
6. Generate comprehensive reports and visualizations

## Troubleshooting

### Import Errors

If you encounter RDKit import errors:
```bash
# Using conda (recommended for RDKit)
conda install -c conda-forge rdkit

# Or using pip
pip install rdkit-pypi
```

### Memory Issues

For large datasets:
- Reduce `N_BITS` in `config.py`
- Disable `USE_MORGAN_COUNTS` or `USE_RDKitFP`
- Use fewer base models

### Slow Training

- Reduce `n_estimators` in LGBM/RF/ETC parameters
- Reduce `OUTER_FOLDS` and `INNER_FOLDS_OOF`
- Disable SHAP analysis

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Contact

For questions or issues, please open an issue on GitHub or contact [bilalcuiatd@gmail.com]

## Acknowledgments

- RDKit team for molecular informatics toolkit
- scikit-learn for machine learning utilities
- LightGBM team for gradient boosting implementation
- SHAP team for explainability tools

