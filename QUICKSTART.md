# Quick Start Guide

Get up and running with the AChE Inhibitor Prediction Pipeline in 5 minutes!

## Prerequisites

- Python 3.8+
- pip or conda

## Installation (2 minutes)

```bash
# Clone the repository
git clone https://github.com/yourusername/ache-inhibitor-prediction.git
cd ache-inhibitor-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install RDKit (if not already installed)
conda install -c conda-forge rdkit  # OR: pip install rdkit-pypi
```

## Data Setup (1 minute)

Place your data in the `data/` directory:

1. **Training data** → `data/Acetylcholinesterase_5.csv`
   ```csv
   canonical_smiles,class
   CCO,active
   CCC,inactive
   ...
   ```

2. **External validation** (optional) → `data/bilalcuiatd_gmail.com9289.tsv`

## Configuration (1 minute)

Edit `config.py` to set your data paths:

```python
TRAIN_CSV_PATH = "data/Acetylcholinesterase_5.csv"
BINDINGDB_TSV_PATH = "data/bilalcuiatd_gmail.com9289.tsv"
```

## Run Pipeline (1 minute)

```bash
python main.py
```

That's it! The pipeline will:
- ✅ Load and standardize your data
- ✅ Generate molecular features
- ✅ Run nested cross-validation
- ✅ Train the final model
- ✅ Evaluate on external data
- ✅ Save results to `outputs/`

## View Results

Check the `outputs/` directory for:
- `cv_summary.csv` - Cross-validation metrics
- `external_predictions.csv` - External predictions
- `*.png` - Visualization plots

## Making Predictions on New Compounds

```python
from example_predict import predict_new_compounds

# Your new compounds
smiles = [
    "CC(C)Cc1ccc(cc1)C(C)C(O)=O",  # Ibuprofen
    "CC(=O)Oc1ccccc1C(=O)O",        # Aspirin
]

# Get predictions (requires trained model)
results = predict_new_compounds(smiles, model_path="trained_model.pkl")
print(results)
```

## Common Customizations

### Adjust Cross-Validation Folds

In `config.py`:
```python
OUTER_FOLDS = 5  # Change to 3 for faster runs
```

### Disable Certain Features

In `config.py`:
```python
USE_MACCS = False  # Skip MACCS keys
USE_RDKitFP = False  # Skip RDKit fingerprint
```

### Enable SHAP Explainability

In `config.py`:
```python
RUN_SHAP = True  # Generate feature importance plots
```

## Troubleshooting

**RDKit import error?**
```bash
conda install -c conda-forge rdkit
```

**Out of memory?**
```python
# In config.py, reduce feature dimensions:
N_BITS = 2048  # Instead of 4096
USE_MORGAN_COUNTS = False
```

**Slow training?**
```python
# In config.py:
OUTER_FOLDS = 3  # Instead of 5
LGBM_PARAMS['n_estimators'] = 5000  # Instead of 20000
```

## Next Steps

- 📖 Read the full [README.md](README.md) for detailed documentation
- 🔧 Customize hyperparameters in `config.py`
- 🧪 Run `example_predict.py` for prediction examples
- 🤝 Check [CONTRIBUTING.md](CONTRIBUTING.md) to contribute

## Support

- 🐛 Found a bug? [Open an issue](https://github.com/yourusername/ache-inhibitor-prediction/issues)
- 💡 Have a question? [Start a discussion](https://github.com/yourusername/ache-inhibitor-prediction/discussions)
- 📧 Email: your.email@example.com

Happy predicting! 🎉
