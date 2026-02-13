# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-02-13

### Added
- Initial release of AChE Inhibitor Prediction Pipeline
- Comprehensive molecular featurization (Morgan, RDKit, MACCS, descriptors)
- Nested scaffold-based cross-validation
- Ensemble learning with 4 base models + meta-learner
- External validation on BindingDB data
- Applicability domain assessment
- Automatic threshold optimization
- SHAP explainability support
- Complete documentation and examples

### Features
- `config.py`: Centralized configuration management
- `data_utils.py`: Data loading and preprocessing utilities
- `feature_utils.py`: Molecular feature engineering
- `model_utils.py`: Model training and CV strategies
- `evaluation.py`: Metrics and visualization tools
- `main.py`: Complete pipeline execution
- `example_predict.py`: Example prediction script

### Documentation
- Comprehensive README with installation and usage instructions
- CONTRIBUTING.md for contribution guidelines
- MIT LICENSE
- Example scripts and notebooks

## [Unreleased]

### Planned
- Web interface for predictions
- Additional base models (XGBoost, Neural Networks)
- Batch prediction API
- Unit tests and CI/CD
- Docker container for easy deployment
- Pre-trained models for download

---

## Version History

- **1.0.0** (2024-02-13) - Initial public release
