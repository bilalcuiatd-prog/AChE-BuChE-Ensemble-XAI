# Contributing to AChE Inhibitor Prediction Pipeline

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Your environment (Python version, OS, etc.)
- Relevant error messages or logs

### Suggesting Enhancements

Feature requests are welcome! Please open an issue with:
- A clear description of the feature
- Use cases and benefits
- Potential implementation approach (if you have ideas)

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes**:
   - Follow the existing code style
   - Add docstrings to new functions
   - Update documentation if needed
3. **Test your changes**:
   - Ensure the pipeline runs without errors
   - Add tests for new functionality if applicable
4. **Update the README** if you've added features
5. **Submit a pull request** with:
   - Clear description of changes
   - Reference to related issues
   - Screenshots/examples if relevant

## Development Setup

1. Clone your fork:
```bash
git clone https://github.com/yourusername/ache-inhibitor-prediction.git
cd ache-inhibitor-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to all functions (Google style preferred)
- Keep functions focused and modular
- Add type hints where appropriate

Example:
```python
def calculate_similarity(smiles1: str, smiles2: str) -> float:
    """
    Calculate Tanimoto similarity between two molecules
    
    Args:
        smiles1: First SMILES string
        smiles2: Second SMILES string
        
    Returns:
        Tanimoto similarity score (0-1)
    """
    # Implementation
    pass
```

## Testing

While we don't currently have a formal test suite, please:
- Test your changes on a small dataset
- Verify that the pipeline completes without errors
- Check that outputs are reasonable

Future contributions of unit tests are highly welcome!

## Areas for Contribution

We welcome contributions in:

### Code Enhancements
- Additional base models (XGBoost, Neural Networks, etc.)
- Alternative feature representations
- Hyperparameter optimization
- Performance improvements

### Documentation
- Tutorial notebooks
- Additional examples
- API documentation
- Improved explanations

### Testing
- Unit tests for individual modules
- Integration tests for the pipeline
- Test datasets

### Features
- Model serialization/deployment tools
- Web interface or REST API
- Support for additional file formats
- Multi-target prediction

### Bug Fixes
- Data loading edge cases
- Memory optimization
- Error handling improvements

## Commit Messages

Write clear commit messages:
- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and PRs when relevant

Example:
```
Add SHAP waterfall plots for individual predictions

- Implement waterfall plot generation function
- Add option to save plots to outputs/
- Update README with new visualization options

Fixes #42
```

## Documentation

Update documentation for:
- New features (README.md)
- Configuration changes (config.py comments)
- API changes (docstrings)
- Breaking changes (CHANGELOG if we add one)

## Questions?

Feel free to:
- Open an issue for questions
- Reach out via [contact method]
- Join discussions on existing issues/PRs

## Code of Conduct

Be respectful and constructive in all interactions. We aim to maintain a welcoming environment for contributors of all backgrounds and skill levels.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be acknowledged in the README and/or a CONTRIBUTORS file.

Thank you for helping improve this project! 🎉
