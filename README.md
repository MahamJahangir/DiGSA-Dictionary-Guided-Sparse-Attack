# DiGSA (Dictionary-Guided Sparse Attack)

This repository contains a refactored version of a MNIST-based adversarial pipeline.
The original single-file script was split into modular Python files without changing
the underlying logic or algorithms.

## Contents
- `data.py` - MNIST loading and preprocessing functions
- `model.py` - Model build, train, save/load utilities
- `sampling.py` - Digit dictionary creation and sampling utilities
- `activations.py` - Activation map extraction and loading
- `dictionary.py` - Dictionary learning and sparse reconstruction utilities
- `sparse_save.py` - Utilities to save sparse visualizations
- `attack.py` - Adversarial example creation and evaluation utilities
- `main.py` - Orchestrator that runs the full workflow
- `requirements.txt` - Python packages used
- `.gitignore` - common ignores

## Usage
1. Create a Python environment (recommended: python 3.8+)
2. Install dependencies: `pip install -r requirements.txt`
3. Run the full pipeline: `python main.py`

**Notes**
- The code is a direct refactor; it preserves behavior from the single-file version.
- Large computations (training, dictionary learning, saving many images) may take time and disk space.
- You may want to run fewer epochs or smaller samples during testing.
