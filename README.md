# DiGSA (Dictionary-Guided Sparse Attack)

DiGSA (Dictionary-Guided Sparse Attack) generates sparse, low-ℓ₂ adversarial examples using feature maps and dictionary learning. It creates fast, one-shot perturbations that highlight key image regions, achieving competitive results with reduced computation and strong resistance against defense strategies.

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
- Large computations (training, dictionary learning, saving many images) may take time and disk space.
- You may want to run fewer epochs or smaller samples during testing.
