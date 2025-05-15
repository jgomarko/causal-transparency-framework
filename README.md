# Causal Transparency Framework (CTF)

![CTF Framework Overview](Image%20May%208%2C%202025%20at%2008_08_00%20PM.png)

## Overview

The Causal Transparency Framework (CTF) is a comprehensive methodology for evaluating and enhancing the transparency of machine learning models through causal reasoning. This framework combines causal discovery, model evaluation, and counterfactual analysis to provide structured insights into model behavior and decision-making processes.

## Key Components

### 1. Causal Discovery

The framework implements multiple causal discovery methods:
- Correlation-based analysis
- Constraint-based algorithms (PC)
- Score-based algorithms (Hill Climbing)
- Domain-specific approaches (GNN-based)
- Ensemble integration of multiple methods

### 2. Transparency Metrics

CTF evaluates model transparency through four key metrics:

- **Causal Influence Index (CII)**: Measures how strongly each feature causally influences model predictions
- **Causal Complexity Measure (CCM)**: Quantifies the complexity of the causal model
- **Transparency Entropy (TE)**: Measures model interpretability and decision clarity
- **Counterfactual Stability (CS)**: Tests model robustness to perturbations and fairness

### 3. Model Evaluation

The framework supports evaluation of multiple model types:
- Causal models (using only causally relevant features)
- Standard machine learning models (logistic regression, random forests, etc.)
- XGBoost and neural network models
- Comparative analysis across model architectures

### 4. Visualization & Reporting

Comprehensive visualization suite including:
- Causal graph visualizations
- Radar charts for CTF metrics
- Feature importance plots
- Model performance comparisons
- Interactive dashboards

## Applications

The CTF has been applied to:

1. **Clinical Decision Support (MIMIC-III)**
   - Mortality prediction
   - Treatment recommendation
   - Risk stratification

2. **Criminal Justice (COMPAS)**
   - Recidivism prediction
   - Fairness evaluation
   - Bias mitigation

## Features

- **Causal Discovery**: Automated discovery of causal relationships in data
- **Model Training**: Train both causal and standard ML models
  - Supports Logistic Regression, Random Forest, XGBoost
  - **NEW**: Deep Neural Network (DNN) support with TensorFlow/Keras
- **Transparency Metrics**: Calculate CII, CCM, TE, and CS metrics
- **Visualization**: Generate comprehensive visualizations and reports
- **Model Comparison**: Compare traditional ML models with DNNs
- **Extensibility**: Easy to extend with new models and metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/causal-transparency-framework.git
cd causal-transparency-framework

# Install dependencies
pip install -r requirements.txt

# Install as a package (optional)
pip install -e .
```

## Usage

### Basic Example

```python
from ctf.framework import CausalTransparencyFramework

# Initialize the framework
ctf = CausalTransparencyFramework(
    data_path="your_data.csv",
    target_col="target_variable",
    output_dir="./results"
)

# Discover causal structure
ctf.discover_causal_structure()

# Train models including DNNs
models = ctf.train_models(
    dnn_epochs=50,       # Number of epochs for DNN training
    dnn_batch_size=32,   # Batch size for DNN training
    verbose=1            # Show training progress
)

# Calculate transparency metrics
metrics = ctf.calculate_transparency_metrics()

# Generate report
report_path = ctf.generate_report()
```

### Example Notebooks

- [MIMIC-III Example](examples/mimic_iii_example.ipynb): Demonstrates CTF application to clinical mortality prediction
- [COMPAS Example](examples/compas_example.ipynb): Demonstrates CTF application to criminal justice with fairness analysis

## Directory Structure

```
ctf/
├── causal_discovery.py      # Causal structure learning algorithms
├── transparency_metrics.py  # CTF metrics implementation
├── framework.py            # Main CTF implementation (with DNN support)
├── framework_enhanced.py   # Enhanced framework with advanced DNN features
├── __init__.py             # Package initialization
examples/
├── mimic_iii_example.ipynb  # Example for clinical data
├── compas_example.ipynb    # Example for criminal justice
├── test_dnn_support.py     # Test script for DNN functionality
data/                      # Sample datasets
docs/
├── dnn_support.md         # Documentation for DNN features
tests/                     # Unit tests
```

## Citation

If you use this framework in your research, please cite:

```
@article{CTF2025,
  title={The Causal Transparency Framework: A Multi-Metric Approach to Algorithmic Accountability},
  author={Marko, John Gabriel O. and Neagu, Ciprian Daniel and Anand, P.B},
  journal={},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This research was supported by University of Bradford 
- Special thanks to contributors and collaborators
