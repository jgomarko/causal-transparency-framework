# DNN Support in CTF Framework

## Overview

The Causal Transparency Framework (CTF) has been enhanced to support Deep Neural Networks (DNNs) alongside traditional machine learning models. This enhancement enables comparison between interpretable models (like Random Forest and XGBoost) and more complex neural networks in terms of both performance and transparency metrics.

## New Features

### 1. DNN Model Integration

The framework now supports:
- Causal DNNs: Neural networks trained only on causal features
- Full DNNs: Neural networks trained on all available features
- Automatic scaling of features for neural network training
- Configurable architecture (hidden layers, activation functions, dropout)

### 2. Enhanced Model Training

The `train_models()` method now accepts additional parameters:
- `dnn_epochs`: Number of epochs for DNN training (default: 50)
- `dnn_batch_size`: Batch size for DNN training (default: 32)
- `verbose`: Verbosity level for training progress (default: 0)

### 3. Architecture Configuration

DNNs are built with:
- Configurable hidden layers (default: [64, 32])
- ReLU activation functions
- Dropout regularization (default: 0.3)
- Sigmoid output for binary classification
- Adam optimizer with binary crossentropy loss

## Usage

### Basic Example

```python
from ctf.framework import CausalTransparencyFramework

# Initialize CTF
ctf = CausalTransparencyFramework(
    data_path="data.csv",
    target_col="target",
    output_dir="results",
    random_state=42
)

# Train models including DNNs
models = ctf.train_models(
    dnn_epochs=50,
    dnn_batch_size=32,
    verbose=1
)

# Calculate transparency metrics
metrics = ctf.calculate_transparency_metrics()
```

### Enhanced Framework Usage

For more advanced usage, you can use the `EnhancedCausalTransparencyFramework`:

```python
from ctf.framework_enhanced import EnhancedCausalTransparencyFramework

# Initialize enhanced framework
ctf = EnhancedCausalTransparencyFramework(
    data_path="data.csv",
    target_col="target",
    output_dir="results",
    random_state=42
)

# Configure DNN architecture
ctf._build_dnn_model(
    input_dim=10,
    hidden_layers=[128, 64, 32],
    activation='relu',
    dropout_rate=0.3
)
```

## Requirements

To use DNN support, ensure TensorFlow is installed:

```bash
pip install tensorflow>=2.0
```

The framework will automatically detect if TensorFlow is available and skip DNN models if it's not installed.

## Model Comparison

The framework now provides comprehensive comparison between:

1. **Model Types**: Traditional ML vs. DNNs
2. **Feature Sets**: Causal features vs. all features
3. **Performance Metrics**: AUC, Accuracy, F1 Score
4. **Transparency Metrics**: CII, CCM, TE, CS

## Output and Visualization

The enhanced framework generates:
- Performance comparison charts showing DNN vs traditional models
- Transparency metric comparisons
- Detailed reports distinguishing between model types
- Radar charts for each model type

## Benefits

1. **Performance Assessment**: Compare complex DNNs with interpretable models
2. **Transparency Trade-offs**: Evaluate the transparency cost of using DNNs
3. **Causal Analysis**: Assess how DNNs perform with only causal features
4. **Comprehensive Evaluation**: Get a complete picture of model capabilities

## Example Results

When running the CTF with DNN support, you'll see output like:

```
Training predictive models...
Model causal_lr: Accuracy=0.8500, AUC=0.8750, F1=0.8800
Model causal_rf: Accuracy=0.8700, AUC=0.8900, F1=0.8950
Model causal_xgb: Accuracy=0.8800, AUC=0.9000, F1=0.9050
Model causal_dnn: Accuracy=0.8600, AUC=0.8850, F1=0.8900
Model full_lr: Accuracy=0.8600, AUC=0.8800, F1=0.8850
Model full_rf: Accuracy=0.8800, AUC=0.8950, F1=0.9000
Model full_xgb: Accuracy=0.8900, AUC=0.9100, F1=0.9150
Model full_dnn: Accuracy=0.8750, AUC=0.8975, F1=0.9025
```

This allows direct comparison of DNNs with traditional models in both causal and full feature settings.

## Best Practices

1. **Feature Scaling**: The framework automatically scales features for DNNs
2. **Model Selection**: Compare DNN performance against interpretable models
3. **Transparency Trade-offs**: Use CTF metrics to evaluate if DNN complexity is justified
4. **Early Stopping**: Consider implementing early stopping for DNNs to prevent overfitting

## Future Enhancements

Potential future improvements include:
- Support for multi-class classification
- Custom DNN architectures
- Transfer learning capabilities
- Model ensemble methods
- Explainability tools for DNNs (SHAP, LIME integration)

## Citations

If you use the DNN-enhanced CTF framework in your research, please cite:

```bibtex
@software{ctf_dnn,
  title={Causal Transparency Framework with Deep Neural Network Support},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/ctf}
}
```