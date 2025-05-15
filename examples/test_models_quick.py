#!/usr/bin/env python3
"""Quick test to verify model training works correctly"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from ctf.framework import CausalTransparencyFramework


def test_models_quick():
    """Quick test of model training with minimal data"""
    print("Quick test of CTF model training...")
    
    # Create small synthetic dataset
    np.random.seed(42)
    n_samples = 100
    
    # Create simple features
    data = pd.DataFrame({
        'age': np.random.normal(50, 10, n_samples),
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'target': np.random.binomial(1, 0.3, n_samples)
    })
    
    # Save to temp file
    temp_data_path = '/tmp/test_data.csv'
    data.to_csv(temp_data_path, index=False)
    
    # Initialize CTF
    ctf = CausalTransparencyFramework(
        data_path=temp_data_path,
        target_col="target",
        output_dir="/tmp/test_ctf",
        random_state=42
    )
    
    # Simple domain knowledge
    domain_knowledge = {
        "edges": [
            ["age", "target", 0.8],
            ["feature1", "target", 0.6]
        ]
    }
    
    # Test causal discovery
    print("\n1. Testing causal discovery...")
    ctf.discover_causal_structure(domain_knowledge=domain_knowledge)
    print(f"   Causal graph: {len(ctf.causal_graph.nodes())} nodes, {len(ctf.causal_graph.edges())} edges")
    
    # Test model training
    print("\n2. Testing model training...")
    models = ctf.train_models(dnn_epochs=5, verbose=0)
    
    print("\n3. Trained models:")
    for model_name in ctf.models.keys():
        if 'features' not in model_name:
            print(f"   - {model_name}")
    
    # Check DNN models
    dnn_models = [name for name in ctf.models.keys() if 'dnn' in name and 'features' not in name]
    if dnn_models:
        print(f"\n✓ SUCCESS: DNN models trained: {dnn_models}")
    else:
        print("\n✗ WARNING: No DNN models found")
    
    # Clean up
    os.remove(temp_data_path)
    
    return ctf


if __name__ == "__main__":
    ctf = test_models_quick()
    print("\nTest completed!")