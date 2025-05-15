#!/usr/bin/env python3
"""Test script to verify DNN support in the CTF framework"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctf.framework import CausalTransparencyFramework


def test_dnn_support():
    """Test if DNN models are properly integrated into the framework"""
    print("Testing DNN support in CTF framework...")
    
    # Initialize CTF with MIMIC-III dataset
    ctf = CausalTransparencyFramework(
        data_path="../data/mimic_processed_for_ctf.csv",
        target_col="mortality",
        output_dir="../results/test_dnn",
        random_state=42
    )
    
    # Add domain knowledge
    domain_knowledge = {
        "edges": [
            ["age", "mortality", 0.8],
            ["sofa_score", "mortality", 0.9],
            ["lactate", "mortality", 0.7],
            ["creatinine", "mortality", 0.6]
        ]
    }
    
    # Discover causal structure
    print("\n1. Discovering causal structure...")
    ctf.discover_causal_structure(domain_knowledge=domain_knowledge)
    
    # Train models including DNNs
    print("\n2. Training models (including DNNs)...")
    models = ctf.train_models(dnn_epochs=10, verbose=1)  # Use fewer epochs for testing
    
    # Check which models were trained
    print("\n3. Trained models:")
    for model_name in ctf.models.keys():
        if 'features' not in model_name:
            print(f"   - {model_name}")
    
    # Calculate transparency metrics
    print("\n4. Calculating transparency metrics...")
    metrics = ctf.calculate_transparency_metrics()
    
    # Generate report
    print("\n5. Generating report...")
    report_path = ctf.generate_report()
    
    # Summary
    print("\n=== Test Summary ===")
    
    # Check if DNN models were trained
    dnn_models = [name for name in ctf.models.keys() if 'dnn' in name and 'features' not in name]
    if dnn_models:
        print(f"✓ DNN models successfully trained: {dnn_models}")
        
        # Show performance
        for model_name in dnn_models:
            if model_name in ctf.model_performance:
                perf = ctf.model_performance[model_name]
                print(f"  {model_name}: AUC={perf['auc']:.4f}, Accuracy={perf['accuracy']:.4f}")
        
        print(f"\n✓ Results saved to: {ctf.output_dir}")
        print(f"✓ Report available at: {report_path}")
    else:
        print("✗ No DNN models found - TensorFlow may not be installed")
        print("  Install with: pip install tensorflow")
    
    return ctf


if __name__ == "__main__":
    test_dnn_support()