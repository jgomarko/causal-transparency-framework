#!/usr/bin/env python3
"""Test the Enhanced Causal Transparency Framework with DNN support"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctf.framework_enhanced import EnhancedCausalTransparencyFramework


def test_enhanced_framework_mimic():
    """Test the enhanced framework on MIMIC-III dataset"""
    print("Testing Enhanced CTF on MIMIC-III dataset...")
    
    # Initialize framework
    ctf = EnhancedCausalTransparencyFramework(
        data_path="../data/mimic_processed_for_ctf.csv",
        target_col="mortality",
        output_dir="../results/mimic_enhanced",
        random_state=42
    )
    
    # Add domain knowledge for clinical data
    domain_knowledge = {
        "edges": [
            ["age", "mortality", 0.8],
            ["sofa_score", "mortality", 0.9],
            ["lactate", "mortality", 0.7],
            ["creatinine", "mortality", 0.6],
            ["age", "creatinine", 0.3],
            ["sofa_score", "lactate", 0.5]
        ]
    }
    
    # Run the full pipeline
    print("\n1. Discovering causal structure...")
    ctf.discover_causal_structure(domain_knowledge=domain_knowledge)
    
    print("\n2. Training models (including DNNs)...")
    models = ctf.train_models(dnn_epochs=30, verbose=1)
    
    print("\n3. Calculating transparency metrics...")
    metrics = ctf.calculate_transparency_metrics()
    
    print("\n4. Generating report...")
    report_path = ctf.generate_report()
    
    print(f"\nAnalysis complete! Results saved to: {ctf.output_dir}")
    print(f"Report available at: {report_path}")
    
    # Print summary comparison
    print("\n=== Model Performance Summary ===")
    for name, perf in ctf.model_performance.items():
        if 'features' not in name:
            print(f"{name}: AUC={perf['auc']:.4f}, Accuracy={perf['accuracy']:.4f}")
    
    return ctf


def test_enhanced_framework_compas():
    """Test the enhanced framework on COMPAS dataset"""
    print("Testing Enhanced CTF on COMPAS dataset...")
    
    # Initialize framework
    ctf = EnhancedCausalTransparencyFramework(
        data_path="../data/processed_compas_data.csv",
        target_col="two_year_recid",
        output_dir="../results/compas_enhanced",
        random_state=42
    )
    
    # Add domain knowledge for criminal justice
    domain_knowledge = {
        "edges": [
            ["priors_count", "two_year_recid", 0.8],
            ["age", "two_year_recid", 0.6],
            ["c_charge_degree_F", "two_year_recid", 0.4],
            ["race_African-American", "priors_count", 0.3],
            ["sex_Male", "priors_count", 0.2]
        ]
    }
    
    # Run the full pipeline
    print("\n1. Discovering causal structure...")
    ctf.discover_causal_structure(domain_knowledge=domain_knowledge)
    
    print("\n2. Training models (including DNNs)...")
    models = ctf.train_models(dnn_epochs=30, verbose=1)
    
    print("\n3. Calculating transparency metrics...")
    metrics = ctf.calculate_transparency_metrics()
    
    print("\n4. Generating report...")
    report_path = ctf.generate_report()
    
    print(f"\nAnalysis complete! Results saved to: {ctf.output_dir}")
    print(f"Report available at: {report_path}")
    
    # Print summary comparison
    print("\n=== Model Performance Summary ===")
    for name, perf in ctf.model_performance.items():
        if 'features' not in name:
            print(f"{name}: AUC={perf['auc']:.4f}, Accuracy={perf['accuracy']:.4f}")
    
    return ctf


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Enhanced CTF Framework")
    parser.add_argument("--dataset", choices=["mimic", "compas", "both"], 
                       default="mimic", help="Dataset to test on")
    
    args = parser.parse_args()
    
    if args.dataset == "mimic" or args.dataset == "both":
        test_enhanced_framework_mimic()
    
    if args.dataset == "compas" or args.dataset == "both":
        test_enhanced_framework_compas()