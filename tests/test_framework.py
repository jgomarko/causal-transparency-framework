"""
Unit tests for the Causal Transparency Framework.
"""

import unittest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil

from ctf.framework import CausalTransparencyFramework

class TestCausalTransparencyFramework(unittest.TestCase):
    """Test cases for the CausalTransparencyFramework class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for output
        self.test_dir = tempfile.mkdtemp()
        
        # Create synthetic test data
        np.random.seed(42)
        n_samples = 100
        
        # Create features with causal relationships
        age = np.random.normal(65, 10, n_samples)
        
        # SOFA score influenced by age
        sofa = 0.2 * age + np.random.normal(0, 3, n_samples)
        sofa = np.clip(sofa, 0, 24).astype(int)
        
        # Target influenced by age and SOFA
        logits = -5 + 0.02 * age + 0.3 * sofa
        p_mortality = 1 / (1 + np.exp(-logits))
        mortality = np.random.binomial(1, p_mortality)
        
        # Create DataFrame
        self.df = pd.DataFrame({
            'age': age,
            'sofa': sofa,
            'mortality': mortality
        })
        
        # Save DataFrame to temp file
        self.data_path = os.path.join(self.test_dir, 'test_data.csv')
        self.df.to_csv(self.data_path, index=False)
        
        # Domain knowledge for testing
        self.domain_knowledge = {
            "edges": [
                ["age", "mortality", 0.9],
                ["sofa", "mortality", 0.8]
            ]
        }
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_initialization_with_data(self):
        """Test that the framework initializes correctly with dataframe."""
        ctf = CausalTransparencyFramework(
            data=self.df,
            target_col="mortality",
            output_dir=os.path.join(self.test_dir, 'test_output'),
            random_state=42
        )
        
        self.assertEqual(ctf.target_col, "mortality")
        self.assertEqual(ctf.random_state, 42)
        self.assertTrue((ctf.df == self.df).all().all())
    
    def test_initialization_with_path(self):
        """Test that the framework initializes correctly with data path."""
        ctf = CausalTransparencyFramework(
            data_path=self.data_path,
            target_col="mortality",
            output_dir=os.path.join(self.test_dir, 'test_output'),
            random_state=42
        )
        
        self.assertEqual(ctf.target_col, "mortality")
        self.assertEqual(ctf.random_state, 42)
        self.assertTrue(set(ctf.df.columns) == set(self.df.columns))
        self.assertEqual(len(ctf.df), len(self.df))
    
    def test_discover_causal_structure(self):
        """Test causal structure discovery."""
        ctf = CausalTransparencyFramework(
            data=self.df,
            target_col="mortality",
            output_dir=os.path.join(self.test_dir, 'test_output'),
            random_state=42
        )
        
        G = ctf.discover_causal_structure()
        
        # Check that the graph was created
        self.assertIsNotNone(G)
        self.assertEqual(len(G.nodes()), 3)  # 3 variables
        
        # Check that the causal graph was saved
        self.assertIsNotNone(ctf.causal_graph)
        self.assertEqual(ctf.causal_graph, G)
        
        # Test with domain knowledge
        G = ctf.discover_causal_structure(domain_knowledge=self.domain_knowledge)
        
        # Check that domain knowledge edges exist
        self.assertTrue(G.has_edge("age", "mortality"))
        self.assertTrue(G.has_edge("sofa", "mortality"))
    
    def test_train_models(self):
        """Test model training."""
        ctf = CausalTransparencyFramework(
            data=self.df,
            target_col="mortality",
            output_dir=os.path.join(self.test_dir, 'test_output'),
            random_state=42
        )
        
        # Discover causal structure first
        ctf.discover_causal_structure()
        
        # Train models
        models = ctf.train_models(test_size=0.2)
        
        # Check that models were created
        self.assertIsNotNone(models)
        self.assertTrue(len(models) > 0)
        
        # Check that train/test data was split
        self.assertIsNotNone(ctf.train_data)
        self.assertIsNotNone(ctf.test_data)
        
        # Check that model performance was calculated
        self.assertTrue(len(ctf.model_performance) > 0)
        
        # Check that at least one causal model was created
        causal_models = [name for name in models.keys() if name.startswith('causal_')]
        self.assertTrue(len(causal_models) > 0)
    
    def test_calculate_transparency_metrics(self):
        """Test transparency metrics calculation."""
        ctf = CausalTransparencyFramework(
            data=self.df,
            target_col="mortality",
            output_dir=os.path.join(self.test_dir, 'test_output'),
            random_state=42
        )
        
        # Discover causal structure
        ctf.discover_causal_structure()
        
        # Train models
        ctf.train_models(test_size=0.2)
        
        # Calculate transparency metrics
        metrics = ctf.calculate_transparency_metrics()
        
        # Check that metrics were calculated
        self.assertIsNotNone(metrics)
        self.assertIn('cii', metrics)
        self.assertIn('ccm', metrics)
        self.assertIn('te', metrics)
        self.assertIn('cs', metrics)
        
        # Check that metrics files were created
        self.assertTrue(os.path.exists(os.path.join(ctf.output_dir, 'ctf_metrics.json')))
        self.assertTrue(os.path.exists(os.path.join(ctf.output_dir, 'model_performance.json')))
    
    def test_generate_report(self):
        """Test report generation."""
        ctf = CausalTransparencyFramework(
            data=self.df,
            target_col="mortality",
            output_dir=os.path.join(self.test_dir, 'test_output'),
            random_state=42
        )
        
        # Run the full pipeline up to report generation
        ctf.discover_causal_structure()
        ctf.train_models(test_size=0.2)
        ctf.calculate_transparency_metrics()
        
        # Generate report
        report_path = ctf.generate_report()
        
        # Check that report was created
        self.assertTrue(os.path.exists(report_path))
    
    def test_run_complete_pipeline(self):
        """Test the complete CTF pipeline."""
        ctf = CausalTransparencyFramework(
            data=self.df,
            target_col="mortality",
            output_dir=os.path.join(self.test_dir, 'test_output'),
            random_state=42
        )
        
        # Run the complete pipeline
        report_path = ctf.run_complete_pipeline(domain_knowledge=self.domain_knowledge)
        
        # Check that report was created
        self.assertTrue(os.path.exists(report_path))
        
        # Check that all key components were executed
        self.assertIsNotNone(ctf.causal_graph)
        self.assertTrue(len(ctf.models) > 0)
        self.assertTrue(len(ctf.metrics) > 0)
        self.assertTrue(len(ctf.model_performance) > 0)

if __name__ == '__main__':
    unittest.main()