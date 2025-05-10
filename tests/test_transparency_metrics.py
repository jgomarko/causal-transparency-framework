"""
Unit tests for the transparency metrics module.
"""

import unittest
import pandas as pd
import numpy as np
import networkx as nx
import os
import tempfile
import shutil

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from ctf.transparency_metrics import TransparencyMetrics

class TestTransparencyMetrics(unittest.TestCase):
    """Test cases for the TransparencyMetrics class."""
    
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
        
        # Create causal graph
        self.G = nx.DiGraph()
        self.G.add_nodes_from(['age', 'sofa', 'mortality'])
        self.G.add_edges_from([
            ('age', 'sofa', {'weight': 0.7}),
            ('age', 'mortality', {'weight': 0.6}),
            ('sofa', 'mortality', {'weight': 0.8})
        ])
        
        # Train a simple model
        X = self.df[['age', 'sofa']]
        y = self.df['mortality']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model = LogisticRegression(random_state=42)
        self.model.fit(self.X_train, self.y_train)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test that the TransparencyMetrics class initializes correctly."""
        tm = TransparencyMetrics(
            causal_graph=self.G,
            data=self.df,
            target_col="mortality",
            output_dir=self.test_dir
        )
        
        self.assertEqual(tm.target_col, "mortality")
        self.assertEqual(tm.output_dir, self.test_dir)
        self.assertTrue((tm.data == self.df).all().all())
        self.assertEqual(tm.causal_graph, self.G)
        self.assertEqual(tm.metrics, {})
    
    def test_calculate_cii(self):
        """Test CII calculation."""
        tm = TransparencyMetrics(
            causal_graph=self.G,
            data=self.df,
            target_col="mortality",
            output_dir=self.test_dir
        )
        
        cii = tm.calculate_cii()
        
        # Check that CII was calculated for each parent of mortality
        self.assertIn('age', cii)
        self.assertIn('sofa', cii)
        
        # Check that CII values are normalized
        self.assertAlmostEqual(sum(cii.values()), 1.0, places=5)
        
        # Check that CII values are reasonable
        for value in cii.values():
            self.assertTrue(0 <= value <= 1)
    
    def test_calculate_ccm(self):
        """Test CCM calculation."""
        tm = TransparencyMetrics(
            causal_graph=self.G,
            data=self.df,
            target_col="mortality",
            output_dir=self.test_dir
        )
        
        ccm = tm.calculate_ccm()
        
        # Check that CCM returns the expected keys
        self.assertIn('raw_ccm', ccm)
        self.assertIn('scaled_ccm', ccm)
        self.assertIn('k_g', ccm)
        self.assertIn('k_f', ccm)
        self.assertIn('graph_stats', ccm)
        
        # Check that CCM values are reasonable
        self.assertTrue(ccm['raw_ccm'] > 0)
        self.assertTrue(ccm['k_g'] > 0)
    
    def test_calculate_te(self):
        """Test TE calculation."""
        tm = TransparencyMetrics(
            causal_graph=self.G,
            data=self.df,
            target_col="mortality",
            output_dir=self.test_dir
        )
        
        te = tm.calculate_te(self.model, self.X_test, self.y_test)
        
        # Check that TE returns the expected keys
        self.assertIn('te', te)
        self.assertIn('h_m_given_o', te)
        self.assertIn('h_m_given_io', te)
        
        # Check that TE values are reasonable
        self.assertTrue(0 <= te['te'] <= 1)
        self.assertTrue(te['h_m_given_o'] >= te['h_m_given_io'])
    
    def test_calculate_cs(self):
        """Test CS calculation."""
        tm = TransparencyMetrics(
            causal_graph=self.G,
            data=self.df,
            target_col="mortality",
            output_dir=self.test_dir
        )
        
        cs = tm.calculate_cs(self.model, self.X_test)
        
        # Check that CS returns the expected keys
        self.assertIn('overall', cs)
        
        # CS should include individual feature scores
        for feature in self.X_test.columns:
            if feature in cs:
                self.assertTrue(0 <= cs[feature] <= 1)
        
        # Check that overall CS is reasonable
        self.assertTrue(0 <= cs['overall'] <= 1)
    
    def test_visualize_metrics(self):
        """Test metrics visualization."""
        tm = TransparencyMetrics(
            causal_graph=self.G,
            data=self.df,
            target_col="mortality",
            output_dir=self.test_dir
        )
        
        # Calculate all metrics
        tm.calculate_cii()
        tm.calculate_ccm()
        tm.calculate_te(self.model, self.X_test, self.y_test)
        tm.calculate_cs(self.model, self.X_test)
        
        # Create visualization
        output_path = tm.visualize_metrics(model_name="test_model")
        
        # Check that visualization was created
        self.assertTrue(os.path.exists(output_path))

if __name__ == '__main__':
    unittest.main()