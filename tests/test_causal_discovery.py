"""
Unit tests for the causal discovery module.
"""

import unittest
import pandas as pd
import numpy as np
import networkx as nx
import os
import tempfile
import shutil

from ctf.causal_discovery import CausalDiscovery

class TestCausalDiscovery(unittest.TestCase):
    """Test cases for the CausalDiscovery class."""
    
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
        
        # Lab values influenced by SOFA
        creatinine = 0.1 * sofa + 0.01 * age + np.random.normal(0, 0.5, n_samples)
        creatinine = np.clip(creatinine, 0.3, 7)
        
        # Target influenced by age, SOFA, and creatinine
        logits = -5 + 0.02 * age + 0.3 * sofa + 0.4 * creatinine
        p_mortality = 1 / (1 + np.exp(-logits))
        mortality = np.random.binomial(1, p_mortality)
        
        # Create DataFrame
        self.df = pd.DataFrame({
            'age': age,
            'sofa': sofa,
            'creatinine': creatinine,
            'mortality': mortality
        })
        
        # Expected causal relationships
        self.expected_edges = [
            ('age', 'sofa'),
            ('age', 'creatinine'),
            ('age', 'mortality'),
            ('sofa', 'creatinine'),
            ('sofa', 'mortality'),
            ('creatinine', 'mortality')
        ]
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test that the CausalDiscovery class initializes correctly."""
        cd = CausalDiscovery(
            data=self.df,
            target_col="mortality",
            output_dir=self.test_dir
        )
        
        self.assertEqual(cd.target_col, "mortality")
        self.assertEqual(cd.output_dir, self.test_dir)
        self.assertTrue((cd.data == self.df).all().all())
        self.assertEqual(len(cd.graphs), 0)
    
    def test_correlation_graph(self):
        """Test that the correlation graph is created correctly."""
        cd = CausalDiscovery(
            data=self.df,
            target_col="mortality",
            output_dir=self.test_dir
        )
        
        G = cd.create_correlation_graph()
        
        # Check that the graph was created
        self.assertIsInstance(G, nx.DiGraph)
        self.assertEqual(len(G.nodes()), 4)  # 4 variables
        
        # Check that the target has incoming edges
        self.assertTrue(any(e[1] == "mortality" for e in G.edges()))
        
        # Check that key files were created
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'graphs', 'correlation_graph.json')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'graphs', 'correlation_graph.png')))
    
    def test_ensemble_graph(self):
        """Test that the ensemble graph is created correctly."""
        cd = CausalDiscovery(
            data=self.df,
            target_col="mortality",
            output_dir=self.test_dir
        )
        
        # Create ensemble graph
        G = cd.create_ensemble_graph()
        
        # Check that the graph was created
        self.assertIsInstance(G, nx.DiGraph)
        self.assertEqual(len(G.nodes()), 4)  # 4 variables
        
        # Check that the graph is acyclic
        self.assertTrue(nx.is_directed_acyclic_graph(G))
        
        # Check that key files were created
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'graphs', 'ensemble_graph.json')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'graphs', 'ensemble_graph.png')))
    
    def test_domain_knowledge_integration(self):
        """Test that domain knowledge is correctly integrated."""
        cd = CausalDiscovery(
            data=self.df,
            target_col="mortality",
            output_dir=self.test_dir
        )
        
        # Define domain knowledge
        domain_knowledge = {
            "edges": [
                ["age", "mortality", 0.9],
                ["sofa", "mortality", 0.8]
            ]
        }
        
        # Create ensemble graph with domain knowledge
        G = cd.create_ensemble_graph(domain_knowledge=domain_knowledge)
        
        # Check that domain knowledge edges exist
        self.assertTrue(G.has_edge("age", "mortality"))
        self.assertTrue(G.has_edge("sofa", "mortality"))
        
        # Check edge weights
        self.assertAlmostEqual(G["age"]["mortality"]["weight"], 0.9, places=1)
        self.assertAlmostEqual(G["sofa"]["mortality"]["weight"], 0.8, places=1)

if __name__ == '__main__':
    unittest.main()