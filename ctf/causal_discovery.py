"""
Causal Discovery Module

Implements multiple causal discovery methods and ensemble techniques
for learning causal structure from observational data.
"""

import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

class CausalDiscovery:
    """
    Enhanced causal discovery methods for the Causal Transparency Framework.
    Provides multiple algorithms for causal structure learning and ensemble methods.
    """
    
    def __init__(self, data, target_col="target", output_dir="./results/causal_models"):
        """
        Initialize causal discovery with data and output settings.
        
        Args:
            data: DataFrame with the dataset
            target_col: Name of target column
            output_dir: Directory to save causal models
        """
        self.data = data
        self.target_col = target_col
        self.output_dir = output_dir
        
        # Ensure output directories exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'graphs'), exist_ok=True)
        
        # Initialize causal graphs
        self.graphs = {}
        
    def create_correlation_graph(self):
        """Create a causal graph based on correlations"""
        print("Creating correlation-based causal graph")
        
        df = self.data.copy()
        target_col = self.target_col
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add all columns as nodes
        for col in df.columns:
            G.add_node(col)
        
        # Calculate correlations
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        corr_matrix = df[numeric_cols].corr().abs()
        
        # Add edges for strong correlations, directed toward target
        threshold = 0.3  # Minimum correlation to consider
        
        # For numeric features
        for col1 in numeric_cols:
            for col2 in numeric_cols:
                if col1 != col2 and col1 in corr_matrix and col2 in corr_matrix:
                    corr = corr_matrix.loc[col1, col2]
                    
                    if corr > threshold:
                        # Direct edges toward target or based on domain knowledge
                        if col2 == target_col:
                            G.add_edge(col1, col2, weight=corr, method='correlation')
                        elif col1 == target_col:
                            G.add_edge(col2, col1, weight=corr, method='correlation')
                        else:
                            # For non-target variables, use a heuristic
                            precursors = ['age', 'gender', 'admission_type', 'ethnicity', 'insurance']
                            
                            if any(pre in col1.lower() for pre in precursors):
                                G.add_edge(col1, col2, weight=corr, method='correlation')
                            elif any(pre in col2.lower() for pre in precursors):
                                G.add_edge(col2, col1, weight=corr, method='correlation')
                            # Otherwise, add based on column order as a simple heuristic
                            elif df.columns.get_loc(col1) < df.columns.get_loc(col2):
                                G.add_edge(col1, col2, weight=corr, method='correlation')
                            else:
                                G.add_edge(col2, col1, weight=corr, method='correlation')
        
        # Handle categorical variables
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for cat_col in categorical_cols:
            if cat_col != target_col:
                # Check association with target (if numeric)
                if target_col in numeric_cols:
                    # Calculate mean target value for each category
                    grouped = df.groupby(cat_col)[target_col].mean()
                    if grouped.nunique() > 1:  # If there's variation
                        overall_mean = df[target_col].mean()
                        # Calculate weighted variation from mean
                        counts = df[cat_col].value_counts(normalize=True)
                        strength = sum(abs(v - overall_mean) * counts[k] for k, v in grouped.items())
                        if strength > 0.1:  # Only add if strength is significant
                            G.add_edge(cat_col, target_col, weight=min(strength*2, 0.99), method='category_effect')
                
                # Check associations with numeric variables
                for num_col in numeric_cols:
                    if num_col != target_col:
                        # Check if grouping by category shows significant differences
                        grouped = df.groupby(cat_col)[num_col].mean()
                        if grouped.nunique() > 1:  # If there's variation
                            overall_mean = df[num_col].mean()
                            # Calculate weighted variation from mean
                            counts = df[cat_col].value_counts(normalize=True)
                            strength = sum(abs(v - overall_mean) * counts[k] for k, v in grouped.items())
                            if strength > 0.1:  # Only add if strength is significant
                                G.add_edge(cat_col, num_col, weight=min(strength*2, 0.99), method='category_effect')
        
        # Save the graph
        self.graphs['correlation'] = G
        self._save_graph(G, 'correlation_graph')
        
        print(f"Correlation graph created with {len(G.nodes())} nodes and {len(G.edges())} edges")
        return G
        
    def create_ensemble_graph(self, domain_knowledge=None):
        """Create an ensemble causal graph by combining multiple methods"""
        print("Creating ensemble causal graph")
        
        # Create correlation graph if it doesn't exist
        if 'correlation' not in self.graphs:
            self.create_correlation_graph()
            
        # Create PC and Hill Climbing graphs if available
        try:
            if 'pc' not in self.graphs:
                self.create_pc_graph()
        except Exception as e:
            print(f"Skipping PC graph: {e}")
            
        try:
            if 'hc' not in self.graphs:
                self.create_hill_climbing_graph()
        except Exception as e:
            print(f"Skipping Hill Climbing graph: {e}")
            
        # Create ensemble graph by combining all available graphs
        G_ensemble = nx.DiGraph()
        
        # Add nodes from all graphs
        all_nodes = set()
        for graph_name, G in self.graphs.items():
            all_nodes.update(G.nodes())
        
        for node in all_nodes:
            G_ensemble.add_node(node)
        
        # Track edges and their weights across methods
        edge_weights = defaultdict(list)
        edge_methods = defaultdict(list)
        
        # Collect edges and weights from all graphs
        for graph_name, G in self.graphs.items():
            for u, v, data in G.edges(data=True):
                edge_weights[(u, v)].append(data.get('weight', 0.5))
                edge_methods[(u, v)].append(data.get('method', graph_name))
        
        # Add weighted edges to ensemble graph
        for (u, v), weights in edge_weights.items():
            # Add edge if it appears in multiple methods or with strong weight
            if len(weights) > 1 or max(weights) > 0.7:
                avg_weight = sum(weights) / len(weights)
                methods = edge_methods[(u, v)]
                G_ensemble.add_edge(u, v, weight=avg_weight, methods=','.join(methods))
        
        # Add domain knowledge if provided
        if domain_knowledge and 'edges' in domain_knowledge:
            for edge in domain_knowledge['edges']:
                source, target, weight = edge
                if source in G_ensemble.nodes and target in G_ensemble.nodes:
                    G_ensemble.add_edge(source, target, weight=weight, method='domain_knowledge')
        
        # Ensure the graph is acyclic (DAG)
        cycles = list(nx.simple_cycles(G_ensemble))
        if cycles:
            print(f"Found {len(cycles)} cycles in ensemble graph. Removing edges to create DAG.")
            for cycle in cycles:
                # Break cycle by removing the weakest edge
                weakest_weight = float('inf')
                weakest_edge = None
                
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    if G_ensemble.has_edge(u, v):
                        weight = G_ensemble[u][v].get('weight', 0.5)
                        if weight < weakest_weight:
                            weakest_weight = weight
                            weakest_edge = (u, v)
                
                if weakest_edge:
                    print(f"Breaking cycle by removing edge {weakest_edge}")
                    G_ensemble.remove_edge(*weakest_edge)
        
        # Save the ensemble graph
        self.graphs['ensemble'] = G_ensemble
        self._save_graph(G_ensemble, 'ensemble_graph')
        
        print(f"Ensemble graph created with {len(G_ensemble.nodes())} nodes and {len(G_ensemble.edges())} edges")
        return G_ensemble
    
    def _save_graph(self, G, name):
        """Save graph as JSON and visualization as PNG"""
        # Save graph structure as JSON for later use
        graph_data = {
            'nodes': list(G.nodes()),
            'edges': [(u, v, G[u][v]) for u, v in G.edges()]
        }
        
        # Convert edge data to serializable format
        for i, (u, v, data) in enumerate(graph_data['edges']):
            serializable_data = {}
            for k, val in data.items():
                if isinstance(val, (int, float, str, bool, list, dict)) or val is None:
                    serializable_data[k] = val
                else:
                    serializable_data[k] = str(val)
            graph_data['edges'][i] = (u, v, serializable_data)
        
        with open(os.path.join(self.output_dir, 'graphs', f'{name}.json'), 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        # Create layout
        pos = nx.spring_layout(G, seed=42)
        
        # Get edge weights for line width
        edge_weights = [G[u][v].get('weight', 0.5) * 2 for u, v in G.edges()]
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', alpha=0.8)
        
        # Highlight target node
        if self.target_col in G.nodes():
            nx.draw_networkx_nodes(G, pos, nodelist=[self.target_col], 
                                  node_size=1000, node_color='red', alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.6, 
                              edge_color='gray', arrowsize=15)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
        
        # Save figure
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'graphs', f'{name}.png'), dpi=300, bbox_inches='tight')
        plt.close()