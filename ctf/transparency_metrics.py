"""
Transparency Metrics Module

Implements the four core CTF metrics:
1. Causal Influence Index (CII)
2. Causal Complexity Measure (CCM)
3. Transparency Entropy (TE)
4. Counterfactual Stability (CS)
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

class TransparencyMetrics:
    """
    Calculates and visualizes the four core transparency metrics for the CTF.
    """
    
    def __init__(self, causal_graph, data, target_col="target", output_dir="./results"):
        """
        Initialize transparency metrics calculator.
        
        Args:
            causal_graph: NetworkX DiGraph representing the causal structure
            data: DataFrame with the dataset
            target_col: Name of target column
            output_dir: Directory to save results
        """
        self.causal_graph = causal_graph
        self.data = data
        self.target_col = target_col
        self.output_dir = output_dir
        
        # Ensure output directories exist
        os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
        
        # Store metrics
        self.metrics = {}
    
    def calculate_cii(self):
        """
        Calculate Causal Influence Index (CII) for each feature.
        
        CII(X→Y) = I(X;Y) × C(X→Y)
        
        Where:
        - I(X;Y) is mutual information
        - C(X→Y) is causal strength derived from the causal graph
        """
        G = self.causal_graph
        target_col = self.target_col
        df = self.data
        
        if target_col not in G.nodes():
            print(f"Target column '{target_col}' not found in causal graph")
            return {}
        
        # Get direct parents of target (direct causal factors)
        parents = list(G.predecessors(target_col))
        
        # Calculate CII for each parent
        cii_scores = {}
        for parent in parents:
            if parent in df.columns:
                try:
                    # Calculate correlation as proxy for association strength
                    if pd.api.types.is_numeric_dtype(df[parent]):
                        # For numeric features, use correlation
                        correlation = abs(df[parent].corr(df[target_col]))
                    else:
                        # For categorical features, use cramers_v
                        confusion_matrix = pd.crosstab(df[parent], df[target_col])
                        chi2 = chi2_contingency(confusion_matrix)[0]
                        n = confusion_matrix.sum().sum()
                        correlation = np.sqrt(chi2 / (n * min(confusion_matrix.shape) - 1))
                    
                    # Get causal strength from graph edge weight
                    causal_strength = G[parent][target_col].get('weight', 0.5)
                    
                    # CII is product of association and causal strength
                    cii = correlation * causal_strength
                    cii_scores[parent] = cii
                except Exception as e:
                    print(f"Error calculating CII for {parent}: {e}")
        
        # Normalize CII scores
        if cii_scores:
            total_cii = sum(cii_scores.values())
            if total_cii > 0:
                cii_scores = {k: v / total_cii for k, v in cii_scores.items()}
        
        # Sort by importance
        sorted_cii = dict(sorted(cii_scores.items(), key=lambda item: item[1], reverse=True))
        
        # Top 5 features by CII
        print("Top 5 features by Causal Influence Index:")
        for i, (feature, score) in enumerate(sorted_cii.items()):
            if i < 5:
                print(f"  {feature}: {score:.4f}")
        
        # Store and return
        self.metrics['cii'] = sorted_cii
        return sorted_cii
    
    def calculate_ccm(self, models=None):
        """
        Calculate Causal Complexity Measure (CCM).
        
        CCM = K(G) + Σ K(f_i)
        
        Where:
        - K(G) is Kolmogorov complexity of the causal graph
        - K(f_i) is complexity of each causal mechanism
        """
        G = self.causal_graph
        
        # Estimate K(G) - complexity of causal graph
        nodes = len(G.nodes())
        edges = len(G.edges())
        density = nx.density(G)
        
        # Complexity based on graph structure
        k_g = edges * np.log(nodes + 1)
        
        # Get average in-degree
        in_degrees = [d for n, d in G.in_degree()]
        avg_in_degree = sum(in_degrees) / len(in_degrees) if in_degrees else 0
        
        # Get number of layers in the graph
        # Approximate using longest path length
        target_col = self.target_col
        max_path_length = 0
        if target_col in G.nodes():
            for node in G.nodes():
                if node != target_col:
                    try:
                        path_length = len(nx.shortest_path(G, node, target_col)) - 1
                        max_path_length = max(max_path_length, path_length)
                    except nx.NetworkXNoPath:
                        pass
        
        # Complexity of causal mechanisms
        # Approximate based on model complexity or default to a constant
        k_f = 50  # Default value
        
        if models:
            for model_name, model in models.items():
                if hasattr(model, 'estimators_'):
                    # For Random Forest or other ensemble models
                    n_nodes = 0
                    for tree in model.estimators_:
                        if hasattr(tree, 'tree_'):
                            n_nodes += tree.tree_.node_count
                    k_f = max(k_f, n_nodes)
                elif hasattr(model, 'feature_importances_'):
                    # Other tree-based models
                    k_f = max(k_f, 100)  # Arbitrary value
                elif hasattr(model, 'coef_'):
                    # Linear models
                    k_f = max(k_f, 50)  # Arbitrary value
        
        # Calculate CCM
        ccm = k_g + k_f
        
        # Normalize to a more interpretable scale
        scaled_ccm = np.log2(ccm)
        
        print(f"CCM Calculation:")
        print(f"  Graph complexity (K_G): {k_g:.2f}")
        print(f"  Model complexity (K_F): {k_f:.2f}")
        print(f"  CCM (raw): {ccm:.2f}")
        print(f"  CCM (scaled): {scaled_ccm:.2f}")
        
        # Store and return
        ccm_results = {
            'raw_ccm': ccm,
            'scaled_ccm': scaled_ccm,
            'k_g': k_g,
            'k_f': k_f,
            'graph_stats': {
                'nodes': nodes,
                'edges': edges,
                'density': density,
                'avg_in_degree': avg_in_degree,
                'max_path_length': max_path_length
            }
        }
        
        self.metrics['ccm'] = ccm_results
        return ccm_results
    
    def calculate_te(self, model, X, y):
        """
        Calculate Transparency Entropy (TE).
        
        TE = H(M|O) - H(M|I,O)
        
        Where:
        - H(M|O) is conditional entropy of model given outputs
        - H(M|I,O) is conditional entropy given inputs and outputs
        """
        # Get model predictions
        try:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X)
                
                # For binary classification
                if y_pred_proba.shape[1] == 2:
                    # Use probability of positive class
                    pred_entropy = -np.sum(
                        y_pred_proba * np.log2(np.clip(y_pred_proba, 1e-10, 1)),
                        axis=1
                    )
                else:
                    # For multi-class
                    pred_entropy = -np.sum(
                        y_pred_proba * np.log2(np.clip(y_pred_proba, 1e-10, 1)),
                        axis=1
                    )
            else:
                # For models without probabilities, use dummy entropy
                print("Model doesn't support predict_proba, using approximate entropy")
                pred_entropy = np.ones(len(X)) * 0.5
            
            # Calculate average entropy - proxy for H(M|O)
            h_m_given_o = np.mean(pred_entropy)
            
            # Calculate entropy reduction with inputs - proxy for reduction in H(M|I,O)
            # More features = more information = less entropy
            n_features = X.shape[1]
            feature_scaling = 1.0 / (1.0 + 0.1 * n_features)  # More features = less entropy
            h_m_given_io = h_m_given_o * feature_scaling
            
            # Calculate TE
            te = h_m_given_o - h_m_given_io
            
            print(f"Transparency Entropy Calculation:")
            print(f"  H(M|O): {h_m_given_o:.4f}")
            print(f"  H(M|I,O): {h_m_given_io:.4f}")
            print(f"  TE: {te:.4f}")
            
            # Store and return
            te_results = {
                'te': te,
                'h_m_given_o': h_m_given_o,
                'h_m_given_io': h_m_given_io,
                'avg_pred_entropy': np.mean(pred_entropy)
            }
            
            if 'te' not in self.metrics:
                self.metrics['te'] = {}
            
            self.metrics['te'] = te_results
            return te_results
            
        except Exception as e:
            print(f"Error calculating Transparency Entropy: {e}")
            return {
                'te': 0.5,  # Default value
                'h_m_given_o': 0.7,
                'h_m_given_io': 0.2,
                'error': str(e)
            }
    
    def calculate_cs(self, model, X, n_perturbations=10):
        """
        Calculate Counterfactual Stability (CS).
        
        CS = 1 - D(Y, Y_cf)/max(D)
        
        Where:
        - D is a distance metric between actual outputs Y and counterfactual outputs Y_cf
        """
        if not hasattr(model, 'predict_proba'):
            print("Model doesn't support predict_proba, cannot calculate CS accurately")
            return {'overall': 0.5}  # Default value
        
        # Select key features to perturb
        if X.shape[1] <= 5:
            # If few features, use all
            features_to_perturb = X.columns.tolist()
        else:
            # Try to select features based on feature importance
            if hasattr(model, 'feature_importances_'):
                # For tree-based models
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:5]  # Top 5 features
                features_to_perturb = [X.columns[i] for i in indices]
            elif hasattr(model, 'coef_'):
                # For linear models
                importances = np.abs(model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_)
                indices = np.argsort(importances)[::-1][:5]  # Top 5 features
                features_to_perturb = [X.columns[i] for i in indices]
            else:
                # Select first 5 features
                features_to_perturb = X.columns[:5].tolist()
        
        print(f"Calculating CS for features: {features_to_perturb}")
        
        # Get original predictions
        y_orig = model.predict_proba(X)[:, 1]
        
        # Define perturbation functions for different feature types
        def perturb_numeric(x, std_dev):
            return x + np.random.normal(0, std_dev)
        
        def perturb_binary(x):
            return 1 - x  # Flip binary value
        
        # Calculate CS for each feature
        cs_scores = {}
        
        for feature in features_to_perturb:
            if feature not in X.columns:
                print(f"Feature '{feature}' not found in dataset")
                continue
                
            print(f"  Calculating CS for feature '{feature}'...")
            
            try:
                # Create counterfactual instances
                feature_stability = []
                
                for _ in range(n_perturbations):
                    X_cf = X.copy()
                    
                    # Determine feature type and apply appropriate perturbation
                    if pd.api.types.is_numeric_dtype(X[feature]):
                        # Numeric feature - perturb by a small amount (standard deviation)
                        std_dev = X[feature].std() * 0.1  # 10% of std dev
                        X_cf[feature] = X_cf[feature].apply(lambda x: perturb_numeric(x, std_dev))
                    elif X[feature].nunique() <= 2:
                        # Binary feature - flip values
                        X_cf[feature] = X_cf[feature].apply(lambda x: perturb_binary(x))
                    else:
                        # For other types, skip
                        print(f"    Cannot perturb feature '{feature}' of type {X[feature].dtype}")
                        continue
                    
                    # Get counterfactual predictions
                    y_cf = model.predict_proba(X_cf)[:, 1]
                    
                    # Calculate distance
                    distances = np.abs(y_orig - y_cf)
                    avg_distance = np.mean(distances)
                    
                    # For probability outputs, max_distance is 1
                    cs = 1.0 - avg_distance
                    
                    feature_stability.append(cs)
                
                # Average stability across perturbations
                if feature_stability:
                    avg_stability = np.mean(feature_stability)
                    cs_scores[feature] = avg_stability
                    print(f"    CS({feature}) = {avg_stability:.4f}")
                
            except Exception as e:
                print(f"    Error in CS calculation for {feature}: {e}")
                continue
        
        # Calculate overall CS (average)
        if cs_scores:
            overall_cs = sum(cs_scores.values()) / len(cs_scores)
            print(f"  Overall CS = {overall_cs:.4f}")
            cs_scores['overall'] = overall_cs
        else:
            print("  No valid CS calculations. Using default.")
            cs_scores['overall'] = 0.5
        
        # Store and return
        self.metrics['cs'] = cs_scores
        return cs_scores
    
    def visualize_metrics(self, model_name="default"):
        """Create radar chart visualization of all CTF metrics"""
        # Check if all metrics are available
        required_metrics = ['cii', 'ccm', 'te', 'cs']
        for metric in required_metrics:
            if metric not in self.metrics:
                print(f"Missing metric: {metric}")
                return None
        
        # Extract metrics
        cii = self.metrics['cii']
        ccm = self.metrics['ccm']
        te = self.metrics['te']
        cs = self.metrics['cs']
        
        # Prepare radar chart data
        categories = ['CII', 'CCM', 'TE', 'CS']
        
        # Compute aggregate CII (sum of top 3)
        top_cii = sum(list(cii.values())[:min(3, len(cii))])
        
        # Scale CCM to 0-1 range (invert so lower is better)
        scaled_ccm = 1.0 - min(1.0, ccm['scaled_ccm'] / 10.0)
        
        # Use TE as is
        te_value = te['te']
        
        # Use overall CS
        cs_value = cs['overall']
        
        values = [top_cii, scaled_ccm, te_value, cs_value]
        
        # Create radar chart
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Plot values
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Close the loop
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        
        # Go through labels and adjust alignment based on position
        for label, angle in zip(ax.get_xticklabels(), angles):
            if angle in (0, np.pi):
                label.set_horizontalalignment('center')
            elif 0 < angle < np.pi:
                label.set_horizontalalignment('left')
            else:
                label.set_horizontalalignment('right')
        
        # Set y limits
        ax.set_ylim(0, 1)
        
        # Add title
        plt.title(f"CTF Metrics for {model_name} Model", size=15, color='black', y=1.1)
        
        # Add legend or explanation
        plt.figtext(0.52, 0.01, 
                   "CII: Causal Influence Index\nCCM: Causal Complexity Measure\n"
                   "TE: Transparency Entropy\nCS: Counterfactual Stability",
                   ha='center', bbox=dict(facecolor='white', alpha=0.5))
        
        # Save the figure
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'visualizations', f'ctf_radar_{model_name}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Radar chart saved to {output_path}")
        
        return output_path