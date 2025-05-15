"""Enhanced Causal Transparency Framework with DNN support"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Try importing optional dependencies
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Install with: pip install tensorflow")

from .causal_discovery import CausalDiscovery
from .transparency_metrics import TransparencyMetrics


class EnhancedCausalTransparencyFramework:
    """
    Enhanced Causal Transparency Framework with support for Deep Neural Networks
    
    This framework builds upon the original CTF by adding:
    - Deep Neural Network (DNN) model support via TensorFlow/Keras
    - Improved model architecture configurations
    - Additional model comparison capabilities
    """
    
    def __init__(self, data_path, target_col, output_dir='ctf_results', random_state=42):
        """
        Initialize the Enhanced Causal Transparency Framework
        
        Args:
            data_path: Path to the dataset CSV file
            target_col: Name of the target column
            output_dir: Directory to save results
            random_state: Random seed for reproducibility
        """
        self.data_path = data_path
        self.target_col = target_col
        self.output_dir = output_dir
        self.random_state = random_state
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        self._load_data()
        
        # Initialize attributes
        self.causal_graph = None
        self.models = {}
        self.model_performance = {}
        self.transparency_metrics = {}
        self.test_data = None
        
    def _load_data(self):
        """Load and preprocess the dataset"""
        print(f"Loading data from {self.data_path}")
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        
        # Basic preprocessing
        self.df = self.df.dropna()
        
        # Separate features and target
        self.X = self.df.drop(columns=[self.target_col])
        self.y = self.df[self.target_col]
        
        print(f"Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        
    def discover_causal_structure(self, domain_knowledge=None):
        """
        Discover causal structure using CausalDiscovery
        
        Args:
            domain_knowledge: Dict with prior knowledge about causal relationships
            
        Returns:
            networkx.DiGraph: Causal graph
        """
        print("Discovering causal structure...")
        
        # Initialize CausalDiscovery
        cd = CausalDiscovery(
            data=self.df,
            target_col=self.target_col,
            output_dir=os.path.join(self.output_dir, 'causal_models')
        )
        
        # Discover causal structure
        self.causal_graph = cd.create_ensemble_graph(domain_knowledge)
        
        # Save the graph
        cd.save_graphs({'ensemble': self.causal_graph})
        
        print(f"Causal graph discovered with {len(self.causal_graph.nodes())} nodes and {len(self.causal_graph.edges())} edges")
        
        return self.causal_graph
        
    def _build_dnn_model(self, input_dim, hidden_layers=[64, 32], activation='relu', dropout_rate=0.3):
        """
        Build a Deep Neural Network model
        
        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes
            activation: Activation function
            dropout_rate: Dropout rate for regularization
            
        Returns:
            keras.Model: Compiled DNN model
        """
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.InputLayer(input_shape=(input_dim,)))
        
        # Hidden layers with dropout
        for units in hidden_layers:
            model.add(layers.Dense(units, activation=activation))
            model.add(layers.Dropout(dropout_rate))
        
        # Output layer for binary classification
        model.add(layers.Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_models(self, test_size=0.2, dnn_epochs=50, dnn_batch_size=32, verbose=0):
        """
        Train multiple models including DNNs: causal and full versions
        
        Args:
            test_size: Proportion of data to use for testing
            dnn_epochs: Number of epochs for DNN training
            dnn_batch_size: Batch size for DNN training
            verbose: Verbosity level for DNN training
            
        Returns:
            Dict of trained models
        """
        print("Training predictive models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=self.random_state
        )
        
        self.test_data = (X_test, y_test)
        
        # Get causal features
        if self.causal_graph is None:
            print("Causal graph not found. Discovering causal structure first...")
            self.discover_causal_structure()
        
        # Get causal parents of target
        causal_parents = list(self.causal_graph.predecessors(self.target_col))
        causal_features = [f for f in causal_parents if f in self.X.columns]
        
        if not causal_features:
            print("Warning: No causal features found. Using all features for causal models.")
            causal_features = self.X.columns.tolist()
        
        print(f"Using {len(causal_features)} causal features: {causal_features}")
        
        # Scale features for DNN
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scaler = scaler
        
        # Train causal models
        # Logistic Regression
        causal_lr = LogisticRegression(random_state=self.random_state)
        causal_lr.fit(X_train[causal_features], y_train)
        self.models['causal_lr'] = causal_lr
        
        # Random Forest
        causal_rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        causal_rf.fit(X_train[causal_features], y_train)
        self.models['causal_rf'] = causal_rf
        
        # XGBoost (if available)
        if XGBOOST_AVAILABLE:
            causal_xgb = xgb.XGBClassifier(random_state=self.random_state)
            causal_xgb.fit(X_train[causal_features], y_train)
            self.models['causal_xgb'] = causal_xgb
        
        # DNN (if TensorFlow available)
        if TENSORFLOW_AVAILABLE:
            # Get causal feature indices
            causal_indices = [i for i, col in enumerate(X_train.columns) if col in causal_features]
            X_train_causal = X_train_scaled[:, causal_indices]
            X_test_causal = X_test_scaled[:, causal_indices]
            
            # Build and train causal DNN
            causal_dnn = self._build_dnn_model(len(causal_indices))
            causal_dnn.fit(
                X_train_causal, y_train,
                validation_split=0.2,
                epochs=dnn_epochs,
                batch_size=dnn_batch_size,
                verbose=verbose
            )
            
            # Store model with feature indices
            self.models['causal_dnn'] = causal_dnn
            self.models['causal_dnn_features'] = causal_indices
        
        # Train full models (using all features)
        # Logistic Regression
        full_lr = LogisticRegression(random_state=self.random_state)
        full_lr.fit(X_train, y_train)
        self.models['full_lr'] = full_lr
        
        # Random Forest
        full_rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        full_rf.fit(X_train, y_train)
        self.models['full_rf'] = full_rf
        
        # XGBoost (if available)
        if XGBOOST_AVAILABLE:
            full_xgb = xgb.XGBClassifier(random_state=self.random_state)
            full_xgb.fit(X_train, y_train)
            self.models['full_xgb'] = full_xgb
        
        # DNN (if TensorFlow available)
        if TENSORFLOW_AVAILABLE:
            # Build and train full DNN
            full_dnn = self._build_dnn_model(X_train_scaled.shape[1])
            full_dnn.fit(
                X_train_scaled, y_train,
                validation_split=0.2,
                epochs=dnn_epochs,
                batch_size=dnn_batch_size,
                verbose=verbose
            )
            self.models['full_dnn'] = full_dnn
        
        # Evaluate all models
        self._evaluate_models()
        
        print(f"Trained {len(self.models)} models")
        return self.models
    
    def _evaluate_models(self):
        """Evaluate all models and store performance metrics"""
        X_test, y_test = self.test_data
        X_test_scaled = self.scaler.transform(X_test)
        
        for name, model in self.models.items():
            if 'features' in name:
                continue  # Skip feature indices entries
                
            # Get features used by this model
            if name.startswith('causal_'):
                # Get causal parents
                parents = list(self.causal_graph.predecessors(self.target_col))
                features = [p for p in parents if p in X_test.columns]
                
                if 'dnn' in name:
                    # Use scaled data for DNN
                    feature_indices = self.models.get(f'{name}_features', [])
                    X_test_model = X_test_scaled[:, feature_indices]
                else:
                    X_test_model = X_test[features]
            else:
                # Use all features
                features = X_test.columns.tolist()
                
                if 'dnn' in name:
                    # Use scaled data for DNN
                    X_test_model = X_test_scaled
                else:
                    X_test_model = X_test
            
            # Calculate performance metrics
            try:
                y_pred = model.predict(X_test_model)
                
                # For DNNs, convert probabilities to binary predictions
                if 'dnn' in name:
                    y_pred_proba = y_pred.flatten()
                    y_pred = (y_pred_proba > 0.5).astype(int)
                else:
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_test_model)[:, 1]
                    else:
                        y_pred_proba = y_pred
                
                from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
                
                accuracy = accuracy_score(y_test, y_pred)
                
                # Handle binary vs multi-class
                if len(np.unique(y_test)) <= 2:
                    auc = roc_auc_score(y_test, y_pred_proba)
                    f1 = f1_score(y_test, y_pred)
                else:
                    # For multi-class
                    auc = 0  # Skip or calculate macro AUC
                    f1 = f1_score(y_test, y_pred, average='macro')
                
                self.model_performance[name] = {
                    'accuracy': accuracy,
                    'auc': auc,
                    'f1': f1,
                    'n_features': len(features),
                    'model_type': 'DNN' if 'dnn' in name else 'Traditional'
                }
                
                print(f"Model {name}: Accuracy={accuracy:.4f}, AUC={auc:.4f}, F1={f1:.4f}")
                
            except Exception as e:
                print(f"Error evaluating model {name}: {e}")
                self.model_performance[name] = {
                    'accuracy': 0,
                    'auc': 0,
                    'f1': 0,
                    'n_features': len(features),
                    'error': str(e)
                }
    
    def calculate_transparency_metrics(self):
        """
        Calculate all transparency metrics for all models
        
        Returns:
            Dict of transparency metrics
        """
        print("Calculating transparency metrics...")
        
        if not self.models:
            print("No models found. Training models first...")
            self.train_models()
        
        # Initialize TransparencyMetrics
        tm = TransparencyMetrics(
            causal_graph=self.causal_graph,
            data=self.df,
            target_col=self.target_col,
            output_dir=self.output_dir
        )
        
        # Calculate CII (common for all models)
        cii = tm.calculate_cii()
        
        # Calculate CCM (common for all models)
        ccm = tm.calculate_ccm()
        
        # Calculate TE and CS for each model
        self.transparency_metrics = {'cii': cii, 'ccm': ccm, 'te': {}, 'cs': {}}
        X_test, y_test = self.test_data
        X_test_scaled = self.scaler.transform(X_test)
        
        for name, model in self.models.items():
            if 'features' in name:
                continue  # Skip feature indices entries
                
            print(f"\nCalculating metrics for model: {name}")
            
            # Get features used by this model
            if name.startswith('causal_'):
                parents = list(self.causal_graph.predecessors(self.target_col))
                features = [p for p in parents if p in X_test.columns]
                
                if 'dnn' in name:
                    feature_indices = self.models.get(f'{name}_features', [])
                    X_test_model = X_test_scaled[:, feature_indices]
                else:
                    X_test_model = X_test[features]
            else:
                features = X_test.columns.tolist()
                
                if 'dnn' in name:
                    X_test_model = X_test_scaled
                else:
                    X_test_model = X_test
            
            # Calculate TE
            te = tm.calculate_te(model, X_test_model)
            self.transparency_metrics['te'][name] = te
            
            # Calculate CS
            cs = tm.calculate_cs(model, X_test_model)
            self.transparency_metrics['cs'][name] = cs
            
            # Generate and save radar chart
            self._generate_radar_chart(name, cii, ccm, te, cs)
        
        # Save all metrics
        self._save_metrics()
        
        print("Transparency metrics calculated")
        return self.transparency_metrics
    
    def _generate_radar_chart(self, model_name, cii, ccm, te, cs):
        """Generate radar chart for a specific model"""
        # Handle metrics
        cii_value = np.mean(list(cii.values()))
        ccm_value = ccm.get('ccm_scaled', 0)
        te_value = te.get('te', 0)
        cs_value = cs.get('overall', 0)
        
        # Create radar chart
        categories = ['CII', 'CCM', 'TE', 'CS']
        values = [cii_value, ccm_value, te_value, cs_value]
        
        # Normalize values to 0-1 range
        values[0] = values[0]  # CII already in 0-1
        values[1] = 1 - min(values[1] / 100, 1)  # CCM inverted and scaled
        values[2] = values[2]  # TE already in 0-1
        values[3] = values[3]  # CS already in 0-1
        
        # Create the plot
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
        values_plot = values + values[:1]
        angles_plot = list(angles) + [angles[0]]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles_plot, values_plot, 'o-', linewidth=2)
        ax.fill(angles_plot, values_plot, alpha=0.25)
        ax.set_xticks(angles)
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.set_title(f'CTF Metrics - {model_name}', pad=20)
        
        # Add value labels
        for angle, value, cat in zip(angles, values, categories):
            ax.text(angle, value + 0.1, f'{value:.3f}', ha='center', va='center')
        
        # Save the chart
        output_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'ctf_radar_{model_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Radar chart saved to {output_dir}/ctf_radar_{model_name}.png")
    
    def _save_metrics(self):
        """Save transparency metrics to JSON file"""
        import json
        
        # Prepare metrics for JSON serialization
        metrics_json = {
            'cii': self.transparency_metrics['cii'],
            'ccm': self.transparency_metrics['ccm'],
            'te': self.transparency_metrics['te'],
            'cs': self.transparency_metrics['cs'],
            'model_performance': self.model_performance
        }
        
        # Save to JSON
        output_path = os.path.join(self.output_dir, 'ctf_metrics.json')
        with open(output_path, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        
        print(f"Metrics saved to {output_path}")
    
    def generate_report(self):
        """Generate comprehensive CTF report"""
        report_dir = os.path.join(self.output_dir, 'report')
        os.makedirs(report_dir, exist_ok=True)
        
        report_path = os.path.join(report_dir, 'ctf_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Causal Transparency Framework Report\n\n")
            
            # Dataset information
            f.write("## Dataset Information\n")
            f.write(f"- Dataset: {os.path.basename(self.data_path)}\n")
            f.write(f"- Target variable: {self.target_col}\n")
            f.write(f"- Number of samples: {len(self.df)}\n")
            f.write(f"- Number of features: {len(self.X.columns)}\n\n")
            
            # Causal structure
            f.write("## Causal Structure\n")
            f.write(f"- Nodes: {len(self.causal_graph.nodes())}\n")
            f.write(f"- Edges: {len(self.causal_graph.edges())}\n")
            f.write(f"- Causal parents of target: {list(self.causal_graph.predecessors(self.target_col))}\n\n")
            
            # Model performance
            f.write("## Model Performance\n")
            f.write("| Model | Type | Accuracy | AUC | F1 Score | Features |\n")
            f.write("|-------|------|----------|-----|----------|----------|\n")
            
            for name, perf in sorted(self.model_performance.items()):
                if 'features' not in name:
                    model_type = perf.get('model_type', 'Traditional')
                    f.write(f"| {name} | {model_type} | {perf['accuracy']:.4f} | "
                           f"{perf['auc']:.4f} | {perf['f1']:.4f} | {perf['n_features']} |\n")
            
            # Transparency metrics
            f.write("\n## Transparency Metrics\n")
            
            # CII
            f.write("\n### Causal Influence Index (CII)\n")
            for feature, value in sorted(self.transparency_metrics['cii'].items(), 
                                       key=lambda x: x[1], reverse=True)[:10]:
                f.write(f"- {feature}: {value:.4f}\n")
            
            # CCM
            f.write("\n### Causal Complexity Measure (CCM)\n")
            ccm = self.transparency_metrics['ccm']
            f.write(f"- Graph complexity: {ccm.get('graph_complexity', 0):.2f}\n")
            f.write(f"- Model complexity: {ccm.get('model_complexity', 0):.2f}\n")
            f.write(f"- CCM (scaled): {ccm.get('ccm_scaled', 0):.2f}\n")
            
            # TE and CS by model
            f.write("\n### Transparency Entropy (TE) and Counterfactual Stability (CS)\n")
            f.write("| Model | TE | CS |\n")
            f.write("|-------|----|----|\n")
            
            for name in sorted(self.models.keys()):
                if 'features' not in name:
                    te = self.transparency_metrics['te'].get(name, {}).get('te', 0)
                    cs = self.transparency_metrics['cs'].get(name, {}).get('overall', 0)
                    f.write(f"| {name} | {te:.4f} | {cs:.4f} |\n")
            
            f.write("\n## Summary\n")
            f.write("The Causal Transparency Framework analysis reveals:\n")
            
            # Find best performing model
            best_model = max(self.model_performance.items(), 
                           key=lambda x: x[1]['auc'] if 'features' not in x[0] else 0)[0]
            f.write(f"- Best performing model: {best_model} (AUC: {self.model_performance[best_model]['auc']:.4f})\n")
            
            # Compare traditional vs DNN models
            traditional_models = [name for name in self.models.keys() 
                               if 'dnn' not in name and 'features' not in name]
            dnn_models = [name for name in self.models.keys() 
                         if 'dnn' in name and 'features' not in name]
            
            if traditional_models and dnn_models:
                avg_traditional_auc = np.mean([self.model_performance[m]['auc'] 
                                             for m in traditional_models])
                avg_dnn_auc = np.mean([self.model_performance[m]['auc'] 
                                      for m in dnn_models])
                
                f.write(f"- Average AUC - Traditional models: {avg_traditional_auc:.4f}\n")
                f.write(f"- Average AUC - DNN models: {avg_dnn_auc:.4f}\n")
            
            # Compare causal vs full models
            causal_models = [name for name in self.models.keys() 
                           if name.startswith('causal_') and 'features' not in name]
            full_models = [name for name in self.models.keys() 
                          if name.startswith('full_') and 'features' not in name]
            
            if causal_models and full_models:
                avg_causal_auc = np.mean([self.model_performance[m]['auc'] 
                                        for m in causal_models])
                avg_full_auc = np.mean([self.model_performance[m]['auc'] 
                                      for m in full_models])
                
                f.write(f"- Average AUC - Causal models: {avg_causal_auc:.4f}\n")
                f.write(f"- Average AUC - Full models: {avg_full_auc:.4f}\n")
            
            f.write("\nVisualizations have been saved to the visualizations directory.\n")
        
        print(f"CTF report generated at {report_path}")
        return report_path