"""
Causal Transparency Framework

The main module that implements the complete Causal Transparency Framework.
Integrates causal discovery, model training, transparency metrics, and visualization.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Will use only LogisticRegression and RandomForest models.")

try:
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. DNN models will not be available.")

from .causal_discovery import CausalDiscovery
from .transparency_metrics import TransparencyMetrics

class CausalTransparencyFramework:
    """
    Complete implementation of the Causal Transparency Framework.
    """
    
    def __init__(self, data_path=None, data=None, target_col="target",
                output_dir="./ctf_results", random_state=42):
        """
        Initialize the Causal Transparency Framework.
        
        Args:
            data_path: Path to data file (CSV)
            data: Pandas DataFrame (if data_path not provided)
            target_col: Name of the target column
            output_dir: Directory for saving results
            random_state: Random seed for reproducibility
        """
        self.data_path = data_path
        self.target_col = target_col
        self.output_dir = output_dir
        self.random_state = random_state
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'causal_models'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'counterfactuals'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'report'), exist_ok=True)
        
        # Initialize components
        self.causal_discovery = None
        self.causal_graph = None
        self.models = {}
        self.metrics = {}
        self.model_performance = {}
        
        # Load data
        if data is not None:
            self.df = data.copy()
        elif data_path is not None:
            self.df = self._load_data()
        else:
            raise ValueError("Either data_path or data must be provided")
    
    def _load_data(self):
        """Load data from file"""
        print(f"Loading data from {self.data_path}")
        
        try:
            # Try to infer file format from extension
            if self.data_path.endswith('.csv'):
                df = pd.read_csv(self.data_path)
            elif self.data_path.endswith('.pkl'):
                df = pd.read_pickle(self.data_path)
            else:
                # Default to CSV
                df = pd.read_csv(self.data_path)
            
            print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def discover_causal_structure(self, domain_knowledge=None):
        """
        Discover causal structure from data.
        
        Args:
            domain_knowledge: Optional dict with domain-specific causal information
        
        Returns:
            NetworkX DiGraph representing the causal structure
        """
        print("Discovering causal structure...")
        
        # Initialize causal discovery
        causal_models_dir = os.path.join(self.output_dir, 'causal_models')
        self.causal_discovery = CausalDiscovery(
            data=self.df,
            target_col=self.target_col,
            output_dir=causal_models_dir
        )
        
        # Create ensemble causal graph
        self.causal_graph = self.causal_discovery.create_ensemble_graph(domain_knowledge)
        
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
        Train predictive models using both causal and standard approaches.
        
        Args:
            test_size: Proportion of data to use for testing
            dnn_epochs: Number of epochs for DNN training
            dnn_batch_size: Batch size for DNN training
            verbose: Verbosity level for DNN training
        
        Returns:
            Dict of trained models
        """
        print("Training predictive models...")
        
        if self.causal_graph is None:
            print("Causal graph not found. Discovering causal structure first...")
            self.discover_causal_structure()
        
        # Prepare data
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Store split data for later use
        self.train_data = (X_train, y_train)
        self.test_data = (X_test, y_test)
        
        # Scale features for DNN (if available)
        if TENSORFLOW_AVAILABLE:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        
        # Get causal parents (direct causes of target)
        parents = list(self.causal_graph.predecessors(self.target_col))
        
        # Filter to only include columns present in the data
        causal_features = [p for p in parents if p in X.columns]
        print(f"Using {len(causal_features)} causal features: {causal_features}")
        
        # Causal models (using only causal features)
        if causal_features:
            # Logistic Regression on causal features
            causal_lr = LogisticRegression(random_state=self.random_state)
            causal_lr.fit(X_train[causal_features], y_train)
            self.models['causal_lr'] = causal_lr
            
            # Random Forest on causal features
            causal_rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            causal_rf.fit(X_train[causal_features], y_train)
            self.models['causal_rf'] = causal_rf
            
            # XGBoost on causal features (if available)
            if XGBOOST_AVAILABLE:
                causal_xgb = xgb.XGBClassifier(random_state=self.random_state)
                causal_xgb.fit(X_train[causal_features], y_train)
                self.models['causal_xgb'] = causal_xgb
            
            # DNN on causal features (if available)
            if TENSORFLOW_AVAILABLE:
                # Get causal feature indices
                causal_indices = [i for i, col in enumerate(X_train.columns) if col in causal_features]
                X_train_causal = X_train_scaled[:, causal_indices]
                
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
        
        # Full models (using all features)
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
        
        # DNN (if available)
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
        
        return self.models
    
    def _evaluate_models(self):
        """Evaluate all models and store performance metrics"""
        X_test, y_test = self.test_data
        X_test_scaled = None
        
        # Scale test data for DNNs
        if TENSORFLOW_AVAILABLE and hasattr(self, 'scaler'):
            X_test_scaled = self.scaler.transform(X_test)
        
        for name, model in self.models.items():
            if 'features' in name:
                continue  # Skip feature indices entries
                
            # Get features used by this model
            if name.startswith('causal_'):
                # Get causal parents
                parents = list(self.causal_graph.predecessors(self.target_col))
                features = [p for p in parents if p in X_test.columns]
                
                if 'dnn' in name and X_test_scaled is not None:
                    # Use scaled data for DNN
                    feature_indices = self.models.get(f'{name}_features', [])
                    X_test_model = X_test_scaled[:, feature_indices]
                else:
                    X_test_model = X_test[features]
            else:
                # Use all features
                features = X_test.columns.tolist()
                
                if 'dnn' in name and X_test_scaled is not None:
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
                    'n_features': len(features)
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
        Calculate all transparency metrics for all models.
        
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
        ccm = tm.calculate_ccm(self.models)
        
        # Initialize metrics structure
        self.metrics = {
            'cii': cii,
            'ccm': ccm,
            'te': {},
            'cs': {}
        }
        
        # Calculate TE and CS for each model
        X_test, y_test = self.test_data
        X_test_scaled = None
        
        # Scale test data for DNNs
        if TENSORFLOW_AVAILABLE and hasattr(self, 'scaler'):
            X_test_scaled = self.scaler.transform(X_test)
        
        for name, model in self.models.items():
            if 'features' in name:
                continue  # Skip feature indices entries
                
            print(f"\nCalculating metrics for model: {name}")
            
            # Get features used by this model
            if name.startswith('causal_'):
                # Get causal parents
                parents = list(self.causal_graph.predecessors(self.target_col))
                features = [p for p in parents if p in X_test.columns]
                
                if 'dnn' in name and X_test_scaled is not None:
                    # Use scaled data for DNN
                    feature_indices = self.models.get(f'{name}_features', [])
                    X_test_model = X_test_scaled[:, feature_indices]
                else:
                    X_test_model = X_test[features]
            else:
                # Use all features
                features = X_test.columns.tolist()
                
                if 'dnn' in name and X_test_scaled is not None:
                    # Use scaled data for DNN
                    X_test_model = X_test_scaled
                else:
                    X_test_model = X_test
            
            # Calculate TE
            te = tm.calculate_te(model, X_test_model, y_test)
            self.metrics['te'][name] = te
            
            # Calculate CS
            cs = tm.calculate_cs(model, X_test_model)
            self.metrics['cs'][name] = cs
            
            # Create visualization for this model
            tm.visualize_metrics(model_name=name)
        
        # Create comparison visualizations
        self._visualize_model_comparison()
        
        # Save metrics to file
        self._save_metrics()
        
        return self.metrics
    
    def _visualize_model_comparison(self):
        """Create visualizations comparing different models"""
        if not self.model_performance:
            print("No model performance data available")
            return
        
        # Create a DataFrame from model performance
        df = []
        for model_name, metrics in self.model_performance.items():
            if 'features' not in model_name:  # Skip feature indices entries
                model_type = 'DNN' if 'dnn' in model_name else 'Traditional'
                df.append({
                    'Model': model_name,
                    'AUC': metrics.get('auc', 0),
                    'Accuracy': metrics.get('accuracy', 0),
                    'F1 Score': metrics.get('f1', 0),
                    'Features': metrics.get('n_features', 0),
                    'Causal': model_name.startswith('causal_'),
                    'Type': model_type
                })
        
        df = pd.DataFrame(df)
        
        # Create a comparison chart
        plt.figure(figsize=(12, 8))
        
        # Sort by AUC
        df = df.sort_values('AUC', ascending=False)
        
        # Color by causal vs. all features
        colors = ['#3366cc' if row['Causal'] else '#cc3366' for _, row in df.iterrows()]
        
        # Plot AUC, accuracy, and F1
        bar_width = 0.25
        index = np.arange(len(df))
        
        plt.bar(index, df['AUC'], bar_width, label='AUC', color=colors, alpha=0.8)
        plt.bar(index + bar_width, df['Accuracy'], bar_width, label='Accuracy', color=colors, alpha=0.6)
        plt.bar(index + 2*bar_width, df['F1 Score'], bar_width, label='F1 Score', color=colors, alpha=0.4)
        
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(index + bar_width, df['Model'], rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save figure
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'visualizations', 'model_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Model comparison chart saved to {output_path}")
        
        # Create CTF metrics comparison
        if self.metrics and 'te' in self.metrics and 'cs' in self.metrics:
            plt.figure(figsize=(12, 8))
            
            # Prepare data
            model_names = list(self.metrics['te'].keys())
            te_values = [self.metrics['te'][m].get('te', 0) for m in model_names]
            cs_values = [self.metrics['cs'][m].get('overall', 0) for m in model_names]
            auc_values = [self.model_performance[m].get('auc', 0) for m in model_names]
            
            # Sort by AUC
            sorted_indices = np.argsort(auc_values)[::-1]
            model_names = [model_names[i] for i in sorted_indices]
            te_values = [te_values[i] for i in sorted_indices]
            cs_values = [cs_values[i] for i in sorted_indices]
            auc_values = [auc_values[i] for i in sorted_indices]
            
            # Color by causal vs. all features
            colors = ['#3366cc' if m.startswith('causal_') else '#cc3366' for m in model_names]
            
            # Plot TE, CS, and AUC
            bar_width = 0.25
            index = np.arange(len(model_names))
            
            plt.bar(index, te_values, bar_width, label='TE', color=colors, alpha=0.8)
            plt.bar(index + bar_width, cs_values, bar_width, label='CS', color=colors, alpha=0.6)
            plt.bar(index + 2*bar_width, auc_values, bar_width, label='AUC', color=colors, alpha=0.4)
            
            plt.xlabel('Model')
            plt.ylabel('Score')
            plt.title('CTF Metrics Comparison')
            plt.xticks(index + bar_width, model_names, rotation=45, ha='right')
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Save figure
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, 'visualizations', 'ctf_metrics_comparison.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"CTF metrics comparison chart saved to {output_path}")
    
    def _save_metrics(self):
        """Save metrics to JSON file"""
        # Convert all values to basic Python types for JSON
        def to_serializable(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_serializable(i) for i in obj]
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            else:
                return obj
        
        # Convert metrics
        serializable_metrics = to_serializable(self.metrics)
        
        # Save to file
        with open(os.path.join(self.output_dir, 'ctf_metrics.json'), 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
            
        # Save model performance
        serializable_performance = to_serializable(self.model_performance)
        
        with open(os.path.join(self.output_dir, 'model_performance.json'), 'w') as f:
            json.dump(serializable_performance, f, indent=2)
        
        print("Metrics saved to output directory")
    
    def generate_report(self):
        """Generate comprehensive CTF report"""
        report_path = os.path.join(self.output_dir, 'report', 'ctf_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Causal Transparency Framework Report\n\n")
            
            # Dataset summary
            f.write("## Dataset Summary\n\n")
            f.write(f"- Dataset size: {self.df.shape[0]} rows, {self.df.shape[1]} columns\n")
            f.write(f"- Target variable: {self.target_col}\n")
            
            # Causal graph summary
            if self.causal_graph:
                f.write("\n## Causal Graph Summary\n\n")
                f.write(f"- Nodes: {len(self.causal_graph.nodes())}\n")
                f.write(f"- Edges: {len(self.causal_graph.edges())}\n")
                
                # List top incoming edges to target
                f.write("\n### Key Causal Relationships\n\n")
                
                parents = list(self.causal_graph.predecessors(self.target_col))
                if parents:
                    f.write("| Feature | Causal Strength |\n")
                    f.write("|---------|----------------|\n")
                    
                    for parent in parents:
                        strength = self.causal_graph[parent][self.target_col].get('weight', 0.0)
                        f.write(f"| {parent} | {strength:.4f} |\n")
                else:
                    f.write("No direct causal relationships found.\n")
            
            # Model performance
            f.write("\n## Model Performance\n\n")
            
            if self.model_performance:
                f.write("| Model | AUC | Accuracy | F1 Score | Features |\n")
                f.write("|-------|-----|----------|----------|----------|\n")
                
                for model_name, metrics in self.model_performance.items():
                    auc = metrics.get('auc', 0)
                    accuracy = metrics.get('accuracy', 0)
                    f1 = metrics.get('f1', 0)
                    features = metrics.get('n_features', 0)
                    
                    f.write(f"| {model_name} | {auc:.4f} | {accuracy:.4f} | {f1:.4f} | {features} |\n")
            else:
                f.write("No model performance data available.\n")
            
            # CTF Metrics
            f.write("\n## Transparency Metrics\n\n")
            
            # CII
            if 'cii' in self.metrics:
                f.write("### Causal Influence Index (CII)\n\n")
                f.write("| Feature | CII |\n")
                f.write("|---------|-----|\n")
                
                top_features = list(self.metrics['cii'].items())
                for feature, score in top_features[:10]:  # Top 10 features
                    f.write(f"| {feature} | {score:.4f} |\n")
                
                if len(top_features) > 10:
                    f.write("*Only top 10 features shown*\n")
            
            # CCM
            if 'ccm' in self.metrics:
                f.write("\n### Causal Complexity Measure (CCM)\n\n")
                f.write(f"- Graph complexity: {self.metrics['ccm'].get('k_g', 0):.2f}\n")
                f.write(f"- Model complexity: {self.metrics['ccm'].get('k_f', 0):.2f}\n")
                f.write(f"- Overall CCM: {self.metrics['ccm'].get('raw_ccm', 0):.2f}\n")
                f.write(f"- Scaled CCM: {self.metrics['ccm'].get('scaled_ccm', 0):.2f}\n")
            
            # TE and CS per model
            if 'te' in self.metrics and 'cs' in self.metrics:
                f.write("\n### Transparency Entropy (TE) and Counterfactual Stability (CS)\n\n")
                
                f.write("| Model | TE | CS |\n")
                f.write("|-------|----|----||\n")
                
                for model_name in self.metrics['te'].keys():
                    te = self.metrics['te'][model_name].get('te', 0)
                    cs = self.metrics['cs'][model_name].get('overall', 0)
                    
                    f.write(f"| {model_name} | {te:.4f} | {cs:.4f} |\n")
            
            # Add visualizations
            f.write("\n## Visualizations\n\n")
            
            # Causal graph
            f.write("### Causal Graph\n\n")
            f.write("![Causal Graph](../causal_models/graphs/ensemble_graph.png)\n\n")
            
            # CTF Radar Chart
            f.write("### CTF Metrics Radar Chart\n\n")
            
            if 'te' in self.metrics:
                for model_name in self.metrics['te'].keys():
                    f.write(f"#### {model_name}\n\n")
                    f.write(f"![CTF Radar Chart](../visualizations/ctf_radar_{model_name}.png)\n\n")
            
            # Model Comparison
            f.write("### Model Comparison\n\n")
            f.write("![Model Comparison](../visualizations/model_comparison.png)\n\n")
            
            # CTF Metrics Comparison
            f.write("### CTF Metrics Comparison\n\n")
            f.write("![CTF Metrics Comparison](../visualizations/ctf_metrics_comparison.png)\n\n")
            
            # Conclusions
            f.write("\n## Conclusions\n\n")
            
            f.write("Based on the analysis, the following conclusions can be drawn:\n\n")
            
            # Check if we have causal models
            causal_models = [m for m in self.model_performance.keys() if m.startswith('causal_')]
            full_models = [m for m in self.model_performance.keys() if not m.startswith('causal_')]
            
            if causal_models and full_models:
                # Compare best causal model with best full model
                best_causal = max(causal_models, key=lambda m: self.model_performance[m].get('auc', 0))
                best_full = max(full_models, key=lambda m: self.model_performance[m].get('auc', 0))
                
                causal_auc = self.model_performance[best_causal].get('auc', 0)
                full_auc = self.model_performance[best_full].get('auc', 0)
                
                if causal_auc >= 0.95 * full_auc:
                    f.write("1. The causal model achieves comparable performance to the full model, suggesting that the identified causal features capture most of the predictive signal.\n")
                else:
                    f.write("1. The full model outperforms the causal model, indicating that there are predictive features not captured in the causal structure.\n")
                
                # Compare TE and CS
                if 'te' in self.metrics and 'cs' in self.metrics:
                    causal_te = self.metrics['te'][best_causal].get('te', 0)
                    full_te = self.metrics['te'][best_full].get('te', 0)
                    
                    causal_cs = self.metrics['cs'][best_causal].get('overall', 0)
                    full_cs = self.metrics['cs'][best_full].get('overall', 0)
                    
                    if causal_te > full_te:
                        f.write("2. The causal model offers higher transparency entropy, making it more interpretable.\n")
                    else:
                        f.write("2. The full model offers higher transparency entropy, despite using more features.\n")
                        
                    if causal_cs > full_cs:
                        f.write("3. The causal model provides better counterfactual stability, making its predictions more robust.\n")
                    else:
                        f.write("3. The full model provides better counterfactual stability, potentially due to its use of redundant features.\n")
            
            # Add top causal drivers
            if 'cii' in self.metrics and self.metrics['cii']:
                f.write("\n4. The key causal drivers of the prediction are:\n")
                
                for i, (feature, score) in enumerate(list(self.metrics['cii'].items())[:3]):
                    f.write(f"   - {feature} (CII: {score:.4f})\n")
            
            f.write("\n*This report was automatically generated by the Causal Transparency Framework.*\n")
        
        print(f"CTF report generated at {report_path}")
        return report_path
    
    def run_complete_pipeline(self, domain_knowledge=None):
        """
        Run the complete CTF pipeline from data loading to report generation.
        
        Args:
            domain_knowledge: Optional dict with domain-specific causal information
            
        Returns:
            Path to generated report
        """
        # Step 1: Discover causal structure
        self.discover_causal_structure(domain_knowledge)
        
        # Step 2: Train models
        self.train_models()
        
        # Step 3: Calculate transparency metrics
        self.calculate_transparency_metrics()
        
        # Step 4: Generate report
        report_path = self.generate_report()
        
        print("Complete CTF pipeline execution finished!")
        return report_path