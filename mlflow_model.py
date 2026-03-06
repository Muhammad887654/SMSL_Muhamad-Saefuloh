"""
MLflow Integration for Corn Fertilizer Classification Model
Muhamad Saefuloh - SMSL Project

This module integrates:
- MLflow for experiment tracking
- DagsHub for remote storage and collaboration
- Model versioning and registry
- Dashboard for model monitoring
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# MLflow imports
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Try to import XGBoost and LightGBM
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Try to import DagsHub
try:
    import dagshub
    DAGSHUB_AVAILABLE = True
except ImportError:
    DAGSHUB_AVAILABLE = False


class MLflowExperimentTracker:
    """
    MLflow experiment tracker for ML model training with DagsHub integration.
    """
    
    def __init__(self, experiment_name="SMSL_Corn_Classification", 
                 tracking_uri=None, dagshub_repo=None):
        """
        Initialize MLflow tracker.
        
        Parameters:
        -----------
        experiment_name : str
            Name of the MLflow experiment
        tracking_uri : str
            MLflow tracking URI (local or DagsHub)
        dagshub_repo : str
            DagsHub repository in format "owner/repo"
        """
        self.experiment_name = experiment_name
        self.dagshub_repo = dagshub_repo
        self.tracking_uri = tracking_uri
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models_results = {}
        
    def setup_mlflow(self, use_dagshub=True):
        """
        Setup MLflow with local or DagsHub tracking.
        
        Parameters:
        -----------
        use_dagshub : bool
            Whether to use DagsHub for remote tracking
        """
        if use_dagshub and DAGSHUB_AVAILABLE and self.dagshub_repo:
            # Setup DagsHub MLflow
            dagshub.init(repo_name=self.dagshub_repo.split('/')[-1], 
                        repo_owner=self.dagshub_repo.split('/')[0],
                        mlflow=True)
            self.tracking_uri = mlflow.get_tracking_uri()
            print(f"✅ MLflow connected to DagsHub: {self.tracking_uri}")
        elif self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
            print(f"✅ MLflow connected to: {self.tracking_uri}")
        else:
            # Use local MLflow
            mlflow.set_tracking_uri("mlruns")
            print("✅ MLflow connected to local storage")
        
        # Set experiment
        mlflow.set_experiment(self.experiment_name)
        print(f"✅ Experiment set: {self.experiment_name}")
        
        return self
    
    def load_and_prepare_data(self, data_path):
        """Load and prepare data for training."""
        print("\n" + "="*60)
        print("LOADING DATA")
        print("="*60)
        
        self.df = pd.read_csv(data_path)
        target_column = 'Fertilizer_Category'
        
        # Prepare features and target
        self.X = self.df.drop(columns=[target_column, 'County', 'Farmer', 'Crop'])
        self.y = self.df[target_column]
        
        # Encode target
        self.y = self.label_encoder.fit_transform(self.y)
        self.target_classes = self.label_encoder.classes_
        
        print(f"Dataset: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        print(f"Classes: {list(self.target_classes)}")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"Train: {self.X_train.shape[0]}, Test: {self.X_test.shape[0]}")
        
        return self
    
    def get_models(self):
        """Get dictionary of models to train."""
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100, random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                max_iter=1000, random_state=42
            ),
            'SVM': SVC(
                probability=True, random_state=42
            ),
            'DecisionTree': DecisionTreeClassifier(
                random_state=42
            ),
            'KNN': KNeighborsClassifier(),
        }
        
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = XGBClassifier(
                random_state=42, use_label_encoder=False, 
                eval_metric='mlogloss', verbosity=0
            )
        
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = LGBMClassifier(
                random_state=42, verbose=-1
            )
        
        return models
    
    def log_model_metrics(self, model, model_name, X_test, y_test):
        """Calculate and return model metrics."""
        y_pred = model.predict(X_test)
        
        # Get probabilities for ROC-AUC
        try:
            y_proba = model.predict_proba(X_test)
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        except:
            y_proba = None
            roc_auc = 0.0
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc
        }
        
        return metrics, y_pred
    
    def create_plots(self, model, model_name, y_test, y_pred):
        """Create visualization plots for the model."""
        plots = {}
        
        # 1. Confusion Matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.target_classes,
                   yticklabels=self.target_classes, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()
        
        # Save to temp file
        cm_path = f'confusion_matrix_{model_name}.png'
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        plots['confusion_matrix'] = cm_path
        
        # 2. Feature Importance (if available)
        if hasattr(model, 'feature_importances_'):
            fig, ax = plt.subplots(figsize=(10, 6))
            importance_df = pd.DataFrame({
                'feature': self.X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True).tail(15)
            
            ax.barh(importance_df['feature'], importance_df['importance'], color='steelblue')
            ax.set_xlabel('Importance')
            ax.set_title(f'Top 15 Feature Importance - {model_name}')
            plt.tight_layout()
            
            fi_path = f'feature_importance_{model_name}.png'
            plt.savefig(fi_path, dpi=150, bbox_inches='tight')
            plt.close()
            plots['feature_importance'] = fi_path
        
        return plots
    
    def train_and_log_model(self, model, model_name):
        """
        Train a single model and log to MLflow.
        """
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print("="*60)
        
        with mlflow.start_run(run_name=model_name) as run:
            # Log parameters
            mlflow.log_param('model_type', model_name)
            mlflow.log_param('n_features', self.X.shape[1])
            mlflow.log_param('n_train_samples', self.X_train.shape[0])
            mlflow.log_param('n_test_samples', self.X_test.shape[0])
            
            # Log model parameters
            if hasattr(model, 'get_params'):
                for param, value in model.get_params().items():
                    mlflow.log_param(param, value)
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Calculate metrics
            metrics, y_pred = self.log_model_metrics(
                model, model_name, self.X_test, self.y_test
            )
            
            # Log metrics
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)
            
            # Cross-validation scores
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv)
            mlflow.log_metric('cv_mean', cv_scores.mean())
            mlflow.log_metric('cv_std', cv_scores.std())
            
            # Log model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=f"models/{model_name}",
                registered_model_name=model_name if self.dagshub_repo else None
            )
            
            # Create and log plots
            plots = self.create_plots(model, model_name, self.y_test, y_pred)
            for plot_name, plot_path in plots.items():
                mlflow.log_artifact(plot_path)
            
            # Log classification report
            report = classification_report(
                self.y_test, y_pred, 
                target_names=self.target_classes,
                output_dict=True
            )
            
            # Log additional info as JSON
            mlflow.log_dict(report, 'classification_report.json')
            
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
            print(f"  ROC-AUC:  {metrics['roc_auc']:.4f}")
            print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
            
            # Store results
            self.models_results[model_name] = {
                'model': model,
                'metrics': metrics,
                'run_id': run.info.run_id,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            return model, metrics
    
    def train_all_models(self, data_path):
        """
        Train all models and track with MLflow.
        """
        # Setup
        self.setup_mlflow(use_dagshub=True)
        
        # Load data
        self.load_and_prepare_data(data_path)
        
        # Get models
        models = self.get_models()
        
        # Train each model
        for model_name, model in models.items():
            self.train_and_log_model(model, model_name)
        
        # Print summary
        self.print_summary()
        
        # Register best model
        self.register_best_model()
        
        return self.models_results
    
    def print_summary(self):
        """Print summary of all trained models."""
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        
        summary_data = []
        for name, result in self.models_results.items():
            summary_data.append({
                'Model': name,
                'Accuracy': result['metrics']['accuracy'],
                'F1-Score': result['metrics']['f1_score'],
                'ROC-AUC': result['metrics']['roc_auc'],
                'CV Mean': result['cv_mean']
            })
        
        summary_df = pd.DataFrame(summary_data).sort_values('F1-Score', ascending=False)
        print(summary_df.to_string(index=False))
        
        # Best model
        best_model = summary_df.iloc[0]['Model']
        print(f"\n🏆 Best Model: {best_model}")
        
        return summary_df
    
    def register_best_model(self):
        """Register the best model to MLflow Model Registry."""
        if not self.models_results:
            return
        
        # Find best model by F1-score
        best_name = max(self.models_results, 
                       key=lambda x: self.models_results[x]['metrics']['f1_score'])
        best_result = self.models_results[best_name]
        
        client = MlflowClient()
        
        try:
            # Create model registry
            model_name = "CornFertilizerClassifier"
            
            # Register model
            model_uri = f"runs:/{best_result['run_id']}/models/{best_name}"
            registered_model = mlflow.register_model(model_uri, model_name)
            
            # Add description
            client.set_model_version_tag(
                model_name, 
                registered_model.version,
                "best_model",
                "true"
            )
            
            print(f"\n✅ Best model registered: {model_name} (version {registered_model.version})")
            
        except Exception as e:
            print(f"\n⚠️  Could not register model: {e}")
            print("   Model tracking will still work in MLflow UI")


class MLflowDashboard:
    """
    Dashboard class for visualizing MLflow experiment results.
    """
    
    def __init__(self, experiment_name="SMSL_Corn_Classification"):
        self.experiment_name = experiment_name
        self.client = MlflowClient()
    
    def get_experiment(self):
        """Get experiment by name."""
        exp = mlflow.get_experiment_by_name(self.experiment_name)
        if exp:
            return mlflow.get_experiment(exp.experiment_id)
        return None
    
    def get_runs(self):
        """Get all runs for the experiment."""
        exp = self.get_experiment()
        if not exp:
            print(f"Experiment '{self.experiment_name}' not found!")
            return []
        
        runs = mlflow.search_runs([exp.experiment_id])
        return runs
    
    def create_dashboard_plot(self, save_path="mlflow_dashboard.png"):
        """Create a comprehensive dashboard plot."""
        runs = self.get_runs()
        
        if runs.empty:
            print("No runs found!")
            return
        
        # Create figure
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('MLflow Experiment Dashboard\nSMSL Corn Fertilizer Classification', 
                    fontsize=16, fontweight='bold')
        
        # 1. Model Comparison - Bar Chart
        ax1 = fig.add_subplot(2, 2, 1)
        metrics_to_plot = ['metrics.f1_score', 'metrics.accuracy', 'metrics.roc_auc']
        x = np.arange(len(runs))
        width = 0.25
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in runs.columns:
                values = runs[metric].fillna(0)
                ax1.bar(x + i*width, values, width, label=metric.split('.')[-1])
        
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels([r.split('_')[0][:3] for r in runs['tags.mlflow.runName']], rotation=45)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Accuracy vs F1-Score Scatter
        ax2 = fig.add_subplot(2, 2, 2)
        if 'metrics.accuracy' in runs.columns and 'metrics.f1_score' in runs.columns:
            ax2.scatter(runs['metrics.accuracy'], runs['metrics.f1_score'], 
                        s=100, alpha=0.7, c='steelblue', edgecolors='black')
            for i, row in runs.iterrows():
                ax2.annotate(row['tags.mlflow.runName'].split('_')[0][:3], 
                           (row['metrics.accuracy'], row['metrics.f1_score']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax2.set_xlabel('Accuracy')
        ax2.set_ylabel('F1-Score')
        ax2.set_title('Accuracy vs F1-Score')
        ax2.grid(alpha=0.3)
        
        # 3. Cross-Validation Scores
        ax3 = fig.add_subplot(2, 2, 3)
        if 'metrics.cv_mean' in runs.columns:
            cv_means = runs['metrics.cv_mean']
            cv_stds = runs['metrics.cv_std'].fillna(0)
            ax3.barh(range(len(cv_means)), cv_means, xerr=cv_stds, 
                    color='lightgreen', edgecolor='darkgreen', capsize=3)
            ax3.set_yticks(range(len(cv_means)))
            ax3.set_yticklabels([r.split('_')[0][:3] for r in runs['tags.mlflow.runName']])
            ax3.set_xlabel('CV Score')
            ax3.set_title('Cross-Validation Scores')
            ax3.grid(axis='x', alpha=0.3)
        
        # 4. Training Duration
        ax4 = fig.add_subplot(2, 2, 4)
        if 'metrics.precision' in runs.columns and 'metrics.recall' in runs.columns:
            ax4.scatter(runs['metrics.recall'], runs['metrics.precision'],
                       s=runs['metrics.f1_score']*200, alpha=0.6, c='coral', edgecolors='black')
            for i, row in runs.iterrows():
                ax4.annotate(row['tags.mlflow.runName'].split('_')[0][:3],
                           (row['metrics.recall'], row['metrics.precision']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax4.set_xlabel('Recall')
        ax4.set_ylabel('Precision')
        ax4.set_title('Precision vs Recall (bubble size = F1)')
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Dashboard saved to: {save_path}")
        return save_path
    
    def print_model_leaderboard(self):
        """Print a leaderboard of all models."""
        runs = self.get_runs()
        
        if runs.empty:
            return
        
        print("\n" + "="*60)
        print("MODEL LEADERBOARD")
        print("="*60)
        
        leaderboard = runs[['tags.mlflow.runName', 
                           'metrics.accuracy', 
                           'metrics.f1_score',
                           'metrics.roc_auc',
                           'metrics.cv_mean']].copy()
        
        leaderboard.columns = ['Model', 'Accuracy', 'F1-Score', 'ROC-AUC', 'CV Mean']
        leaderboard = leaderboard.sort_values('F1-Score', ascending=False)
        
        print(leaderboard.to_string(index=False))
        
        return leaderboard


def run_training_with_mlflow(experiment_name="SMSL_Corn_Classification", 
                             data_path="preprosecessing/workflow/preprocessed_corn_data.csv",
                             dagshub_repo=None):
    """
    Main function to run training with MLflow tracking.
    
    Parameters:
    -----------
    experiment_name : str
        Name of the MLflow experiment
    data_path : str
        Path to the preprocessed data
    dagshub_repo : str
        DagsHub repository in format "owner/repo"
    
    Returns:
    --------
    dict : Results of all trained models
    """
    print("\n" + "="*70)
    print("CORN FERTILIZER CLASSIFICATION - MLflow Training")
    print("="*70)
    
    # Initialize tracker
    tracker = MLflowExperimentTracker(
        experiment_name=experiment_name,
        dagshub_repo=dagshub_repo
    )
    
    # Train all models
    results = tracker.train_all_models(data_path)
    
    # Create dashboard
    dashboard = MLflowDashboard(experiment_name)
    dashboard.create_dashboard_plot()
    dashboard.print_model_leaderboard()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nTo view results:")
    print("  1. Local: mlflow ui")
    print("  2. DagsHub: Visit your DagsHub repository")
    
    return results


def launch_mlflow_ui(port=5000):
    """
    Launch MLflow UI.
    
    Parameters:
    -----------
    port : int
        Port to run MLflow UI on
    """
    import subprocess
    subprocess.run(["mlflow", "ui", "-p", str(port), "-h", "0.0.0.0"])
    print(f"MLflow UI running at http://localhost:{port}")


if __name__ == "__main__":
    # Run training with MLflow
    results = run_training_with_mlflow(
        experiment_name="SMSL_Corn_Classification_v1",
        dagshub_repo="your_username/SMSL_Muhamad-Saefuloh-ML"  # Update with your DagsHub repo
    )
