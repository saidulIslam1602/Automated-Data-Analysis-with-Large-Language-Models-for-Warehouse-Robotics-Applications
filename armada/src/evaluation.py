"""
Model Evaluation for Warehouse Robotics Maintenance

This module contains evaluation metrics and utilities for assessing model performance:
1. Standard classification metrics
2. Time-to-failure prediction metrics
3. Maintenance cost optimization evaluation
4. Domain adaptation evaluation
5. Visualization utilities for evaluation results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import learning_curve
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MaintenanceModelEvaluator:
    """
    Evaluator for maintenance prediction models.
    
    Provides metrics specifically designed for predictive maintenance
    and methods to compare different models.
    """
    
    def __init__(self, metrics=None):
        """
        Initialize the evaluator.
        
        Args:
            metrics: List of metrics to calculate
        """
        self.metrics = metrics or [
            'accuracy', 'precision', 'recall', 'f1',
            'roc_auc', 'confusion_matrix'
        ]
        self.results = {}
        
    def evaluate(self, y_true, y_pred):
        """
        Evaluate model performance using specified metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = {}
        
        # Calculate standard metrics
        if 'accuracy' in self.metrics:
            results['accuracy'] = accuracy_score(y_true, y_pred)
        if 'precision' in self.metrics:
            results['precision'] = precision_score(y_true, y_pred)
        if 'recall' in self.metrics:
            results['recall'] = recall_score(y_true, y_pred)
        if 'f1' in self.metrics:
            results['f1'] = f1_score(y_true, y_pred)
        if 'roc_auc' in self.metrics:
            results['roc_auc'] = auc(roc_curve(y_true, y_pred)[0], roc_curve(y_true, y_pred)[1])
        if 'confusion_matrix' in self.metrics:
            results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        return results
    
    def evaluate_classifier(self, y_true, y_pred, y_proba=None, model_name='model'):
        """
        Evaluate a binary classification model for maintenance prediction.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (if available)
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating classifier: {model_name}")
        
        results = self.evaluate(y_true, y_pred)
        
        # Log results
        logger.info(f"Evaluation results for {model_name}:")
        for metric, value in results.items():
            if not isinstance(value, dict) and not isinstance(value, np.ndarray):
                logger.info(f"  {metric}: {value:.4f}")
        
        # Store results
        self.results[model_name] = results
        
        return results
    
    def evaluate_regression(self, y_true, y_pred, model_name='model'):
        """
        Evaluate a regression model for remaining useful life prediction.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating regression model: {model_name}")
        
        results = self.evaluate(y_true, y_pred)
        
        # Log results
        logger.info(f"Evaluation results for {model_name}:")
        for metric, value in results.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
        
        # Store results
        self.results[model_name] = results
        
        return results
    
    def compare_models(self, metrics=None):
        """
        Compare multiple models on selected metrics.
        
        Args:
            metrics: List of metrics to compare
            
        Returns:
            DataFrame with comparison results
        """
        if not self.results:
            logger.warning("No evaluation results available for comparison")
            return None
            
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'confusion_matrix']
            
        # Extract metrics from results
        comparison = {}
        
        for model_name, results in self.results.items():
            model_metrics = {}
            
            for metric in metrics:
                if metric in results and not isinstance(results[metric], (dict, np.ndarray)):
                    model_metrics[metric] = results[metric]
                else:
                    model_metrics[metric] = None
                    
            comparison[model_name] = model_metrics
            
        return pd.DataFrame(comparison).T
    
    def evaluate_learning_curve(self, estimator, X, y, cv=5, train_sizes=None, model_name='model'):
        """
        Evaluate learning curve for a model.
        
        Args:
            estimator: Model to evaluate
            X: Features
            y: Target
            cv: Cross-validation folds
            train_sizes: List of training set sizes
            model_name: Name of the model
            
        Returns:
            Dictionary with learning curve data
        """
        logger.info(f"Evaluating learning curve for {model_name}")
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 5)
            
        # Calculate learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, train_sizes=train_sizes, scoring='f1',
            n_jobs=-1, verbose=1
        )
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        results = {
            'train_sizes': train_sizes,
            'train_mean': train_mean,
            'train_std': train_std,
            'test_mean': test_mean,
            'test_std': test_std
        }
        
        # Store results
        if model_name not in self.results:
            self.results[model_name] = {}
            
        self.results[model_name]['learning_curve'] = results
        
        return results
    
    def plot_confusion_matrix(self, model_name=None, ax=None, cmap='Blues'):
        """
        Plot confusion matrix.
        
        Args:
            model_name: Name of the model
            ax: Matplotlib axis
            cmap: Colormap
            
        Returns:
            Matplotlib axis
        """
        if model_name is None:
            if not self.results:
                logger.warning("No evaluation results available")
                return None
                
            model_name = list(self.results.keys())[0]
            
        if model_name not in self.results:
            logger.warning(f"No results found for model: {model_name}")
            return None
            
        results = self.results[model_name]
        
        if 'confusion_matrix' not in results:
            logger.warning(f"No confusion matrix found for model: {model_name}")
            return None
            
        cm = results['confusion_matrix']
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title(f'Confusion Matrix - {model_name}')
        
        # Add class labels
        ax.set_xticklabels(['No Maintenance', 'Maintenance'])
        ax.set_yticklabels(['No Maintenance', 'Maintenance'])
        
        return ax
    
    def plot_roc_curve(self, model_names=None, ax=None):
        """
        Plot ROC curve.
        
        Args:
            model_names: List of model names to plot
            ax: Matplotlib axis
            
        Returns:
            Matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            
        if model_names is None:
            model_names = list(self.results.keys())
            
        # Plot ROC curve for each model
        for model_name in model_names:
            if model_name not in self.results:
                logger.warning(f"No results found for model: {model_name}")
                continue
                
            results = self.results[model_name]
            
            if 'roc_curve' not in results or 'roc_auc' not in results:
                logger.warning(f"No ROC curve data found for model: {model_name}")
                continue
                
            roc_data = results['roc_curve']
            roc_auc = results['roc_auc']
            
            ax.plot(roc_data['fpr'], roc_data['tpr'], 
                   label=f'{model_name} (AUC = {roc_auc:.3f})')
            
        # Add diagonal line
        ax.plot([0, 1], [0, 1], 'k--')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc='lower right')
        
        return ax
    
    def plot_precision_recall_curve(self, model_names=None, ax=None):
        """
        Plot precision-recall curve.
        
        Args:
            model_names: List of model names to plot
            ax: Matplotlib axis
            
        Returns:
            Matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            
        if model_names is None:
            model_names = list(self.results.keys())
            
        # Plot precision-recall curve for each model
        for model_name in model_names:
            if model_name not in self.results:
                logger.warning(f"No results found for model: {model_name}")
                continue
                
            results = self.results[model_name]
            
            if 'pr_curve' not in results or 'pr_auc' not in results:
                logger.warning(f"No precision-recall curve data found for model: {model_name}")
                continue
                
            pr_data = results['pr_curve']
            pr_auc = results['pr_auc']
            
            ax.plot(pr_data['recall'], pr_data['precision'], 
                   label=f'{model_name} (AUC = {pr_auc:.3f})')
            
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc='lower left')
        
        return ax
    
    def plot_learning_curve(self, model_name=None, ax=None):
        """
        Plot learning curve.
        
        Args:
            model_name: Name of the model
            ax: Matplotlib axis
            
        Returns:
            Matplotlib axis
        """
        if model_name is None:
            if not self.results:
                logger.warning("No evaluation results available")
                return None
                
            model_name = list(self.results.keys())[0]
            
        if model_name not in self.results:
            logger.warning(f"No results found for model: {model_name}")
            return None
            
        results = self.results[model_name]
        
        if 'learning_curve' not in results:
            logger.warning(f"No learning curve data found for model: {model_name}")
            return None
            
        lc_data = results['learning_curve']
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        # Plot learning curve
        ax.plot(lc_data['train_sizes'], lc_data['train_mean'], 'o-', color='r', label='Training score')
        ax.fill_between(lc_data['train_sizes'], 
                        lc_data['train_mean'] - lc_data['train_std'],
                        lc_data['train_mean'] + lc_data['train_std'], 
                        alpha=0.1, color='r')
        
        ax.plot(lc_data['train_sizes'], lc_data['test_mean'], 'o-', color='g', label='Validation score')
        ax.fill_between(lc_data['train_sizes'], 
                        lc_data['test_mean'] - lc_data['test_std'],
                        lc_data['test_mean'] + lc_data['test_std'], 
                        alpha=0.1, color='g')
        
        ax.set_xlabel('Training Examples')
        ax.set_ylabel('F1 Score')
        ax.set_title(f'Learning Curve - {model_name}')
        ax.grid(True)
        ax.legend(loc='best')
        
        return ax
    
    def plot_model_comparison(self, metrics=None, figsize=(12, 8)):
        """
        Plot model comparison for selected metrics.
        
        Args:
            metrics: List of metrics to compare
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        comparison_df = self.compare_models(metrics)
        
        if comparison_df is None:
            return None
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot comparison
        comparison_df.plot(kind='bar', ax=ax)
        
        ax.set_title('Model Comparison')
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.legend(title='Metric')
        
        plt.tight_layout()
        
        return fig
    
    def plot_cost_analysis(self, model_names=None, figsize=(12, 8)):
        """
        Plot cost analysis for selected models.
        
        Args:
            model_names: List of model names to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if model_names is None:
            model_names = list(self.results.keys())
            
        # Extract cost metrics
        cost_metrics = ['false_positive_cost', 'false_negative_cost', 'maintenance_cost']
        cost_data = []
        
        for model_name in model_names:
            if model_name not in self.results:
                logger.warning(f"No results found for model: {model_name}")
                continue
                
            results = self.results[model_name]
            
            model_costs = {}
            for metric in cost_metrics:
                if metric in results:
                    model_costs[metric] = results[metric]
                else:
                    model_costs[metric] = None
                    
            if all(model_costs.values()):
                cost_data.append({
                    'model': model_name,
                    **model_costs
                })
                
        if not cost_data:
            logger.warning("No cost data available for plotting")
            return None
            
        # Create DataFrame
        cost_df = pd.DataFrame(cost_data)
        cost_df = cost_df.set_index('model')
        
        # Plot stacked bar chart
        fig, ax = plt.subplots(figsize=figsize)
        
        cost_df.plot(kind='bar', stacked=True, ax=ax)
        
        ax.set_title('Cost Analysis by Model')
        ax.set_xlabel('Model')
        ax.set_ylabel('Cost')
        ax.legend(title='Cost Type')
        
        # Add total cost as text above each bar
        for i, model in enumerate(cost_df.index):
            total = cost_df.loc[model].sum()
            ax.text(i, total + 100, f'Total: {total:.0f}', ha='center')
            
        plt.tight_layout()
        
        return fig
    
    def save_results(self, output_dir, prefix=None):
        """
        Save evaluation results and plots to files.
        
        Args:
            output_dir: Output directory
            prefix: Prefix for filenames
            
        Returns:
            Dictionary with saved file paths
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        if prefix is None:
            prefix = datetime.now().strftime('%Y%m%d_%H%M%S')
            
        saved_files = {}
        
        # Save comparison DataFrame
        comparison_df = self.compare_models()
        if comparison_df is not None:
            comparison_path = os.path.join(output_dir, f'{prefix}_model_comparison.csv')
            comparison_df.to_csv(comparison_path)
            saved_files['comparison_csv'] = comparison_path
            
        # Save confusion matrices
        for model_name in self.results:
            if 'confusion_matrix' in self.results[model_name]:
                fig, ax = plt.subplots(figsize=(8, 6))
                self.plot_confusion_matrix(model_name=model_name, ax=ax)
                
                cm_path = os.path.join(output_dir, f'{prefix}_{model_name}_confusion_matrix.png')
                fig.savefig(cm_path)
                plt.close(fig)
                
                saved_files[f'{model_name}_confusion_matrix'] = cm_path
                
        # Save ROC curve
        fig, ax = plt.subplots(figsize=(8, 6))
        self.plot_roc_curve(ax=ax)
        
        roc_path = os.path.join(output_dir, f'{prefix}_roc_curve.png')
        fig.savefig(roc_path)
        plt.close(fig)
        
        saved_files['roc_curve'] = roc_path
        
        # Save precision-recall curve
        fig, ax = plt.subplots(figsize=(8, 6))
        self.plot_precision_recall_curve(ax=ax)
        
        pr_path = os.path.join(output_dir, f'{prefix}_precision_recall_curve.png')
        fig.savefig(pr_path)
        plt.close(fig)
        
        saved_files['precision_recall_curve'] = pr_path
        
        # Save model comparison plot
        fig = self.plot_model_comparison()
        if fig is not None:
            comparison_plot_path = os.path.join(output_dir, f'{prefix}_model_comparison.png')
            fig.savefig(comparison_plot_path)
            plt.close(fig)
            
            saved_files['model_comparison_plot'] = comparison_plot_path
            
        # Save cost analysis plot
        fig = self.plot_cost_analysis()
        if fig is not None:
            cost_plot_path = os.path.join(output_dir, f'{prefix}_cost_analysis.png')
            fig.savefig(cost_plot_path)
            plt.close(fig)
            
            saved_files['cost_analysis_plot'] = cost_plot_path
            
        # Save raw results as JSON
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for model_name, results in self.results.items():
            json_results[model_name] = {}
            
            for metric, value in results.items():
                if isinstance(value, np.ndarray):
                    json_results[model_name][metric] = value.tolist()
                elif isinstance(value, dict):
                    json_results[model_name][metric] = {}
                    for k, v in value.items():
                        if isinstance(v, np.ndarray):
                            json_results[model_name][metric][k] = v.tolist()
                        else:
                            json_results[model_name][metric][k] = v
                else:
                    json_results[model_name][metric] = value
                    
        results_path = os.path.join(output_dir, f'{prefix}_results.json')
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
            
        saved_files['results_json'] = results_path
        
        logger.info(f"Saved evaluation results to {output_dir}")
        
        return saved_files

def evaluate_domain_adaptation(model, X_source, y_source, X_target, y_target):
    """
    Evaluate domain adaptation performance.
    
    Args:
        model: Trained domain adaptation model
        X_source: Source domain features
        y_source: Source domain labels
        X_target: Target domain features
        y_target: Target domain labels
        
    Returns:
        Dictionary with domain adaptation metrics
    """
    # Evaluate on source domain
    y_pred_source = model.predict(X_source)
    source_metrics = {
        'source_accuracy': accuracy_score(y_source, y_pred_source),
        'source_f1': f1_score(y_source, y_pred_source, zero_division=0)
    }
    
    # Evaluate on target domain
    y_pred_target = model.predict(X_target)
    target_metrics = {
        'target_accuracy': accuracy_score(y_target, y_pred_target),
        'target_f1': f1_score(y_target, y_pred_target, zero_division=0)
    }
    
    # Calculate domain adaptation metrics
    adaptation_gap = source_metrics['source_f1'] - target_metrics['target_f1']
    
    metrics = {
        **source_metrics,
        **target_metrics,
        'adaptation_gap': adaptation_gap,
        'adaptation_ratio': target_metrics['target_f1'] / source_metrics['source_f1'] if source_metrics['source_f1'] > 0 else 0
    }
    
    return metrics

def plot_domain_adaptation_results(results, figsize=(12, 8)):
    """
    Plot domain adaptation results.
    
    Args:
        results: Dictionary with domain adaptation results
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot accuracy comparison
    models = list(results.keys())
    source_acc = [results[model]['source_accuracy'] for model in models]
    target_acc = [results[model]['target_accuracy'] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax1.bar(x - width/2, source_acc, width, label='Source Domain')
    ax1.bar(x + width/2, target_acc, width, label='Target Domain')
    
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Domain Adaptation: Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    
    # Plot adaptation ratio
    adaptation_ratio = [results[model]['adaptation_ratio'] for model in models]
    
    ax2.bar(x, adaptation_ratio)
    ax2.axhline(y=1.0, color='r', linestyle='--', label='Perfect Adaptation')
    
    ax2.set_ylabel('Adaptation Ratio (Target F1 / Source F1)')
    ax2.set_title('Domain Adaptation Efficiency')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend()
    
    plt.tight_layout()
    
    return fig

# Example usage if run directly
if __name__ == "__main__":
    # Generate some example data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_proba = np.random.random(100)
    
    # Create evaluator
    evaluator = MaintenanceModelEvaluator()
    
    # Evaluate model
    results = evaluator.evaluate_classifier(y_true, y_pred, y_proba, model_name='example_model')
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    evaluator.plot_confusion_matrix(model_name='example_model')
    plt.tight_layout()
    plt.show()
    
    print("Evaluation metrics:")
    for metric, value in results.items():
        if not isinstance(value, dict) and not isinstance(value, np.ndarray):
            print(f"  {metric}: {value:.4f}")