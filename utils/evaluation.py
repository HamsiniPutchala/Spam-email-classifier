"""
Evaluation metrics computation
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, auc, precision_recall_curve
)

class EvaluationMetrics:
    """
    Class for computing evaluation metrics
    """
    
    @staticmethod
    def compute_metrics(y_true, y_pred, y_proba=None):
        """
        Compute comprehensive evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            )
        }
        
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1], zero_division=0)
            fpr, tpr, thresholds = roc_curve(y_true, y_proba[:, 1])
            metrics['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist()
            }
        
        return metrics
    
    @staticmethod
    def get_confusion_matrix(y_true, y_pred):
        """Get confusion matrix"""
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def get_classification_report(y_true, y_pred):
        """Get classification report as string"""
        return classification_report(y_true, y_pred, zero_division=0)
