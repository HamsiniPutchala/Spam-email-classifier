"""
Visualization utilities for plots and charts
"""

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np

class Visualizations:
    """
    Class for generating visualizations
    """
    
    @staticmethod
    def plot_confusion_matrix(cm, labels=['Ham', 'Spam'], save_path=None):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            labels: Class labels
            save_path: Path to save figure
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_roc_curve(fpr, tpr, auc_score, save_path=None):
        """
        Plot ROC curve
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            auc_score: AUC score
            save_path: Path to save figure
        """
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_wordcloud(texts, title='Word Cloud', save_path=None):
        """
        Generate word cloud from texts
        
        Args:
            texts: List of texts
            title: Title for the plot
            save_path: Path to save figure
        """
        text_combined = ' '.join(texts)
        wordcloud = WordCloud(width=800, height=400,
                            background_color='white').generate(text_combined)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_metrics_comparison(metrics_dict, save_path=None):
        """
        Plot model metrics comparison
        
        Args:
            metrics_dict: Dictionary of metrics for different models
            save_path: Path to save figure
        """
        models = list(metrics_dict.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.figure(figsize=(10, 6))
        for i, model in enumerate(models):
            values = [metrics_dict[model].get(m, 0) for m in metrics]
            plt.bar(x + i * width, values, width, label=model)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Model Metrics Comparison')
        plt.xticks(x + width / 2, metrics)
        plt.legend()
        plt.ylim([0.95, 1.0])
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
