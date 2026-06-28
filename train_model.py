"""
Model training script
Trains the Naive Bayes and SVM models on the dataset
"""

import os
import sys
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import joblib

# Setup path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import ModelConfig, DataConfig, EvalConfig
from utils.preprocessing import TextPreprocessor
from utils.evaluation import EvaluationMetrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Trainer class for spam classification models
    """
    
    def __init__(self):
        """Initialize the trainer"""
        self.preprocessor = TextPreprocessor()
        self.evaluator = EvaluationMetrics()
        self.vectorizer = None
        self.nb_model = None
        self.svm_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.metrics = {}
        
    def load_data(self, filepath):
        """
        Load dataset from CSV file
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Tuple of (X, y)
        """
        logger.info(f"Loading data from {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Data shape: {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()}")
            
            # Map labels
            if DataConfig.LABEL_COLUMN in df.columns:
                label_col = DataConfig.LABEL_COLUMN
            else:
                # Try common label column names
                for col in ['label', 'target', 'class', 'spam']:
                    if col in df.columns:
                        label_col = col
                        break
                else:
                    raise ValueError(f"Could not find label column. Available: {df.columns.tolist()}")
            
            if DataConfig.TEXT_COLUMN in df.columns:
                text_col = DataConfig.TEXT_COLUMN
            else:
                # Try common text column names
                for col in ['text', 'message', 'content', 'body']:
                    if col in df.columns:
                        text_col = col
                        break
                else:
                    raise ValueError(f"Could not find text column. Available: {df.columns.tolist()}")
            
            X = df[text_col].astype(str)
            y = df[label_col].map(DataConfig.LABEL_MAP)
            
            # Handle unmapped labels
            y = y.fillna(df[label_col].apply(lambda x: 1 if 'spam' in str(x).lower() else 0))
            
            logger.info(f"Class distribution:")
            logger.info(f"  Ham: {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)")
            logger.info(f"  Spam: {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_texts(self, X):
        """
        Preprocess text data
        
        Args:
            X: Array of text samples
            
        Returns:
            Preprocessed text array
        """
        logger.info("Preprocessing texts...")
        X_processed = X.apply(self.preprocessor.preprocess)
        return X_processed
    
    def vectorize_texts(self, X_train, X_test, fit=True):
        """
        Vectorize texts using TF-IDF
        
        Args:
            X_train: Training texts
            X_test: Test texts
            fit: Whether to fit the vectorizer
            
        Returns:
            Tuple of (X_train_vec, X_test_vec)
        """
        logger.info("Vectorizing texts using TF-IDF...")
        
        if fit:
            self.vectorizer = TfidfVectorizer(
                max_features=ModelConfig.MAX_FEATURES,
                min_df=ModelConfig.MIN_DF,
                max_df=ModelConfig.MAX_DF,
                ngram_range=ModelConfig.NGRAM_RANGE,
                lowercase=ModelConfig.LOWERCASE,
                stop_words=ModelConfig.STOP_WORDS
            )
            X_train_vec = self.vectorizer.fit_transform(X_train)
        else:
            X_train_vec = self.vectorizer.transform(X_train)
        
        X_test_vec = self.vectorizer.transform(X_test)
        
        logger.info(f"Vectorizer shape: {X_train_vec.shape}")
        logger.info(f"Vocabulary size: {len(self.vectorizer.get_feature_names_out())}")
        
        return X_train_vec, X_test_vec
    
    def train_naive_bayes(self, X_train, y_train):
        """
        Train Multinomial Naive Bayes model
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info("Training Multinomial Naive Bayes...")
        
        self.nb_model = MultinomialNB(
            alpha=ModelConfig.NB_ALPHA,
            fit_prior=ModelConfig.NB_FIT_PRIOR
        )
        self.nb_model.fit(X_train, y_train)
        
        logger.info("Naive Bayes training completed")
    
    def train_svm(self, X_train, y_train):
        """
        Train SVM model
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info("Training Linear SVM...")
        
        self.svm_model = LinearSVC(
            C=ModelConfig.SVM_C,
            max_iter=2000,
            random_state=ModelConfig.RANDOM_STATE,
            dual=False
        )
        self.svm_model.fit(X_train, y_train)
        
        logger.info("SVM training completed")
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of model for logging
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        }
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        metrics['classification_report'] = report
        
        # Log results
        logger.info(f"{model_name} Results:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {metrics['f1']:.4f}")
        
        return metrics, y_pred
    
    def save_models(self):
        """
        Save trained models and vectorizer
        """
        logger.info("Saving models...")
        
        # Create models directory
        os.makedirs(ModelConfig.MODEL_DIR, exist_ok=True)
        
        # Save vectorizer
        joblib.dump(self.vectorizer, ModelConfig.VECTORIZER_PATH)
        logger.info(f"Vectorizer saved to {ModelConfig.VECTORIZER_PATH}")
        
        # Save Naive Bayes
        if self.nb_model:
            joblib.dump(self.nb_model, ModelConfig.NB_MODEL_PATH)
            logger.info(f"Naive Bayes saved to {ModelConfig.NB_MODEL_PATH}")
        
        # Save SVM
        if self.svm_model:
            joblib.dump(self.svm_model, ModelConfig.SVM_MODEL_PATH)
            logger.info(f"SVM saved to {ModelConfig.SVM_MODEL_PATH}")
        
        # Save metrics
        with open(ModelConfig.METRICS_PATH, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        logger.info(f"Metrics saved to {ModelConfig.METRICS_PATH}")
    
    def train(self, data_path):
        """
        Complete training pipeline
        
        Args:
            data_path: Path to training data
        """
        logger.info("Starting training pipeline...")
        
        # Load data
        X, y = self.load_data(data_path)
        
        # Preprocess
        X = self.preprocess_texts(X)
        
        # Split data
        logger.info(f"Splitting data: {ModelConfig.TRAIN_TEST_SPLIT * 100}% test")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=ModelConfig.TRAIN_TEST_SPLIT,
            random_state=ModelConfig.RANDOM_STATE,
            stratify=y
        )
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        logger.info(f"Train set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        
        # Vectorize
        X_train_vec, X_test_vec = self.vectorize_texts(X_train, X_test, fit=True)
        
        # Train models
        self.train_naive_bayes(X_train_vec, y_train)
        self.train_svm(X_train_vec, y_train)
        
        # Evaluate
        nb_metrics, nb_pred = self.evaluate_model(
            self.nb_model, X_test_vec, y_test, "Naive Bayes"
        )
        svm_metrics, svm_pred = self.evaluate_model(
            self.svm_model, X_test_vec, y_test, "SVM"
        )
        
        # Store metrics
        self.metrics = {
            'naive_bayes': nb_metrics,
            'svm': svm_metrics,
            'data_info': {
                'total_samples': len(X),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_count': X_train_vec.shape[1],
                'vocabulary_size': len(self.vectorizer.get_feature_names_out())
            }
        }
        
        # Save models
        self.save_models()
        
        logger.info("Training pipeline completed!")
        return self.metrics

def main():
    """
    Main entry point for training
    """
    trainer = ModelTrainer()
    
    # Use sample data if available
    if os.path.exists(DataConfig.SAMPLE_DATA_PATH):
        data_path = DataConfig.SAMPLE_DATA_PATH
    else:
        # Create sample data path if doesn't exist
        os.makedirs(DataConfig.DATA_DIR, exist_ok=True)
        data_path = DataConfig.SAMPLE_DATA_PATH
        logger.warning(f"Data file not found at {data_path}")
        logger.info("Please ensure sample_data.csv is in the data/ directory")
        return
    
    try:
        metrics = trainer.train(data_path)
        logger.info("\n" + "="*50)
        logger.info("Training Summary")
        logger.info("="*50)
        logger.info(f"Naive Bayes Accuracy: {metrics['naive_bayes']['accuracy']:.4f}")
        logger.info(f"SVM Accuracy: {metrics['svm']['accuracy']:.4f}")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
