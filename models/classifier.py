"""
Main classifier module
Handles model loading, prediction, and evaluation
"""

import os
import json
import numpy as np
import joblib
from typing import Dict, Tuple, Any

from config import ModelConfig, DataConfig
from utils.preprocessing import TextPreprocessor

class SpamClassifier:
    """
    Main spam classification class
    Handles loading models and making predictions
    """
    
    def __init__(self):
        """Initialize the classifier"""
        self.nb_model = None
        self.svm_model = None
        self.vectorizer = None
        self.preprocessor = TextPreprocessor()
        self.is_loaded = False
        self.metrics = None
        
    def load_model(self):
        """
        Load trained models and vectorizer from disk
        
        Raises:
            FileNotFoundError: If model files not found
        """
        try:
            # Load vectorizer
            if os.path.exists(ModelConfig.VECTORIZER_PATH):
                self.vectorizer = joblib.load(ModelConfig.VECTORIZER_PATH)
                print(f"Vectorizer loaded from {ModelConfig.VECTORIZER_PATH}")
            else:
                raise FileNotFoundError(f"Vectorizer not found at {ModelConfig.VECTORIZER_PATH}")
            
            # Load Naive Bayes model
            if os.path.exists(ModelConfig.NB_MODEL_PATH):
                self.nb_model = joblib.load(ModelConfig.NB_MODEL_PATH)
                print(f"Naive Bayes model loaded from {ModelConfig.NB_MODEL_PATH}")
            else:
                raise FileNotFoundError(f"Naive Bayes model not found at {ModelConfig.NB_MODEL_PATH}")
            
            # Load SVM model
            if os.path.exists(ModelConfig.SVM_MODEL_PATH):
                self.svm_model = joblib.load(ModelConfig.SVM_MODEL_PATH)
                print(f"SVM model loaded from {ModelConfig.SVM_MODEL_PATH}")
            else:
                print(f"Warning: SVM model not found at {ModelConfig.SVM_MODEL_PATH}")
            
            # Load metrics
            if os.path.exists(ModelConfig.METRICS_PATH):
                with open(ModelConfig.METRICS_PATH, 'r') as f:
                    self.metrics = json.load(f)
            
            self.is_loaded = True
            print("All models loaded successfully!")
            
        except FileNotFoundError as e:
            print(f"Error loading models: {e}")
            self.is_loaded = False
            raise
    
    def preprocess(self, text: str) -> str:
        """
        Preprocess text before vectorization
        
        Args:
            text: Raw text input
            
        Returns:
            Preprocessed text
        """
        return self.preprocessor.preprocess(text)
    
    def predict(self, text: str, model_type: str = 'naive_bayes') -> Dict[str, Any]:
        """
        Predict if text is spam or ham
        
        Args:
            text: Text to classify
            model_type: 'naive_bayes' or 'svm'
            
        Returns:
            Dictionary with prediction results
            
        Raises:
            ValueError: If models not loaded or invalid model_type
        """
        if not self.is_loaded:
            raise ValueError("Models not loaded. Call load_model() first.")
        
        if model_type not in ['naive_bayes', 'svm']:
            raise ValueError(f"Invalid model_type: {model_type}. Use 'naive_bayes' or 'svm'")
        
        # Select model
        if model_type == 'naive_bayes':
            model = self.nb_model
        else:
            if self.svm_model is None:
                raise ValueError("SVM model not available. Please train it first.")
            model = self.svm_model
        
        # Preprocess text
        processed_text = self.preprocess(text)
        
        # Vectorize
        X = self.vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)[0]
        else:
            # For SVM without probability
            decision = model.decision_function(X)[0]
            probabilities = self._sigmoid(np.array([decision]))[0]
            probabilities = np.array([1 - probabilities, probabilities])
        
        # Get confidence
        confidence = probabilities[prediction]
        
        # Map label
        label = DataConfig.REVERSE_LABEL_MAP.get(int(prediction), 'unknown')
        
        return {
            'text': text,
            'processed_text': processed_text,
            'prediction': int(prediction),
            'label': label,
            'confidence': float(confidence),
            'probabilities': {
                'ham': float(probabilities[0]),
                'spam': float(probabilities[1])
            },
            'model': model_type
        }
    
    def predict_batch(self, texts: list, model_type: str = 'naive_bayes') -> list:
        """
        Predict multiple texts
        
        Args:
            texts: List of texts to classify
            model_type: 'naive_bayes' or 'svm'
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for text in texts:
            try:
                result = self.predict(text, model_type)
                results.append(result)
            except Exception as e:
                results.append({'error': str(e), 'text': text})
        return results
    
    @staticmethod
    def _sigmoid(x):
        """
        Sigmoid function for probability conversion
        
        Args:
            x: Input value
            
        Returns:
            Sigmoid of x
        """
        return 1 / (1 + np.exp(-x))
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get model metrics
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics
