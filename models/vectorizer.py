"""
Vectorizer utilities for text transformation
"""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from config import ModelConfig

class TfidfVectorizer:
    """
    TF-IDF vectorizer wrapper
    """
    
    @staticmethod
    def create_vectorizer():
        """
        Create a new TF-IDF vectorizer
        
        Returns:
            Configured TfidfVectorizer instance
        """
        return TfidfVectorizer(
            max_features=ModelConfig.MAX_FEATURES,
            min_df=ModelConfig.MIN_DF,
            max_df=ModelConfig.MAX_DF,
            ngram_range=ModelConfig.NGRAM_RANGE,
            lowercase=ModelConfig.LOWERCASE,
            stop_words=ModelConfig.STOP_WORDS
        )
    
    @staticmethod
    def save_vectorizer(vectorizer, path):
        """
        Save vectorizer to disk
        
        Args:
            vectorizer: Vectorizer instance
            path: Path to save
        """
        joblib.dump(vectorizer, path)
    
    @staticmethod
    def load_vectorizer(path):
        """
        Load vectorizer from disk
        
        Args:
            path: Path to vectorizer file
            
        Returns:
            Loaded vectorizer
        """
        return joblib.load(path)
