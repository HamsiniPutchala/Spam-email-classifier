"""
Init file for models package
"""

from models.classifier import SpamClassifier
from models.vectorizer import TfidfVectorizer

__all__ = ['SpamClassifier', 'TfidfVectorizer']
