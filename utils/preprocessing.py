"""
Text preprocessing utilities
"""

import re
import string
from typing import List
from config import PreprocessConfig

class TextPreprocessor:
    """
    Text preprocessing class for email/SMS cleaning
    """
    
    def __init__(self):
        """Initialize preprocessor"""
        self.config = PreprocessConfig()
    
    def preprocess(self, text: str) -> str:
        """
        Complete preprocessing pipeline
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove URLs
        if self.config.REMOVE_URLS:
            text = self._remove_urls(text)
        
        # Remove emails
        if self.config.REMOVE_EMAILS:
            text = self._remove_emails(text)
        
        # Remove special characters
        if self.config.REMOVE_SPECIAL_CHARS:
            text = self._remove_special_chars(text)
        
        # Remove punctuation
        if self.config.REMOVE_PUNCTUATION:
            text = self._remove_punctuation(text)
        
        # Remove numbers
        if self.config.REMOVE_NUMBERS:
            text = self._remove_numbers(text)
        
        # Lowercase
        if self.config.LOWERCASE:
            text = text.lower()
        
        # Remove extra whitespace
        if self.config.REMOVE_EXTRA_WHITESPACE:
            text = self._remove_extra_whitespace(text)
        
        # Remove stopwords
        if self.config.REMOVE_STOPWORDS:
            text = self._remove_stopwords(text)
        
        return text.strip()
    
    @staticmethod
    def _remove_urls(text: str) -> str:
        """Remove URLs from text"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)
    
    @staticmethod
    def _remove_emails(text: str) -> str:
        """Remove email addresses from text"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, '', text)
    
    @staticmethod
    def _remove_special_chars(text: str) -> str:
        """Remove special characters"""
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    @staticmethod
    def _remove_punctuation(text: str) -> str:
        """Remove punctuation"""
        return text.translate(str.maketrans('', '', string.punctuation))
    
    @staticmethod
    def _remove_numbers(text: str) -> str:
        """Remove numbers"""
        return re.sub(r'\d+', '', text)
    
    @staticmethod
    def _remove_extra_whitespace(text: str) -> str:
        """Remove extra whitespace"""
        return re.sub(r'\s+', ' ', text).strip()
    
    @staticmethod
    def _remove_stopwords(text: str) -> str:
        """
        Remove stopwords
        
        Uses a basic set of English stopwords
        """
        stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
            'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
            'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'why', 'how', 'all',
            'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            'can', 'will', 'just', 'don', 'should', 'now', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'ought', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'what', 'which', 'who', 'when', 'where', 'why', 'how', 'a', 'an', 'the',
            'and', 'or', 'but', 'in', 'out', 'on', 'off', 'of', 'to', 'for', 'at'
        }
        
        words = text.split()
        filtered = [w for w in words if w.lower() not in stopwords]
        return ' '.join(filtered)
    
    @staticmethod
    def get_tokens(text: str) -> List[str]:
        """Get tokens from text"""
        return text.split()
    
    @staticmethod
    def get_token_count(text: str) -> int:
        """Count tokens in text"""
        return len(text.split())
