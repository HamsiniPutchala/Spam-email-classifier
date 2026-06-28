"""
Configuration module for Spam Email Classifier
Contains all configuration settings for the application
"""

import os
from datetime import timedelta

# Environment
ENV = os.getenv('FLASK_ENV', 'production')
DEBUG = ENV == 'development'

# Flask Configuration
class Config:
    """Base configuration"""
    FLASK_APP = 'run.py'
    FLASK_ENV = ENV
    DEBUG = DEBUG
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # Server settings
    HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    PORT = int(os.getenv('FLASK_PORT', 5000))
    
    # CORS settings
    CORS_HEADERS = 'Content-Type'
    CORS_ORIGINS = ['*']
    
    # Session settings
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    SESSION_COOKIE_SECURE = not DEBUG
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

# Model Configuration
class ModelConfig:
    """Model-specific configuration"""
    
    # Data splitting
    TRAIN_TEST_SPLIT = 0.2
    VALIDATION_SPLIT = 0.1
    RANDOM_STATE = 42
    
    # TF-IDF Vectorizer settings
    MAX_FEATURES = 5000
    MIN_DF = 2  # Minimum document frequency
    MAX_DF = 0.95  # Maximum document frequency
    NGRAM_RANGE = (1, 2)  # Unigrams and bigrams
    LOWERCASE = True
    STOP_WORDS = 'english'
    
    # Naive Bayes
    NB_ALPHA = 1.0  # Laplace smoothing
    NB_FIT_PRIOR = True
    
    # SVM
    SVM_KERNEL = 'rbf'
    SVM_C = 1.0
    SVM_GAMMA = 'scale'
    SVM_PROBABILITY = True
    
    # Training
    BATCH_SIZE = 32
    EPOCHS = 50
    EARLY_STOPPING = True
    PATIENCE = 5
    
    # Model paths
    MODEL_DIR = 'models'
    NB_MODEL_PATH = os.path.join(MODEL_DIR, 'naive_bayes_model.joblib')
    SVM_MODEL_PATH = os.path.join(MODEL_DIR, 'svm_model.joblib')
    VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')
    METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.json')

# Data Configuration
class DataConfig:
    """Data-related configuration"""
    
    DATA_DIR = 'data'
    SAMPLE_DATA_PATH = os.path.join(DATA_DIR, 'sample_data.csv')
    PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed')
    RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw')
    
    # Column names
    TEXT_COLUMN = 'message'
    LABEL_COLUMN = 'label'
    
    # Label mapping
    LABEL_MAP = {
        'ham': 0,
        'spam': 1,
        'legitimate': 0,
        'not spam': 0
    }
    
    REVERSE_LABEL_MAP = {
        0: 'ham',
        1: 'spam'
    }

# Evaluation Configuration
class EvalConfig:
    """Evaluation and metrics configuration"""
    
    # Metrics to compute
    COMPUTE_ACCURACY = True
    COMPUTE_PRECISION = True
    COMPUTE_RECALL = True
    COMPUTE_F1 = True
    COMPUTE_ROC = True
    COMPUTE_CONFUSION_MATRIX = True
    
    # Visualization
    PLOT_CONFUSION_MATRIX = True
    PLOT_ROC_CURVE = True
    PLOT_PRECISION_RECALL = True
    PLOT_WORDCLOUD = True
    PLOT_FEATURE_IMPORTANCE = False  # For tree-based models
    
    # Plot settings
    FIGURE_SIZE = (10, 6)
    DPI = 100
    FONT_SIZE = 12
    COLOR_PALETTE = 'husl'

# Logging Configuration
class LogConfig:
    """Logging configuration"""
    
    LOG_DIR = 'logs'
    LOG_FILE = os.path.join(LOG_DIR, 'app.log')
    LOG_LEVEL = 'INFO' if DEBUG else 'WARNING'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_MAX_BYTES = 10485760  # 10MB
    LOG_BACKUP_COUNT = 5

# Text Preprocessing Configuration
class PreprocessConfig:
    """Text preprocessing configuration"""
    
    # Preprocessing steps
    LOWERCASE = True
    REMOVE_SPECIAL_CHARS = True
    REMOVE_NUMBERS = False
    REMOVE_STOPWORDS = True
    REMOVE_PUNCTUATION = True
    REMOVE_EXTRA_WHITESPACE = True
    REMOVE_URLS = True
    REMOVE_EMAILS = True
    
    # Stopwords
    STOPWORDS_FILE = None  # Uses sklearn default if None
    
    # Text length limits
    MIN_TEXT_LENGTH = 2
    MAX_TEXT_LENGTH = 10000

# API Configuration
class APIConfig:
    """API-specific configuration"""
    
    # Rate limiting
    RATE_LIMIT = '100/hour'
    RATE_LIMIT_STORAGE = 'memory'  # or 'redis'
    
    # Response settings
    JSON_SORT_KEYS = False
    JSONIFY_PRETTYPRINT_REGULAR = not DEBUG
    
    # Timeout settings
    REQUEST_TIMEOUT = 30
    PREDICTION_TIMEOUT = 10

# Create configuration dictionary
config_dict = {
    'development': Config,
    'testing': Config,
    'production': Config
}

def get_config(env=None):
    """Get configuration object for given environment"""
    if env is None:
        env = ENV
    return config_dict.get(env, Config)

# Convenience imports
if __name__ == '__main__':
    print(f"Environment: {ENV}")
    print(f"Debug Mode: {DEBUG}")
    print(f"Model Directory: {ModelConfig.MODEL_DIR}")
    print(f"Data Directory: {DataConfig.DATA_DIR}")
