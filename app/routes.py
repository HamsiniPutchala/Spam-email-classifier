"""
API routes and web interface handlers
"""

import os
import json
from datetime import datetime

from flask import Blueprint, render_template, request, jsonify

from models.classifier import SpamClassifier
from utils.preprocessing import TextPreprocessor

# Create blueprints
main_bp = Blueprint('main', __name__)
api_bp = Blueprint('api', __name__)

# Initialize classifier (singleton)
_classifier = None
_preprocessor = None

def get_classifier():
    """Get classifier instance"""
    global _classifier
    if _classifier is None:
        _classifier = SpamClassifier()
        try:
            _classifier.load_model()
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
    return _classifier

def get_preprocessor():
    """Get preprocessor instance"""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = TextPreprocessor()
    return _preprocessor

# ==================== Web Interface Routes ====================

@main_bp.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@main_bp.route('/classifier')
def classifier_page():
    """Classifier interface page"""
    return render_template('classifier.html')

@main_bp.route('/metrics')
def metrics_page():
    """Metrics and evaluation page"""
    return render_template('metrics.html')

@main_bp.route('/about')
def about():
    """About page"""
    return render_template('base.html', page='about')

# ==================== API Endpoints ====================

@api_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    
    Returns:
        JSON with health status
    """
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }), 200

@api_bp.route('/classify', methods=['POST'])
def classify():
    """
    Classify text as spam or ham
    
    Request JSON:
        {
            "text": "Email/SMS text to classify",
            "model": "naive_bayes" or "svm" (optional, default: naive_bayes)
        }
    
    Returns:
        JSON with classification results
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing required field: text'}), 400
        
        text = data.get('text', '').strip()
        model_type = data.get('model', 'naive_bayes')
        
        # Validate input
        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        if len(text) > 10000:
            return jsonify({'error': 'Text too long (max 10000 characters)'}), 400
        
        # Get classifier
        classifier = get_classifier()
        
        # Make prediction
        result = classifier.predict(text, model_type=model_type)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/classify-batch', methods=['POST'])
def classify_batch():
    """
    Classify multiple texts
    
    Request JSON:
        {
            "texts": ["text1", "text2", ...],
            "model": "naive_bayes" (optional)
        }
    
    Returns:
        JSON with batch classification results
    """
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({'error': 'Missing required field: texts'}), 400
        
        texts = data.get('texts', [])
        model_type = data.get('model', 'naive_bayes')
        
        if not isinstance(texts, list):
            return jsonify({'error': 'texts must be a list'}), 400
        
        if len(texts) == 0:
            return jsonify({'error': 'texts list cannot be empty'}), 400
        
        if len(texts) > 100:
            return jsonify({'error': 'Maximum 100 texts per request'}), 400
        
        # Get classifier
        classifier = get_classifier()
        
        # Classify all texts
        results = []
        for text in texts:
            if isinstance(text, str) and text.strip():
                result = classifier.predict(text, model_type=model_type)
                results.append(result)
        
        return jsonify({
            'total': len(texts),
            'processed': len(results),
            'results': results
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/metrics', methods=['GET'])
def get_metrics():
    """
    Get model metrics and evaluation results
    
    Returns:
        JSON with model metrics
    """
    try:
        metrics_path = 'models/metrics.json'
        
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            return jsonify(metrics), 200
        else:
            return jsonify({
                'error': 'Metrics not found',
                'message': 'Please run train_model.py first'
            }), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/compare', methods=['POST'])
def compare_models():
    """
    Compare predictions between models
    
    Request JSON:
        {
            "text": "Text to classify"
        }
    
    Returns:
        JSON with predictions from both models
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing required field: text'}), 400
        
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        # Get classifier
        classifier = get_classifier()
        
        # Get predictions from both models
        nb_result = classifier.predict(text, model_type='naive_bayes')
        svm_result = classifier.predict(text, model_type='svm')
        
        return jsonify({
            'text': text,
            'naive_bayes': nb_result,
            'svm': svm_result,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/preprocess', methods=['POST'])
def preprocess():
    """
    Preprocess text and return processed version
    
    Request JSON:
        {
            "text": "Text to preprocess"
        }
    
    Returns:
        JSON with preprocessed text
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing required field: text'}), 400
        
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        # Get preprocessor
        preprocessor = get_preprocessor()
        
        # Preprocess
        processed = preprocessor.preprocess(text)
        
        return jsonify({
            'original': text,
            'processed': processed,
            'length_original': len(text),
            'length_processed': len(processed),
            'tokens': len(processed.split())
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/stats', methods=['GET'])
def get_stats():
    """
    Get application statistics
    
    Returns:
        JSON with app statistics
    """
    try:
        metrics_path = 'models/metrics.json'
        
        stats = {
            'models': {
                'naive_bayes': {
                    'available': False,
                    'accuracy': None
                },
                'svm': {
                    'available': False,
                    'accuracy': None
                }
            },
            'data': {
                'total_samples': 0,
                'feature_count': 0
            }
        }
        
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            if 'naive_bayes' in metrics:
                stats['models']['naive_bayes'] = {
                    'available': True,
                    'accuracy': metrics['naive_bayes'].get('accuracy')
                }
            
            if 'svm' in metrics:
                stats['models']['svm'] = {
                    'available': True,
                    'accuracy': metrics['svm'].get('accuracy')
                }
            
            if 'data_info' in metrics:
                stats['data'] = metrics['data_info']
        
        return jsonify(stats), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
