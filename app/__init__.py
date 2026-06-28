"""
Flask application factory
Creates and configures the Flask application
"""

import os
import logging
from logging.handlers import RotatingFileHandler

from flask import Flask
from flask_cors import CORS

from config import Config, LogConfig, ENV

def create_app(config=None):
    """
    Create and configure Flask application
    
    Args:
        config: Configuration object (optional)
        
    Returns:
        Configured Flask application
    """
    # Create Flask app
    app = Flask(
        __name__,
        template_folder='templates',
        static_folder='static'
    )
    
    # Load configuration
    if config is None:
        app.config.from_object(Config)
    else:
        app.config.from_object(config)
    
    # Enable CORS
    CORS(app)
    
    # Setup logging
    setup_logging(app)
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Register blueprints
    from app.routes import main_bp, api_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Error handlers
    register_error_handlers(app)
    
    # Log app creation
    app.logger.info(f"Flask app created successfully")
    app.logger.info(f"Environment: {ENV}")
    
    return app

def setup_logging(app):
    """
    Configure application logging
    
    Args:
        app: Flask application
    """
    # Create log directory
    os.makedirs(LogConfig.LOG_DIR, exist_ok=True)
    
    # Create file handler
    file_handler = RotatingFileHandler(
        LogConfig.LOG_FILE,
        maxBytes=LogConfig.LOG_MAX_BYTES,
        backupCount=LogConfig.LOG_BACKUP_COUNT
    )
    
    # Create formatter
    formatter = logging.Formatter(LogConfig.LOG_FORMAT)
    file_handler.setFormatter(formatter)
    
    # Set log level
    log_level = logging.INFO if LogConfig.LOG_LEVEL == 'INFO' else logging.WARNING
    file_handler.setLevel(log_level)
    app.logger.addHandler(file_handler)
    
    # Set app logger level
    app.logger.setLevel(log_level)

def register_error_handlers(app):
    """
    Register error handlers
    
    Args:
        app: Flask application
    """
    
    @app.errorhandler(400)
    def bad_request(error):
        """Handle 400 Bad Request"""
        return {'error': 'Bad request', 'message': str(error)}, 400
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 Not Found"""
        return {'error': 'Not found', 'message': 'Resource not found'}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 Internal Server Error"""
        app.logger.error(f"Internal error: {error}")
        return {'error': 'Internal server error'}, 500
