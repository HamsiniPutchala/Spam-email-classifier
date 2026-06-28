"""
Application entry point
Runs the Flask development server or production server
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app import create_app
from config import Config, ENV, DEBUG

def main():
    """
    Main entry point for the application
    """
    # Create Flask app
    app = create_app()
    
    # Ensure directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Log application info
    app.logger.info(f"Starting Spam Email Classifier")
    app.logger.info(f"Environment: {ENV}")
    app.logger.info(f"Debug Mode: {DEBUG}")
    
    # Run application
    if ENV == 'development':
        # Development server with auto-reload
        app.run(
            host=Config.HOST,
            port=Config.PORT,
            debug=True,
            use_reloader=True
        )
    else:
        # Production server
        app.run(
            host=Config.HOST,
            port=Config.PORT,
            debug=False
        )

if __name__ == '__main__':
    main()
