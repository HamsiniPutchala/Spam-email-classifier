# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git
- Virtual environment tool (recommended: venv)

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/HamsiniPutchala/Spam-Email-Classifier.git
cd Spam-Email-Classifier
```

### 2. Create Virtual Environment

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import flask, sklearn, pandas; print('All dependencies installed!')"
```

### 5. Train the Model (Optional)

If you want to retrain the model with your own data:

```bash
python train_model.py
```

### 6. Run the Application

```bash
python run.py
```

The application will be available at `http://localhost:5000`

## Docker Installation

### Using Docker Compose

```bash
docker-compose up
```

### Using Docker

```bash
# Build image
docker build -t spam-classifier .

# Run container
docker run -p 5000:5000 spam-classifier
```

## Troubleshooting

### Port 5000 Already in Use

```bash
# Change port
FLASK_PORT=5001 python run.py
```

### Models Not Found

```bash
# Train the models
python train_model.py
```

### Permission Denied on macOS/Linux

```bash
chmod +x run.py train_model.py
```

### Module Import Errors

```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

## Development Setup

### Additional Development Dependencies

```bash
pip install black flake8 pylint jupyter notebook
```

### Running Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=models --cov=utils
```

### Code Formatting

```bash
black . --line-length 100
flake8 . --max-line-length 100
```

## Production Deployment

### Using Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 run:app
```

### Using uWSGI

```bash
pip install uwsgi
uwsgi --http :5000 --wsgi-file run.py --callable app --processes 4 --threads 2
```

### Environment Variables

Create a `.env` file:

```
FLASK_ENV=production
FLASK_DEBUG=0
SECRET_KEY=your-secret-key-here
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
```

## Next Steps

- Read [Usage Guide](USAGE_GUIDE.md)
- Check [API Documentation](API_DOCUMENTATION.md)
- Review [Model Details](MODEL_DETAILS.md)
