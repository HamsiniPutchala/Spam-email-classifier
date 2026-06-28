# Usage Guide

## Web Interface

### Accessing the Application

1. Start the application:
   ```bash
   python run.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

### Home Page

The home page displays:
- Project overview
- Key features
- Quick classification form
- Model statistics
- Technology stack

### Classifier Page

1. Navigate to the **Classifier** section
2. Enter email or SMS text in the text area
3. Select the model (Naive Bayes or SVM)
4. Click **Classify**
5. View the results:
   - Prediction (Ham or Spam)
   - Confidence score
   - Probability distribution

### Metrics Page

View comprehensive model performance metrics:
- Accuracy, Precision, Recall, F1-Score
- Confusion matrices
- Dataset information
- Model comparison

---

## REST API Usage

### Basic Classification

#### Using cURL

```bash
curl -X POST http://localhost:5000/api/classify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Congratulations! You have won a prize!",
    "model": "naive_bayes"
  }'
```

#### Using Python

```python
import requests

url = 'http://localhost:5000/api/classify'
data = {
    'text': 'Congratulations! You have won a prize!',
    'model': 'naive_bayes'
}

response = requests.post(url, json=data)
result = response.json()

print(f"Text: {result['text']}")
print(f"Prediction: {result['label'].upper()}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Probabilities: {result['probabilities']}")
```

#### Using JavaScript

```javascript
const text = 'Congratulations! You have won a prize!';

fetch('/api/classify', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    text: text,
    model: 'naive_bayes'
  })
})
.then(response => response.json())
.then(data => {
  console.log(`Prediction: ${data.label}`);
  console.log(`Confidence: ${(data.confidence * 100).toFixed(1)}%`);
  console.log(`Probabilities:`, data.probabilities);
});
```

### Batch Classification

#### Using Python

```python
import requests

url = 'http://localhost:5000/api/classify-batch'
data = {
    'texts': [
        'Hello, how are you?',
        'Click here to win FREE money now!!!',
        'Can we schedule a meeting tomorrow?',
        'Limited time offer: get 50% off now'
    ],
    'model': 'naive_bayes'
}

response = requests.post(url, json=data)
result = response.json()

print(f"Processed: {result['processed']} of {result['total']} texts")
for i, res in enumerate(result['results'], 1):
    print(f"\n{i}. {res['text']}")
    print(f"   Prediction: {res['label'].upper()}")
    print(f"   Confidence: {res['confidence']:.1%}")
```

### Compare Models

```python
import requests

url = 'http://localhost:5000/api/compare'
data = {'text': 'Your email text here'}

response = requests.post(url, json=data)
result = response.json()

print("Naive Bayes:")
print(f"  Prediction: {result['naive_bayes']['label']}")
print(f"  Confidence: {result['naive_bayes']['confidence']:.1%}")

print("\nSVM:")
print(f"  Prediction: {result['svm']['label']}")
print(f"  Confidence: {result['svm']['confidence']:.1%}")
```

### Preprocess Text

```python
import requests

url = 'http://localhost:5000/api/preprocess'
data = {'text': 'Check out https://example.com or email us at test@example.com!'}

response = requests.post(url, json=data)
result = response.json()

print(f"Original: {result['original']}")
print(f"Processed: {result['processed']}")
print(f"Tokens: {result['tokens']}")
```

---

## Command Line Usage

### Using the Classifier Class

```python
from models.classifier import SpamClassifier

# Initialize and load model
classifier = SpamClassifier()
classifier.load_model()

# Single prediction
text = "Win FREE money now!"
result = classifier.predict(text, model_type='naive_bayes')

print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")

# Batch prediction
texts = [
    "Hello there",
    "WIN FREE PRIZE NOW!!!",
    "Meeting at 10 AM tomorrow"
]

results = classifier.predict_batch(texts)
for text, result in zip(texts, results):
    print(f"{text} -> {result['label']} ({result['confidence']:.1%})")
```

### Text Preprocessing

```python
from utils.preprocessing import TextPreprocessor

preprocessor = TextPreprocessor()

# Preprocess a text
text = "Check https://example.com! Contact us at test@example.com"
processed = preprocessor.preprocess(text)
print(f"Original: {text}")
print(f"Processed: {processed}")

# Get tokens
tokens = preprocessor.get_tokens(processed)
token_count = preprocessor.get_token_count(processed)
print(f"Tokens: {tokens}")
print(f"Token count: {token_count}")
```

---

## Training Custom Models

### Using Sample Data

```bash
python train_model.py
```

### Using Custom Data

Prepare a CSV file with columns: `message` and `label`

```csv
message,label
"Hello, how are you?",ham
"CLICK HERE TO WIN NOW!!!",spam
"Meeting scheduled for tomorrow",ham
```

Then modify `train_model.py` to use your data:

```python
from train_model import ModelTrainer

trainer = ModelTrainer()
metrics = trainer.train('path/to/your/data.csv')
print(f"Model accuracy: {metrics['naive_bayes']['accuracy']:.2%}")
```

---

## Tips and Best Practices

### For Best Results

1. **Keep texts concise**: The model works best with typical email/SMS length texts
2. **Include context**: Full sentences provide better predictions than fragments
3. **Avoid special characters**: The preprocessor handles most, but clean input helps
4. **Batch processing**: Use batch API for multiple texts for better performance

### Model Selection

- **Naive Bayes**: Faster, slightly better accuracy (98.2%)
- **SVM**: Slower, good accuracy (97.9%), more robust

For production, use Naive Bayes for speed or SVM for robustness.

### Error Handling

```python
import requests
from requests.exceptions import RequestException

try:
    response = requests.post(
        'http://localhost:5000/api/classify',
        json={'text': 'Your text'},
        timeout=5
    )
    response.raise_for_status()
    result = response.json()
except RequestException as e:
    print(f"API Error: {e}")
except Exception as e:
    print(f"Error: {e}")
```

---

## Performance Optimization

### Caching

Implement caching for repeated classifications:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_classify(text):
    return classifier.predict(text)['label']
```

### Batch Processing

For multiple texts, use the batch endpoint:

```python
# Efficient: 1 API call
response = requests.post('/api/classify-batch', json={'texts': texts})

# Inefficient: N API calls
for text in texts:
    response = requests.post('/api/classify', json={'text': text})
```

---

For more details, see:
- [API Documentation](API_DOCUMENTATION.md)
- [Model Details](MODEL_DETAILS.md)
- [README](../README.md)
