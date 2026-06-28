# API Documentation

## Overview

The Spam Email Classifier provides a RESTful API for classifying emails and SMS messages as spam or ham.

## Base URL

```
http://localhost:5000/api
```

## Authentication

Currently, no authentication is required. (Add JWT in production)

## Endpoints

### 1. Health Check

**Endpoint:** `GET /api/health`

**Description:** Check if the API is running

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-06-28T12:00:00Z",
  "version": "1.0.0"
}
```

---

### 2. Classify Text

**Endpoint:** `POST /api/classify`

**Description:** Classify a single text as spam or ham

**Request Body:**
```json
{
  "text": "Your email or SMS text here",
  "model": "naive_bayes"  // Optional: "naive_bayes" or "svm"
}
```

**Response:**
```json
{
  "text": "Your email or SMS text here",
  "processed_text": "processed version of text",
  "prediction": 0,
  "label": "ham",
  "confidence": 0.98,
  "probabilities": {
    "ham": 0.98,
    "spam": 0.02
  },
  "model": "naive_bayes"
}
```

**Status Codes:**
- `200`: Success
- `400`: Bad request (missing or invalid text)
- `500`: Server error

**Example:**
```bash
curl -X POST http://localhost:5000/api/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Click here to claim your FREE prize!", "model": "naive_bayes"}'
```

---

### 3. Classify Batch

**Endpoint:** `POST /api/classify-batch`

**Description:** Classify multiple texts

**Request Body:**
```json
{
  "texts": ["text1", "text2", "text3"],
  "model": "naive_bayes"  // Optional
}
```

**Response:**
```json
{
  "total": 3,
  "processed": 3,
  "results": [
    {
      "text": "text1",
      "prediction": 0,
      "label": "ham",
      "confidence": 0.95,
      "probabilities": {"ham": 0.95, "spam": 0.05},
      "model": "naive_bayes"
    },
    // ... more results
  ]
}
```

**Constraints:**
- Maximum 100 texts per request
- Each text must be non-empty

---

### 4. Get Metrics

**Endpoint:** `GET /api/metrics`

**Description:** Get model evaluation metrics

**Response:**
```json
{
  "naive_bayes": {
    "accuracy": 0.982,
    "precision": 0.978,
    "recall": 0.985,
    "f1": 0.981,
    "confusion_matrix": [[1387, 15], [8, 536]]
  },
  "svm": {
    "accuracy": 0.979,
    "precision": 0.975,
    "recall": 0.982,
    "f1": 0.978,
    "confusion_matrix": [[1382, 20], [10, 534]]
  },
  "data_info": {
    "total_samples": 5574,
    "train_samples": 4459,
    "test_samples": 1115,
    "feature_count": 5000,
    "vocabulary_size": 7486
  }
}
```

---

### 5. Compare Models

**Endpoint:** `POST /api/compare`

**Description:** Get predictions from both models for comparison

**Request Body:**
```json
{
  "text": "Text to classify"
}
```

**Response:**
```json
{
  "text": "Text to classify",
  "naive_bayes": {
    "label": "ham",
    "confidence": 0.95,
    "probabilities": {"ham": 0.95, "spam": 0.05}
  },
  "svm": {
    "label": "ham",
    "confidence": 0.92,
    "probabilities": {"ham": 0.92, "spam": 0.08}
  },
  "timestamp": "2024-06-28T12:00:00Z"
}
```

---

### 6. Preprocess Text

**Endpoint:** `POST /api/preprocess`

**Description:** Preprocess text (remove URLs, emails, punctuation, etc.)

**Request Body:**
```json
{
  "text": "Your text here with URLs and emails@example.com"
}
```

**Response:**
```json
{
  "original": "Original text",
  "processed": "processed text",
  "length_original": 50,
  "length_processed": 35,
  "tokens": 7
}
```

---

### 7. Get Statistics

**Endpoint:** `GET /api/stats`

**Description:** Get application statistics

**Response:**
```json
{
  "models": {
    "naive_bayes": {
      "available": true,
      "accuracy": 0.982
    },
    "svm": {
      "available": true,
      "accuracy": 0.979
    }
  },
  "data": {
    "total_samples": 5574,
    "feature_count": 5000
  }
}
```

---

## Error Handling

All error responses follow this format:

```json
{
  "error": "Error message"
}
```

**Common Errors:**
- `400 Bad Request`: Missing or invalid input
- `404 Not Found`: Metrics file not found (train model first)
- `500 Internal Server Error`: Server-side error

---

## Rate Limiting

No rate limiting is currently implemented. Add rate limiting in production.

---

## Response Codes

| Code | Meaning |
|------|----------|
| 200 | Success |
| 400 | Bad Request |
| 404 | Not Found |
| 500 | Internal Server Error |

---

## Examples

### JavaScript/Fetch

```javascript
fetch('/api/classify', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    text: 'Check this out!',
    model: 'naive_bayes'
  })
})
.then(response => response.json())
.then(data => console.log(data))
.catch(error => console.error('Error:', error));
```

### Python

```python
import requests
import json

url = 'http://localhost:5000/api/classify'
data = {
    'text': 'Check this out!',
    'model': 'naive_bayes'
}

response = requests.post(url, json=data)
result = response.json()
print(f"Prediction: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### cURL

```bash
curl -X POST http://localhost:5000/api/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here"}'
```

---

## Versioning

Current API Version: **1.0.0**

The API uses semantic versioning. Breaking changes will increment the major version.

---

For more information, see [README.md](../README.md) and [MODEL_DETAILS.md](MODEL_DETAILS.md)
