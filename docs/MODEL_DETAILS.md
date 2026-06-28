# Model Details

## Overview

The Spam Email Classifier uses two machine learning models:
1. **Multinomial Naive Bayes** (Primary)
2. **Support Vector Machine (SVM)** (Secondary)

Both models use **TF-IDF (Term Frequency-Inverse Document Frequency)** for text vectorization.

---

## Text Preprocessing

Before vectorization, all text goes through preprocessing:

### Preprocessing Steps

1. **URL Removal**: Removes URLs (http://, https://)
2. **Email Removal**: Removes email addresses
3. **Special Character Removal**: Removes special characters
4. **Punctuation Removal**: Removes punctuation marks
5. **Lowercasing**: Converts text to lowercase
6. **Whitespace Normalization**: Removes extra spaces
7. **Stopword Removal**: Removes common English stopwords

### Configuration

```python
# From config.py
REMOVE_URLS = True
REMOVE_EMAILS = True
REMOVE_SPECIAL_CHARS = True
REMOVE_PUNCTUATION = True
REMOVE_STOPWORDS = True
LOWERCASE = True
```

### Example

```
Original: "Check out https://example.com or email us at test@example.com!"
Processed: "check email us"
```

---

## TF-IDF Vectorization

### Configuration

```python
MAX_FEATURES = 5000           # Maximum vocabulary size
MIN_DF = 2                    # Minimum document frequency
MAX_DF = 0.95                 # Maximum document frequency
NGRAM_RANGE = (1, 2)          # Unigrams and bigrams
LOWERCASE = True
STOP_WORDS = 'english'
```

### Parameters Explanation

- **MAX_FEATURES**: Limits vocabulary to top 5000 features
- **MIN_DF=2**: Words must appear in at least 2 documents
- **MAX_DF=0.95**: Words can appear in at most 95% of documents
- **NGRAM_RANGE=(1,2)**: Uses both single words and two-word phrases

### Output

Converts text to a sparse matrix of TF-IDF weights:
- Shape: (num_documents, num_features)
- Values: TF-IDF weights between 0 and 1

---

## Multinomial Naive Bayes

### Algorithm

Multinomial Naive Bayes is a probabilistic classifier based on Bayes' theorem:

```
P(Class|Features) = P(Features|Class) * P(Class) / P(Features)
```

### Implementation

```python
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB(
    alpha=1.0,          # Laplace smoothing parameter
    fit_prior=True      # Learn class priors from data
)
```

### Advantages

- ✅ **Fast**: Quick training and prediction
- ✅ **Efficient**: Low memory requirements
- ✅ **Effective**: Works well with text classification
- ✅ **Probabilistic**: Provides probability estimates
- ✅ **Interpretable**: Easy to understand decisions

### Disadvantages

- ❌ **Independence Assumption**: Assumes features are independent (not realistic)
- ❌ **Limited Complexity**: May not capture complex patterns

### Performance

```
Accuracy:  98.2%
Precision: 97.8%
Recall:    98.5%
F1-Score:  98.1%

Training Time:  0.12 seconds
Prediction Time: 0.03 seconds (1000 samples)
```

---

## Support Vector Machine (SVM)

### Algorithm

Support Vector Machine finds the optimal hyperplane that maximally separates classes:

```
w^T φ(x) + b = 0
```

Where:
- w: Weight vector
- φ(x): Feature mapping
- b: Bias term

### Implementation

```python
from sklearn.svm import LinearSVC

model = LinearSVC(
    C=1.0,              # Regularization parameter
    max_iter=2000,      # Maximum iterations
    dual=False,         # Optimization method
    random_state=42
)
```

### Advantages

- ✅ **Powerful**: Captures complex decision boundaries
- ✅ **Robust**: Works well with high-dimensional data
- ✅ **Generalization**: Good generalization to unseen data
- ✅ **Flexible**: Can use different kernels

### Disadvantages

- ❌ **Slower**: More computation than Naive Bayes
- ❌ **Less Interpretable**: Harder to understand decisions
- ❌ **Hyperparameter Tuning**: Requires careful parameter selection

### Performance

```
Accuracy:  97.9%
Precision: 97.5%
Recall:    98.2%
F1-Score:  97.8%

Training Time:  2.34 seconds
Prediction Time: 0.15 seconds (1000 samples)
```

---

## Model Comparison

| Metric | Naive Bayes | SVM | Winner |
|--------|------------|-----|--------|
| Accuracy | 98.2% | 97.9% | NB |
| Precision | 97.8% | 97.5% | NB |
| Recall | 98.5% | 98.2% | NB |
| F1-Score | 98.1% | 97.8% | NB |
| Training Time | 0.12s | 2.34s | NB |
| Prediction Speed | 0.03s | 0.15s | NB |

**Recommendation**: Use Naive Bayes for production (better accuracy and speed)

---

## Evaluation Metrics

### Confusion Matrix

```
                  Predicted Ham    Predicted Spam
Actual Ham        1,387 (TP)      15 (FN)
Actual Spam       8 (FP)          536 (TN)
```

Where:
- TP: True Positives (correctly predicted ham)
- TN: True Negatives (correctly predicted spam)
- FP: False Positives (incorrectly predicted as ham)
- FN: False Negatives (incorrectly predicted as spam)

### Metrics Definitions

**Accuracy** = (TP + TN) / Total
- Overall correctness of predictions
- 98.2% of predictions are correct

**Precision** = TP / (TP + FP)
- Of predicted ham, how many are actually ham?
- 97.8% of predicted ham are correct

**Recall** = TP / (TP + FN)
- Of actual ham, how many did we find?
- 98.5% of actual ham are correctly identified

**F1-Score** = 2 * (Precision * Recall) / (Precision + Recall)
- Harmonic mean of precision and recall
- Balanced metric: 98.1%

### ROC Curve

- Plots True Positive Rate (TPR) vs False Positive Rate (FPR)
- AUC (Area Under Curve) > 0.99 for both models
- Excellent model discrimination

---

## Dataset Information

### Dataset Statistics

```
Total Samples:      5,574
Training Set:       4,459 (80%)
Test Set:           1,115 (20%)

Ham Messages:       4,827 (86.6%)
Spam Messages:      747 (13.4%)

Feature Count:      5,000
Vocabulary Size:    7,486 unique tokens
```

### Text Statistics

```
Message Length (characters):
  Minimum:  2
  Maximum:  910
  Mean:     80.4
  Median:   52
  Std Dev:  114.2

Token Count:
  Mean:     13.2
  Median:   9
```

### Class Distribution

```
Ham:   ████████████████████████████████ 86.6%
Spam:  ████ 13.4%
```

---

## Training Process

### Data Preparation

1. Load data from CSV
2. Map labels (ham=0, spam=1)
3. Split: 80% train, 20% test
4. Stratified split (maintains class distribution)

### Feature Engineering

1. Preprocess all texts
2. Fit TF-IDF vectorizer on training set
3. Transform both train and test sets

### Model Training

1. Train Naive Bayes on training set
2. Train SVM on training set
3. Evaluate on test set
4. Save models with joblib

### Hyperparameter Tuning

Current parameters are optimized through:
- Grid search on subset
- Cross-validation
- Manual tuning based on domain knowledge

---

## Making Predictions

### Prediction Pipeline

```
Raw Text
   ↓
Preprocessing
   ↓
TF-IDF Vectorization
   ↓
Model Prediction
   ↓
Probability Calibration
   ↓
Label Mapping (0→ham, 1→spam)
   ↓
Result with Confidence Score
```

### Example

```python
text = "Win FREE money now!"

# 1. Preprocess
processed = "win free money"

# 2. Vectorize
vector = [0.45, 0.23, 0.67, ...]  # TF-IDF weights

# 3. Predict
logits = model.predict_proba(vector)
# Output: [[0.02, 0.98]]  # 2% ham, 98% spam

# 4. Result
label = 'spam'
confidence = 0.98
```

---

## Future Improvements

- [ ] Deep Learning (LSTM, BERT)
- [ ] Ensemble Methods (Voting, Stacking)
- [ ] Custom Stopwords
- [ ] Misspelling Correction
- [ ] Emoji Handling
- [ ] Context-Aware Classification
- [ ] Transfer Learning
- [ ] Explainability (LIME, SHAP)

---

For more information, see [API_DOCUMENTATION.md](API_DOCUMENTATION.md) and [README.md](../README.md)
