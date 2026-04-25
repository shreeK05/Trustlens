# TrustLens AI — ML-Grade Upgrade Plan

## Overview

Transform TrustLens from a hardcoded rule-based scorer into a **genuine, multi-model Machine Learning system** capable of:
- Detecting fake reviews using NLP
- Predicting trust scores using a trained classifier
- Identifying price anomalies with statistical/time-series models
- Classifying seller risk
- Performing sentiment analysis on product reviews

This plan is designed to make TrustLens a **high-end ML course project** demonstrating data pipelines, multiple ML paradigms, model evaluation, and a production-grade API.

---

## Architecture Overview

```
TrustLens ML System
├── Data Layer           → Web scraping + public datasets
├── Feature Engineering  → Text, numeric, statistical features
├── ML Pipeline (5 modules)
│   ├── Module 1: Fake Review Detector     (NLP Classification)
│   ├── Module 2: Sentiment Analyzer       (NLP / Lexicon + ML)
│   ├── Module 3: Price Anomaly Detector   (Unsupervised / Statistical)
│   ├── Module 4: Seller Risk Classifier   (Supervised Classification)
│   └── Module 5: Trust Score Predictor   (Ensemble / Regression)
├── Model Registry       → Saved .pkl / .joblib models
├── FastAPI Backend      → Serve all models via REST API
└── Next.js Frontend     → Visualize ML outputs
```

---

## Module 1 — Fake Review Detector (NLP)

### Concept
Amazon listings with fake/incentivized reviews are a major trust signal. This module classifies individual reviews as **genuine** or **fake/suspicious**.

### ML Approach
- **Type**: Binary Text Classification (Supervised)
- **Algorithm**: TF-IDF + Logistic Regression (baseline) → fine-tuned DistilBERT (advanced)

### Dataset
Use the **Yelp/Amazon Fake Reviews Dataset** (publicly available):
- [Kaggle: Fake and Real Product Reviews](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset)
- ~20,000 labeled reviews (CG = Computer Generated, OR = Original)

### Feature Engineering
```python
features = [
    "tfidf_vectors",         # TF-IDF on review text (5000 vocab)
    "review_length",         # Short reviews often fake
    "exclamation_count",     # "Amazing!!!" pattern
    "verified_purchase",     # Binary flag
    "review_age_days",       # Very recent = suspicious
    "helpfulness_ratio",     # upvotes / total votes
    "all_caps_ratio",        # "BEST PRODUCT EVER"
    "sentiment_polarity",    # Extreme polarity = suspicious
]
```

### Model Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ('clf', LogisticRegression(C=1.0, max_iter=1000))
])
```

### Evaluation Metrics
- **Accuracy**, **Precision**, **Recall**, **F1 Score**
- **ROC-AUC Curve**
- Confusion Matrix visualization

> **Course Value**: Demonstrates NLP preprocessing, TF-IDF, text classification, and threshold tuning.

---

## Module 2 — Sentiment Analyzer

### Concept
Analyze the overall emotional tone of product reviews to determine if customer satisfaction matches the star rating (rating-sentiment mismatch = red flag).

### ML Approach
- **Type**: Multi-class Sentiment Classification (Positive / Neutral / Negative)
- **Algorithm**: VADER (lexicon-based, fast, no training needed) + optional fine-tuned model

### Implementation
```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(review_text):
    scores = analyzer.polarity_scores(review_text)
    compound = scores['compound']
    if compound >= 0.05: return "Positive"
    elif compound <= -0.05: return "Negative"
    else: return "Neutral"
```

### Red Flag Detection
```python
# Rating-Sentiment Mismatch
if star_rating >= 4 and sentiment == "Negative":
    flag = "Rating Manipulation Suspected"
```

> **Course Value**: Demonstrates lexicon-based NLP, polarity scoring, and mismatch detection as a feature for the trust score.

---

## Module 3 — Price Anomaly Detector

### Concept
Replace the current `random.uniform()` fake price history with **real statistical anomaly detection** on pricing data.

### ML Approach
- **Type**: Unsupervised Anomaly Detection
- **Algorithms**:
  - **Isolation Forest** — detects outliers in pricing distribution
  - **Z-Score / IQR** — simple statistical baseline
  - **LSTM Autoencoder** (advanced) — for time-series price reconstruction error

### Dataset Strategy
Collect or simulate price history using the **Keepa Amazon Price History API** or use the [Amazon Price History Dataset on Kaggle](https://www.kaggle.com/).

### Implementation (Isolation Forest)
```python
from sklearn.ensemble import IsolationForest
import numpy as np

# Train on historical price distributions
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(price_history_matrix)

def detect_price_anomaly(current_price, history):
    X = np.array([[p] for p in history + [current_price]])
    preds = model.predict(X)
    return preds[-1] == -1  # -1 = anomaly
```

### Output
- `is_anomaly: true/false`
- `anomaly_score: float` (how extreme the price is)
- `price_trend: "rising" | "stable" | "suspicious_drop"`

> **Course Value**: Demonstrates unsupervised learning, Isolation Forest, and anomaly scoring.

---

## Module 4 — Seller Risk Classifier

### Concept
Replace the naive `"Appario" in seller` string check with a trained classifier that predicts seller trustworthiness.

### ML Approach
- **Type**: Multi-class Classification (Low Risk / Medium Risk / High Risk)
- **Algorithm**: Random Forest / XGBoost

### Features (Seller Profile)
```python
seller_features = [
    "seller_name_length",       # Very long = often shady
    "is_amazon_fulfilled",      # FBA vs FBM
    "seller_rating",            # Scraped rating
    "num_products_listed",      # Too many = dropshipper
    "days_active",              # Account age
    "return_policy_present",    # Binary
    "contact_info_present",     # Binary
    "brand_registry",           # Binary
]
```

### Training Dataset
Create a labeled dataset by:
1. Scraping 500+ Amazon seller profiles
2. Manually labeling as Low/Medium/High risk (or use known scam seller lists)
3. Alternatively: use the [Amazon Seller Trust Dataset](https://www.kaggle.com/) if available

```python
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1)
model.fit(X_train, y_train)
```

> **Course Value**: Demonstrates feature engineering, XGBoost, hyperparameter tuning.

---

## Module 5 — Trust Score Predictor (Ensemble)

### Concept
Replace `score = 88 - fixed_penalties` with a **trained regression/classification model** that predicts the trust score from all engineered features.

### ML Approach
- **Type**: Regression (predict 0–100 trust score) OR Binary Classification (trustworthy / untrustworthy)
- **Algorithm**: **Stacking Ensemble** — combines outputs of Modules 1–4 as meta-features

### Meta-Feature Vector
```python
meta_features = [
    fake_review_probability,    # From Module 1
    sentiment_mismatch_score,   # From Module 2
    price_anomaly_score,        # From Module 3
    seller_risk_score,          # From Module 4
    product_rating,             # Raw scraped value
    review_count,               # Proxy for popularity
    discount_percent,           # Raw scraped value
]
```

### Stacking Architecture
```python
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

base_models = [
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('xgb', XGBClassifier(n_estimators=100)),
]
meta_model = LogisticRegression()

stacker = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5
)
```

> **Course Value**: Demonstrates ensemble methods, stacking, meta-learning — the most advanced concept in the project.

---

## Data Strategy

| Module | Dataset Source | Size |
|--------|---------------|------|
| Fake Reviews | [Kaggle Fake Reviews](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset) | ~20K rows |
| Sentiment | Amazon Product Reviews (Public) | 50K+ rows |
| Price Anomaly | Simulated + Keepa API | Generated |
| Seller Risk | Self-labeled scraped data | 500+ sellers |
| Trust Score | Aggregate of above | Derived |

---

## New File Structure

```
backend/
├── main.py                    ← Updated API with ML inference
├── requirements.txt           ← Updated with ML libs
├── ml/
│   ├── train/
│   │   ├── train_fake_review.py
│   │   ├── train_price_anomaly.py
│   │   ├── train_seller_risk.py
│   │   └── train_trust_score.py
│   ├── models/                ← Saved .joblib model files
│   │   ├── fake_review_clf.joblib
│   │   ├── price_anomaly_iso.joblib
│   │   ├── seller_risk_xgb.joblib
│   │   └── trust_score_stacker.joblib
│   ├── features/
│   │   └── feature_engineering.py
│   └── inference/
│       └── predict.py         ← Unified inference pipeline
├── data/
│   ├── raw/                   ← Raw downloaded datasets
│   └── processed/             ← Cleaned, feature-engineered CSVs
└── notebooks/
    ├── 01_EDA.ipynb
    ├── 02_fake_review_model.ipynb
    ├── 03_price_anomaly_model.ipynb
    ├── 04_seller_risk_model.ipynb
    └── 05_trust_score_ensemble.ipynb
```

---

## Updated Requirements

```txt
# Existing
fastapi
uvicorn
pydantic
python-multipart
requests
beautifulsoup4

# ML Core
scikit-learn>=1.3.0
xgboost>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
joblib

# NLP
nltk
vaderSentiment
transformers        # optional for DistilBERT
torch               # optional

# Visualization / Notebooks
matplotlib
seaborn
jupyter
```

---

## Frontend ML Visualizations (Next.js)

New UI components to add:

| Component | What it shows |
|-----------|--------------|
| **Review Authenticity Bar** | % of reviews classified as fake |
| **Sentiment Pie Chart** | Positive / Neutral / Negative breakdown |
| **Price Anomaly Timeline** | Real chart with anomaly markers (not random) |
| **Seller Risk Badge** | Low / Medium / High with confidence % |
| **Trust Score Gauge** | ML-predicted score with feature contributions (SHAP) |
| **ML Confidence Panel** | Model confidence for each prediction |

---

## Evaluation & Reporting (For Course Submission)

Each module must include:

1. **Baseline vs ML comparison** table
2. **Confusion Matrix** (classification)
3. **ROC-AUC Curve** 
4. **Feature Importance Plot** (for tree-based models)
5. **SHAP Values** — explainability (why did the model give this score?)
6. **Cross-Validation Results** (5-fold CV)

---

## ML Concepts Demonstrated

| Concept | Where Used |
|---------|-----------|
| Supervised Learning | Fake Review (Binary), Seller Risk (Multi-class) |
| Unsupervised Learning | Price Anomaly (Isolation Forest) |
| NLP / Text Classification | Fake Review Detector |
| Ensemble Methods | Stacking, Random Forest, XGBoost |
| Feature Engineering | All modules |
| Model Persistence | `joblib.dump()` / `joblib.load()` |
| Hyperparameter Tuning | `GridSearchCV` / `RandomizedSearchCV` |
| Explainability (XAI) | SHAP values for trust score |
| Pipeline Design | `sklearn.pipeline.Pipeline` |
| Cross-Validation | `cross_val_score` |

---

## Implementation Phases

### Phase 1 — Data & Notebooks (Week 1)
- Download fake review dataset
- EDA notebooks for each module
- Feature engineering scripts

### Phase 2 — Model Training (Week 2)
- Train and evaluate all 5 models
- Save `.joblib` files to `ml/models/`

### Phase 3 — API Integration (Week 3)
- Update `main.py` to load and call ML models
- New `/analyze-ml` endpoint with full ML pipeline

### Phase 4 — Frontend (Week 4)
- Add ML visualization components to Next.js
- Sentiment chart, anomaly timeline, SHAP bar chart

---

> **Bottom Line**: This upgrade transforms TrustLens into a genuine multi-model ML system covering NLP, unsupervised learning, ensemble methods, and explainable AI — more than enough for a top-tier ML course project.
