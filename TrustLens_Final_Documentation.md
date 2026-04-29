# TrustLens AI — Final Project Documentation

> **Version:** 2.1.0 (Production)  
> **Date:** April 29, 2026  
> **Architecture:** Neuro-Symbolic Hybrid ML Platform  
> **Status:** ✅ Fully Operational & Refined

---

## 1. Project Overview

**TrustLens AI** is a production-grade, ML-powered e-commerce trust analysis platform that evaluates Amazon product listings in real-time. It scrapes live product data, runs it through a 5-module machine learning pipeline, and produces an explainable Trust Score (0–100) with detailed breakdowns across authenticity, sentiment, pricing, and seller reliability.

### Core Value Proposition
- **Real-time Analysis** — Paste any Amazon.in product URL and get instant ML-powered trust evaluation
- **5-Module ML Pipeline** — Fake review detection, sentiment analysis, price anomaly detection, seller risk classification, and trust score ensemble
- **Cross-Platform Price Comparison** — Category-aware competitor pricing across Flipkart, Croma, Myntra, Nykaa, etc.
- **Explainable AI** — SHAP-style contribution bars show exactly *why* a product scored the way it did
- **Evidence-Based Signals** — Every conclusion is backed by positive/negative signal explanations

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     FRONTEND (React + Vite)                  │
│                    http://localhost:5175                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│  │ URL Input│  │Trust Gauge│  │ SHAP Bars│  │  Competitor   │ │
│  │  + Form  │  │ + Grade  │  │          │  │ Price Grid   │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘ │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│  │  Review  │  │Sentiment │  │  Price   │  │Seller Risk   │ │
│  │Authenticy│  │ Analysis │  │ Anomaly  │  │  + Signals   │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘ │
└──────────────────────┬───────────────────────────────────────┘
                       │ POST /analyze
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                   BACKEND (FastAPI + Uvicorn)                │
│                   http://127.0.0.1:8001                      │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    SCRAPER LAYER                        │ │
│  │  CloudScraper → BeautifulSoup → Jina.ai (fallback)     │ │
│  │  Extracts: title, price, MRP, rating, reviews, seller  │ │
│  └─────────────────────┬───────────────────────────────────┘ │
│                        ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              ML INFERENCE ENGINE                        │ │
│  │  Module 1: Fake Review Detector   (TF-IDF + LogReg)    │ │
│  │  Module 2: Sentiment Analyzer (VADER+TF-IDF+RF)        │ │
│  │  Module 3: Price Anomaly Detector (Isolation Forest)    │ │
│  │  Module 4: Seller Risk Classifier (Deterministic)      │ │
│  │  Module 5: Trust Score Ensemble   (Bayesian Scoring)   │ │
│  └─────────────────────┬───────────────────────────────────┘ │
│                        ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │           EXPLAINABILITY ENGINE                         │ │
│  │  SHAP contributions, Pros/Cons, Competitor Pricing,     │ │
│  │  Price History, Risk Grading (A/B/C/D)                  │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Directory Structure

```
D:\TrustLens\
├── backend\
│   ├── main.py                    # FastAPI app: scraping + API (886 lines)
│   ├── requirements.txt           # Python dependencies
│   ├── Procfile                   # Render deployment config
│   ├── test_full_pipeline.py      # Comprehensive 5-module test script
│   ├── venv\                      # Python virtual environment
│   └── ml\
│       ├── train_models.py        # Training pipeline (411 lines)
│       ├── inference.py           # Inference engine (378 lines)
│       └── models\               # Trained .joblib artifacts
├── frontend\
│   ├── package.json               # React 19 + Vite 6 + Recharts + Lucide
│   ├── vite.config.ts             # Dev server on port 3000
│   └── src\
│       ├── App.tsx                # Full application (1199 lines)
│       └── styles.css             # Design system (1100+ lines)
├── Dataset\                       # 3 Kaggle datasets (see Section 9)
├── start_project.bat              # One-click launcher (refined)
└── TrustLens_Final_Documentation.md
```

---

## 4. ML Pipeline — Detailed Module Breakdown

### Module 1: Fake Review Detector

| Property | Value |
|---|---|
| **Algorithm** | TF-IDF (5000 features, bigrams) + Logistic Regression |
| **Dataset** | Kaggle: Fake and Real Product Reviews |
| **Training Samples** | 40,000 balanced (20K genuine + 20K fake) |
| **Labels** | CG = Computer Generated (Fake), OR = Original (Genuine) |
| **Accuracy** | ~68.6% |
| **F1 (Fake)** | ~0.71 |
| **Features** | TF-IDF bigrams + 10 handcrafted text features (length, word count, exclamation count, caps ratio, unique word ratio, rating, etc.) |
| **Output** | `authenticity_score` (0–100%), `fake_count`, `genuine_count`, `flagged_reviews` |

**How it works:** Each review is vectorized via TF-IDF and enriched with 10 numeric features. The Logistic Regression classifier outputs a per-review fake probability. Reviews with >60% fake probability are flagged.

---

### Module 2: Sentiment Analyzer

| Property | Value |
|---|---|
| **Algorithm** | VADER Sentiment + TF-IDF (3000 features) + Random Forest (150 estimators) |
| **Dataset** | Amazon Fine Food Reviews (200K rows loaded) |
| **Training Samples** | 45,000 balanced (15K per class) |
| **Labels** | Score ≥4 → Positive, Score 3 → Neutral, Score ≤2 → Negative |
| **Accuracy** | ~71% |
| **Features** | TF-IDF bigrams + VADER compound score + helpfulness ratio + text length + punctuation counts |
| **Output** | `overall` (Positive/Neutral/Negative), `sentiment_distribution`, `mismatch_detected` |

**How it works:** Each review gets a VADER compound score and TF-IDF vector. The Random Forest ensemble classifies into 3 sentiment classes. A **rating-sentiment mismatch detector** flags suspicious products (e.g., 5-star rating with negative language).

---

### Module 3: Price Anomaly Detector

| Property | Value |
|---|---|
| **Algorithm** | Isolation Forest (300 estimators, 5% contamination) |
| **Dataset** | Amazon Products Dataset 2023 |
| **Training Samples** | 80,000 real products |
| **Type** | Unsupervised Anomaly Detection |
| **Features** | `discount_pct`, `log(price)`, `price/listPrice` ratio, `price_zscore`, `stars`, `log(reviews)` |
| **Output** | `is_anomaly`, `anomaly_score` (0–1), `price_trend`, `discount_pct`, `vs_avg_history` |

---

### Module 4: Seller Risk Classifier

| Property | Value |
|---|---|
| **Algorithm** | Deterministic Business Logic (rule-based) |
| **Accuracy** | 100% (by design) |
| **Output** | `risk_level` (Low/Medium/High), `confidence`, `probabilities` |

**Business Rules:**

| Classification | Condition |
|---|---|
| **Low Risk** | (rating ≥ 4.2 AND reviews ≥ 500 AND discount ≤ 60%) OR bestseller OR Amazon/Appario seller |
| **High Risk** | (rating < 3.0 AND reviews < 20) OR (discount > 80% AND reviews < 10) OR rating < 2.5 |
| **Medium Risk** | Default |

---

### Module 5: Trust Score Ensemble

| Property | Value |
|---|---|
| **Algorithm** | Calibrated Bayesian Evidence Scoring |
| **Output** | `score` (0–100), `grade` (A/B/C/D), `verdict`, `shap_contributions` |

**Scoring Formula:**
```
Trust Score = 80 (base)
  - fake_prob × 30          (fake review penalty)
  - seller_risk_penalty      (0/10/25 for Low/Med/High)
  - mismatch × 15            (rating-sentiment mismatch)
  - anomaly_score × 15       (price anomaly penalty)
  - discount_penalty          (if discount > 70%)
  + min(10, log_reviews × 1.5)  (volume bonus)
  + (rating - 3.5) × 5 × authenticity  (rating adjustment)
```

| Grade | Score | Verdict |
|---|---|---|
| A | 75–99 | Trusted |
| B | 60–74 | Caution |
| C | 45–59 | Caution |
| D | 5–44 | Risky |

---

## 5. Key Features

### 5.1 Rating-Calibrated Review Fallback
When Amazon blocks review scraping, the system generates reviews matching the product's star rating:

| Stars | Positive : Neutral : Negative |
|---|---|
| ≥ 4.3 | 8 : 1 : 1 |
| 3.8–4.2 | 6 : 2 : 2 |
| 3.0–3.7 | 4 : 3 : 3 |
| 2.0–2.9 | 2 : 2 : 6 |
| < 2.0 | 1 : 1 : 8 |

### 5.2 Category-Aware Competitor Pricing

| Category | Platforms |
|---|---|
| Electronics | Flipkart, Croma, Reliance Digital, Vijay Sales |
| Fashion | Flipkart, Myntra, Ajio, Nykaa Fashion |
| Beauty | Flipkart, Nykaa, Purplle, Tata 1mg |
| General | Flipkart, JioMart, Reliance Digital, Snapdeal |

### 5.3 Explainable AI
- **SHAP Bars** — Color-coded (green ≥70%, amber ≥40%, red <40%)
- **Positive/Risk Signals** — Evidence-based explanations
- **Trust Gauge** — Animated SVG arc with grade

---

## 6. API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/analyze` | Scrape + analyze an Amazon URL |
| `GET` | `/model-stats` | ML model metadata |
| `GET` | `/health` | Health check |
| `POST` | `/retrain` | Background retraining |

---

## 7. Technology Stack

### Backend
FastAPI · Uvicorn · scikit-learn ≥ 1.3 · XGBoost ≥ 2.0 · VADER Sentiment · pandas ≥ 2.0 · numpy ≥ 1.24 · cloudscraper · BeautifulSoup4 · joblib · Python 3.10+

### Frontend
React 19.2 · Vite 6.0 · Recharts 3.7 · Lucide React · TypeScript 5 · Vanilla CSS · Plus Jakarta Sans + Space Grotesk fonts

---

## 8. How to Run

```powershell
# One-click (Windows)
D:\TrustLens\start_project.bat

# Manual — Terminal 1 (Backend)
cd D:\TrustLens\backend
..\.venv\Scripts\python.exe main.py    # → http://127.0.0.1:8001

# Manual — Terminal 2 (Frontend)
cd D:\TrustLens\frontend
npm run dev                          # → http://localhost:5175

# Retrain models manually
.\venv\Scripts\python.exe ml\train_models.py

# Verify pipeline
.\venv\Scripts\python.exe test_full_pipeline.py
```

---

## 9. Datasets

| # | Dataset | Size | Used For |
|---|---|---|---|
| 1 | Fake and Real Product Reviews | 40,432 reviews | Fake Review Detection |
| 2 | Amazon Fine Food Reviews | 568,454 reviews | Sentiment Analysis |
| 3 | Amazon Products Dataset 2023 | 1.4M+ products | Price Anomaly Detection |

---

## 10. Known Limitations

1. Amazon frequently blocks review scraping — fallback generates rating-calibrated reviews
2. Competitor prices are category-aware estimates, not live-scraped
3. Fake review model: 68.6% accuracy (best effort on CG/OR dataset)
4. Sentiment model: 71% accuracy (Neutral class is hardest)
5. Currently supports Amazon.in URLs only

---

*Generated from live project state — April 28, 2026*
