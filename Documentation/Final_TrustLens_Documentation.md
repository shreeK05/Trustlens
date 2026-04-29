# TrustLens AI — Enterprise Edition
**Complete Project Documentation (Production Ready)**

## 1. Project Overview
TrustLens AI is an advanced, machine-learning-powered platform designed to perform real-time trust analysis on Amazon product listings. It protects consumers by identifying fake reviews, detecting deceptive pricing (anomalies/fake MRPs), highlighting sentiment mismatches, and scoring third-party seller risks. 

The project has been fully upgraded from a rule-based heuristic tool into a robust, 5-stage ML pipeline with a dynamic feedback loop and a premium "Neural-Legal" interface.

---

## 2. Architecture Stack
The platform is decoupled into a high-performance backend API and a fast, responsive frontend dashboard.

- **Backend (API & ML):** Python 3.10+, FastAPI, Uvicorn
- **Machine Learning Layer:** `scikit-learn`, `xgboost`, `imbalanced-learn` (SMOTE), `vaderSentiment`, `nltk`
- **Data Extraction:** `cloudscraper` (bypasses Cloudflare & Bot protections), `BeautifulSoup4`
- **Frontend (UI/UX):** React 19, Vite, Tailwind CSS v4, `lucide-react`, `recharts`
- **Deployment Strategy:** Render (Backend API) + Vercel (Frontend Dashboard)

---

## 3. The 5-Stage Machine Learning Pipeline
TrustLens AI evaluates every product using an ensemble of independent models, culminating in a single, explainable "Trust Score".

### Model 1: Review Authenticity (NLP / Logistic Regression)
- **Algorithm:** TF-IDF Vectorization + Logistic Regression.
- **Function:** Analyzes review text to detect unnatural repetition, copy-paste spam, and review bursts. Computes an `authenticity_score` representing the probability that the reviews are genuine.

### Model 2: Sentiment Mismatch (VADER Lexicon)
- **Algorithm:** VADER Sentiment Analysis + Rule-Based Logic.
- **Function:** Compares the actual text sentiment against the star rating. If a product has 5 stars but the text is heavily negative, it flags a *Mismatch Detected* warning to catch manipulative rating inflation.

### Model 3: Price Anomaly Watch (Statistical / Z-Score)
- **Algorithm:** Historical trend comparison + Discount validation.
- **Function:** Accurately scrapes the exact product MRP (ignoring unrelated recommended products). It analyzes the historical pricing curve to identify suspicious drops, fake markdown percentages, or highly unstable pricing patterns.

### Model 4: Seller Risk Classifier (XGBoost + SMOTE)
- **Algorithm:** Multi-class XGBoost Classifier.
- **Function:** Ranks third-party sellers as `Low`, `Medium`, or `High` risk. 
- **Handling Data Imbalance:** Since most sellers are genuine (causing extreme data imbalance), the model uses **SMOTE (Synthetic Minority Over-sampling Technique)** during training to generate synthetic high-risk profiles, heavily penalizing the model for lazily predicting "Trusted".

### Model 5: Trust Stacking Ensemble
- **Algorithm:** Stacking Classifier (Random Forest + XGBoost → Logistic Regression aggregator).
- **Function:** Consolidates the outputs from Models 1-4 into a final Trust Score (0-100) and assigns an alphabetical Grade (A, B, C, F), generating an explainable verdict for the user.

---

## 4. Key Features & Upgrades
1. **Bulletproof Scraper:** Integrated `cloudscraper` to bypass Amazon 503/Captcha blocks, ensuring the API successfully fetches live HTML payloads in production.
2. **Accurate MRP Isolation:** Rewrote CSS selectors to strictly target `#corePriceDisplay_desktop_feature_div` so the parser extracts the true product MRP instead of picking up expensive recommended items from the page footer.
3. **Dynamic Retraining Loop:** Implemented a `/report-inaccuracy` REST endpoint. Users can flag incorrect scores directly from the Vercel dashboard. This data is securely stored, allowing automated cron jobs to trigger `retrain_all.py` and actively fine-tune the ML models on new edge-cases.
4. **Premium UI/UX:** Built a dynamic, responsive dashboard featuring custom CSS Sparklines, interactive gauge charts, and clear "Pros vs Cons" signal breakdowns that visually match a high-end SaaS product.

---

## 5. Deployment Guide
The project has been decoupled and deployed seamlessly:

### Backend (Render)
- Configured using a `render.yaml` Blueprint file, setting the exact root directory and dependencies.
- **CORS Configuration:** `allow_origins=["*"]` ensures the backend accepts incoming requests from any frontend origin.
- The pre-trained `.joblib` files are tracked in Git, preventing Render from wasting memory retraining the models dynamically on boot.

### Frontend (Vercel)
- The Vite/React application is statically built and deployed globally via Vercel Edge networks.
- An environment variable `VITE_TRUSTLENS_API_URL` maps the frontend directly to the live Render endpoint (e.g., `https://trustlens-api-xxxx.onrender.com`).
- The `.gitignore` successfully isolates heavy dataset CSVs and `node_modules`, keeping the production repository lightweight and fast.
