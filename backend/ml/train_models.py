"""
TrustLens AI — Real Dataset ML Training Pipeline
Trains all 4 models using actual datasets:
  Module 1: Fake Reviews  → Fake and Real Product Reviews dataset (CG/OR labels)
  Module 2: Sentiment     → Amazon Fine Food Reviews (Score 1-5)
  Module 3: Price Anomaly → Amazon Products Dataset 2023 (price/discount features)
  Module 4: Seller Risk   → Weak supervision from Products dataset + XGBoost
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest, StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib
import os
import json
import re
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────
BACKEND_DIR   = os.path.dirname(__file__)
MODELS_DIR    = os.path.join(BACKEND_DIR, "models")
DATASET_DIR   = os.path.join(os.path.dirname(os.path.dirname(BACKEND_DIR)), "Dataset")

FAKE_REVIEWS_PATH  = os.path.join(DATASET_DIR, "Fake and Real Product Reviews", "fake reviews dataset.csv")
FOOD_REVIEWS_PATH  = os.path.join(DATASET_DIR, "Amazon Fine Food Reviews", "Reviews.csv")
PRODUCTS_PATH      = os.path.join(DATASET_DIR, "Amazon Products Dataset 2023", "amazon_products.csv")

os.makedirs(MODELS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# MODULE 1 — FAKE REVIEW DETECTOR
# Dataset: "Fake and Real Product Reviews"
# Labels: CG = Computer Generated (Fake), OR = Original (Genuine)
# ═══════════════════════════════════════════════════════════════════

def train_fake_review_model():
    print("\n" + "="*60)
    print("  MODULE 1: Fake Review Detector")
    print("="*60)
    print(f"  Loading: {FAKE_REVIEWS_PATH}")

    df = pd.read_csv(FAKE_REVIEWS_PATH)
    print(f"  Rows loaded: {len(df):,}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Label distribution:\n{df['label'].value_counts().to_string()}")

    # ── Clean ──────────────────────────────────────────────────────
    df = df.dropna(subset=["text_", "label", "rating"]).copy()
    df["text_"] = df["text_"].astype(str).str.strip()
    df = df[df["text_"].str.len() > 5]

    # ── Label: CG=1 (fake), OR=0 (genuine) ────────────────────────
    df["is_fake"] = (df["label"].str.upper() == "CG").astype(int)

    # ── Feature Engineering ────────────────────────────────────────
    df["length"]            = df["text_"].str.len()
    df["word_count"]        = df["text_"].str.split().str.len()
    df["exclamation_count"] = df["text_"].str.count("!")
    df["question_count"]    = df["text_"].str.count(r"\?")
    df["all_caps_words"]    = df["text_"].apply(lambda t: sum(1 for w in t.split() if w.isupper() and len(w) > 1))
    df["all_caps_ratio"]    = df["all_caps_words"] / df["word_count"].clip(lower=1)
    df["avg_word_len"]      = df["text_"].apply(lambda t: np.mean([len(w) for w in t.split()]) if t.split() else 0)
    df["unique_ratio"]      = df["text_"].apply(lambda t: len(set(t.lower().split())) / max(len(t.split()), 1))
    df["rating"]            = pd.to_numeric(df["rating"], errors="coerce").fillna(4.0)
    df["uppercase_ratio"]   = df["text_"].apply(lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1))

    # ── Sample balanced if large ───────────────────────────────────
    min_class = df["is_fake"].value_counts().min()
    sample_size = min(min_class, 20000)
    df_fake  = df[df["is_fake"] == 1].sample(n=sample_size, random_state=42)
    df_real  = df[df["is_fake"] == 0].sample(n=sample_size, random_state=42)
    df_bal   = pd.concat([df_fake, df_real]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  Training on {len(df_bal):,} balanced samples ({sample_size:,} per class)")

    numeric_cols = ["length", "word_count", "exclamation_count", "question_count",
                    "all_caps_ratio", "avg_word_len", "unique_ratio", "rating",
                    "uppercase_ratio", "all_caps_words"]

    # ── TF-IDF on text ─────────────────────────────────────────────
    from scipy.sparse import hstack, csr_matrix
    print("  Building TF-IDF matrix (max_features=5000)...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                            sublinear_tf=True, min_df=3, strip_accents="unicode")
    X_text   = tfidf.fit_transform(df_bal["text_"])
    X_num    = csr_matrix(df_bal[numeric_cols].fillna(0).values)
    X        = hstack([X_text, X_num])
    y        = df_bal["is_fake"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("  Training Logistic Regression...")
    clf = LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced", solver="saga", n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Genuine", "Fake"], output_dict=True)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Fake  : {report['Fake']['f1-score']:.4f}")
    print(f"  F1 Real  : {report['Genuine']['f1-score']:.4f}")

    model_data = {
        "tfidf": tfidf,
        "clf": clf,
        "numeric_cols": numeric_cols,
        "accuracy": round(acc, 4),
        "f1_fake": round(report["Fake"]["f1-score"], 4),
        "f1_genuine": round(report["Genuine"]["f1-score"], 4),
        "n_samples": len(df_bal),
        "trained_on": "real_dataset",
    }
    joblib.dump(model_data, os.path.join(MODELS_DIR, "fake_review_model.joblib"))
    print(f"  Saved -> ml/models/fake_review_model.joblib")
    return {"accuracy": round(acc, 4), "n_samples": len(df_bal), "f1_fake": round(report["Fake"]["f1-score"], 4)}


# ═══════════════════════════════════════════════════════════════════
# MODULE 2 — SENTIMENT ANALYZER
# Dataset: Amazon Fine Food Reviews (Score 1-5)
# Maps Score > 3 = Positive, Score 3 = Neutral, Score < 3 = Negative
# ═══════════════════════════════════════════════════════════════════

def train_sentiment_model():
    print("\n" + "="*60)
    print("  MODULE 2: Sentiment Classifier")
    print("="*60)
    print(f"  Loading: {FOOD_REVIEWS_PATH}")

    # Only load needed columns — huge file
    df = pd.read_csv(FOOD_REVIEWS_PATH,
                     usecols=["Score", "Text", "HelpfulnessNumerator", "HelpfulnessDenominator"],
                     nrows=200000)
    print(f"  Rows loaded: {len(df):,}")

    df = df.dropna(subset=["Score", "Text"]).copy()
    df["Text"] = df["Text"].astype(str).str.strip()
    df = df[df["Text"].str.len() > 10]

    # ── Sentiment label ───────────────────────────────────────────
    def score_to_sentiment(s):
        if s >= 4:   return 2   # Positive
        elif s == 3: return 1   # Neutral
        else:        return 0   # Negative

    df["sentiment"] = df["Score"].astype(int).apply(score_to_sentiment)

    # ── Feature Engineering ───────────────────────────────────────
    df["helpfulness"] = df["HelpfulnessNumerator"] / df["HelpfulnessDenominator"].clip(lower=1)
    df["text_len"]    = df["Text"].str.len()
    df["excl"]        = df["Text"].str.count("!")
    df["ques"]        = df["Text"].str.count(r"\?")

    # ── Balance classes ────────────────────────────────────────────
    min_class = df["sentiment"].value_counts().min()
    sample_size = min(min_class, 15000)
    frames = [df[df["sentiment"] == c].sample(n=sample_size, random_state=42) for c in [0, 1, 2]]
    df_bal = pd.concat(frames).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  Training on {len(df_bal):,} balanced samples ({sample_size:,}/class)")

    numeric_cols = ["helpfulness", "text_len", "excl", "ques"]

    from scipy.sparse import hstack, csr_matrix
    print("  Building TF-IDF matrix (max_features=3000)...")
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True,
                            min_df=3, strip_accents="unicode")
    X_text = tfidf.fit_transform(df_bal["Text"])
    X_num  = csr_matrix(df_bal[numeric_cols].fillna(0).values)
    X      = hstack([X_text, X_num])
    y      = df_bal["sentiment"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("  Training Logistic Regression...")
    clf = LogisticRegression(C=1.5, max_iter=500, class_weight="balanced", solver="saga", n_jobs=-1, multi_class="multinomial")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Negative","Neutral","Positive"], output_dict=True)
    print(f"  Accuracy: {acc:.4f}")

    model_data = {
        "tfidf": tfidf,
        "clf": clf,
        "numeric_cols": numeric_cols,
        "accuracy": round(acc, 4),
        "n_samples": len(df_bal),
        "trained_on": "real_dataset",
        "label_map": {0: "Negative", 1: "Neutral", 2: "Positive"},
    }
    joblib.dump(model_data, os.path.join(MODELS_DIR, "sentiment_model.joblib"))
    print(f"  Saved -> ml/models/sentiment_model.joblib")
    return {"accuracy": round(acc, 4), "n_samples": len(df_bal)}


# ═══════════════════════════════════════════════════════════════════
# MODULE 3 — PRICE ANOMALY DETECTOR
# Dataset: Amazon Products Dataset 2023
# Uses actual price/discount distributions from real product data
# ═══════════════════════════════════════════════════════════════════

def train_price_anomaly_model():
    print("\n" + "="*60)
    print("  MODULE 3: Price Anomaly Detector (Isolation Forest)")
    print("="*60)
    print(f"  Loading: {PRODUCTS_PATH}")

    # Load only price-related columns — huge file
    df = pd.read_csv(PRODUCTS_PATH, nrows=150000,
                     usecols=lambda c: c in [
                         "price", "listPrice", "stars", "reviews",
                         "boughtInLastMonth", "isBestSeller", "category_id"
                     ])
    print(f"  Rows loaded: {len(df):,}")
    print(f"  Columns found: {df.columns.tolist()}")

    df = df.dropna(subset=["price"]).copy()

    # ── Normalize price column (may have $ sign) ───────────────────
    def clean_price(val):
        if pd.isna(val): return np.nan
        return float(re.sub(r"[^\d.]", "", str(val)) or 0)

    df["price_clean"]    = df["price"].apply(clean_price)
    df["list_clean"]     = df["listPrice"].apply(clean_price) if "listPrice" in df.columns else df["price_clean"] * 1.2

    df = df[(df["price_clean"] > 0)].copy()
    df["list_clean"]     = df["list_clean"].fillna(df["price_clean"] * 1.1)
    df["list_clean"]     = df[["list_clean", "price_clean"]].max(axis=1)  # list >= price

    # ── Feature Engineering ────────────────────────────────────────
    df["discount_pct"]     = ((df["list_clean"] - df["price_clean"]) / df["list_clean"] * 100).clip(0, 99)
    df["price_log"]        = np.log1p(df["price_clean"])
    df["price_to_list"]    = df["price_clean"] / df["list_clean"].clip(lower=0.01)

    # Stars
    if "stars" in df.columns:
        df["stars"] = pd.to_numeric(df["stars"], errors="coerce").fillna(0)
    else:
        df["stars"] = 4.0

    # Reviews
    if "reviews" in df.columns:
        df["reviews_log"] = np.log1p(pd.to_numeric(df["reviews"], errors="coerce").fillna(0))
    else:
        df["reviews_log"] = 6.0

    # Price Z-score within dataset
    price_mean = df["price_clean"].mean()
    price_std  = df["price_clean"].std()
    df["price_zscore"] = (df["price_clean"] - price_mean) / (price_std + 1)

    features = ["discount_pct", "price_log", "price_to_list", "price_zscore",
                 "stars", "reviews_log"]
    df_feat = df[features].fillna(0)

    # Cap at 80k for training speed
    df_feat = df_feat.sample(n=min(80000, len(df_feat)), random_state=42)
    print(f"  Training on {len(df_feat):,} real products")

    scaler = StandardScaler()
    X = scaler.fit_transform(df_feat.values)

    print("  Training Isolation Forest (n_estimators=300)...")
    iso = IsolationForest(n_estimators=300, contamination=0.05,
                          max_samples="auto", random_state=42, n_jobs=-1)
    iso.fit(X)

    # Quick sanity: how many flagged as anomalies?
    preds = iso.predict(X[:5000])
    anomaly_rate = (preds == -1).mean()
    print(f"  Contamination rate on training set: {anomaly_rate:.3f}")

    # Price statistics for the inference engine
    price_stats = {
        "mean": round(float(price_mean), 2),
        "std":  round(float(price_std), 2),
        "p25":  round(float(df["price_clean"].quantile(0.25)), 2),
        "p75":  round(float(df["price_clean"].quantile(0.75)), 2),
        "p95":  round(float(df["price_clean"].quantile(0.95)), 2),
    }
    print(f"  Price stats: {price_stats}")

    model_data = {
        "iso_forest":   iso,
        "scaler":       scaler,
        "features":     features,
        "price_stats":  price_stats,
        "anomaly_rate": round(float(anomaly_rate), 4),
        "n_samples":    len(df_feat),
        "trained_on":   "real_dataset",
    }
    joblib.dump(model_data, os.path.join(MODELS_DIR, "price_anomaly_model.joblib"))
    print(f"  Saved -> ml/models/price_anomaly_model.joblib")
    return {"anomaly_detection_rate": round(float(anomaly_rate), 4), "n_samples": len(df_feat)}


# ═══════════════════════════════════════════════════════════════════
# MODULE 4 — SELLER RISK CLASSIFIER
# Approach: Weak Supervision on Products Dataset
#   Low Risk  (0): Amazon/Appario keyword in category, high stars, many reviews
#   Medium Risk(1): Mixed signals
#   High Risk (2): Low stars, very high discount, very few reviews
# Then train XGBoost on the auto-labeled data
# ═══════════════════════════════════════════════════════════════════

def train_seller_risk_model():
    print("\n" + "="*60)
    print("  MODULE 4: Seller Risk Classifier (Weak Supervision + XGBoost)")
    print("="*60)
    print(f"  Loading: {PRODUCTS_PATH}")

    df = pd.read_csv(PRODUCTS_PATH, nrows=100000,
                     usecols=lambda c: c in [
                         "price", "listPrice", "stars", "reviews",
                         "boughtInLastMonth", "isBestSeller", "category_id", "title"
                     ])
    print(f"  Rows loaded: {len(df):,}")

    df = df.dropna(subset=["price", "stars"]).copy()

    # ── Clean numerics ─────────────────────────────────────────────
    def to_float(val):
        try: return float(re.sub(r"[^\d.]", "", str(val)) or 0)
        except: return 0.0

    df["price_val"]     = df["price"].apply(to_float)
    df["list_val"]      = df.get("listPrice", df["price"]).apply(to_float)
    df["stars_val"]     = pd.to_numeric(df["stars"], errors="coerce").fillna(0)
    df["reviews_val"]   = pd.to_numeric(df.get("reviews", 0), errors="coerce").fillna(0)
    df["bought_val"]    = pd.to_numeric(df.get("boughtInLastMonth", 0), errors="coerce").fillna(0)
    df["is_bestseller"] = df.get("isBestSeller", False).astype(int)

    df = df[(df["price_val"] > 0) & (df["stars_val"] > 0)].copy()
    df["list_val"]    = df[["list_val", "price_val"]].max(axis=1)
    df["discount"]    = ((df["list_val"] - df["price_val"]) / df["list_val"].clip(lower=0.01) * 100).clip(0, 99)

    # ── Weak Supervision Labeling ──────────────────────────────────
    # This is the academic approach: auto-label using business rules,
    # then train ML to learn the underlying pattern and generalize.
    def label_seller_risk(row):
        s, r, d, bs = row["stars_val"], row["reviews_val"], row["discount"], row["is_bestseller"]
        # Low Risk
        if s >= 4.2 and r >= 500 and d <= 60:     return 0
        if bs == 1 and s >= 4.0:                  return 0
        # High Risk
        if s < 3.0 and r < 20:                    return 2
        if d > 80 and r < 10:                     return 2
        if s < 2.5:                                return 2
        # Medium Risk (default)
        return 1

    df["risk_label"] = df.apply(label_seller_risk, axis=1)
    print(f"  Weak-supervision label distribution:\n{df['risk_label'].value_counts().to_string()}")

    # ── Features for XGBoost ───────────────────────────────────────
    df["log_reviews"]   = np.log1p(df["reviews_val"])
    df["log_bought"]    = np.log1p(df["bought_val"])
    df["price_log"]     = np.log1p(df["price_val"])

    feature_cols = ["stars_val", "log_reviews", "discount", "is_bestseller",
                    "log_bought", "price_log"]

    # ── SMOTE for Class Imbalance ──────────────────────────────────
    X = df[feature_cols].fillna(0).values
    y = df["risk_label"].values

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=0.2, random_state=42, stratify=y)
    
    from imblearn.over_sampling import SMOTE
    print("  Applying SMOTE to balance classes in training set...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"  Training on {len(y_train):,} SMOTE-balanced samples")
    
    df_bal = pd.DataFrame(y_train, columns=["risk_label"]) # just for the n_samples count below

    print("  Training XGBoost classifier...")
    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.1,
        reg_alpha=0.1,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Low","Medium","High"], output_dict=True)
    importance = dict(zip(feature_cols, clf.feature_importances_.tolist()))

    print(f"  Accuracy: {acc:.4f}")
    print(f"  Feature Importance: {importance}")

    model_data = {
        "clf":              clf,
        "scaler":           scaler,
        "features":         feature_cols,
        "accuracy":         round(acc, 4),
        "feature_importance": {k: round(float(v), 4) for k, v in importance.items()},
        "label_map":        {0: "Low", 1: "Medium", 2: "High"},
        "n_samples":        len(df_bal),
        "trained_on":       "real_dataset_weak_supervision",
        "methodology":      "Weak supervision labels applied to Amazon Products 2023 dataset",
    }
    joblib.dump(model_data, os.path.join(MODELS_DIR, "seller_risk_model.joblib"))
    print(f"  Saved -> ml/models/seller_risk_model.joblib")
    return {"accuracy": round(acc, 4), "n_samples": len(df_bal), "feature_importance": {k: round(float(v), 4) for k, v in importance.items()}}


# ═══════════════════════════════════════════════════════════════════
# MODULE 5 — TRUST SCORE ENSEMBLE (uses outputs of above models)
# Same approach as before but trained on realistic distributions
# derived from the real datasets above
# ═══════════════════════════════════════════════════════════════════

def train_trust_score_model():
    print("\n" + "="*60)
    print("  MODULE 5: Trust Score Ensemble (Stacking)")
    print("="*60)

    # Load products dataset to get real price/rating distributions
    df_prod = pd.read_csv(PRODUCTS_PATH, nrows=50000,
                          usecols=lambda c: c in ["price", "stars", "reviews", "isBestSeller"])
    df_prod = df_prod.dropna(subset=["stars"]).copy()
    df_prod["stars"]   = pd.to_numeric(df_prod["stars"], errors="coerce").fillna(3.5)
    df_prod["reviews"] = pd.to_numeric(df_prod.get("reviews", 0), errors="coerce").fillna(0)

    np.random.seed(42)
    n = min(5000, len(df_prod))
    df_sample = df_prod.sample(n=n, random_state=42).reset_index(drop=True)

    rows = []
    for _, row in df_sample.iterrows():
        rating      = float(row["stars"])
        rev_count   = float(row["reviews"])
        # Simulate ML module outputs with realistic noise
        # Fake review prob: higher for low-review products
        fake_prob   = max(0, min(1, np.random.beta(2, 5) + (0.3 if rev_count < 50 else 0)))
        sent_mismatch = float(np.random.binomial(1, 0.15))
        price_anomaly = float(np.random.beta(1.5, 6))
        seller_risk  = np.random.choice([0, 1, 2], p=[0.55, 0.35, 0.10])
        discount     = float(np.random.uniform(5, 70))

        # Realistic trust score (the target we want the model to learn)
        score = 82.0
        score -= fake_prob * 22
        score -= sent_mismatch * 8
        score -= price_anomaly * 14
        score -= seller_risk * 11
        score += (rating - 3.5) * 4
        score += min(np.log1p(rev_count) * 1.2, 8)
        score -= max(0, (discount - 60) * 0.5)
        score = max(5, min(99, score))

        rows.append({
            "fake_review_prob":    fake_prob,
            "sentiment_mismatch":  sent_mismatch,
            "price_anomaly_score": price_anomaly,
            "seller_risk_encoded": seller_risk,
            "rating":              rating,
            "log_review_count":    np.log1p(rev_count),
            "discount_pct":        discount,
            "raw_score":           score,
        })

    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c != "raw_score"]
    X = df[feature_cols].values
    # Binary: trusted if score > 60
    y = (df["raw_score"] > 60).astype(int).values

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=0.2, random_state=42)

    print(f"  Training on {len(df):,} meta-samples")
    base = [
        ("rf",  RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)),
        ("xgb", xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                    use_label_encoder=False, eval_metric="logloss",
                                    random_state=42, n_jobs=-1)),
    ]
    stacker = StackingClassifier(estimators=base, final_estimator=LogisticRegression(max_iter=500), cv=5)
    stacker.fit(X_train, y_train)

    y_pred = stacker.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"  Stacking Ensemble Accuracy: {acc:.4f}")

    model_data = {
        "stacker":  stacker,
        "scaler":   scaler,
        "features": feature_cols,
        "accuracy": round(acc, 4),
        "n_samples": len(df),
        "trained_on": "real_distribution",
    }
    joblib.dump(model_data, os.path.join(MODELS_DIR, "trust_score_model.joblib"))
    print(f"  Saved -> ml/models/trust_score_model.joblib")
    return {"accuracy": round(acc, 4), "n_samples": len(df)}


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def train_all_models():
    print("\n" + "#"*60)
    print("  TrustLens AI - Full Real-Dataset Training Pipeline")
    print("#"*60)

    stats = {}
    stats["fake_review"]  = train_fake_review_model()
    stats["sentiment"]    = train_sentiment_model()
    stats["price_anomaly"] = train_price_anomaly_model()
    stats["seller_risk"]  = train_seller_risk_model()
    stats["trust_score"]  = train_trust_score_model()

    print("\n" + "#"*60)
    print("  All models trained and saved successfully!")
    print("#"*60)

    stats_path = os.path.join(MODELS_DIR, "model_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats -> {stats_path}\n")
    return stats


def models_exist():
    required = [
        "fake_review_model.joblib",
        "sentiment_model.joblib",
        "price_anomaly_model.joblib",
        "seller_risk_model.joblib",
        "trust_score_model.joblib",
    ]
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in required)


if __name__ == "__main__":
    train_all_models()
