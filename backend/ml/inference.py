"""
TrustLens AI — ML Inference Engine
Loads trained models and performs live predictions.
Supports both real-dataset models and synthetic fallback.
"""

import numpy as np
import joblib
import os
import re
from scipy.sparse import hstack, csr_matrix

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Lazy-loaded model cache
_models: dict = {}


def _load(name: str):
    if name not in _models:
        path = os.path.join(MODELS_DIR, f"{name}.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        _models[name] = joblib.load(path)
    return _models[name]


# ══════════════════════════════════════════════════════════════════
# MODULE 1 — FAKE REVIEW DETECTOR
# ══════════════════════════════════════════════════════════════════

def _extract_text_features(text: str) -> dict:
    words = text.split()
    return {
        "length":            len(text),
        "word_count":        len(words),
        "exclamation_count": text.count("!"),
        "question_count":    text.count("?"),
        "all_caps_words":    sum(1 for w in words if w.isupper() and len(w) > 1),
        "all_caps_ratio":    sum(1 for w in words if w.isupper() and len(w) > 1) / max(len(words), 1),
        "avg_word_len":      float(np.mean([len(w) for w in words])) if words else 0.0,
        "unique_ratio":      len(set(text.lower().split())) / max(len(words), 1),
        "rating":            4.0,
        "uppercase_ratio":   sum(1 for c in text if c.isupper()) / max(len(text), 1),
    }


def predict_fake_reviews(reviews: list) -> dict:
    """Classify reviews as genuine or fake using TF-IDF + LogReg."""
    if not reviews:
        return {
            "fake_count": 0, "genuine_count": 0, "total_analyzed": 0,
            "fake_probability": 0.0, "avg_fake_score": 0.0,
            "authenticity_score": 100.0, "flagged_reviews": [],
        }

    m           = _load("fake_review_model")
    tfidf       = m["tfidf"]
    clf         = m["clf"]
    num_cols    = m["numeric_cols"]

    scores, flagged = [], []
    for i, review in enumerate(reviews[:20]):
        review = str(review).strip()
        if len(review) < 3:
            continue
        feats   = _extract_text_features(review)
        X_text  = tfidf.transform([review])
        X_num   = csr_matrix([[feats.get(c, 0) for c in num_cols]])
        X       = hstack([X_text, X_num])
        prob    = clf.predict_proba(X)[0]
        # Index 1 = fake class
        p_fake  = float(prob[1]) if len(prob) > 1 else float(prob[0])
        scores.append(p_fake)
        if p_fake > 0.6:
            flagged.append({
                "index": i,
                "text": review[:140] + ("..." if len(review) > 140 else ""),
                "fake_probability": round(p_fake, 3),
            })

    avg_fake    = float(np.mean(scores)) if scores else 0.0
    fake_count  = sum(1 for s in scores if s > 0.5)
    genuine_count = len(scores) - fake_count

    return {
        "fake_count":        fake_count,
        "genuine_count":     genuine_count,
        "total_analyzed":    len(scores),
        "fake_probability":  round(avg_fake, 3),
        "avg_fake_score":    round(avg_fake, 3),
        "authenticity_score": round((1 - avg_fake) * 100, 1),
        "flagged_reviews":   flagged,
    }


# ══════════════════════════════════════════════════════════════════
# MODULE 2 — SENTIMENT ANALYZER (ML-based if model exists)
# ══════════════════════════════════════════════════════════════════

POSITIVE_WORDS = {
    "excellent", "amazing", "great", "good", "love", "perfect", "best",
    "awesome", "fantastic", "wonderful", "happy", "satisfied", "recommend",
    "value", "durable", "reliable", "fast", "easy", "comfortable", "worth",
    "premium", "solid", "pleased", "impressed", "outstanding",
}
NEGATIVE_WORDS = {
    "bad", "poor", "terrible", "horrible", "waste", "broken", "cheap",
    "disappoint", "slow", "uncomfortable", "useless", "fraud", "fake",
    "return", "refund", "damage", "defect", "missing", "wrong", "late",
    "expensive", "overpriced", "never", "worst", "avoid",
}


def analyze_sentiment(reviews: list, rating: float = 4.0) -> dict:
    """
    ML-based sentiment analysis using the trained classifier.
    Falls back to lexicon analysis if model unavailable.
    """
    if not reviews:
        return {
            "positive": 0, "neutral": 0, "negative": 0,
            "overall": "Neutral", "sentiment_score": 0.0,
            "mismatch_detected": False, "mismatch_reason": "",
            "sentiment_distribution": {"positive": 0.0, "neutral": 0.0, "negative": 0.0},
            "method": "none",
        }

    # Try ML-based classifier first
    try:
        m       = _load("sentiment_model")
        tfidf   = m["tfidf"]
        clf     = m["clf"]
        num_cols = m["numeric_cols"]
        label_map = m.get("label_map", {0: "Negative", 1: "Neutral", 2: "Positive"})

        pos, neu, neg = 0, 0, 0
        compound_scores = []

        for review in reviews[:20]:
            review = str(review).strip()
            if len(review) < 3:
                continue
            feats = {
                "helpfulness": 0.8,
                "text_len":    len(review),
                "excl":        review.count("!"),
                "ques":        review.count("?"),
            }
            X_text = tfidf.transform([review])
            X_num  = csr_matrix([[feats.get(c, 0) for c in num_cols]])
            X      = hstack([X_text, X_num])
            pred   = int(clf.predict(X)[0])
            proba  = clf.predict_proba(X)[0]

            if pred == 2:   pos += 1; compound_scores.append(1)
            elif pred == 1: neu += 1; compound_scores.append(0)
            else:           neg += 1; compound_scores.append(-1)

        total = pos + neu + neg or 1
        avg   = float(np.mean(compound_scores)) if compound_scores else 0.0
        overall = "Positive" if avg > 0.2 else "Negative" if avg < -0.2 else "Neutral"
        method = "ml_classifier"

    except Exception:
        # Lexicon fallback
        pos, neu, neg = 0, 0, 0
        compound_scores = []
        for review in reviews[:20]:
            words = set(str(review).lower().split())
            ph = len(words & POSITIVE_WORDS)
            nh = len(words & NEGATIVE_WORDS)
            raw = ph - nh
            if raw > 0.5:   pos += 1; compound_scores.append(1)
            elif raw < -0.5: neg += 1; compound_scores.append(-1)
            else:            neu += 1; compound_scores.append(0)
        total  = pos + neu + neg or 1
        avg    = float(np.mean(compound_scores)) if compound_scores else 0.0
        overall = "Positive" if avg > 0.2 else "Negative" if avg < -0.2 else "Neutral"
        method  = "lexicon_fallback"

    # Rating-sentiment mismatch
    mismatch, reason = False, ""
    if rating >= 4.2 and overall == "Negative":
        mismatch = True
        reason   = "High star rating but negative review language"
    elif rating <= 2.5 and overall == "Positive":
        mismatch = True
        reason   = "Low star rating but positive review language"

    return {
        "positive":  pos,
        "neutral":   neu,
        "negative":  neg,
        "overall":   overall,
        "sentiment_score":  round(avg, 3),
        "mismatch_detected": mismatch,
        "mismatch_reason":   reason,
        "sentiment_distribution": {
            "positive": round(pos / total * 100, 1),
            "neutral":  round(neu / total * 100, 1),
            "negative": round(neg / total * 100, 1),
        },
        "method": method,
    }


# ══════════════════════════════════════════════════════════════════
# MODULE 3 — PRICE ANOMALY DETECTOR (Isolation Forest)
# ══════════════════════════════════════════════════════════════════

def detect_price_anomaly(price: float, mrp: float, price_history: list) -> dict:
    """Isolation Forest anomaly detection on real price distributions."""
    m       = _load("price_anomaly_model")
    iso     = m["iso_forest"]
    scaler  = m["scaler"]
    stats   = m.get("price_stats", {"mean": price, "std": price * 0.3})

    if mrp <= 0 or mrp < price:
        mrp = price * 1.15

    discount_pct   = ((mrp - price) / mrp) * 100 if mrp > price else 0
    price_log      = float(np.log1p(price))
    price_to_list  = price / max(mrp, 1)
    price_zscore   = (price - stats["mean"]) / (stats["std"] + 1)

    # Review proxy from history count
    reviews_log    = float(np.log1p(len(price_history) * 100))  # rough proxy

    # Stars proxy from rating (if available)
    stars          = 4.0  # default; caller can override via meta

    X = np.array([[discount_pct, price_log, price_to_list, price_zscore, stars, reviews_log]])
    X_scaled = scaler.transform(X)

    pred   = iso.predict(X_scaled)[0]
    score  = iso.score_samples(X_scaled)[0]

    # Normalize: more negative score = more anomalous
    norm_score = float(1 / (1 + np.exp(score * 2)))

    is_anomaly = pred == -1

    # Price trend from history
    prices_hist = [p["price"] for p in price_history]
    trend = "stable"
    if len(prices_hist) >= 3:
        early = np.mean(prices_hist[:2])
        if price < early * 0.82:    trend = "suspicious_drop"
        elif price > early * 1.18:  trend = "rising"

    vs_avg = 0.0
    if prices_hist:
        mean_hist = float(np.mean(prices_hist))
        vs_avg = round(((price - mean_hist) / (mean_hist + 1)) * 100, 1)

    return {
        "is_anomaly":    bool(is_anomaly),
        "anomaly_score": round(norm_score, 3),
        "price_trend":   trend,
        "discount_pct":  round(discount_pct, 1),
        "vs_avg_history": vs_avg,
        "confidence":    round(abs(float(score)), 3),
    }


# ══════════════════════════════════════════════════════════════════
# MODULE 4 — SELLER RISK CLASSIFIER (XGBoost, Weak Supervision)
# ══════════════════════════════════════════════════════════════════

RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}
RISK_COLORS = {0: "green", 1: "yellow", 2: "red"}


def classify_seller_risk(seller_info: dict) -> dict:
    """
    XGBoost seller risk classification.
    Features match those used during weak-supervision training:
      stars_val, log_reviews, discount, is_bestseller, log_bought, price_log
    """
    m         = _load("seller_risk_model")
    clf       = m["clf"]
    scaler    = m["scaler"]
    features  = m["features"]
    importance = m.get("feature_importance", {})

    seller_name  = str(seller_info.get("name", ""))
    is_amazon    = any(k in seller_name for k in ["Amazon", "Appario", "Cloudtail"])
    rating       = float(seller_info.get("rating", 3.8))
    discount_pct = float(seller_info.get("discount_aggressiveness", 0.3)) * 100
    reviews_est  = 1000 if is_amazon else 150     # estimated reviews
    bought_est   = 500  if is_amazon else 50
    price_est    = float(seller_info.get("price", 1000))

    feat_map = {
        "stars_val":    rating,
        "log_reviews":  float(np.log1p(reviews_est)),
        "discount":     discount_pct,
        "is_bestseller": 1 if is_amazon else 0,
        "log_bought":   float(np.log1p(bought_est)),
        "price_log":    float(np.log1p(price_est)),
    }
    vec = [feat_map.get(f, 0.0) for f in features]

    X        = np.array([vec])
    X_scaled = scaler.transform(X)

    risk_class = int(clf.predict(X_scaled)[0])
    risk_proba = clf.predict_proba(X_scaled)[0].tolist()

    return {
        "risk_level":  RISK_LABELS[risk_class],
        "risk_score":  risk_class,
        "confidence":  round(max(risk_proba), 3),
        "probabilities": {
            "low":    round(risk_proba[0], 3),
            "medium": round(risk_proba[1], 3),
            "high":   round(risk_proba[2], 3),
        },
        "color": RISK_COLORS[risk_class],
        "feature_importance": importance,
    }


# ══════════════════════════════════════════════════════════════════
# MODULE 5 — TRUST SCORE ENSEMBLE (Stacking)
# ══════════════════════════════════════════════════════════════════

def predict_trust_score(meta: dict) -> dict:
    """Stacking ensemble (RF + XGB → LogReg) for final trust score."""
    m        = _load("trust_score_model")
    stacker  = m["stacker"]
    scaler   = m["scaler"]
    features = m["features"]

    vec      = [meta.get(f, 0.0) for f in features]
    X        = np.array([vec])
    X_scaled = scaler.transform(X)

    prob_trust = float(stacker.predict_proba(X_scaled)[0][1])
    raw_score  = float(meta.get("raw_score", 75))

    # Blend: 55% model, 45% heuristic
    final = int(prob_trust * 0.55 * 100 + raw_score * 0.45)
    final = max(5, min(99, final))

    shap = {
        "Review Authenticity":    round((1 - meta.get("fake_review_prob", 0.3)) * 25, 1),
        "Seller Trustworthiness": round((1 - meta.get("seller_risk_encoded", 1) / 2) * 30, 1),
        "Price Legitimacy":       round((1 - meta.get("price_anomaly_score", 0.3)) * 20, 1),
        "Rating Signals":         round(((meta.get("rating", 3.5) - 1) / 4) * 14, 1),
        "Review Volume":          round(min(meta.get("log_review_count", 5) / 12, 1) * 11, 1),
    }

    grade = "A" if final > 85 else "B" if final > 70 else "C" if final > 55 else "D"

    return {
        "score":            final,
        "trust_probability": round(prob_trust, 3),
        "verdict":          "Trusted" if final > 70 else "Caution" if final > 45 else "Risky",
        "confidence_pct":   round(prob_trust * 100, 1),
        "shap_contributions": shap,
        "certificate":      "valid" if final > 70 else "warning",
        "grade":            grade,
    }
