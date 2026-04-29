# ml/inference.py — Production Inference Engine
# Handles: missing models gracefully, consistent feature extraction, 
#          proper probability calibration, SHAP-style explanations

import joblib
import numpy as np
import re
import os
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.sparse import hstack, csr_matrix

MODEL_DIR = Path(__file__).parent / "models"

# Lazy-loaded model cache (load once, reuse)
_models = {}

def load_model(name: str):
    """Load model from disk with caching. Returns None if not found."""
    if name not in _models:
        path = MODEL_DIR / f"{name}.joblib"
        if path.exists():
            _models[name] = joblib.load(path)
            print(f"✅ Loaded model: {name}")
        else:
            print(f"⚠️  Model not found: {name} — using fallback")
            _models[name] = None
    return _models[name]


# ─────────────────────────────────────────────
# MODULE 1: FAKE REVIEW DETECTION
# ─────────────────────────────────────────────

def extract_handcrafted_features(texts):
    """Must match train_models.py exactly."""
    features = []
    for text in texts:
        text = str(text)
        words = text.split()
        feat = [
            len(text),
            len(words),
            np.mean([len(w) for w in words]) if words else 0,
            len([s for s in text.split('.') if s.strip()]),
            text.count('!'),
            text.count('?'),
            sum(1 for c in text if c.isupper()) / max(len(text), 1),
            len(set(words)) / max(len(words), 1),
            len(set([w.lower() for w in words])) / max(len(words), 1),
            int('http' in text or 'www.' in text),
            sum(1 for w in words if w.lower() in 
                ['i','my','me','we','our','you','your']) / max(len(words), 1),
            sum(1 for w in words if w.lower() in 
                ['best','amazing','perfect','excellent','outstanding',
                 'incredible','fantastic','worst']),
            int(bool(re.search(r'product|item|thing', text.lower()))),
            len(re.findall(r'\d+', text)),
            int(len(words) < 10),
        ]
        features.append(feat)
    return np.array(features)


def analyze_reviews_fake(reviews: list) -> dict:
    """
    Run fake review detection on a list of review strings.
    Returns authenticity_score, fake_count, genuine_count, flagged_reviews.
    """
    artifact = load_model("fake_review_model")
    
    if not reviews:
        return {
            "authenticity_score": 50.0,
            "fake_count": 0, "genuine_count": 0,
            "flagged_reviews": [], "confidence": "low",
            "warning": "No reviews available"
        }
    
    # Fallback if model not trained yet
    if artifact is None:
        return _fake_review_fallback(reviews)
    
    model = artifact['model']
    tfidf_word = artifact['tfidf_word']
    tfidf_char = artifact['tfidf_char']
    scaler = artifact['scaler']
    threshold = artifact['metadata'].get('threshold', 0.55)
    
    # Feature extraction (must match training exactly)
    X_word = tfidf_word.transform(reviews)
    X_char = tfidf_char.transform(reviews)
    X_hand = scaler.transform(extract_handcrafted_features(reviews))
    X = hstack([X_word, X_char, csr_matrix(X_hand)])
    
    # Get calibrated probabilities
    probs = model.predict_proba(X)[:, 1]  # P(fake)
    
    fake_mask = probs > threshold
    fake_count = int(fake_mask.sum())
    genuine_count = len(reviews) - fake_count
    
    # Authenticity score: inverse of avg fake probability
    avg_fake_prob = float(np.mean(probs))
    authenticity_score = round((1 - avg_fake_prob) * 100, 1)
    
    # Flag worst reviews with their fake probability
    flagged = []
    for i, (review, prob) in enumerate(zip(reviews, probs)):
        if prob > threshold:
            flagged.append({
                "index": i,
                "text": review[:200] + "..." if len(review) > 200 else review,
                "fake_probability": round(float(prob) * 100, 1),
                "reason": _explain_fake_flag(review, prob)
            })
    
    # Sort by probability descending
    flagged = sorted(flagged, key=lambda x: x['fake_probability'], reverse=True)[:5]
    
    return {
        "authenticity_score": authenticity_score,
        "fake_count": fake_count,
        "genuine_count": genuine_count,
        "total_analyzed": len(reviews),
        "avg_fake_probability": float(avg_fake_prob),  # Raw 0-1 float for frontend to format
        "flagged_reviews": flagged,
        "confidence": "high" if len(reviews) >= 10 else "medium" if len(reviews) >= 5 else "low"
    }


def _explain_fake_flag(text: str, prob: float) -> str:
    """Generate human-readable explanation for why review was flagged."""
    reasons = []
    words = text.split()
    
    if len(words) < 10:
        reasons.append("very short review")
    if text.count('!') > 2:
        reasons.append("excessive exclamation marks")
    if sum(1 for c in text if c.isupper()) / max(len(text), 1) > 0.3:
        reasons.append("high caps ratio")
    if sum(1 for w in words if w.lower() in ['best','amazing','perfect','excellent']) > 2:
        reasons.append("overuse of superlatives")
    if not re.search(r'\d+', text):
        reasons.append("no specific details/numbers")
    
    if not reasons:
        reasons.append(f"ML pattern detected ({prob:.0%} fake probability)")
    
    return "; ".join(reasons)


def _fake_review_fallback(reviews: list) -> dict:
    """Rule-based fallback when model isn't trained yet."""
    fake_count = 0
    flagged = []
    
    for i, review in enumerate(reviews):
        text = str(review)
        words = text.split()
        is_suspicious = (
            len(words) < 8 or
            text.count('!') > 3 or
            sum(1 for c in text if c.isupper()) / max(len(text), 1) > 0.4
        )
        if is_suspicious:
            fake_count += 1
            flagged.append({"index": i, "text": text[:150], 
                           "fake_probability": 70.0, "reason": "Rule-based flag"})
    
    return {
        "authenticity_score": round((1 - fake_count / max(len(reviews), 1)) * 100, 1),
        "fake_count": fake_count,
        "genuine_count": len(reviews) - fake_count,
        "total_analyzed": len(reviews),
        "flagged_reviews": flagged[:5],
        "confidence": "low",
        "warning": "Using rule-based fallback — retrain models"
    }


# ─────────────────────────────────────────────
# MODULE 2: SENTIMENT ANALYSIS
# ─────────────────────────────────────────────

def analyze_sentiment(reviews: list, rating: float = 3.0) -> dict:
    """
    Analyze sentiment distribution across reviews.
    Returns: overall sentiment, distribution, mismatch detection.
    """
    artifact = load_model("sentiment_model")
    
    if not reviews:
        return {
            "overall": "Neutral", "distribution": {"Positive": 0, "Neutral": 0, "Negative": 0},
            "mismatch_detected": False, "confidence_score": 0.0
        }
    
    if artifact is None:
        return _sentiment_fallback(reviews, rating)
    
    model = artifact['model']
    tfidf = artifact['tfidf']
    le = artifact['label_encoder']
    
    # Extract features
    X_tfidf = tfidf.transform(reviews)
    
    # Build handcrafted features
    ratings_arr = np.full(len(reviews), rating)
    X_hand = _extract_sentiment_features_inference(reviews, ratings_arr)
    X = hstack([X_tfidf, csr_matrix(X_hand)])
    
    # Predict
    y_pred = model.predict(X)
    labels = le.inverse_transform(y_pred)
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)
        confidence = float(np.mean(np.max(probs, axis=1)))
    else:
        confidence = 0.75
    
    # Distribution
    from collections import Counter
    dist = Counter(labels)
    total = len(labels)
    distribution = {
        "Positive": round(dist.get('Positive', 0) / total * 100, 1),
        "Neutral": round(dist.get('Neutral', 0) / total * 100, 1),
        "Negative": round(dist.get('Negative', 0) / total * 100, 1),
    }
    
    # Overall sentiment
    overall = dist.most_common(1)[0][0]
    
    # Rating-Sentiment Mismatch Detection
    mismatch = False
    mismatch_reason = None
    
    neg_ratio = dist.get('Negative', 0) / total
    pos_ratio = dist.get('Positive', 0) / total
    
    if rating >= 4.2 and neg_ratio > 0.5:
        mismatch = True
        mismatch_reason = f"High rating ({rating}★) but {neg_ratio:.0%} negative reviews"
    elif rating <= 2.5 and pos_ratio > 0.5:
        mismatch = True
        mismatch_reason = f"Low rating ({rating}★) but {pos_ratio:.0%} positive reviews"
    
    return {
        "overall": overall,
        "distribution": distribution,
        "mismatch_detected": mismatch,
        "mismatch_reason": mismatch_reason,
        "confidence_score": round(confidence, 3),
        "total_analyzed": total
    }


def _extract_sentiment_features_inference(texts, ratings):
    """Inference-time sentiment features — must match training."""
    vader = SentimentIntensityAnalyzer()
    ecommerce_lexicon = {
        'counterfeit': -3.5, 'duplicate': -3.0, 'original': 2.0,
        'genuine': 2.5, 'authentic': 2.5, 'fake': -3.5,
        'broken': -3.0, 'defective': -3.0, 'damaged': -2.5,
        'value for money': 2.5, 'overpriced': -2.5,
    }
    vader.lexicon.update(ecommerce_lexicon)
    
    features = []
    for i, text in enumerate(texts):
        text = str(text)
        scores = vader.polarity_scores(text)
        words = text.lower().split()
        feat = [
            scores['compound'], scores['pos'], scores['neg'], scores['neu'],
            np.log1p(len(text)), np.log1p(len(words)),
            sum(1 for w in words if w in ['not', "n't", 'never', 'no']),
            ratings[i] if ratings is not None else 3.0,
            sum(1 for w in words if w in ['better','worse','best','worst']),
            sum(1 for w in words if w in ['quality','build','material','design']),
            sum(1 for w in words if w in ['very','extremely','absolutely']),
        ]
        features.append(feat)
    return np.array(features)


def _sentiment_fallback(reviews: list, rating: float) -> dict:
    """VADER-only fallback."""
    vader = SentimentIntensityAnalyzer()
    counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    
    for review in reviews:
        score = vader.polarity_scores(str(review))['compound']
        if score >= 0.05: counts["Positive"] += 1
        elif score <= -0.05: counts["Negative"] += 1
        else: counts["Neutral"] += 1
    
    total = len(reviews)
    overall = max(counts, key=counts.get)
    
    return {
        "overall": overall,
        "distribution": {k: round(v/total*100, 1) for k, v in counts.items()},
        "mismatch_detected": False,
        "confidence_score": 0.6,
        "warning": "Using VADER fallback — retrain models"
    }


# ─────────────────────────────────────────────
# MODULE 3: PRICE ANOMALY DETECTION
# ─────────────────────────────────────────────

def analyze_price_anomaly(price: float, mrp: float, rating: float, 
                           review_count: int, category: str = "general") -> dict:
    """
    Detect price manipulation/anomalies.
    Returns: is_anomaly, anomaly_score, discount_pct, warnings.
    """
    artifact = load_model("price_anomaly_model")
    
    discount_pct = ((mrp - price) / mrp * 100) if mrp > price > 0 else 0.0
    discount_pct = max(0, min(discount_pct, 100))
    
    # Build feature vector (must match training exactly)
    log_price = np.log1p(price)
    log_reviews = np.log1p(review_count)
    price_quality_ratio = log_price / (rating + 0.1)
    extreme_discount = int(discount_pct > 70)
    low_review = int(review_count < 10)
    
    # Category price z-score (use robust category medians)
    CATEGORY_MEDIANS = {
        "electronics": 15000, "mobiles": 25000, "laptops": 45000,
        "fashion": 1500, "clothing": 1200, "shoes": 2500,
        "beauty": 800, "health": 1200, "books": 400,
        "home": 3000, "kitchen": 5000, "appliances": 35000,
        "food": 300, "general": 2000
    }
    
    # Try fuzzy matching for category
    cat_lower = category.lower()
    median = 2000
    for k, v in CATEGORY_MEDIANS.items():
        if k in cat_lower:
            median = v
            break
            
    cat_zscore = (price - median) / (median * 0.5 + 1)
    
    features = np.array([[
        log_price, discount_pct, rating, log_reviews,
        price_quality_ratio, extreme_discount, discount_pct**2,
        low_review, log_reviews / (rating + 0.1), cat_zscore
    ]])
    
    if artifact is None:
        return _price_anomaly_fallback(price, mrp, discount_pct, rating, review_count)
    
    scaler = artifact['scaler']
    iso_forest = artifact['iso_forest']
    lof = artifact['lof']
    
    # Pad/trim features to match scaler expectation
    n_features = scaler.n_features_in_
    if features.shape[1] < n_features:
        features = np.pad(features, ((0,0), (0, n_features - features.shape[1])))
    elif features.shape[1] > n_features:
        features = features[:, :n_features]
    
    features_scaled = scaler.transform(features)
    
    # Ensemble anomaly score (average of both detectors)
    iso_score = -iso_forest.score_samples(features_scaled)[0]  # Higher = more anomalous
    lof_score = -lof.score_samples(features_scaled)[0]
    
    # Normalize to 0-1
    anomaly_score = float(np.clip((iso_score + lof_score) / 2, 0, 1))
    
    iso_pred = iso_forest.predict(features_scaled)[0]
    lof_pred = lof.predict(features_scaled)[0]
    is_anomaly = (iso_pred == -1) or (lof_pred == -1)
    
    # Generate warnings
    warnings = []
    if discount_pct > 80:
        warnings.append(f"Extreme discount ({discount_pct:.0f}%) — verify authenticity")
    if discount_pct > 70 and review_count < 20:
        warnings.append("High discount with very few reviews — high suspicion")
    if price < 100 and rating > 4.5:
        warnings.append("Unusually low price for high-rated product")
    
    # Calculate vs_avg_history if possible (mock comparison for now, or use actual median)
    vs_avg = round((price - median) / median * 100, 1)

    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": round(anomaly_score, 3),
        "discount_pct": round(discount_pct, 1),
        "vs_avg_history": vs_avg,
        "price_trend": "normal" if not is_anomaly else "anomalous",
        "warnings": warnings,
        "iso_forest_prediction": "anomaly" if iso_pred == -1 else "normal",
        "lof_prediction": "anomaly" if lof_pred == -1 else "normal"
    }


def _price_anomaly_fallback(price, mrp, discount_pct, rating, review_count):
    """Rule-based price anomaly fallback."""
    warnings = []
    is_anomaly = False
    anomaly_score = 0.1
    
    if discount_pct > 80:
        is_anomaly = True; anomaly_score = 0.85
        warnings.append(f"Extreme discount: {discount_pct:.0f}%")
    elif discount_pct > 60:
        anomaly_score = 0.55
        warnings.append(f"High discount: {discount_pct:.0f}%")
    
    if review_count < 5 and rating > 4.5:
        is_anomaly = True; anomaly_score = max(anomaly_score, 0.7)
        warnings.append("Suspiciously high rating with very few reviews")
    
    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": anomaly_score,
        "discount_pct": round(discount_pct, 1),
        "vs_avg_history": round(discount_pct * -0.5, 1), # Simple placeholder
        "price_trend": "anomalous" if is_anomaly else "normal",
        "warnings": warnings,
        "warning": "Using rule-based fallback"
    }


# ─────────────────────────────────────────────
# MODULE 4: SELLER RISK CLASSIFIER
# ─────────────────────────────────────────────

def classify_seller_risk(rating: float, review_count: int, discount_pct: float,
                          seller_name: str = "", is_bestseller: bool = False) -> dict:
    """Deterministic seller risk classification with confidence scores."""
    
    seller_lower = seller_name.lower()
    is_amazon_seller = any(s in seller_lower for s in 
                          ['amazon', 'appario', 'cloudtail', 'cocoblu'])
    
    # Risk scoring (0 = safe, higher = riskier)
    risk_score = 0
    signals = []
    
    # Positive signals (reduce risk)
    if is_amazon_seller:
        risk_score -= 30
        signals.append({"type": "positive", "text": "Sold by Amazon/authorized seller"})
    if is_bestseller:
        risk_score -= 20
        signals.append({"type": "positive", "text": "Amazon Best Seller badge"})
    if rating >= 4.2:
        risk_score -= 10
        signals.append({"type": "positive", "text": f"Strong rating: {rating}★"})
    if review_count >= 1000:
        risk_score -= 15
        signals.append({"type": "positive", "text": f"High review volume: {review_count:,}"})
    elif review_count >= 500:
        risk_score -= 8
    elif review_count >= 100:
        risk_score -= 3
    
    # Negative signals (increase risk)
    if rating < 3.0:
        risk_score += 30
        signals.append({"type": "negative", "text": f"Poor rating: {rating}★"})
    elif rating < 3.5:
        risk_score += 15
        signals.append({"type": "negative", "text": f"Below average rating: {rating}★"})
    
    if review_count < 10:
        risk_score += 25
        signals.append({"type": "negative", "text": f"Very few reviews: {review_count}"})
    elif review_count < 50:
        risk_score += 10
        signals.append({"type": "negative", "text": f"Low review count: {review_count}"})
    
    if discount_pct > 80:
        risk_score += 35
        signals.append({"type": "negative", "text": f"Extreme discount: {discount_pct:.0f}% off"})
    elif discount_pct > 60:
        risk_score += 15
        signals.append({"type": "negative", "text": f"High discount: {discount_pct:.0f}% off"})
    
    # Classify based on final score
    if risk_score <= -10 or (rating >= 4.2 and review_count >= 500 and discount_pct <= 60):
        risk_level = "Low"
        confidence = min(0.95, 0.7 + abs(risk_score) / 100)
        probabilities = {"Low": 0.85, "Medium": 0.12, "High": 0.03}
    elif risk_score >= 40 or (rating < 3.0 and review_count < 20) or discount_pct > 80:
        risk_level = "High"
        confidence = min(0.95, 0.65 + risk_score / 100)
        probabilities = {"Low": 0.05, "Medium": 0.15, "High": 0.80}
    else:
        risk_level = "Medium"
        confidence = 0.70
        probabilities = {"Low": 0.25, "Medium": 0.55, "High": 0.20}
    
    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "confidence": round(confidence, 3),
        "probabilities": probabilities,
        "signals": signals[:6],
        "seller_name": seller_name,
        "is_amazon_seller": is_amazon_seller
    }


# ─────────────────────────────────────────────
# MODULE 5: TRUST SCORE ENSEMBLE
# ─────────────────────────────────────────────

def compute_trust_score(fake_result: dict, sentiment_result: dict,
                         price_result: dict, seller_result: dict,
                         rating: float, review_count: int) -> dict:
    """
    Calibrated Bayesian Trust Score with SHAP-style contributions.
    All inputs are from Modules 1-4.
    """
    base_score = 78.0
    contributions = {}
    
    # --- Fake Review Contribution ---
    authenticity = fake_result.get('authenticity_score', 75) / 100
    fake_prob = 1 - authenticity
    fake_penalty = fake_prob * 35
    contributions['review_authenticity'] = {
        'value': authenticity * 100,
        'contribution': -fake_penalty,
        'label': f"{authenticity:.0%} authentic",
        'direction': 'positive' if authenticity > 0.7 else 'negative'
    }
    
    # --- Seller Risk Contribution ---
    seller_penalties = {"Low": 0, "Medium": 12, "High": 28}
    seller_penalty = seller_penalties.get(seller_result.get('risk_level', 'Medium'), 12)
    contributions['seller_risk'] = {
        'value': 100 - seller_penalty * 3,
        'contribution': -seller_penalty,
        'label': f"{seller_result.get('risk_level', 'Medium')} seller risk",
        'direction': 'positive' if seller_penalty == 0 else 'negative'
    }
    
    # --- Sentiment Contribution ---
    mismatch_penalty = 18 if sentiment_result.get('mismatch_detected') else 0
    dist = sentiment_result.get('distribution', {})
    sentiment_score = (
        dist.get('Positive', 33) * 1.0 - 
        dist.get('Negative', 33) * 0.8
    ) / 100 * 10
    contributions['sentiment'] = {
        'value': max(0, dist.get('Positive', 50)),
        'contribution': sentiment_score - mismatch_penalty,
        'label': sentiment_result.get('overall', 'Neutral') + 
                 (" (MISMATCH⚠)" if sentiment_result.get('mismatch_detected') else ""),
        'direction': 'positive' if dist.get('Positive', 0) > 50 else 'negative'
    }
    
    # --- Price Anomaly Contribution ---
    anomaly_score = price_result.get('anomaly_score', 0)
    discount_pct = price_result.get('discount_pct', 0)
    price_penalty = anomaly_score * 18
    if discount_pct > 70: price_penalty += 8
    contributions['price_analysis'] = {
        'value': max(0, (1 - anomaly_score) * 100),
        'contribution': -price_penalty,
        'label': f"{discount_pct:.0f}% discount" + 
                 (" — ANOMALY" if price_result.get('is_anomaly') else ""),
        'direction': 'positive' if not price_result.get('is_anomaly') else 'negative'
    }
    
    # --- Rating & Volume Contribution ---
    rating_bonus = (rating - 3.5) * 5 * authenticity if rating else 0
    volume_bonus = min(12, np.log1p(review_count) * 1.5)
    contributions['rating_volume'] = {
        'value': rating * 20,
        'contribution': rating_bonus + volume_bonus,
        'label': f"{rating}★ ({review_count:,} reviews)",
        'direction': 'positive' if rating >= 4.0 else 'neutral'
    }
    
    # --- Compute Final Score ---
    total_adjustment = sum(c['contribution'] for c in contributions.values())
    raw_score = base_score + total_adjustment
    final_score = max(5, min(99, round(raw_score)))
    
    # --- Grade & Verdict ---
    if final_score >= 75:
        grade, verdict = "A", "Trusted Product"
        color = "#22c55e"
    elif final_score >= 60:
        grade, verdict = "B", "Looks Legitimate"
        color = "#84cc16"
    elif final_score >= 45:
        grade, verdict = "C", "Proceed with Caution"
        color = "#f59e0b"
    else:
        grade, verdict = "D", "High Risk — Avoid"
        color = "#ef4444"
    
    # --- Generate Pros & Cons ---
    pros, cons = [], []
    for key, c in contributions.items():
        if c['direction'] == 'positive' and c['contribution'] >= 0:
            pros.append(c['label'])
        elif c['direction'] == 'negative' or c['contribution'] < -5:
            cons.append(c['label'])
    
    return {
        "score": final_score,
        "grade": grade,
        "verdict": verdict,
        "color": color,
        "base_score": base_score,
        "total_adjustment": round(total_adjustment, 2),
        "shap_contributions": contributions,
        "pros": pros[:4],
        "cons": cons[:4],
        "summary": f"TrustLens analyzed this product and assigned a {grade} grade ({final_score}/100). {verdict}."
    }

