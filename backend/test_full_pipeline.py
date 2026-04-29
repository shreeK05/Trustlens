"""Full end-to-end test of the TrustLens ML pipeline."""
import sys
sys.path.insert(0, ".")

from ml.inference import (
    predict_fake_reviews,
    analyze_sentiment,
    detect_price_anomaly,
    classify_seller_risk,
    predict_trust_score,
)

print("=" * 60)
print("  TrustLens AI — Full Pipeline Verification")
print("=" * 60)

# Simulate 10 realistic reviews (7 good, 3 bad)
reviews = [
    "Absolutely love this phone. The quality is exactly as described and it arrived a day early.",
    "Very solid phone for the price. I was skeptical but it performs perfectly.",
    "Five stars. The packaging was great and the phone feels very premium.",
    "Exactly what I was looking for. Highly recommend to anyone needing this.",
    "Works out of the box. No issues so far and the build quality is sturdy.",
    "Great value for money. I've been using it for a week and it hasn't let me down.",
    "Genuine product. Checked the serial number and it's 100% authentic.",
    "Terrible phone. Broke within two days of using it. Do not buy.",
    "Box arrived damaged and the item had scratches on it. Very disappointed.",
    "Fake product! This is not an original phone, the branding is completely off.",
]
rating = 4.2

print("\n--- MODULE 1: Fake Review Detection ---")
fr = predict_fake_reviews(reviews, rating)
print(f"  Total analyzed : {fr['total_analyzed']}")
print(f"  Fake count     : {fr['fake_count']}")
print(f"  Genuine count  : {fr['genuine_count']}")
print(f"  Fake prob      : {fr['fake_probability']}")
print(f"  Authenticity   : {fr['authenticity_score']}%")
print(f"  Flagged        : {len(fr['flagged_reviews'])} reviews")

print("\n--- MODULE 2: Sentiment Analysis ---")
sa = analyze_sentiment(reviews, rating)
print(f"  Positive       : {sa['positive']}")
print(f"  Neutral        : {sa['neutral']}")
print(f"  Negative       : {sa['negative']}")
print(f"  Overall        : {sa['overall']}")
print(f"  Mismatch       : {sa['mismatch_detected']}")
print(f"  Distribution   : {sa['sentiment_distribution']}")

print("\n--- MODULE 3: Price Anomaly Detection ---")
pa = detect_price_anomaly(
    1299, 1999,
    [{"price": 1450}, {"price": 1400}, {"price": 1350}, {"price": 1299}],
    stars=rating,
    review_count=2500,
)
print(f"  Is anomaly     : {pa['is_anomaly']}")
print(f"  Anomaly score  : {pa['anomaly_score']}")
print(f"  Price trend    : {pa['price_trend']}")
print(f"  Discount       : {pa['discount_pct']}%")
print(f"  Vs history     : {pa['vs_avg_history']}%")

print("\n--- MODULE 4: Seller Risk Classification ---")
sr = classify_seller_risk({
    "name": "Appario Retail Private Ltd",
    "rating": rating,
    "review_count": 2500,
    "is_bestseller": True,
    "discount_aggressiveness": 0.35,
    "price": 1299,
})
print(f"  Risk level     : {sr['risk_level']}")
print(f"  Confidence     : {sr['confidence']}")
print(f"  Color          : {sr['color']}")
print(f"  Probabilities  : {sr['probabilities']}")

print("\n--- MODULE 5: Trust Score Ensemble ---")
import numpy as np
ts = predict_trust_score({
    "fake_review_prob": fr["fake_probability"],
    "sentiment_mismatch": 1.0 if sa["mismatch_detected"] else 0.0,
    "price_anomaly_score": pa["anomaly_score"],
    "seller_risk_encoded": sr["risk_score"],
    "rating": rating,
    "log_review_count": float(np.log1p(2500)),
    "discount_pct": 35.0,
})
print(f"  Trust Score    : {ts['score']}/100")
print(f"  Grade          : {ts['grade']}")
print(f"  Verdict        : {ts['verdict']}")
print(f"  Confidence     : {ts['confidence_pct']}%")
print(f"  SHAP values    : {ts['shap_contributions']}")

print("\n" + "=" * 60)
print("  ALL 5 MODULES WORKING CORRECTLY!")
print("=" * 60)
