import sys
import os
sys.path.append('d:/TrustLens/backend')
from ml.inference import predict_fake_reviews, analyze_sentiment, detect_price_anomaly, classify_seller_risk, predict_trust_score

print("Testing fake_reviews:")
print(predict_fake_reviews(["This is an amazing product, I absolutely love it!", "Horrible, broke in one day."]))

print("\nTesting sentiment:")
print(analyze_sentiment(["This is an amazing product, I absolutely love it!", "Horrible, broke in one day."], 4.5))

print("\nTesting anomaly:")
print(detect_price_anomaly(100, 150, [{"price": 100}, {"price": 110}]))

print("\nTesting seller risk:")
print(classify_seller_risk({"name": "Amazon", "rating": 4.5, "discount_aggressiveness": 0.1}))

print("\nTesting trust score:")
print(predict_trust_score({
        "fake_review_prob": 0.1,
        "sentiment_mismatch": 0.0,
        "price_anomaly_score": 0.1,
        "seller_risk_encoded": 0,
        "rating": 4.5,
        "log_review_count": 5.0,
        "discount_pct": 10,
        "raw_score": 85.0,
    }))
