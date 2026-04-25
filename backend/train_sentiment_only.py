"""
Quick script to train ONLY the missing sentiment model.
Run from: d:\TrustLens\backend\
Command:  python train_sentiment_only.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from ml.train_models import train_sentiment_model

if __name__ == "__main__":
    print("Training sentiment model...")
    result = train_sentiment_model()
    print(f"\nDone! Result: {result}")
    
    # Verify file was created
    model_path = os.path.join(os.path.dirname(__file__), "ml", "models", "sentiment_model.joblib")
    if os.path.exists(model_path):
        size_kb = os.path.getsize(model_path) / 1024
        print(f"Model saved: {model_path} ({size_kb:.0f} KB)")
    else:
        print(f"ERROR: Model not found at {model_path}")
