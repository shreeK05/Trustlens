"""Retrain ALL models with current scikit-learn version to fix compatibility."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from ml.train_models import train_all_models

if __name__ == "__main__":
    print("Retraining all 5 models with current scikit-learn...")
    print(f"sklearn version: {__import__('sklearn').__version__}")
    print()
    results = train_all_models()
    print("\n\nAll done! Results:")
    for k, v in results.items():
        print(f"  {k}: {v}")
    
    # Verify
    models_dir = os.path.join(os.path.dirname(__file__), "ml", "models")
    print(f"\nModels in {models_dir}:")
    for f in sorted(os.listdir(models_dir)):
        size = os.path.getsize(os.path.join(models_dir, f))
        print(f"  {f} ({size/1024:.0f} KB)")
