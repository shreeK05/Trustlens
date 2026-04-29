import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

# Import the FastAPI app from backend/main.py
try:
    from main import app
except ImportError:
    # Fallback for different directory structures
    from backend.main import app

if __name__ == "__main__":
    import uvicorn
    # Use the port assigned by Render, defaulting to 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
