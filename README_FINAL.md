TrustLens — Finalized Local Runbook

Quick start (local dev)

1. Create and activate a Python venv (Windows PowerShell):

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
```

2. Run backend (uvicorn):

```powershell
Set-Location backend
python -m uvicorn main:app --host 127.0.0.1 --port 8001
```

3. Run frontend:

```powershell
Set-Location frontend
npm install
npm run dev
```

Training models (long-running)

- To retrain models locally (may take hours and CPU):

```powershell
Set-Location backend
python -u ml/train_models.py
```

- CI: A GitHub Actions workflow `backend/ci_train_and_publish.yml` is included to run training and upload artifacts.

Caching & Refresh

- Analysis results are cached to `backend/cache/` for 12 hours.
- Use `POST /cache/refresh` to refresh specific ASINs or URLs, or to trigger a background refresh of all cached entries.

Proxy & Scraper

- To use a proxy pool for scraping, set the `PROXY_POOL` environment variable to a comma-separated list of proxy URLs (http://user:pass@host:port,...)

Metrics & Logging

- Prometheus metrics exposed at `GET /metrics`.
- Persistent logs are in `backend/logs/trustlens.log` (rotating).

Notes

- The project now includes: retrying requests with UA rotation, file-based cache, cache refresher, Prometheus metrics, rotating file logs, and a CI training workflow.
- Recommended next steps for production: use Redis for rate-limiting and cache, deploy models behind a model server, and use a managed proxy or scraping service.
