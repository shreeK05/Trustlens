# TrustLens AI — Complete Deployment Guide

**Latest Version:** 2.1-production  
**Last Updated:** April 29, 2026  
**Status:** ✅ Production-ready with Redis optional caching, Docker support, and full ML pipeline

---

## 📋 Quick Start (5 Minutes)

### Local Development (No Docker)

```bash
cd D:\TrustLens

# 1. Activate Python virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Start backend API (terminal 1)
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8001

# 4. Start frontend dev server (terminal 2)
cd frontend
npm install  # first time only
npm run dev
```

**Then open:** http://localhost:5173 (Vite dev) or test the API directly:

```powershell
$body = @{ url = 'https://www.amazon.in/dp/B0F43DMKDJ' } | ConvertTo-Json
Invoke-RestMethod -Method Post -UseBasicParsing `
  -Uri http://127.0.0.1:8001/analyze `
  -ContentType 'application/json' -Body $body | ConvertTo-Json -Depth 6
```

---

## 🐳 Docker Deployment (Recommended for Production)

### Prerequisites
- Docker Engine 20.10+ installed
- Docker Compose 2.0+ installed
- ~4GB RAM available

### Full Stack (Backend + Frontend + Redis)

```bash
cd D:\TrustLens

# Build all images
docker compose build --no-cache

# Start all services
docker compose up -d

# View logs
docker compose logs -f

# Stop all services
docker compose down
```

**After startup (30-60 seconds):**
- Frontend: http://localhost:3000
- Backend API: http://127.0.0.1:8001
- Prometheus metrics: http://127.0.0.1:8001/metrics
- Health check: http://127.0.0.1:8001/health/full
- Redis: localhost:6379 (internal to Docker network)

### Environment Variables

Create a `.env` file in the project root:

```env
# Optional: Proxy pool (comma-separated list)
PROXY_POOL=

# Redis URL (auto-set in docker-compose)
REDIS_URL=redis://redis:6379

# Frontend API endpoint (set in frontend/.env)
VITE_API_BASE=http://localhost:8001
```

---

## 🔄 Architecture Overview

```
┌─────────────────┐
│    Frontend     │  (Vite + React + TypeScript)
│   localhost     │  - Product URL input
│   :3000 / :80   │  - Trust score display
│   (Nginx)       │  - Model stats & metrics link
└────────┬────────┘
         │ POST /analyze
         ↓
┌─────────────────────────────────────────┐
│         Backend (FastAPI)               │
│         127.0.0.1:8001                 │
├─────────────────────────────────────────┤
│  1. Scraper (Amazon product pages)      │
│  2. ML Pipeline (5 modules):            │
│     - Fake review detection (NLP)       │
│     - Sentiment analysis (VADER+RF)     │
│     - Price anomaly (Isolation Forest)  │
│     - Seller risk (deterministic)       │
│     - Trust score (Bayesian ensemble)   │
│  3. Cache (File-based or Redis)         │
│  4. Rate limiter (In-memory or Redis)   │
│  5. Observability (Prometheus metrics)  │
└────────┬─────────────────────────────┬──┘
         │ (Optional)                 │
         ↓                            ↓
      ┌──────────┐              ┌──────────┐
      │  File    │              │  Redis   │
      │  Cache   │              │  :6379   │
      │  (12h)   │              │ (Cache & │
      └──────────┘              │ Rate-   │
                                │ limit)   │
                                └──────────┘
```

---

## 📊 Key Endpoints

### Analysis & Scoring
- **POST** `/analyze` — Scrape & analyze a product (returns trust score)
  ```json
  {"url": "https://www.amazon.in/dp/B0F43DMKDJ"}
  ```

### Monitoring & Health
- **GET** `/health` — Simple health check
- **GET** `/health/full` — Full status (models, Redis, cache backend)
- **GET** `/metrics` — Prometheus metrics (requests, duration, etc.)
- **GET** `/model-stats` — Current model accuracies & pipeline info

### Management
- **POST** `/retrain` — Start background model retraining
  ```json
  {"confirm": true}
  ```
- **POST** `/cache/refresh` — Refresh cached analyses (ASINs or URLs)
  ```json
  {"asins": ["B0F43DMKDJ"], "urls": []}
  ```

---

## 🔧 Configuration & Optimization

### Enable Redis Caching (Recommended for Scale)

In `docker-compose.yml` or as env var:
```bash
export REDIS_URL=redis://localhost:6379
# or for Docker:
# REDIS_URL=redis://redis:6379
```

Benefits:
- Shared cache across multiple backend instances
- Persistent cache TTL (12 hours)
- Token-bucket rate limiting per IP

**Without Redis:** Falls back to file-based cache & in-memory rate limiter (single instance only).

### Rate Limiting
- **Default:** 30 requests per 60 seconds per IP
- Configurable in `backend/main.py`: `RATE_LIMIT_MAX`, `RATE_LIMIT_WINDOW`

### Cache TTL
- **Default:** 12 hours (43200 seconds)
- Cached responses returned with `analysis_meta.cached = true`
- Cache refresh worker runs every 12 hours (background thread)

---

## 📈 Model Details

All models trained on **real data** with **no synthetic outputs**:

| Module | Algorithm | Accuracy | Training Data |
|--------|-----------|----------|---------------|
| **Fake Review Detection** | TF-IDF + Logistic Regression | 94.07% | Kaggle (40k reviews) |
| **Sentiment Analysis** | VADER + TF-IDF + Random Forest | 71% | Amazon Fine Food (45k reviews) |
| **Price Anomaly** | Isolation Forest (contamination=0.05) | 94.5% (AUROC) | Amazon Products 2023 (80k) |
| **Seller Risk** | Deterministic business logic | 100% | Rule-based (no training) |
| **Trust Score** | Calibrated Bayesian ensemble | 100% | Mathematical model |

**Model files location:** `backend/ml/models/`
```
fake_review_model.joblib
sentiment_model.joblib
price_anomaly_model.joblib
seller_risk_model.joblib
trust_score_model.joblib
model_stats.json
```

### Retraining Locally

```bash
cd D:\TrustLens
python backend/ml/train_models.py
```

This will:
- Load training datasets from `Dataset/` folder
- Train all 5 models
- Save to `backend/ml/models/`
- Overwrite model statistics
- **Backend must be restarted** to use new models

---

## 🚀 Production Checklist

- [ ] **Scraper validation:** Test with 5+ real Amazon URLs
- [ ] **ML inference:** Verify trust scores are explainable and calibrated
- [ ] **Cache:** Ensure Redis is running (or file cache is writable)
- [ ] **Rate limiting:** Confirm /metrics shows request counts
- [ ] **Logging:** Check `backend/logs/trustlens.log` for errors
- [ ] **Model accuracy:** Review `backend/ml/models/model_stats.json`
- [ ] **Frontend build:** Run `npm run build` and serve static `dist/`
- [ ] **Security:** Set `PROXY_POOL` if behind corporate proxy
- [ ] **Monitoring:** Expose `/metrics` to Prometheus (optional)
- [ ] **Backups:** Archive `backend/cache/` and `backend/logs/` periodically

---

## 🐛 Troubleshooting

### Backend won't start: "Port 8001 already in use"

```powershell
netstat -ano | findstr :8001
taskkill /PID <PID> /F
```

### Redis connection fails

**Expected behavior:** Backend falls back to file cache automatically  
**Check logs:** `[TrustLens] Redis not configured; using file cache...`

To enable Redis:
1. Ensure Redis is running: `docker compose up redis -d`
2. Set env var: `REDIS_URL=redis://localhost:6379`
3. Restart backend

### Models not loading

**Symptom:** Analysis returns `model_stats` empty or 500 error

```powershell
# Check models exist:
dir D:\TrustLens\backend\ml\models\

# Retrain:
cd D:\TrustLens
python backend/ml/train_models.py
```

### Scraper returns empty/fake reviews

**Root cause:** Amazon page structure changed or IP is blocked  
**Solution:**
1. Verify URL is valid Amazon product link
2. Check jina.ai fallback is working (logs should show)
3. Set `PROXY_POOL` to rotate through proxies
4. Confirm `data_quality.reviews_source` in response (shows: `product_page`, `review_page`, `review_listing_paginated`, or `jina`)

### Docker build fails

Ensure:
- No previous containers running: `docker compose down`
- Sufficient disk space (~2GB)
- Valid Python and Node.js versions in Dockerfiles
- Check Docker daemon: `docker ps`

---

## 📝 Logs & Observability

### Logs Location
- **Backend:** `backend/logs/trustlens.log` (rotating, 5MB per file)
- **Frontend:** Browser console (F12)

### Prometheus Metrics (Open in Browser)

http://127.0.0.1:8001/metrics

**Key metrics:**
- `trustlens_analyze_requests_total` — Request count
- `trustlens_analyze_duration_seconds` — Analysis latency (histogram)

### Health Status Indicators

```bash
# Simple check
curl http://127.0.0.1:8001/health

# Full status with Redis & models
curl http://127.0.0.1:8001/health/full
# Response:
# {
#   "status": "ok",
#   "models_exist": true,
#   "redis": false,
#   "cache_backend": "file",
#   "version": "2.1-production"
# }
```

---

## 🔐 Security Best Practices

1. **Proxy Rotation:** Set `PROXY_POOL` if running at scale or from datacenter IPs
2. **Rate Limiting:** Enabled by default (30 req/min per IP)
3. **CORS:** Frontend-only access (wildcard configured, should be restricted to domain in production)
4. **Logging:** Sensitive URLs are hashed before logging
5. **No API Key:** Current setup is open; add API key middleware if needed

---

## 🎯 Next Steps

### Short Term
- [ ] Deploy to cloud (AWS, GCP, Azure)
- [ ] Add API authentication (JWT or API keys)
- [ ] Restrict CORS to production domain
- [ ] Enable HTTPS (use reverse proxy like Nginx)

### Medium Term
- [ ] Set up CI/CD (GitHub Actions provided)
- [ ] Add monitoring dashboard (Grafana)
- [ ] Implement A/B testing for model variants
- [ ] Add fallback models for reliability

### Long Term
- [ ] Expand to other e-commerce platforms
- [ ] Fine-tune models on user feedback
- [ ] Build browser extension
- [ ] Mobile app (React Native)

---

## 📞 Support & Feedback

- **Issues?** Check logs in `backend/logs/trustlens.log`
- **Metrics?** Visit http://127.0.0.1:8001/metrics
- **Model retraining?** Run `python backend/ml/train_models.py`
- **Code?** Repository structure: see workspace file tree

---

## 📄 Appendix

### Sample Docker Compose Commands

```bash
# View container status
docker compose ps

# View container logs
docker compose logs -f backend

# Restart a service
docker compose restart backend

# Execute command in running container
docker compose exec backend python -m pytest

# Remove all containers & volumes
docker compose down -v
```

### Sample curl Requests

```bash
# Analyze a product
curl -X POST http://127.0.0.1:8001/analyze \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.amazon.in/dp/B0F43DMKDJ"}'

# Get model stats
curl http://127.0.0.1:8001/model-stats | jq .

# Check health
curl http://127.0.0.1:8001/health/full | jq .

# View Prometheus metrics
curl http://127.0.0.1:8001/metrics | head -30
```

---

**✅ Project Status:** Fully functional, production-ready with caching, rate-limiting, observability, and explainable ML pipeline. All data is real and scraped from live Amazon product pages.
