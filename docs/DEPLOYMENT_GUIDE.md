# Deployment Guide

Complete guide for deploying, monitoring, and troubleshooting the Sentinel Environment.

## Table of Contents

- [Deployment Options](#deployment-options)
- [Docker Deployment](#docker-deployment)
- [Hugging Face Space Deployment](#hugging-face-space-deployment)
- [Local Development](#local-development)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Backup and Recovery](#backup-and-recovery)

---

## Deployment Options

The Sentinel Environment supports multiple deployment targets:

| Deployment | Use Case | Complexity | Cost |
|-----------|----------|------------|------|
| **Docker Container** | Production, CI/CD | Medium | Infrastructure cost |
| **Hugging Face Space** | Demo, hackathons, testing | Low | Free (GPU optional) |
| **Local Development** | Development, testing | Low | Free |

### Deployment Comparison

| Feature | Docker | HF Space | Local |
|---------|--------|----------|-------|
| Scalability | ✅ Horizontal scaling | ❌ Single instance | ❌ Single instance |
| GPU Support | ✅ Full support | ⚠️ Limited (paid tier) | ✅ Local GPU |
| Persistent Storage | ✅ Volumes | ❌ Ephemeral | ✅ Local disk |
| Custom Domains | ✅ | ❌ | ❌ |
| Monitoring | ✅ Full stack | ⚠️ Basic health | ❌ Manual |
| Cost | Variable | Free | Free |

---

## Docker Deployment

### Building the Image

```bash
# Build production image
docker build -t sentinel-env:latest .

# Build with custom tag
docker build -t sentinel-env:1.1.0 --target production .
```

**Build Process:**
```
Stage 1: Builder (python:3.11-slim)
  └─ Install build-essential
  └─ Install dependencies via uv
  └─ Output: /install (Python packages)

Stage 2: Production (python:3.11-slim)
  └─ Copy /install from builder
  └─ Copy application code
  └─ Create non-root user (appuser)
  └─ Set ownership
  └─ Configure health check
```

### Running the Container

```bash
# Basic run
docker run -p 7860:7860 sentinel-env:latest

# With environment variables
docker run -p 7860:7860 \
  -e SENTINEL_API_KEY=your-secret-key \
  -e SENTRY_DSN=https://xxx@xxx.ingest.sentry.io/xxx \
  sentinel-env:latest

# With resource limits
docker run -p 7860:7860 \
  --memory=2g \
  --cpus=2 \
  sentinel-env:latest

# Detached mode with restart policy
docker run -d \
  --name sentinel \
  -p 7860:7860 \
  --restart unless-stopped \
  sentinel-env:latest
```

### Docker Compose

```yaml
# docker-compose.yml
version: "3.8"

services:
  sentinel:
    build: .
    ports:
      - "7860:7860"
    environment:
      - SENTINEL_API_KEY=${SENTINEL_API_KEY}
      - SENTRY_DSN=${SENTRY_DSN}
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7860/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: "2"

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    depends_on:
      - sentinel

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
    depends_on:
      - prometheus

volumes:
  grafana-storage:
```

**Prometheus Configuration:**
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "sentinel"
    static_configs:
      - targets: ["sentinel:7860"]
```

```bash
# Start full stack
docker-compose up -d

# View logs
docker-compose logs -f sentinel

# Stop
docker-compose down
```

### Health Check

The Dockerfile includes a built-in health check:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1
```

**Check container health:**
```bash
docker inspect --format='{{.State.Health.Status}}' sentinel

# Expected output: healthy (after 60s start period)
```

### Updating the Container

```bash
# Pull latest code
git pull origin main

# Rebuild image
docker build -t sentinel-env:latest .

# Stop and remove old container
docker stop sentinel
docker rm sentinel

# Start new container
docker run -d --name sentinel -p 7860:7860 --restart unless-stopped sentinel-env:latest

# Clean up old images
docker image prune -f
```

---

## Hugging Face Space Deployment

### Prerequisites

- Hugging Face account with Space creation permissions
- `huggingface_hub` Python package installed
- Write access to target Space repository

### Quick Deploy

```bash
# Install HF CLI
pip install huggingface_hub

# Login to Hugging Face
huggingface-cli login

# Deploy to Space
python deploy-hf.py
```

### What Gets Deployed

**Included Files:**
- `server/` (all server code)
- `client.py`
- `models.py`
- `inference.py`
- `inference_logging.py`
- `pyproject.toml`
- `Dockerfile`
- `openenv.yaml`
- `.dockerignore`
- `.pre-commit-config.yaml`

**Excluded Files:**
- `jailbreak-prompts/` (proprietary IP)
- `model_checkpoints_hyperion/` (large files)
- `docs/` (not needed in production)
- `.agents/`, `.qwen/` (IDE files)
- `codebase_analysis.md`, `code_review.md` (internal docs)
- `QUICK_REFERENCE.md`, `IMPLEMENTATION_COMPLETE.md` (internal docs)
- `scripts/`, `tools/` (internal utilities)
- `*.log`, `*.out`, `*.err` (runtime files)
- `uv.lock` (regenerated by HF)

### Deployment Process

```python
# deploy-hf.py workflow
from huggingface_hub import HfApi

api = HfApi()

# Upload folder with exclusions
api.upload_folder(
    folder_path=".",
    repo_id="PranavaKumar09/sentinel-env",
    repo_type="space",
    ignore_patterns=[
        ".git",
        "jailbreak-prompts",
        "model_checkpoints_hyperion",
        "docs/",
        "*.log",
        # ... more patterns
    ],
    commit_message="fix: resolve all code review issues",
)
```

### Space Configuration

**SDK:** Docker  
**Port:** 7860  
**Hardware:** CPU (free tier) or GPU (paid tier)

**Environment Variables (set in Space Settings):**
```
SENTINEL_API_KEY=your-secret-key
SENTRY_DSN=https://xxx@xxx.ingest.sentry.io/xxx
```

### Fixing Space Issues

```bash
# Fix README and clean up unwanted files
python fix-hf-space.py
```

**What this does:**
- Updates README.md with proper HF Space metadata
- Deletes internal documentation files
- Removes graph visualization artifacts
- Cleans up submission-related files

### Verifying Deployment

```bash
# Check Space health
curl https://pranavakumar09-sentinel-env.hf.space/health

# Expected response
{"status":"healthy","service":"sentinel-env","version":"1.1.0",...}

# Test endpoint
curl -X POST "https://pranavakumar09-sentinel-env.hf.space/reset?task_name=basic-injection" \
  -H "X-API-Key: your-api-key"
```

---

## Local Development

### Quick Start

```bash
# Clone repository
git clone <repository-url>
cd "E:\OpenENV RL Challenge"

# Create virtual environment
uv venv
.venv\Scripts\activate  # Windows

# Install dependencies
uv pip install -e ".[dev]"

# Run server
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# Verify
curl http://localhost:7860/health
```

### Running with Environment Variables

```bash
# Windows (PowerShell)
$env:SENTINEL_API_KEY="dev-key"
$env:SENTRY_DSN=""
uvicorn server.app:app --reload

# Linux/Mac
export SENTINEL_API_KEY="dev-key"
export SENTRY_DSN=""
uvicorn server.app:app --reload
```

### Testing Locally

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=server --cov-report=html

# Open coverage report
start htmlcov/index.html  # Windows
open htmlcov/index.html   # Mac
```

---

## Configuration

### Environment Variables

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `SENTINEL_API_KEY` | `None` | No (dev) | API key for endpoint authentication |
| `SENTRY_DSN` | `None` | No | Sentry error tracking DSN |
| `PORT` | `7860` | No | Server port |
| `HOST` | `0.0.0.0` | No | Server bind address |
| `ENVIRONMENT` | `production` | No | Environment name (production, staging, dev) |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | No | LLM API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | No | LLM model name |
| `HF_TOKEN` | `""` | No | Hugging Face API token |
| `BASE_URL` | `http://localhost:7860` | No | Local server URL (for inference.py) |

### Configuration Files

**pyproject.toml:**
```toml
[project]
name = "sentinel-env"
version = "1.0.0"
requires-python = ">=3.11"
```

**openenv.yaml:**
```yaml
name: sentinel-env
sdk: docker
port: 7860

metadata:
  tasks:
    - name: basic-injection
      difficulty: easy
    - name: social-engineering
      difficulty: medium
    - name: stealth-exfiltration
      difficulty: hard
```

**.dockerignore:**
```
tests/
scripts/
docs/
*.md
!.dockerignore
!.pre-commit-config.yaml
.git
.venv
```

---

## Monitoring

### Health Check Endpoint

```bash
curl http://localhost:7860/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "sentinel-env",
  "version": "1.1.0",
  "features": {
    "structured_logging": true,
    "prometheus_metrics": true,
    "sentry_tracking": true,
    "rate_limiting": true,
    "concurrent_episodes": true,
    "jailbreak_prompts": true,
    "wandb_tracking": true
  },
  "episode_manager": {
    "max_episodes": 1000,
    "ttl_seconds": 3600
  },
  "rate_limiter": {
    "max_requests": 100,
    "window_seconds": 60
  }
}
```

### Prometheus Metrics

```bash
# Scrape metrics
curl http://localhost:7860/metrics

# Query specific metrics (Prometheus)
sentinel_requests_total{status="200"}
sentinel_detection_rate
histogram_quantile(0.95, rate(sentinel_request_duration_seconds_bucket[5m]))
```

### Structured Logging

Logs are output in JSON format:

```json
{
  "event": "Request completed",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "method": "POST",
  "path": "/reset",
  "status_code": 200,
  "duration_ms": 45.23,
  "timestamp": "2026-04-12T10:30:45.123456Z"
}
```

**Viewing Logs:**
```bash
# Docker
docker logs sentinel --follow

# Local
# Logs output to stdout
```

### Sentry Error Tracking

If `SENTRY_DSN` is configured, errors are automatically sent to Sentry:

- Stack traces with context
- Request ID for correlation
- Environment and version tags
- User feedback collection

---

## Troubleshooting

### Common Issues

#### Server Won't Start

| Symptom | Possible Cause | Solution |
|---------|---------------|----------|
| `Address already in use` | Port 7860 occupied | `lsof -i :7860` (Linux/Mac) or `netstat -ano | findstr :7860` (Windows) |
| `ModuleNotFoundError` | Missing dependencies | `uv pip install -e ".[dev]"` |
| `Permission denied` | Port < 1024 without sudo | Use port > 1024 or run as root |
| Build fails (Docker) | Missing files | Check `.dockerignore` isn't excluding needed files |

**Example: Port Already in Use**
```bash
# Find process using port 7860
# Linux/Mac:
lsof -i :7860
kill -9 <PID>

# Windows:
netstat -ano | findstr :7860
taskkill /PID <PID> /F

# Or use different port:
uvicorn server.app:app --port 8080
```

#### Health Check Failing

```bash
# Check if server is running
curl http://localhost:7860/health

# If connection refused, server isn't running
# Start server:
uvicorn server.app:app --host 0.0.0.0 --port 7860

# If 500 error, check logs:
docker logs sentinel
```

**Health Check Timeout (Docker):**
```bash
# Increase start period
docker run --health-cmd="curl -f http://localhost:7860/health" \
  --health-start-period=120s \
  sentinel-env:latest
```

#### Rate Limiting Issues

| Symptom | Cause | Solution |
|---------|-------|----------|
| 429 Too Many Requests | Exceeded 100 req/min | Wait and retry with backoff |
| Missing rate limit headers | Old client version | Update client to receive headers |

**Retry with Exponential Backoff:**
```python
import asyncio
from httpx import AsyncClient, HTTPStatusError

async def request_with_retry(client, url, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await client.post(url)
        except HTTPStatusError as e:
            if e.response.status_code == 429:
                wait = 2 ** attempt
                await asyncio.sleep(wait)
            else:
                raise
    raise Exception("Max retries exceeded")
```

#### Episode Not Found (404)

| Cause | Solution |
|-------|----------|
| Episode expired (1hr TTL) | Start new episode with `/reset` |
| Invalid episode ID | Check `X-Episode-ID` header value |
| Episode cleanup ran | Episodes > TTL are removed automatically |

**Check Episode Status:**
```bash
curl http://localhost:7860/state \
  -H "X-Episode-ID: ep_abc123"

# If 404, episode doesn't exist or expired
```

#### High Memory Usage

```bash
# Check container memory
docker stats sentinel

# Check for memory leaks (Python)
import sys
sys.getsizeof(object)  # Check object sizes

# Restart container (if needed)
docker restart sentinel
```

**Memory Optimization:**
```python
# Reduce max episodes
episode_manager = EpisodeManager(max_episodes=500, ttl_seconds=1800)  # From 1000, 3600s

# Enable garbage collection
import gc
gc.collect()  # Force garbage collection
```

### HF Space Issues

#### Space Stuck in "Building" State

**Causes:**
- Dockerfile syntax error
- Missing dependencies
- Build timeout (15 min limit)

**Solution:**
```bash
# Check build logs
# Go to Space page → "Logs" tab

# Test Dockerfile locally
docker build -t test-build .

# Fix issues and redeploy
python deploy-hf.py
```

#### Space Shows "Error"

**Check Runtime Logs:**
1. Go to Space page
2. Click "Logs" tab
3. Look for stack traces

**Common Errors:**

| Error | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError` | Missing package | Add to `pyproject.toml` dependencies |
| `OSError: [Errno 98]` | Port already in use | Check Dockerfile CMD uses correct port (7860) |
| `Permission denied` | File ownership | Ensure `appuser` owns `/app` directory |

#### Space Slow to Respond

**Causes:**
- CPU-only instance (HF free tier)
- Cold start (Space slept)
- High concurrent load

**Solutions:**
1. **Upgrade to GPU:** Space Settings → Hardware → GPU
2. **Keep Space awake:** Ping health endpoint every 5 min
3. **Optimize startup:** Reduce dependencies, use multi-stage Docker build

```bash
# Keep-alive script (run every 5 min via cron)
curl -f https://pranavakumar09-sentinel-env.hf.space/health
```

### Training Issues

See [HyperionRL Training Guide](HYPERIONRL_TRAINING_GUIDE.md#troubleshooting) for training-specific issues.

---

## Backup and Recovery

### Backing Up Data

**What to Back Up:**
- Model checkpoints (`model_checkpoints_hyperion/`)
- Trackio database (`~/.cache/huggingface/trackio/`)
- Configuration files (`.env`, custom configs)
- Logs (if stored persistently)

**Backup Script:**
```bash
#!/bin/bash
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup checkpoints
cp -r model_checkpoints_hyperion "$BACKUP_DIR/"

# Backup Trackio DB
cp ~/.cache/huggingface/trackio/*.db "$BACKUP_DIR/"

# Backup config
cp pyproject.toml openenv.yaml "$BACKUP_DIR/"

# Compress
tar -czf "$BACKUP_DIR.tar.gz" "$BACKUP_DIR"
rm -rf "$BACKUP_DIR"

echo "Backup saved to $BACKUP_DIR.tar.gz"
```

### Restoring from Backup

```bash
# Extract backup
tar -xzf backups/20260412_103000.tar.gz

# Restore checkpoints
cp -r backups/20260412_103000/model_checkpoints_hyperion ./

# Resume training
python train_hyperion.py --resume
```

### Disaster Recovery

**Complete Redeploy:**
```bash
# 1. Clone fresh repository
git clone <repository-url>
cd "E:\OpenENV RL Challenge"

# 2. Restore backups
tar -xzf latest-backup.tar.gz
cp -r backups/*/model_checkpoints_hyperion ./

# 3. Deploy
python deploy-hf.py

# 4. Verify
curl https://pranavakumar09-sentinel-env.hf.space/health
```

---

## Performance Tuning

### Docker Performance

```bash
# Allocate more resources
docker run --memory=4g --cpus=4 sentinel-env:latest

# Use host networking (Linux only, reduces latency)
docker run --network=host sentinel-env:latest
```

### Uvicorn Tuning

```bash
# Production with multiple workers
uvicorn server.app:app \
  --host 0.0.0.0 \
  --port 7860 \
  --workers 4 \
  --loop uvloop \
  --http httptools \
  --log-level info
```

**Worker Calculation:** `(2 × CPU cores) + 1`
- 2 cores → 5 workers
- 4 cores → 9 workers
- 8 cores → 17 workers

**Note:** Multiple workers share episode state via in-memory dict (not distributed). For distributed episodes, use external state store (Redis).

### Episode Tuning

```python
# Adjust episode limits
episode_manager = EpisodeManager(
    max_episodes=2000,      # Increase from 1000
    ttl_seconds=7200        # Increase from 3600 (2 hours)
)
```

**Trade-offs:**
- More episodes → More memory usage
- Longer TTL → Episodes stay alive longer (good for slow clients, bad for cleanup)

---

## Security Checklist

- [x] Non-root user in Docker
- [x] API key authentication enabled
- [x] Rate limiting active (100 req/min)
- [x] Request size limits (1MB)
- [x] Sentry error tracking (optional)
- [x] Structured logging with request IDs
- [x] Jailbreak prompts excluded from deployment
- [ ] Regular dependency updates (`uv pip install --upgrade`)
- [ ] Periodic security scans (`bandit -r server/`)
- [ ] HTTPS termination (reverse proxy)

---

## Support

### Getting Help

- **Documentation:** See `/docs` directory
- **Issues:** GitHub Issues for bugs
- **Discussions:** GitHub Discussions for questions
- **Emergency:** Contact maintainers directly

### Reporting Issues

Include:
1. Deployment type (Docker, HF Space, local)
2. Version (`/health` endpoint shows version)
3. Steps to reproduce
4. Expected vs actual behavior
5. Logs/stack traces
6. Environment variables (redact secrets)

---

**Last Updated:** April 12, 2026  
**Version:** 1.1.0  
**Maintainer:** DevOps Team
