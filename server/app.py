"""FastAPI server for the Sentinel Environment.

Implements OpenEnv-compatible endpoints:
- POST /reset  — Start a new episode
- POST /step   — Execute one step
- GET  /state  — Get current episode state
- GET  /health — Health check
"""

import logging
import asyncio
import os
import time
from collections import defaultdict
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from models import SentinelAction, SentinelObservation, SentinelState
from server.sentinel_environment import SentinelEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────
SENTINEL_API_KEY = os.getenv("SENTINEL_API_KEY")

# ── Simple Rate Limiting ──────────────────────────────────────────
class RateLimiter:
    """Simple in-memory rate limiter (100 requests/minute per IP)."""
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
    
    async def check_rate_limit(self, client_ip: str) -> bool:
        now = time.time()
        # Clean old requests
        self.requests[client_ip] = [
            t for t in self.requests[client_ip]
            if now - t < self.window_seconds
        ]
        if len(self.requests[client_ip]) >= self.max_requests:
            return False
        self.requests[client_ip].append(now)
        return True

rate_limiter = RateLimiter()

async def get_client_ip(request: Request) -> str:
    return request.client.host

async def check_rate_limit(request: Request, client_ip: str = Depends(get_client_ip)):
    if not await rate_limiter.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")

# Global environment instance and lock for thread safety
env: Optional[SentinelEnvironment] = None
env_lock: asyncio.Lock = asyncio.Lock()


async def verify_api_key(x_api_key: str = Header(None)):
    """Verify the API key from X-API-Key header."""
    if SENTINEL_API_KEY and x_api_key != SENTINEL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global env, env_lock
    env = SentinelEnvironment()
    env_lock = asyncio.Lock()
    logger.info("Sentinel Environment initialized")
    yield
    logger.info("Sentinel Environment shutdown")


app = FastAPI(
    title="Sentinel Environment",
    description="AI Agent Safety & Jailbreak Detection Environment",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/reset")
async def reset(
    request: Request,
    task_name: str = "basic-injection",
    seed: int = 42,
    api_key: str = Depends(verify_api_key),
    rate_limit: bool = Depends(check_rate_limit),
):
    """Start a new episode.

    Args:
        task_name: basic-injection, social-engineering, or stealth-exfiltration
        seed: Random seed for reproducibility
    """
    global env, env_lock
    async with env_lock:
        try:
            observation = env.reset(task_name=task_name, seed=seed)
            return JSONResponse(content=observation.model_dump())
        except Exception as e:
            logger.error(f"reset() failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(
    request: Request,
    action: SentinelAction,
    api_key: str = Depends(verify_api_key),
    rate_limit: bool = Depends(check_rate_limit),
):
    """Execute one step in the current episode."""
    global env, env_lock
    async with env_lock:
        try:
            observation, reward, done, info = env.step(action)
            return JSONResponse(content={
                "observation": observation.model_dump(),
                "reward": reward,
                "done": done,
                "info": info,
            })
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"step() failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def state(api_key: str = Depends(verify_api_key)):
    """Get current episode state."""
    global env
    try:
        state = env.state()
        return JSONResponse(content=state.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return JSONResponse(content={
        "status": "healthy",
        "service": "sentinel-env",
        "version": "1.0.0",
    })


@app.get("/grade")
async def grade(api_key: str = Depends(verify_api_key)):
    """Grade the current episode (helper endpoint)."""
    global env
    try:
        result = env.get_episode_grade()
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/resilience-profile")
async def resilience_profile(api_key: str = Depends(verify_api_key)):
    """Get resilience profile for current episode (helper endpoint)."""
    global env
    try:
        profile = env.get_resilience_profile()
        return JSONResponse(content=profile)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
