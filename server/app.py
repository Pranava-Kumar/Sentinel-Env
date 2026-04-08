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
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from models import SentinelAction, SentinelObservation, SentinelState
from server.sentinel_environment import SentinelEnvironment
from server.rate_limiter import RateLimiter
from server.episode_manager import EpisodeManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────
SENTINEL_API_KEY = os.getenv("SENTINEL_API_KEY")

# ── Rate Limiting ──────────────────────────────────────────────────
rate_limiter = RateLimiter(max_requests=100, window_seconds=60, max_entries=10000)

async def get_client_ip(request: Request) -> str:
    return request.client.host

async def check_rate_limit(request: Request, client_ip: str = Depends(get_client_ip)):
    if not await rate_limiter.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")

# Episode manager for concurrent episode support
episode_manager = EpisodeManager(max_episodes=1000, ttl_seconds=3600)


async def verify_api_key(x_api_key: str = Header(None)):
    """Verify the API key from X-API-Key header."""
    if SENTINEL_API_KEY and x_api_key != SENTINEL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    try:
        episode_id = episode_manager.create_episode(task_name=task_name, seed=seed)
        env = episode_manager.get_episode(episode_id)
        observation = env.reset(task_name=task_name, seed=seed)
        
        response_data = observation.model_dump()
        response_data["episode_id"] = episode_id
        return JSONResponse(content=response_data)
    except Exception as e:
        logger.error(f"reset() failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/step")
async def step(
    request: Request,
    action: SentinelAction,
    episode_id: str = Header(None, alias="X-Episode-ID"),
    api_key: str = Depends(verify_api_key),
    rate_limit: bool = Depends(check_rate_limit),
):
    """Execute one step in the current episode."""
    if not episode_id:
        raise HTTPException(status_code=400, detail="X-Episode-ID header required")
    
    env = episode_manager.get_episode(episode_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    
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
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/state")
async def state(
    episode_id: str = Header(None, alias="X-Episode-ID"),
    api_key: str = Depends(verify_api_key),
):
    """Get current episode state."""
    if not episode_id:
        raise HTTPException(status_code=400, detail="X-Episode-ID header required")
    
    env = episode_manager.get_episode(episode_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    
    try:
        state = env.state()
        return JSONResponse(content=state.model_dump())
    except Exception as e:
        logger.error(f"state() failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return JSONResponse(content={
        "status": "healthy",
        "service": "sentinel-env",
        "version": "1.0.0",
    })


@app.get("/grade")
async def grade(
    episode_id: str = Header(None, alias="X-Episode-ID"),
    api_key: str = Depends(verify_api_key),
):
    """Grade the current episode (helper endpoint)."""
    if not episode_id:
        raise HTTPException(status_code=400, detail="X-Episode-ID header required")
    
    env = episode_manager.get_episode(episode_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    
    try:
        result = env.get_episode_grade()
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"grade() failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/resilience-profile")
async def resilience_profile(
    episode_id: str = Header(None, alias="X-Episode-ID"),
    api_key: str = Depends(verify_api_key),
):
    """Get resilience profile for current episode (helper endpoint)."""
    if not episode_id:
        raise HTTPException(status_code=400, detail="X-Episode-ID header required")
    
    env = episode_manager.get_episode(episode_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    
    try:
        profile = env.get_resilience_profile()
        return JSONResponse(content=profile)
    except Exception as e:
        logger.error(f"resilience_profile() failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


def main():
    """Run the Sentinel Environment server."""
    import uvicorn

    port = int(os.getenv("PORT", 7860))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
