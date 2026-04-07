"""FastAPI server for the Sentinel Environment.

Implements OpenEnv-compatible endpoints:
- POST /reset  — Start a new episode
- POST /step   — Execute one step
- GET  /state  — Get current episode state
- GET  /health — Health check
"""

import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from models import SentinelAction, SentinelObservation, SentinelState
from server.sentinel_environment import SentinelEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global environment instance
env: Optional[SentinelEnvironment] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global env
    env = SentinelEnvironment()
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
    task_name: str = "basic-injection",
    seed: int = 42,
):
    """Start a new episode.

    Args:
        task_name: basic-injection, social-engineering, or stealth-exfiltration
        seed: Random seed for reproducibility
    """
    global env
    try:
        observation = env.reset(task_name=task_name, seed=seed)
        return JSONResponse(content=observation.model_dump())
    except Exception as e:
        logger.error(f"reset() failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(action: SentinelAction):
    """Execute one step in the current episode."""
    global env
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
async def state():
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
async def grade():
    """Grade the current episode (helper endpoint)."""
    global env
    try:
        result = env.get_episode_grade()
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/resilience-profile")
async def resilience_profile():
    """Get resilience profile for current episode (helper endpoint)."""
    global env
    try:
        profile = env.get_resilience_profile()
        return JSONResponse(content=profile)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
