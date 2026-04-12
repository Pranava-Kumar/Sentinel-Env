"""FastAPI server for the Sentinel Environment.

Implements OpenEnv-compatible endpoints:
- POST /reset  — Start a new episode
- POST /step   — Execute one step
- GET  /state  — Get current episode state
- GET  /health — Health check
- GET  /grade  — Grade current episode
- GET  /resilience-profile — Get resilience profile
- GET  /metrics — Prometheus metrics

Production features:
- Structured logging with structlog
- Request/response audit logging
- Prometheus metrics
- Sentry error tracking
- Request body size limits
"""

import hmac
import os
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from models import SentinelAction
from server.dependencies import episode_manager, rate_limiter
from server.middleware import setup_production_middleware

# ── Response Models ──────────────────────────────────────────────────


class ResetResponse(BaseModel):
    """Response model for POST /reset."""

    episode_id: str
    user_prompt: str
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)
    attack_metadata: dict[str, Any]
    resilience_metrics: dict[str, Any]
    step_number: int
    max_steps: int
    is_safe_prompt: bool


class StepResponse(BaseModel):
    """Response model for POST /step."""

    observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]


class StateResponse(BaseModel):
    """Response model for GET /state."""

    episode_id: str
    task_name: str
    step_count: int
    total_attacks_presented: int
    attacks_correctly_detected: int
    false_positives: int
    current_resilience_score: float
    done: bool


class GradeResponse(BaseModel):
    """Response model for GET /grade."""

    score: float
    detection_rate: float
    false_positive_rate: float
    correct_detections: int = 0
    missed_attacks: int = 0
    false_positives: int = 0
    total_attacks: int = 0
    total_safe: int = 0
    total_steps: int = 0
    avg_reasoning_score: float = 0.0


# Configure structlog for JSON output
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()

# ── Configuration ──────────────────────────────────────────────────
SENTINEL_API_KEY = os.getenv("SENTINEL_API_KEY")

# Initialize Sentry if DSN is provided
SENTRY_DSN = os.getenv("SENTRY_DSN")
if SENTRY_DSN:
    try:
        import sentry_sdk

        sentry_sdk.init(
            dsn=SENTRY_DSN,
            traces_sample_rate=0.1,
            environment=os.getenv("ENVIRONMENT", "production"),
        )
        logger.info("Sentry initialized")
    except ImportError:
        logger.warning("sentry-sdk not installed, error tracking disabled")


async def get_client_ip(request: Request) -> str:
    if request.client is None:
        return "unknown"
    return request.client.host


async def check_rate_limit(request: Request, client_ip: str = Depends(get_client_ip)) -> bool:
    """Check rate limit and store remaining count in request state."""
    allowed, remaining = await rate_limiter.check_rate_limit(client_ip)
    request.state.rate_limit_remaining = remaining
    request.state.rate_limit_limit = rate_limiter.max_requests
    if not allowed:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
    return True


async def verify_api_key(x_api_key: str = Header(None)):
    """Verify the API key from X-API-Key header."""
    if SENTINEL_API_KEY and not hmac.compare_digest(x_api_key, SENTINEL_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _sanitize_exception_message(exc: Exception) -> str:
    """Truncate exception messages to prevent log flooding."""
    msg = str(exc)
    if len(msg) > 200:
        return msg[:200] + "...[truncated]"
    return msg


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Sentinel Environment initialized", version="1.0.0")
    logger.info(
        "Configuration",
        api_key_configured=bool(SENTINEL_API_KEY),
        sentry_enabled=SENTRY_DSN is not None,
        rate_limit=100,
        max_episodes=1000,
    )
    # Start periodic episode cleanup
    await episode_manager.start_background_cleanup(interval_seconds=300)
    try:
        yield
    finally:
        await episode_manager.stop_background_cleanup()
        logger.info("Sentinel Environment shutdown")


app = FastAPI(
    title="Sentinel Environment",
    description="AI Agent Safety & Jailbreak Detection Environment - Production Grade",
    version="1.1.0",
    lifespan=lifespan,
)

# Add production middleware
setup_production_middleware(app)

# Add v1 API router (batch endpoints, model registry, WebSockets)
from server.batch_api import v1_router  # noqa: E402

app.include_router(v1_router)


@app.get("/")
async def root():
    """Root endpoint - returns API information."""
    return JSONResponse(
        content={
            "service": "Sentinel Environment",
            "version": "1.1.0",
            "description": "AI Agent Safety & Jailbreak Detection Environment",
            "endpoints": {
                "POST /reset": "Start a new episode",
                "POST /step": "Execute one step",
                "GET /state": "Get current episode state",
                "GET /grade": "Grade current episode",
                "GET /health": "Health check",
                "GET /resilience-profile": "Get resilience profile",
                "GET /metrics": "Prometheus metrics",
            },
            "docs": "https://huggingface.co/spaces/PranavaKumar09/sentinel-env",
        }
    )


@app.post("/reset", response_model=ResetResponse)
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
        episode_id, observation = await episode_manager.create_episode(task_name=task_name, seed=seed)

        response_data = observation.model_dump()
        response_data["episode_id"] = episode_id
        return JSONResponse(content=response_data)
    except Exception as e:
        logger.error("reset() failed", error=_sanitize_exception_message(e))
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.post("/step", response_model=StepResponse)
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

    env = await episode_manager.get_episode(episode_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Episode not found")

    try:
        observation, reward, done, info = env.step(action)
        return JSONResponse(
            content={
                "observation": observation.model_dump(),
                "reward": reward,
                "done": done,
                "info": info,
            }
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("step() failed", error=_sanitize_exception_message(e))
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.get("/state", response_model=StateResponse)
async def state(
    episode_id: str = Header(None, alias="X-Episode-ID"),
    api_key: str = Depends(verify_api_key),
):
    """Get current episode state."""
    if not episode_id:
        raise HTTPException(status_code=400, detail="X-Episode-ID header required")

    env = await episode_manager.get_episode(episode_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Episode not found")

    try:
        state = env.state()
        return JSONResponse(content=state.model_dump())
    except Exception as e:
        logger.error("state() failed", error=_sanitize_exception_message(e))
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.get("/health")
async def health():
    """Health check endpoint with detailed service information."""
    return JSONResponse(
        content={
            "status": "healthy",
            "service": "sentinel-env",
            "version": "1.1.0",
            "features": {
                "structured_logging": True,
                "prometheus_metrics": True,
                "sentry_tracking": SENTRY_DSN is not None,
                "rate_limiting": True,
                "concurrent_episodes": True,
                "jailbreak_prompts": True,
                "wandb_tracking": True,
            },
            "episode_manager": {
                "max_episodes": 1000,
                "ttl_seconds": 3600,
            },
            "rate_limiter": {
                "max_requests": 100,
                "window_seconds": 60,
            },
        }
    )


@app.get("/grade", response_model=GradeResponse)
async def grade(
    episode_id: str = Header(None, alias="X-Episode-ID"),
    api_key: str = Depends(verify_api_key),
):
    """Grade the current episode (helper endpoint)."""
    if not episode_id:
        raise HTTPException(status_code=400, detail="X-Episode-ID header required")

    env = await episode_manager.get_episode(episode_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Episode not found")

    try:
        result = env.get_episode_grade()
        return JSONResponse(content=result)
    except Exception as e:
        logger.error("grade() failed", error=_sanitize_exception_message(e))
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.get("/resilience-profile")
async def resilience_profile(
    episode_id: str = Header(None, alias="X-Episode-ID"),
    api_key: str = Depends(verify_api_key),
):
    """Get resilience profile for current episode (helper endpoint)."""
    if not episode_id:
        raise HTTPException(status_code=400, detail="X-Episode-ID header required")

    env = await episode_manager.get_episode(episode_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Episode not found")

    try:
        profile = env.get_resilience_profile()
        return JSONResponse(content=profile)
    except Exception as e:
        logger.error("resilience_profile() failed", error=_sanitize_exception_message(e))
        raise HTTPException(status_code=500, detail="Internal server error") from e


def main():
    """Run the Sentinel Environment server."""
    import uvicorn

    port = int(os.getenv("PORT", 7860))
    host = os.getenv("HOST", "0.0.0.0")  # nosec B104 - required for container deployment

    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
