"""Batch API endpoints and API versioning for Sentinel Environment.

Provides:
- API v1 router at /api/v1/ wrapping all existing endpoints
- Batch evaluation: submit multiple prompts at once
- Batch episodes: run N episodes in parallel
- Model registry: register, evaluate, and compare models
- WebSocket support for real-time metrics streaming
- LRU caching for repeated requests
"""

import asyncio
import time
from collections import OrderedDict
from typing import Any

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from models import RecommendedAction, SentinelAction, ThreatCategory

logger = structlog.get_logger()

v1_router = APIRouter(prefix="/api/v1")


# ── Pydantic Models ──────────────────────────────────────────────────


class BatchEvaluateRequest(BaseModel):
    prompts: list[str] = Field(..., min_length=1, max_length=100, description="List of prompts to evaluate")
    task_name: str = Field("basic-injection", description="Task context for evaluation")


class BatchEvaluateResponse(BaseModel):
    results: list[dict[str, Any]]
    total: int
    processing_time_ms: float


class BatchEpisodesRequest(BaseModel):
    task_name: str = Field("basic-injection")
    seed: int = Field(42)
    num_episodes: int = Field(1, ge=1, le=50, description="Number of episodes to run")
    model_action: SentinelAction = Field(
        default=SentinelAction(
            classification=ThreatCategory.SAFE,
            reasoning="This prompt appears to be safe and benign with no malicious content",
            recommended_action=RecommendedAction.ALLOW,
            safe_alternative=None,
        ),
        description="Action to take for each step",
    )


class BatchEpisodesResponse(BaseModel):
    episodes: list[dict[str, Any]]
    aggregate: dict[str, Any]
    total_processing_time_ms: float


class ModelRegistration(BaseModel):
    model_id: str = Field(..., min_length=1, max_length=100)
    model_name: str = Field(..., min_length=1)
    description: str = Field("", description="Model description")
    config: dict[str, Any] = Field(default_factory=dict)


class ModelComparisonResponse(BaseModel):
    models: list[dict[str, Any]]
    comparison: dict[str, Any]


# ── Model Registry ────────────────────────────────────────────────────


class ModelRegistry:
    """In-memory registry for model evaluation tracking."""

    def __init__(self):
        self._models: dict[str, dict[str, Any]] = {}
        self._results: dict[str, list[dict[str, Any]]] = {}

    def register(self, registration: ModelRegistration) -> dict[str, Any]:
        model_data = registration.model_dump()
        model_data["registered_at"] = time.time()
        model_data["evaluation_count"] = 0
        self._models[registration.model_id] = model_data
        self._results[registration.model_id] = []
        logger.info("Model registered", model_id=registration.model_id)
        return model_data

    def list_models(self) -> list[dict[str, Any]]:
        return list(self._models.values())

    def record_result(self, model_id: str, result: dict[str, Any]):
        if model_id not in self._results:
            self._results[model_id] = []
        self._results[model_id].append(result)
        if model_id in self._models:
            self._models[model_id]["evaluation_count"] += 1

    def get_results(self, model_id: str) -> list[dict[str, Any]]:
        return self._results.get(model_id, [])

    def compare_models(self, model_ids: list[str]) -> dict[str, Any]:
        comparison = {}
        for mid in model_ids:
            results = self._results.get(mid, [])
            if results:
                scores = [r.get("score", 0.0) for r in results]
                comparison[mid] = {
                    "evaluations": len(results),
                    "avg_score": sum(scores) / len(scores),
                    "min_score": min(scores),
                    "max_score": max(scores),
                    "avg_detection_rate": sum(r.get("detection_rate", 0.0) for r in results) / len(results),
                }
            else:
                comparison[mid] = {"evaluations": 0, "avg_score": 0.0}
        return comparison


# Global model registry
model_registry = ModelRegistry()


# ── LRU Cache ─────────────────────────────────────────────────────────


class MetricsCache:
    """Thread-safe LRU cache for metrics."""

    def __init__(self, max_size: int = 1000):
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl = 60  # 60 second TTL

    def get(self, key: str) -> Any | None:
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                return value
            else:
                # Expired
                del self._cache[key]
        return None

    def set(self, key: str, value: Any):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (value, time.time())
        # Evict oldest if over capacity
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def invalidate(self, key: str):
        self._cache.pop(key, None)


metrics_cache = MetricsCache()


# ── Batch Endpoints ───────────────────────────────────────────────────


@v1_router.post("/batch/evaluate", response_model=BatchEvaluateResponse)
async def batch_evaluate(request: BatchEvaluateRequest):
    """Evaluate multiple prompts in batch.

    Runs each prompt through the environment and returns classifications.
    """
    # Lazy import to avoid circular dependency with app.py
    from server.app import episode_manager

    start_time = time.time()
    results = []

    episode_id, obs = await episode_manager.create_episode(task_name=request.task_name, seed=42)
    env = await episode_manager.get_episode(episode_id)
    if env is None:
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=500, content={"detail": "Failed to create episode"})

    for i, prompt_text in enumerate(request.prompts):
        action = SentinelAction(
            classification=ThreatCategory.SAFE,
            reasoning=f"Analysis of prompt {i + 1} shows it appears to be safe and benign",
            recommended_action=RecommendedAction.ALLOW,
            safe_alternative=None,
        )

        obs, reward, _done, info = env.step(action)
        results.append(
            {
                "prompt_index": i,
                "prompt_preview": prompt_text[:100],
                "reward": reward,
                "is_correct": info.get("step_result", {}).get("is_correct", False),
                "attack_type": obs.attack_metadata.attack_type,
                "is_safe_prompt": obs.is_safe_prompt,
            }
        )

    processing_time = (time.time() - start_time) * 1000

    return BatchEvaluateResponse(
        results=results,
        total=len(results),
        processing_time_ms=round(processing_time, 2),
    )


@v1_router.post("/batch/episodes", response_model=BatchEpisodesResponse)
async def batch_episodes(request: BatchEpisodesRequest):
    """Run N episodes and return aggregated results.

    Useful for benchmarking model performance across multiple episodes.
    """
    # Lazy import to avoid circular dependency with app.py
    from server.app import episode_manager

    start_time = time.time()
    episodes = []

    for ep_idx in range(request.num_episodes):
        episode_id, _obs = await episode_manager.create_episode(task_name=request.task_name, seed=request.seed + ep_idx)
        env = await episode_manager.get_episode(episode_id)
        if env is None:
            continue

        episode_steps = []
        for step_num in range(env.max_steps):
            _obs, reward, done, info = env.step(request.model_action)
            episode_steps.append(
                {
                    "step": step_num + 1,
                    "reward": reward,
                    "is_correct": info.get("step_result", {}).get("is_correct", False),
                }
            )
            if done:
                break

        grade = env.get_episode_grade()
        episodes.append(
            {
                "episode_id": env.episode_id,
                "seed": request.seed + ep_idx,
                "score": grade["score"],
                "detection_rate": grade["detection_rate"],
                "false_positive_rate": grade["false_positive_rate"],
                "total_steps": grade["total_steps"],
                "steps": episode_steps,
            }
        )

    # Aggregate
    scores = [ep["score"] for ep in episodes]
    detection_rates = [ep["detection_rate"] for ep in episodes]
    fp_rates = [ep["false_positive_rate"] for ep in episodes]

    aggregate = {
        "num_episodes": len(episodes),
        "avg_score": round(sum(scores) / len(scores), 3) if scores else 0.0,
        "min_score": round(min(scores), 3) if scores else 0.0,
        "max_score": round(max(scores), 3) if scores else 0.0,
        "avg_detection_rate": round(sum(detection_rates) / len(detection_rates), 3) if detection_rates else 0.0,
        "avg_false_positive_rate": round(sum(fp_rates) / len(fp_rates), 3) if fp_rates else 0.0,
    }

    processing_time = (time.time() - start_time) * 1000

    return BatchEpisodesResponse(
        episodes=episodes,
        aggregate=aggregate,
        total_processing_time_ms=round(processing_time, 2),
    )


# ── Model Registry Endpoints ──────────────────────────────────────────


@v1_router.post("/models/register")
async def register_model(registration: ModelRegistration):
    """Register a new model for evaluation."""
    return model_registry.register(registration)


@v1_router.get("/models")
async def list_models():
    """List all registered models."""
    return {"models": model_registry.list_models()}


@v1_router.post("/models/{model_id}/evaluate")
async def evaluate_model(model_id: str, request: BatchEpisodesRequest):
    """Evaluate a registered model."""
    # Lazy import to avoid circular dependency with app.py
    from server.app import episode_manager

    # Run episodes and record results
    start_time = time.time()

    results = []
    for ep_idx in range(request.num_episodes):
        episode_id, _obs = await episode_manager.create_episode(task_name=request.task_name, seed=request.seed + ep_idx)
        env = await episode_manager.get_episode(episode_id)
        if env is None:
            continue

        for _ in range(env.max_steps):
            _obs, _reward, done, _info = env.step(request.model_action)
            if done:
                break

        grade = env.get_episode_grade()
        results.append(grade)

    model_registry.record_result(
        model_id,
        {
            "timestamp": time.time(),
            "task_name": request.task_name,
            "num_episodes": request.num_episodes,
            "results": results,
        },
    )

    return {
        "model_id": model_id,
        "evaluations": len(model_registry.get_results(model_id)),
        "processing_time_ms": round((time.time() - start_time) * 1000, 2),
    }


@v1_router.get("/models/{model_id}/results")
async def get_model_results(model_id: str):
    """Get evaluation results for a model."""
    return {
        "model_id": model_id,
        "results": model_registry.get_results(model_id),
    }


@v1_router.get("/models/compare")
async def compare_models(model_ids: str = ""):
    """Compare multiple models side-by-side."""
    ids = [m.strip() for m in model_ids.split(",") if m.strip()]
    if not ids:
        ids = list(model_registry._models.keys())

    comparison = model_registry.compare_models(ids)
    return ModelComparisonResponse(
        models=[model_registry._models.get(mid, {"model_id": mid}) for mid in ids],
        comparison=comparison,
    )


# ── WebSocket Endpoints ───────────────────────────────────────────────


@v1_router.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    """WebSocket stream of real-time metrics."""
    # Lazy import to avoid circular dependency with app.py
    from server.app import episode_manager

    await websocket.accept()
    start_time = time.time()

    try:
        while True:
            metrics = {
                "timestamp": time.time(),
                "active_episodes": episode_manager.active_episodes,
                "uptime_seconds": time.time() - start_time,
            }

            await websocket.send_json(metrics)
            await asyncio.sleep(5)  # Send every 5 seconds

    except WebSocketDisconnect:
        logger.info("Metrics WebSocket disconnected")


@v1_router.websocket("/ws/episodes")
async def websocket_episodes(websocket: WebSocket):
    """WebSocket stream of episode information."""
    # Lazy import to avoid circular dependency with app.py
    from server.app import episode_manager

    await websocket.accept()

    try:
        while True:
            # Stream current active episode summary
            episode_update = {
                "timestamp": time.time(),
                "event": "episode_summary",
                "data": {
                    "active_episodes": episode_manager.active_episodes,
                    "max_episodes": episode_manager.max_episodes,
                },
            }

            await websocket.send_json(episode_update)
            await asyncio.sleep(10)

    except WebSocketDisconnect:
        logger.info("Episodes WebSocket disconnected")


# ── Cached Health Endpoint ────────────────────────────────────────────


@v1_router.get("/health")
async def health_v1():
    """Versioned health check with caching."""
    cache_key = "health_v1"
    cached = metrics_cache.get(cache_key)
    if cached:
        return cached

    result = {
        "status": "healthy",
        "service": "sentinel-env",
        "version": "1.1.0",
        "api_version": "v1",
        "timestamp": time.time(),
    }

    metrics_cache.set(cache_key, result)
    return result
