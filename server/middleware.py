"""Production middleware for Sentinel FastAPI server.

Provides:
- Structured logging with structlog (JSON output, trace IDs)
- Request/response middleware with audit logging
- Prometheus metrics collection
- Request body size limits
- Error tracking context for Sentry
"""

import time
import uuid

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

# Maximum request body size (1MB)
MAX_BODY_SIZE = 1_048_576

logger = structlog.get_logger()


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured request/response logging.

    Adds:
    - Request ID tracing (X-Request-ID)
    - Request/response timing
    - Status code tracking
    - User agent and IP logging
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start_time = time.time()

        # Log request
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent", "unknown"),
        )

        logger.info("Request started")

        try:
            response = await call_next(request)

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            # Add rate limit headers if available
            rate_limit = getattr(request.state, "rate_limit_limit", None)
            rate_remaining = getattr(request.state, "rate_limit_remaining", None)
            if rate_limit is not None:
                response.headers["X-RateLimit-Limit"] = str(rate_limit)
            if rate_remaining is not None:
                response.headers["X-RateLimit-Remaining"] = str(rate_remaining)

            # Log response
            process_time = time.time() - start_time
            logger.info(
                "Request completed",
                status_code=response.status_code,
                duration_ms=round(process_time * 1000, 2),
            )

            return response

        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                "Request failed",
                error=str(e),
                duration_ms=round(process_time * 1000, 2),
                exc_info=True,
            )
            raise


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce request body size limits."""

    def __init__(self, app, max_body_size: int = MAX_BODY_SIZE):
        super().__init__(app)
        self.max_body_size = max_body_size

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_body_size:
            return JSONResponse(
                status_code=413,
                content={"detail": "Request body too large"},
            )

        return await call_next(request)


class PrometheusMetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting Prometheus metrics.

    Tracks:
    - Request count by endpoint
    - Request duration histogram
    - Error rate
    - Active episodes
    """

    def __init__(self, app):
        super().__init__(app)
        self._setup_metrics()

    def _setup_metrics(self):
        """Initialize Prometheus metrics."""
        try:
            from prometheus_client import Counter, Gauge, Histogram

            self.request_count = Counter(
                "sentinel_requests_total",
                "Total HTTP requests",
                ["method", "endpoint", "status"],
            )
            self.request_duration = Histogram(
                "sentinel_request_duration_seconds",
                "Request duration in seconds",
                ["method", "endpoint"],
                buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            )
            self.active_episodes = Gauge(
                "sentinel_active_episodes",
                "Number of active episodes",
            )
            self.episode_score = Histogram(
                "sentinel_episode_score",
                "Episode scores",
                buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            )
            self.detection_rate = Gauge(
                "sentinel_detection_rate",
                "Current detection rate",
            )
            self.false_positive_rate = Gauge(
                "sentinel_false_positive_rate",
                "Current false positive rate",
            )
            self._enabled = True
        except ImportError:
            self._enabled = False

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not self._enabled:
            return await call_next(request)

        start_time = time.time()

        response = await call_next(request)

        # Record metrics
        endpoint = request.url.path
        method = request.method
        status = response.status_code

        self.request_count.labels(method=method, endpoint=endpoint, status=status).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(time.time() - start_time)

        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global error handling with context for Sentry.

    Catches unhandled exceptions and returns consistent error responses.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        try:
            return await call_next(request)
        except Exception as e:
            request_id = getattr(request.state, "request_id", "unknown")

            logger.error(
                "Unhandled exception",
                error=str(e),
                request_id=request_id,
                path=request.url.path,
                exc_info=True,
            )

            # Try to send to Sentry if available
            try:
                import sentry_sdk

                sentry_sdk.capture_exception(e)
            except ImportError:
                pass

            return JSONResponse(
                status_code=500,
                content={
                    "detail": "Internal server error",
                    "request_id": request_id,
                },
            )


def setup_production_middleware(app: FastAPI):
    """Configure all production middleware on the FastAPI app.

    Call this during application startup to add:
    - Structured logging
    - Request size limits
    - Prometheus metrics
    - Error handling
    """
    # Order matters: outermost middleware runs first
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(PrometheusMetricsMiddleware)
    app.add_middleware(RequestSizeLimitMiddleware)
    app.add_middleware(StructuredLoggingMiddleware)

    # Add metrics endpoint (lazy import to avoid hard dependency in tests)
    from fastapi import APIRouter
    from starlette.responses import Response

    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        router = APIRouter()

        @router.get("/metrics", include_in_schema=False)
        async def get_metrics():
            """Prometheus metrics endpoint."""
            return Response(
                content=generate_latest(),
                media_type=CONTENT_TYPE_LATEST,
            )

        app.include_router(router)
    except ImportError:
        pass  # prometheus-client not installed; metrics endpoint unavailable
