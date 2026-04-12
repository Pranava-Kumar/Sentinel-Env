# Middleware Architecture

This document describes the middleware pipeline design, implementation, and rationale for the Sentinel Environment FastAPI server.

## Overview

The Sentinel Environment uses a **layered middleware architecture** to handle cross-cutting concerns such as logging, metrics, error handling, and request validation. Each middleware wraps the next, forming a pipeline that processes requests before they reach endpoint handlers and processes responses on the way back.

## Middleware Stack

### Execution Order

```
Request →
  ErrorHandlingMiddleware (outermost)
    → PrometheusMetricsMiddleware
      → RequestSizeLimitMiddleware
        → StructuredLoggingMiddleware (innermost)
          → Endpoint Handler
Response ←
  ErrorHandlingMiddleware
    ← PrometheusMetricsMiddleware
      ← RequestSizeLimitMiddleware
        ← StructuredLoggingMiddleware
          ← Endpoint Handler
```

### Why This Order Matters

| Position | Middleware | Reason |
|----------|-----------|--------|
| **1st (Outermost)** | `ErrorHandlingMiddleware` | Must catch exceptions from ALL downstream middleware, including request parsing errors |
| **2nd** | `PrometheusMetricsMiddleware` | Must measure timing of ALL processing (logging, size checks, endpoint logic) |
| **3rd** | `RequestSizeLimitMiddleware` | Must reject oversized requests BEFORE logging/processing them (resource protection) |
| **4th (Innermost)** | `StructuredLoggingMiddleware` | Closest to endpoint, logs actual request/response details with full context |

**Incorrect order would cause:**
- Error handler after metrics → Miss metrics for error responses
- Size limit after logging → Log flooded with oversized request data
- Logging after metrics → No request context in error traces

---

## Middleware Implementations

### 1. ErrorHandlingMiddleware

**Purpose:** Global error catching with Sentry integration and consistent error responses.

**Position:** Outermost (first to process requests, last to process responses)

**Key Features:**
- Catches all unhandled exceptions from downstream middleware
- Integrates with Sentry for error tracking
- Returns consistent error format with request ID for tracing
- Sanitizes exception messages to prevent data leakage

**Implementation:**
```python
class ErrorHandlingMiddleware(BaseHTTPMiddleware):
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
            
            # Send to Sentry if available
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
```

**Error Response Format:**
```json
{
  "detail": "Internal server error",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Design Decisions:**
- **Why outermost?** Must catch exceptions from all other middleware, including JSON parsing errors, rate limiter failures, etc.
- **Why sanitize messages?** Prevents stack traces and sensitive data from leaking to clients
- **Why optional Sentry?** Not all deployments need error tracking; degrades gracefully

**Edge Cases Handled:**
- Exceptions before request ID is generated (fallback to "unknown")
- Sentry SDK not installed (graceful degradation)
- Exception message truncation (200 char limit to prevent log flooding)

---

### 2. PrometheusMetricsMiddleware

**Purpose:** Collect request/response metrics for monitoring and alerting.

**Position:** Second (after error handler, before size limits)

**Key Features:**
- Request count by endpoint, method, status
- Request duration histogram with custom buckets
- Active episodes gauge
- Episode score histogram
- Detection rate and false positive rate gauges

**Metrics Collected:**

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `sentinel_requests_total` | Counter | `method`, `endpoint`, `status` | Total HTTP requests |
| `sentinel_request_duration_seconds` | Histogram | `method`, `endpoint` | Request latency (9 buckets) |
| `sentinel_active_episodes` | Gauge | — | Current active episodes |
| `sentinel_episode_score` | Histogram | — | Episode scores (11 buckets) |
| `sentinel_detection_rate` | Gauge | — | Current detection rate |
| `sentinel_false_positive_rate` | Gauge | — | Current false positive rate |

**Implementation:**
```python
class PrometheusMetricsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self._setup_metrics()
    
    def _setup_metrics(self):
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
            # ... more metrics ...
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
        self.request_duration.labels(method=method, endpoint=endpoint).observe(
            time.time() - start_time
        )
        
        return response
```

**Histogram Buckets:**

**Request Duration:**
```
[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0] seconds
```
- Covers fast health checks (10ms) to slow reset calls (10s)
- P50 target: <100ms, P95 target: <1s, P99 target: <5s

**Episode Score:**
```
[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
```
- Fine-grained tracking of model improvement over time

**Design Decisions:**
- **Why after ErrorHandling?** Errors are caught downstream, but we still want to count them
- **Why before Size Limits?** Want to track rejected requests too (for monitoring abuse)
- **Why graceful degradation?** Prometheus is optional; server works without it

**Performance Considerations:**
- Metrics collection adds ~0.1ms per request (negligible)
- Counter/Histogram operations are thread-safe and lock-free
- No blocking I/O in metrics path

---

### 3. RequestSizeLimitMiddleware

**Purpose:** Prevent resource exhaustion by rejecting oversized requests.

**Position:** Third (after metrics, before logging)

**Key Features:**
- 1MB request body limit
- Early rejection based on `Content-Length` header
- Returns 413 Payload Too Large without processing body

**Implementation:**
```python
class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_body_size: int = 1_048_576):
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
```

**Why 1MB?**
- Sentinel actions are ~200-500 bytes
- Batch requests (max 100 episodes) are ~50KB
- 1MB provides 10x headroom for future batch sizes
- Prevents memory exhaustion attacks

**Limitations:**
- Only checks `Content-Length` header (not chunked transfers)
- Clients can omit header to bypass check (mitigated by downstream body size limits in FastAPI)

**Design Decisions:**
- **Why after Metrics?** Track oversized requests for monitoring (detect abuse patterns)
- **Why before Logging?** Don't log massive request bodies (wastes log storage)

**Error Response:**
```json
{
  "detail": "Request body too large"
}
```

---

### 4. StructuredLoggingMiddleware

**Purpose:** Provide structured JSON logging with request ID tracing.

**Position:** Innermost (closest to endpoint handlers)

**Key Features:**
- UUID request ID generation (`X-Request-ID`)
- Request/response timing
- IP and user agent logging
- Status code tracking
- Context variable binding for downstream logging

**Implementation:**
```python
class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        start_time = time.time()
        
        # Bind context variables
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
            msg = str(e)
            if len(msg) > 200:
                msg = msg[:200] + "...[truncated]"
            logger.error(
                "Request failed",
                error=msg,
                duration_ms=round((time.time() - start_time) * 1000, 2),
                exc_info=True,
            )
            raise
```

**Log Format (JSON):**
```json
{
  "event": "Request completed",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "method": "POST",
  "path": "/reset",
  "client_ip": "127.0.0.1",
  "user_agent": "python-httpx/0.27.0",
  "status_code": 200,
  "duration_ms": 45.23,
  "timestamp": "2026-04-12T10:30:45.123456Z"
}
```

**Design Decisions:**
- **Why innermost?** Needs full request context; closest to actual endpoint logic
- **Why UUID for request ID?** Globally unique, traceable across services
- **Why context variables?** Allows downstream code to include request ID in their logs automatically
- **Why truncate exceptions?** Prevents log flooding from verbose error messages

**Request ID Flow:**
```
Client Request
  ↓
[Generate UUID: 550e8400-...]
  ↓
[Bind to context vars]
  ↓
[Endpoint Handler] → Can access via request.state.request_id
  ↓
[Add to Response Headers: X-Request-ID]
  ↓
Client Response (includes request ID for debugging)
```

---

## Setup and Configuration

### Adding Middleware to FastAPI

```python
from fastapi import FastAPI
from server.middleware import setup_production_middleware

app = FastAPI()

# Add all production middleware in correct order
setup_production_middleware(app)
```

### `setup_production_middleware()` Function

```python
def setup_production_middleware(app: FastAPI):
    """Configure all production middleware on the FastAPI app.
    
    Order matters: outermost middleware runs first.
    """
    # Add middleware in REVERSE order (FastAPI unwraps from bottom)
    app.add_middleware(ErrorHandlingMiddleware)         # 1st (outermost)
    app.add_middleware(PrometheusMetricsMiddleware)     # 2nd
    app.add_middleware(RequestSizeLimitMiddleware)      # 3rd
    app.add_middleware(StructuredLoggingMiddleware)     # 4th (innermost)
    
    # Add metrics endpoint
    from fastapi import APIRouter
    from starlette.responses import Response
    
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
        
        router = APIRouter()
        
        @router.get("/metrics", include_in_schema=False)
        async def get_metrics():
            return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
        
        app.include_router(router)
    except ImportError:
        pass  # prometheus-client not installed
```

**Why reverse order?**
FastAPI's `add_middleware()` wraps middleware in reverse order of addition. The last middleware added is the innermost (closest to endpoint).

---

## Middleware Interaction Examples

### Example 1: Successful Request

```
POST /reset
  ↓
1. ErrorHandlingMiddleware: try { call_next() }
  ↓
2. PrometheusMetricsMiddleware: start_timer = now()
  ↓
3. RequestSizeLimitMiddleware: check Content-Length < 1MB ✓
  ↓
4. StructuredLoggingMiddleware: 
     - Generate request_id = "abc-123"
     - Log "Request started"
     - Bind context vars
  ↓
5. Endpoint Handler: create episode, return response
  ↓
6. StructuredLoggingMiddleware:
     - Calculate duration = 45ms
     - Add X-Request-ID to response
     - Log "Request completed" (status=200, duration=45ms)
  ↓
7. PrometheusMetricsMiddleware:
     - Record request_count++ (POST, /reset, 200)
     - Record duration (0.045s)
  ↓
8. ErrorHandlingMiddleware: return response
  ↓
Client receives: 200 OK + X-Request-ID: abc-123
```

### Example 2: Oversized Request

```
POST /step (Content-Length: 2MB)
  ↓
1. ErrorHandlingMiddleware: try { call_next() }
  ↓
2. PrometheusMetricsMiddleware: start_timer
  ↓
3. RequestSizeLimitMiddleware: Content-Length > 1MB ✗
     → Return 413 immediately
  ↓
4. PrometheusMetricsMiddleware:
     - Record request_count++ (POST, /step, 413)
     - Record duration
  ↓
5. ErrorHandlingMiddleware: return response
  ↓
Client receives: 413 Payload Too Large
```

**Note:** StructuredLoggingMiddleware never sees the request (rejected before it).

### Example 3: Unhandled Exception

```
POST /step (invalid action JSON)
  ↓
1. ErrorHandlingMiddleware: try { call_next() }
  ↓
2. PrometheusMetricsMiddleware: start_timer
  ↓
3. RequestSizeLimitMiddleware: OK
  ↓
4. StructuredLoggingMiddleware: Log "Request started"
  ↓
5. Endpoint Handler: Pydantic validation fails → raise ValueError
  ↓
6. StructuredLoggingMiddleware: Exception caught, log "Request failed"
     → Re-raise exception
  ↓
7. PrometheusMetricsMiddleware: Exception propagates
     → Record metrics (POST, /step, 500)
  ↓
8. ErrorHandlingMiddleware: Exception caught
     → Log error with Sentry
     → Return 500 JSON
  ↓
Client receives: 500 Internal Server Error + X-Request-ID
```

---

## Adding Custom Middleware

### Guidelines

1. **Determine Position:** Where should your middleware run?
   - Before all others? Add to beginning of `setup_production_middleware()`
   - After all others? Add to end

2. **Implement BaseHTTPMiddleware:**
   ```python
   from fastapi import Request, Response
   from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
   
   class CustomMiddleware(BaseHTTPMiddleware):
       async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
           # Pre-processing
           response = await call_next(request)
           # Post-processing
           return response
   ```

3. **Handle Errors Gracefully:**
   - Don't let exceptions escape (catch and return error response)
   - Or let them propagate to ErrorHandlingMiddleware

4. **Add Tests:**
   - Test normal request flow
   - Test error cases
   - Test interaction with other middleware

### Example: CORS Middleware

```python
from starlette.middleware.base import BaseHTTPMiddleware

class CORSMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Handle preflight
        if request.method == "OPTIONS":
            response = Response(status_code=200)
        else:
            response = await call_next(request)
        
        # Add CORS headers
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "X-API-Key, X-Episode-ID"
        
        return response
```

**Add to setup:**
```python
def setup_production_middleware(app: FastAPI):
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(CORSMiddleware)  # After error handling (so CORS on errors too)
    app.add_middleware(PrometheusMetricsMiddleware)
    # ... rest
```

---

## Performance Considerations

### Middleware Overhead

| Middleware | Overhead per Request | Notes |
|-----------|---------------------|-------|
| ErrorHandling | ~0.01ms | try/except only (fast path) |
| PrometheusMetrics | ~0.1ms | Counter increment + histogram observe |
| RequestSizeLimit | ~0.01ms | Header check (no body read) |
| StructuredLogging | ~0.2ms | UUID generation + JSON serialization |
| **Total** | **~0.3ms** | Negligible for typical requests |

### Optimization Opportunities

1. **Skip metrics for health checks:**
   ```python
   if endpoint == "/health":
       return await call_next(request)  # Skip metrics recording
   ```

2. **Batch log writes:**
   - Currently: Log per request
   - Future: Async batch writes every 100ms

3. **Cache rate limit lookups:**
   - Currently: Check every request
   - Future: LRU cache with TTL

---

## Monitoring and Debugging

### Tracing Requests

Use `X-Request-ID` to trace requests across logs:

```bash
# Search logs for specific request
grep "550e8400-e29b-41d4-a716-446655440000" logs.json

# Follow request through middleware
cat logs.json | jq 'select(.request_id == "550e8400-...")'
```

### Middleware-Specific Debugging

**ErrorHandlingMiddleware Issues:**
- Check Sentry dashboard for captured exceptions
- Look for 500 responses with `request_id`

**PrometheusMetrics Issues:**
- Query metrics: `sentinel_requests_total{status="500"}`
- Check `/metrics` endpoint is accessible

**RequestSizeLimit Issues:**
- Look for 413 responses in logs
- Check `Content-Length` headers

**StructuredLogging Issues:**
- Verify `X-Request-ID` in response headers
- Check JSON log format is valid

---

## Testing Middleware

### Unit Tests

```python
import pytest
from fastapi.testclient import TestClient
from server.app import app

client = TestClient(app)

def test_request_id_in_response():
    """Verify StructuredLoggingMiddleware adds X-Request-ID."""
    response = client.post("/reset", params={"task_name": "basic-injection"})
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) == 36  # UUID format

def test_rate_limit_headers():
    """Verify rate limit headers are present."""
    response = client.post("/reset")
    assert "X-RateLimit-Limit" in response.headers
    assert "X-RateLimit-Remaining" in response.headers

def test_oversized_request_rejected():
    """Verify RequestSizeLimitMiddleware rejects large requests."""
    response = client.post("/step", headers={"Content-Length": "2000000"})
    assert response.status_code == 413

def test_error_handling_middleware():
    """Verify 500 errors include request_id."""
    # Trigger an error (mock endpoint)
    response = client.get("/force-error")
    assert response.status_code == 500
    assert "request_id" in response.json()
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_middleware_order():
    """Verify middleware execute in correct order."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/reset")
        
        # All middleware should have run
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers  # Logging
        # Check Prometheus metrics incremented (via /metrics endpoint)
```

---

## Future Enhancements

### Planned Improvements

1. **Request Body Logging (Opt-in):**
   - Log request/response bodies for debugging (dev only)
   - Configurable via environment variable

2. **Circuit Breaker Middleware:**
   - Detect cascading failures
   - Return fast-fail responses when downstream services are unhealthy

3. **Request Deduplication:**
   - Cache identical requests (same URL + body)
   - Return cached response with `X-Cache: HIT` header

4. **API Versioning Middleware:**
   - Route requests to v1/v2 endpoints based on header
   - Deprecation warnings in response headers

### Monitoring Improvements

1. **Distributed Tracing:**
   - Integrate OpenTelemetry
   - Propagate trace context across services

2. **Custom Metrics:**
   - Episode completion time
   - Attack type distribution
   - Model inference latency

---

## References

- [Starlette Middleware Documentation](https://www.starlette.io/middleware/)
- [FastAPI Middleware Guide](https://fastapi.tiangolo.com/tutorial/middleware/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/naming/)
- [structlog Documentation](https://www.structlog.org/en/stable/)

---

**Last Updated:** April 12, 2026  
**Version:** 1.1.0  
**Maintainer:** Backend Team
