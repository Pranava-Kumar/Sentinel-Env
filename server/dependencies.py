"""Shared dependencies for Sentinel Environment.

Centralizes singleton instances to avoid circular imports between
app.py, batch_api.py, and other modules.
"""

from server.episode_manager import EpisodeManager
from server.rate_limiter import RateLimiter

# Episode manager for concurrent episode support
episode_manager = EpisodeManager(max_episodes=1000, ttl_seconds=3600)

# Rate limiter: 100 requests per minute per IP
rate_limiter = RateLimiter(max_requests=100, window_seconds=60, max_entries=10000)
