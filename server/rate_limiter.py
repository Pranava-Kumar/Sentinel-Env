"""In-memory rate limiter with bounded storage using deque.

Uses sliding window algorithm with per-IP request tracking.
Automatically cleans up stale entries to prevent memory leaks.
Uses collections.deque for O(1) popleft operations.
"""

import time
from collections import OrderedDict, deque


class RateLimiter:
    """Sliding window rate limiter with automatic cleanup.

    Args:
        max_requests: Maximum requests allowed per window
        window_seconds: Time window in seconds
        max_entries: Maximum number of IP entries to track (evicts oldest)
    """

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
        max_entries: int = 10000,
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_entries = max_entries
        self._cleanup_threshold = int(max_entries * 0.8)  # Pre-computed for performance
        self.requests: OrderedDict[str, deque[float]] = OrderedDict()

    async def check_rate_limit(self, client_ip: str) -> tuple[bool, int]:
        """Check if request is within rate limit.

        Uses deque.popleft() for O(1) cleanup instead of list comprehension.

        Args:
            client_ip: Client IP address

        Returns:
            Tuple of (allowed: bool, remaining: int) — remaining count of requests left.
        """
        now = time.time()
        window_start = now - self.window_seconds

        # Create entry if IP not seen before
        if client_ip not in self.requests:
            self.requests[client_ip] = deque()

        # Remove expired entries for this IP (O(1) per item with deque)
        ip_requests = self.requests[client_ip]
        while ip_requests and ip_requests[0] < window_start:
            ip_requests.popleft()

        # Check limit
        if len(ip_requests) >= self.max_requests:
            return False, 0

        ip_requests.append(now)

        remaining = self.max_requests - len(ip_requests)

        # Proactive cleanup: when approaching max_entries, clean expired entries
        if len(self.requests) > self._cleanup_threshold:
            self._cleanup_expired()

        # Evict oldest entries if we've exceeded max_entries
        if len(self.requests) > self.max_entries:
            self.requests.popitem(last=False)

        return True, remaining

    def _cleanup_expired(self) -> int:
        """Remove all expired timestamps from all IPs.

        Returns:
            Number of IPs fully removed.
        """
        now = time.time()
        window_start = now - self.window_seconds
        cleaned = 0

        empty_ips = []
        for ip, timestamps in self.requests.items():
            while timestamps and timestamps[0] < window_start:
                timestamps.popleft()
            if not timestamps:
                empty_ips.append(ip)

        for ip in empty_ips:
            del self.requests[ip]
            cleaned += 1

        return cleaned

    def cleanup(self) -> int:
        """Remove all expired entries. Call periodically.

        Returns:
            Number of IPs cleaned up
        """
        return self._cleanup_expired()
