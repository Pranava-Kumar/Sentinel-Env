"""In-memory rate limiter with bounded storage using deque.

Uses sliding window algorithm with per-IP request tracking.
Automatically cleans up stale entries to prevent memory leaks.
Uses collections.deque for O(1) popleft operations.
"""

import time
from collections import OrderedDict
from typing import Dict, Deque


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
        self.requests: Dict[str, Deque[float]] = OrderedDict()
    
    async def check_rate_limit(self, client_ip: str) -> bool:
        """Check if request is within rate limit.

        Uses deque.popleft() for O(1) cleanup instead of list comprehension.

        Args:
            client_ip: Client IP address

        Returns:
            True if request is allowed, False if rate limited
        """
        from collections import deque

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
            return False

        ip_requests.append(now)

        # Evict oldest entries if we've exceeded max_entries
        if len(self.requests) > self.max_entries:
            self.requests.popitem(last=False)

        return True
    
    def cleanup(self) -> int:
        """Remove all expired entries. Call periodically.
        
        Returns:
            Number of IPs cleaned up
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
