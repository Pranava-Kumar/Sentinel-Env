"""Tests for the rate limiter module."""

import asyncio
import time

import pytest

from server.rate_limiter import RateLimiter


class TestRateLimiter:
    @pytest.fixture
    def limiter(self):
        return RateLimiter(max_requests=5, window_seconds=1)

    @pytest.mark.asyncio
    async def test_allows_within_limit(self, limiter):
        for _ in range(5):
            assert await limiter.check_rate_limit("127.0.0.1") is True

    @pytest.mark.asyncio
    async def test_blocks_over_limit(self, limiter):
        for _ in range(5):
            await limiter.check_rate_limit("127.0.0.1")
        assert await limiter.check_rate_limit("127.0.0.1") is False

    @pytest.mark.asyncio
    async def test_different_ips_independent(self, limiter):
        for _ in range(5):
            await limiter.check_rate_limit("127.0.0.1")
        assert await limiter.check_rate_limit("192.168.1.1") is True

    @pytest.mark.asyncio
    async def test_window_expiry_resets_limit(self, limiter):
        for _ in range(5):
            await limiter.check_rate_limit("127.0.0.1")
        assert await limiter.check_rate_limit("127.0.0.1") is False

        await asyncio.sleep(1.1)
        assert await limiter.check_rate_limit("127.0.0.1") is True

    def test_cleanup_removes_expired(self):
        limiter = RateLimiter(max_requests=5, window_seconds=1)

        loop = asyncio.new_event_loop()
        try:
            for _ in range(5):
                loop.run_until_complete(limiter.check_rate_limit("127.0.0.1"))
        finally:
            loop.close()

        time.sleep(1.1)
        cleaned = limiter.cleanup()
        assert cleaned == 1
        assert "127.0.0.1" not in limiter.requests

    def test_max_entries_eviction(self):
        limiter = RateLimiter(max_requests=1, window_seconds=60, max_entries=3)

        loop = asyncio.new_event_loop()
        try:
            for i in range(5):
                loop.run_until_complete(limiter.check_rate_limit(f"10.0.0.{i}"))
        finally:
            loop.close()

        assert len(limiter.requests) <= 3
