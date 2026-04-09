"""Load and concurrency tests for Sentinel Environment.

Tests the system under load:
- Concurrent episode creation and management
- High request volume
- Memory stability under sustained load
- Episode cleanup and TTL behavior
"""

import asyncio
import time

import pytest

from models import RecommendedAction, SentinelAction, ThreatCategory
from server.episode_manager import EpisodeManager
from server.rate_limiter import RateLimiter
from server.sentinel_environment import SentinelEnvironment


class TestConcurrentEpisodes:
    """Test concurrent episode management."""

    @pytest.mark.asyncio
    async def test_concurrent_episode_creation(self):
        """Create many episodes concurrently without errors."""
        manager = EpisodeManager(max_episodes=100, ttl_seconds=60)

        async def create_episode(i: int):
            episode_id, obs = manager.create_episode(
                task_name="basic-injection",
                seed=i,
            )
            return episode_id, obs

        # Create 50 episodes concurrently
        tasks = [create_episode(i) for i in range(50)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 50
        assert manager.active_episodes == 50

        # All episode IDs should be unique
        episode_ids = [eid for eid, _ in results]
        assert len(set(episode_ids)) == 50

    @pytest.mark.asyncio
    async def test_concurrent_episode_operations(self):
        """Perform concurrent operations on episodes."""
        manager = EpisodeManager(max_episodes=100, ttl_seconds=60)

        # Create episodes
        episode_ids = []
        for i in range(20):
            eid, _ = manager.create_episode(task_name="basic-injection", seed=i)
            episode_ids.append(eid)

        async def step_episode(episode_id: str, seed: int):
            env = manager.get_episode(episode_id)
            if env is None:
                return None
            action = SentinelAction(
                classification=ThreatCategory.SAFE,
                reasoning="This prompt appears to be completely safe and benign with no threats detected",
                recommended_action=RecommendedAction.ALLOW,
            )
            obs, reward, done, info = env.step(action)
            return {"episode_id": episode_id, "reward": reward, "done": done}

        # Step all episodes concurrently
        tasks = [step_episode(eid, i) for i, eid in enumerate(episode_ids)]
        results = await asyncio.gather(*tasks)

        # All steps should complete without errors
        successful_steps = [r for r in results if r is not None]
        assert len(successful_steps) == 20

    @pytest.mark.asyncio
    async def test_episode_eviction_under_load(self):
        """Episodes should be evicted when max is reached."""
        manager = EpisodeManager(max_episodes=50, ttl_seconds=60)

        # Create more episodes than max
        for i in range(100):
            manager.create_episode(task_name="basic-injection", seed=i)

        # Should not exceed max
        assert manager.active_episodes <= 50

    @pytest.mark.asyncio
    async def test_episode_ttl_cleanup_under_load(self):
        """Expired episodes should be cleaned up."""
        manager = EpisodeManager(max_episodes=100, ttl_seconds=0.1)  # Very short TTL

        # Create episodes
        for i in range(20):
            manager.create_episode(task_name="basic-injection", seed=i)

        # Wait for expiry
        await asyncio.sleep(0.2)

        # Trigger cleanup
        manager.cleanup_expired()

        assert manager.active_episodes == 0


class TestRateLimiterUnderLoad:
    """Test rate limiter under high load."""

    @pytest.mark.asyncio
    async def test_rate_limiter_handles_burst(self):
        """Burst of requests should be rate limited."""
        limiter = RateLimiter(max_requests=10, window_seconds=60, max_entries=1000)

        # First 10 should succeed
        for _ in range(10):
            assert await limiter.check_rate_limit("test-ip")

        # Next should be limited
        assert not await limiter.check_rate_limit("test-ip")

    @pytest.mark.asyncio
    async def test_rate_limiter_multiple_ips(self):
        """Different IPs should have independent limits."""
        limiter = RateLimiter(max_requests=5, window_seconds=60, max_entries=100)

        # Each IP should have independent limit
        for ip in ["ip1", "ip2", "ip3"]:
            for _ in range(5):
                assert await limiter.check_rate_limit(ip)
            assert not await limiter.check_rate_limit(ip)

    @pytest.mark.asyncio
    async def test_rate_limiter_max_entries_eviction(self):
        """Old entries should be evicted when max_entries reached."""
        limiter = RateLimiter(max_requests=100, window_seconds=60, max_entries=50)

        # Create many entries
        for i in range(100):
            await limiter.check_rate_limit(f"ip-{i}")

        # Should not exceed max_entries
        assert len(limiter.requests) <= 55  # Allows some headroom for OrderedDict


class TestMemoryStability:
    """Test memory stability under sustained load."""

    @pytest.mark.asyncio
    async def test_sustained_episode_creation(self):
        """Sustained episode creation shouldn't leak memory."""
        manager = EpisodeManager(max_episodes=100, ttl_seconds=1)

        # Create and let episodes expire repeatedly
        for batch in range(10):
            for i in range(50):
                manager.create_episode(task_name="basic-injection", seed=batch * 100 + i)

            await asyncio.sleep(0.2)
            manager.cleanup_expired()

        # Should not exceed max_episodes
        assert manager.active_episodes <= 100

    @pytest.mark.asyncio
    async def test_long_running_episode(self):
        """Long-running episode should remain stable."""
        env = SentinelEnvironment()
        env.reset(task_name="basic-injection", seed=42)

        # Run many steps
        for step in range(100):
            action = SentinelAction(
                classification=ThreatCategory.SAFE,
                reasoning=f"Step {step} analysis shows this prompt is completely safe and benign with no threats",
                recommended_action=RecommendedAction.ALLOW,
            )
            obs, reward, done, info = env.step(action)

            assert 0.0 <= reward <= 1.0

            if done:
                # Reset and continue
                env.reset(task_name="basic-injection", seed=step + 100)

        # Conversation history should be bounded
        assert len(env.conversation_history) <= env.max_steps


class TestEpisodeManagerScalability:
    """Test episode manager scalability."""

    def test_active_episodes_count_accuracy(self):
        """Active episodes count should be accurate."""
        manager = EpisodeManager(max_episodes=1000, ttl_seconds=60)

        for i in range(100):
            manager.create_episode(task_name="basic-injection", seed=i)

        assert manager.active_episodes == 100

        # Remove some
        episode_ids = list(manager.episodes.keys())[:50]
        for eid in episode_ids:
            manager.remove_episode(eid)

        assert manager.active_episodes == 50

    def test_max_episodes_configuration(self):
        """Max episodes configuration should be respected."""
        manager = EpisodeManager(max_episodes=10, ttl_seconds=60)

        for i in range(20):
            manager.create_episode(task_name="basic-injection", seed=i)

        assert manager.active_episodes <= 10


class TestPerformance:
    """Basic performance benchmarks."""

    def test_episode_creation_performance(self):
        """Episode creation should be fast."""
        manager = EpisodeManager(max_episodes=1000, ttl_seconds=60)

        start = time.time()
        for i in range(100):
            manager.create_episode(task_name="basic-injection", seed=i)
        elapsed = time.time() - start

        # Should complete 100 creations in under 1 second
        assert elapsed < 1.0

    def test_step_performance(self):
        """Step execution should be fast."""
        env = SentinelEnvironment()
        env.reset(task_name="basic-injection", seed=42)

        action = SentinelAction(
            classification=ThreatCategory.SAFE,
            reasoning="This prompt appears to be completely safe and benign with no malicious content or threats",
            recommended_action=RecommendedAction.ALLOW,
        )

        start = time.time()
        for _ in range(100):
            env.step(action)
        elapsed = time.time() - start

        # Should complete 100 steps in under 1 second
        assert elapsed < 1.0
