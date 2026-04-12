"""Concurrent episode stress tests.

Validates EpisodeManager's ability to handle 100+ simultaneous episodes.
Tests concurrency, TTL cleanup, and memory limits under load.
"""

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from server.app import app
from server.episode_manager import EpisodeManager


@pytest.fixture
def server_client():
    """Create a TestClient with lifespan context."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def episode_manager():
    """Create a fresh EpisodeManager for isolated tests."""
    return EpisodeManager(max_episodes=100, ttl_seconds=60)


class TestConcurrentEpisodes:
    """Test concurrent episode handling."""

    @pytest.mark.asyncio
    async def test_100_concurrent_episodes(self, episode_manager):
        """Test creating and accessing 100 episodes concurrently."""

        async def create_and_access_episode(task_name: str, seed: int) -> dict:
            episode_id, obs = await episode_manager.create_episode(task_name=task_name, seed=seed)
            # Verify we can retrieve it
            env = await episode_manager.get_episode(episode_id)
            assert env is not None
            assert env.episode_id is not None
            return episode_id

        # Create 100 episodes in parallel
        tasks = [
            create_and_access_episode(
                task_name=["basic-injection", "social-engineering", "stealth-exfiltration"][i % 3],
                seed=i,
            )
            for i in range(100)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"{len(exceptions)} episodes failed: {exceptions[:3]}"

        # Verify episode count
        assert episode_manager.active_episodes == 100

    @pytest.mark.asyncio
    async def test_concurrent_episode_isolation(self, episode_manager):
        """Test that concurrent episodes don't interfere with each other."""

        async def run_episode_step(task_name: str, seed: int):
            episode_id, obs = await episode_manager.create_episode(task_name=task_name, seed=seed)
            env = await episode_manager.get_episode(episode_id)
            assert env is not None
            # Verify episode has correct task_name
            assert env.task_name == task_name
            # Take one step
            from models import RecommendedAction, SentinelAction, ThreatCategory

            action = SentinelAction(
                classification=ThreatCategory.SAFE,
                reasoning="Safe prompt analysis with detailed reasoning",
                recommended_action=RecommendedAction.ALLOW,
            )
            obs2, reward, done, info = env.step(action)
            assert obs2.step_number == 2
            return episode_id

        # Run 20 episodes concurrently
        tasks = [run_episode_step("basic-injection", i) for i in range(20)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"{len(exceptions)} episodes failed"
        assert len(results) == 20

    @pytest.mark.asyncio
    async def test_episode_eviction_under_pressure(self, episode_manager):
        """Test that old episodes are evicted when max_episodes is reached."""
        # Create more episodes than max_episodes (100)
        episode_ids = []
        for i in range(120):
            episode_id, _ = await episode_manager.create_episode(task_name="basic-injection", seed=i)
            episode_ids.append(episode_id)

        # Should have evicted some old episodes
        assert episode_manager.active_episodes <= 100

        # First episodes should have been evicted
        early_episodes_exist = []
        for eid in episode_ids[:20]:
            env = await episode_manager.get_episode(eid)
            early_episodes_exist.append(env is not None)

        # At least some early episodes should be gone
        assert not all(early_episodes_exist), "Early episodes were not evicted"


class TestEpisodeTTLCleanup:
    """Test TTL-based episode cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_removes_expired_episodes(self, episode_manager):
        """Test that cleanup removes episodes older than TTL."""
        # Create episodes
        for i in range(10):
            await episode_manager.create_episode(task_name="basic-injection", seed=i)

        assert episode_manager.active_episodes == 10

        # Manually expire episodes by modifying their creation time
        for eid in episode_manager.episode_metadata:
            episode_manager.episode_metadata[eid]["created_at"] = time.time() - 120

        # Run cleanup
        removed = await episode_manager.cleanup_expired()
        assert removed == 10
        assert episode_manager.active_episodes == 0

    @pytest.mark.asyncio
    async def test_background_cleanup_runs_periodically(self):
        """Test that background cleanup task runs and removes expired episodes."""
        manager = EpisodeManager(max_episodes=50, ttl_seconds=1)

        await manager.start_background_cleanup(interval_seconds=1)

        try:
            # Create episodes
            for i in range(10):
                await manager.create_episode(task_name="basic-injection", seed=i)

            assert manager.active_episodes == 10

            # Wait for cleanup to run (TTL is 1 second, cleanup interval is 1 second)
            await asyncio.sleep(3)

            # Episodes should have been cleaned
            assert manager.active_episodes < 10, f"Expected cleanup, but still have {manager.active_episodes} episodes"
        finally:
            await manager.stop_background_cleanup()


class TestEpisodeManagerLimits:
    """Test EpisodeManager capacity limits."""

    @pytest.mark.asyncio
    async def test_max_episodes_limit(self):
        """Test that EpisodeManager respects max_episodes limit."""
        manager = EpisodeManager(max_episodes=10, ttl_seconds=3600)

        # Create 15 episodes
        for i in range(15):
            await manager.create_episode(task_name="basic-injection", seed=i)

        # Should not exceed max_episodes
        assert manager.active_episodes <= 10

    @pytest.mark.asyncio
    async def test_eviction_removes_oldest(self):
        """Test that eviction removes the oldest episodes."""
        manager = EpisodeManager(max_episodes=5, ttl_seconds=3600)

        # Create episodes with known order
        episode_ids = []
        for i in range(8):
            episode_id, _ = await manager.create_episode(task_name="basic-injection", seed=i)
            episode_ids.append(episode_id)
            await asyncio.sleep(0.01)  # Ensure distinct creation times

        # Should have evicted 3 oldest
        assert manager.active_episodes == 5

        # First 3 episodes should be gone
        for eid in episode_ids[:3]:
            env = await manager.get_episode(eid)
            assert env is None, f"Episode {eid} should have been evicted"

        # Last 5 episodes should still exist
        for eid in episode_ids[3:]:
            env = await manager.get_episode(eid)
            assert env is not None, f"Episode {eid} should still exist"


class TestHighConcurrencyStress:
    """High concurrency stress tests."""

    @pytest.mark.asyncio
    async def test_500_rapid_create_access_cycles(self, episode_manager):
        """Test 500 rapid create-access cycles under concurrency."""

        async def cycle(i: int):
            episode_id, obs = await episode_manager.create_episode(task_name="basic-injection", seed=i)
            env = await episode_manager.get_episode(episode_id)
            assert env is not None
            return i

        # Run 500 cycles concurrently in batches to avoid overwhelming
        batch_size = 50
        for batch_start in range(0, 500, batch_size):
            tasks = [cycle(i) for i in range(batch_start, batch_start + batch_size)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            exceptions = [r for r in results if isinstance(r, Exception)]
            assert len(exceptions) == 0, f"Batch {batch_start} had {len(exceptions)} failures"

    @pytest.mark.asyncio
    async def test_concurrent_create_same_episode(self, episode_manager):
        """Test concurrent creation of episodes with same seed (should be independent)."""

        async def create_episode():
            episode_id, obs = await episode_manager.create_episode(task_name="basic-injection", seed=42)
            return episode_id, obs

        # Create 10 episodes with same seed concurrently
        results = await asyncio.gather(*[create_episode() for _ in range(10)])

        # All should have different episode IDs (UUID-based)
        episode_ids = [r[0] for r in results]
        assert len(set(episode_ids)) == 10, "Episode IDs should be unique"

        # Verify all episodes were created successfully
        assert episode_manager.active_episodes == 10
