"""Tests for the episode manager module."""

import asyncio

import pytest

from server.episode_manager import EpisodeManager


class TestEpisodeManager:
    @pytest.fixture
    def manager(self):
        return EpisodeManager(max_episodes=5, ttl_seconds=1)

    @pytest.mark.asyncio
    async def test_create_episode_returns_id_and_observation(self, manager):
        """Creating episode should return tuple of (id, observation)."""
        episode_id, observation = await manager.create_episode("basic-injection", seed=42)
        assert episode_id is not None
        assert isinstance(episode_id, str)
        assert len(episode_id) > 0
        assert observation is not None
        assert observation.step_number == 1

    @pytest.mark.asyncio
    async def test_get_episode_returns_env(self, manager):
        """Getting episode by ID should return valid environment."""
        episode_id, _ = await manager.create_episode("basic-injection", seed=42)
        env = await manager.get_episode(episode_id)
        assert env is not None
        assert env.episode_id is not None

    @pytest.mark.asyncio
    async def test_get_nonexistent_episode_returns_none(self, manager):
        """Getting non-existent episode should return None."""
        env = await manager.get_episode("nonexistent-id")
        assert env is None

    @pytest.mark.asyncio
    async def test_remove_episode(self, manager):
        """Removing episode should succeed and return True."""
        episode_id, _ = await manager.create_episode("basic-injection", seed=42)
        assert await manager.remove_episode(episode_id) is True
        assert await manager.get_episode(episode_id) is None

    @pytest.mark.asyncio
    async def test_remove_nonexistent_episode(self, manager):
        """Removing non-existent episode should return False."""
        assert await manager.remove_episode("nonexistent-id") is False

    @pytest.mark.asyncio
    async def test_max_episodes_eviction(self, manager):
        """Test that old episodes are evicted when at capacity."""
        episode_ids = []
        for i in range(6):
            eid, _ = await manager.create_episode("basic-injection", seed=i)
            episode_ids.append(eid)

        assert manager.active_episodes <= 5

    @pytest.mark.asyncio
    async def test_cleanup_expired_episodes(self, manager):
        """Expired episodes should be cleaned up."""
        await manager.create_episode("basic-injection", seed=42)
        await asyncio.sleep(1.1)
        removed = await manager.cleanup_expired()
        assert removed == 1
        assert manager.active_episodes == 0

    @pytest.mark.asyncio
    async def test_active_episodes_count(self, manager):
        """Active episodes count should match creations."""
        await manager.create_episode("basic-injection", seed=42)
        await manager.create_episode("social-engineering", seed=99)
        assert manager.active_episodes == 2

    @pytest.mark.asyncio
    async def test_episode_updates_last_accessed(self, manager):
        """Getting episode should update last_accessed timestamp."""
        episode_id, _ = await manager.create_episode("basic-injection", seed=42)
        await asyncio.sleep(0.1)
        await manager.get_episode(episode_id)
        assert "last_accessed" in manager.episode_metadata[episode_id]

    @pytest.mark.asyncio
    async def test_multiple_episodes_independent(self, manager):
        """Multiple episodes should be independent."""
        id1, _ = await manager.create_episode("basic-injection", seed=42)
        id2, _ = await manager.create_episode("basic-injection", seed=99)

        env1 = await manager.get_episode(id1)
        env2 = await manager.get_episode(id2)

        assert env1 is not None
        assert env2 is not None
        assert env1.episode_id != env2.episode_id
