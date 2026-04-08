"""Tests for the episode manager module."""

import pytest
import time
from server.episode_manager import EpisodeManager


class TestEpisodeManager:
    @pytest.fixture
    def manager(self):
        return EpisodeManager(max_episodes=5, ttl_seconds=1)

    def test_create_episode_returns_id_and_observation(self, manager):
        """Creating episode should return tuple of (id, observation)."""
        episode_id, observation = manager.create_episode("basic-injection", seed=42)
        assert episode_id is not None
        assert isinstance(episode_id, str)
        assert len(episode_id) > 0
        assert observation is not None
        assert observation.step_number == 1

    def test_get_episode_returns_env(self, manager):
        """Getting episode by ID should return valid environment."""
        episode_id, _ = manager.create_episode("basic-injection", seed=42)
        env = manager.get_episode(episode_id)
        assert env is not None
        assert env.episode_id is not None

    def test_get_nonexistent_episode_returns_none(self, manager):
        """Getting non-existent episode should return None."""
        env = manager.get_episode("nonexistent-id")
        assert env is None

    def test_remove_episode(self, manager):
        """Removing episode should succeed and return True."""
        episode_id, _ = manager.create_episode("basic-injection", seed=42)
        assert manager.remove_episode(episode_id) is True
        assert manager.get_episode(episode_id) is None

    def test_remove_nonexistent_episode(self, manager):
        """Removing non-existent episode should return False."""
        assert manager.remove_episode("nonexistent-id") is False

    def test_max_episodes_eviction(self, manager):
        """Test that old episodes are evicted when at capacity."""
        episode_ids = []
        for i in range(6):
            eid, _ = manager.create_episode("basic-injection", seed=i)
            episode_ids.append(eid)

        assert manager.active_episodes <= 5

    def test_cleanup_expired_episodes(self, manager):
        """Expired episodes should be cleaned up."""
        manager.create_episode("basic-injection", seed=42)
        time.sleep(1.1)
        removed = manager.cleanup_expired()
        assert removed == 1
        assert manager.active_episodes == 0

    def test_active_episodes_count(self, manager):
        """Active episodes count should match creations."""
        manager.create_episode("basic-injection", seed=42)
        manager.create_episode("social-engineering", seed=99)
        assert manager.active_episodes == 2

    def test_episode_updates_last_accessed(self, manager):
        """Getting episode should update last_accessed timestamp."""
        episode_id, _ = manager.create_episode("basic-injection", seed=42)
        time.sleep(0.1)
        manager.get_episode(episode_id)
        assert "last_accessed" in manager.episode_metadata[episode_id]

    def test_multiple_episodes_independent(self, manager):
        """Multiple episodes should be independent."""
        id1, _ = manager.create_episode("basic-injection", seed=42)
        id2, _ = manager.create_episode("basic-injection", seed=99)

        env1 = manager.get_episode(id1)
        env2 = manager.get_episode(id2)

        assert env1 is not None
        assert env2 is not None
        assert env1.episode_id != env2.episode_id
