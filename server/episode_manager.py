"""Episode lifecycle management for concurrent support.

Manages multiple simultaneous episodes with session-based tracking.
Thread-safe via asyncio.Lock for all mutation operations.
"""

import asyncio
import contextlib
import time
import uuid
from typing import Any

from server.sentinel_environment import SentinelEnvironment


class EpisodeManager:
    """Manages multiple concurrent episodes.

    Each episode is identified by a unique session ID and can be
    independently reset, stepped, and queried.
    """

    def __init__(self, max_episodes: int = 1000, ttl_seconds: int = 3600):
        self.max_episodes = max_episodes
        self.ttl_seconds = ttl_seconds
        self.episodes: dict[str, SentinelEnvironment] = {}
        self.episode_metadata: dict[str, dict] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None

    async def create_episode(self, task_name: str, seed: int) -> tuple[str, Any]:
        """Create a new episode and return its ID with the initial observation.

        Args:
            task_name: Task name for the episode
            seed: Random seed

        Returns:
            Tuple of (episode_id, initial_observation)
        """
        # Create environment and reset OUTSIDE the lock to avoid blocking
        # other episode operations during the potentially slow reset()
        env = SentinelEnvironment()
        observation = env.reset(task_name=task_name, seed=seed)

        episode_id = str(uuid.uuid4())

        async with self._lock:
            # Evict old episodes if at capacity
            if len(self.episodes) >= self.max_episodes:
                await self._evict_old_episodes()

            self.episodes[episode_id] = env
            self.episode_metadata[episode_id] = {
                "created_at": time.time(),
                "task_name": task_name,
                "seed": seed,
            }

            return episode_id, observation

    async def get_episode(self, episode_id: str) -> SentinelEnvironment | None:
        """Get an episode by ID.

        Args:
            episode_id: Episode ID

        Returns:
            SentinelEnvironment instance or None if not found
        """
        async with self._lock:
            episode = self.episodes.get(episode_id)
            if episode and episode_id in self.episode_metadata:
                self.episode_metadata[episode_id]["last_accessed"] = time.time()
            return episode

    async def remove_episode(self, episode_id: str) -> bool:
        """Remove an episode.

        Args:
            episode_id: Episode ID

        Returns:
            True if removed, False if not found
        """
        async with self._lock:
            if episode_id in self.episodes:
                del self.episodes[episode_id]
                self.episode_metadata.pop(episode_id, None)
                return True
            return False

    async def cleanup_expired(self) -> int:
        """Remove episodes older than TTL.

        Returns:
            Number of episodes removed
        """
        async with self._lock:
            now = time.time()
            expired = [
                eid for eid, meta in self.episode_metadata.items() if now - meta["created_at"] > self.ttl_seconds
            ]

            for episode_id in expired:
                del self.episodes[episode_id]
                del self.episode_metadata[episode_id]

            return len(expired)

    async def _evict_old_episodes(self):
        """Evict oldest episodes by creation time. Must be called with lock held."""
        if not self.episode_metadata:
            return

        # Sort by creation time and remove oldest 10%
        sorted_episodes = sorted(
            self.episode_metadata.items(),
            key=lambda x: x[1]["created_at"],
        )
        num_to_evict = max(1, len(sorted_episodes) // 10)

        for episode_id, _ in sorted_episodes[:num_to_evict]:
            del self.episodes[episode_id]
            del self.episode_metadata[episode_id]

    async def start_background_cleanup(self, interval_seconds: int = 300) -> None:
        """Start a periodic background cleanup task.

        Args:
            interval_seconds: How often to run cleanup (default 5 minutes)
        """

        async def _cleanup_loop():
            while True:
                await asyncio.sleep(interval_seconds)
                try:
                    removed = await self.cleanup_expired()
                    if removed:
                        from structlog import get_logger

                        get_logger().info("Background cleanup removed expired episodes", removed=removed)
                except Exception:
                    pass  # Never let cleanup crash the loop

        self._cleanup_task = asyncio.create_task(_cleanup_loop(), name="episode-cleanup")

    async def stop_background_cleanup(self) -> None:
        """Stop the periodic background cleanup task."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task
            self._cleanup_task = None

    @property
    def active_episodes(self) -> int:
        """Number of active episodes."""
        return len(self.episodes)
