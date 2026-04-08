"""Episode lifecycle management for concurrent support.

Manages multiple simultaneous episodes with session-based tracking.
"""

import uuid
import time
from typing import Dict, Optional, Any
from server.sentinel_environment import SentinelEnvironment


class EpisodeManager:
    """Manages multiple concurrent episodes.
    
    Each episode is identified by a unique session ID and can be
    independently reset, stepped, and queried.
    """
    
    def __init__(self, max_episodes: int = 1000, ttl_seconds: int = 3600):
        self.max_episodes = max_episodes
        self.ttl_seconds = ttl_seconds
        self.episodes: Dict[str, SentinelEnvironment] = {}
        self.episode_metadata: Dict[str, dict] = {}
    
    def create_episode(self, task_name: str, seed: int) -> str:
        """Create a new episode and return its ID.
        
        Args:
            task_name: Task name for the episode
            seed: Random seed
            
        Returns:
            Episode ID (UUID string)
        """
        # Evict old episodes if at capacity
        if len(self.episodes) >= self.max_episodes:
            self._evict_old_episodes()
        
        episode_id = str(uuid.uuid4())
        env = SentinelEnvironment()
        env.reset(task_name=task_name, seed=seed)
        
        self.episodes[episode_id] = env
        self.episode_metadata[episode_id] = {
            "created_at": time.time(),
            "task_name": task_name,
            "seed": seed,
        }
        
        return episode_id
    
    def get_episode(self, episode_id: str) -> Optional[SentinelEnvironment]:
        """Get an episode by ID.
        
        Args:
            episode_id: Episode ID
            
        Returns:
            SentinelEnvironment instance or None if not found
        """
        episode = self.episodes.get(episode_id)
        if episode:
            self.episode_metadata[episode_id]["last_accessed"] = time.time()
        return episode
    
    def remove_episode(self, episode_id: str) -> bool:
        """Remove an episode.
        
        Args:
            episode_id: Episode ID
            
        Returns:
            True if removed, False if not found
        """
        if episode_id in self.episodes:
            del self.episodes[episode_id]
            del self.episode_metadata[episode_id]
            return True
        return False
    
    def cleanup_expired(self) -> int:
        """Remove episodes older than TTL.
        
        Returns:
            Number of episodes removed
        """
        now = time.time()
        expired = [
            eid for eid, meta in self.episode_metadata.items()
            if now - meta["created_at"] > self.ttl_seconds
        ]
        
        for episode_id in expired:
            self.remove_episode(episode_id)
        
        return len(expired)
    
    def _evict_old_episodes(self):
        """Evict oldest episodes by creation time."""
        if not self.episode_metadata:
            return
        
        # Sort by creation time and remove oldest 10%
        sorted_episodes = sorted(
            self.episode_metadata.items(),
            key=lambda x: x[1]["created_at"],
        )
        num_to_evict = max(1, len(sorted_episodes) // 10)
        
        for episode_id, _ in sorted_episodes[:num_to_evict]:
            self.remove_episode(episode_id)
    
    @property
    def active_episodes(self) -> int:
        """Number of active episodes."""
        return len(self.episodes)
