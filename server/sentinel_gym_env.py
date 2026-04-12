"""Gymnasium-compatible environment wrapper for Sentinel.

Wraps the SentinelEnvironment to work with standard RL frameworks
like Stable Baselines3, Ray RLlib, etc.
"""

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from models import RecommendedAction, SentinelAction, SentinelObservation, ThreatCategory
from server.sentinel_environment import SentinelEnvironment


class SentinelGymEnv(gym.Env):
    """Gymnasium wrapper for Sentinel Environment.

    Converts the Sentinel RL environment to standard Gym interface:
    - Observation: Text embedding (384-dim vector)
    - Action: Threat category index (0-15)
    - Reward: Shaped reward from environment
    - Done: Episode termination
    """

    metadata: ClassVar[dict] = {"render_modes": ["human"]}

    def __init__(
        self,
        task_name: str = "basic-injection",
        embedder=None,  # TextEmbedder instance
        reward_shaper=None,  # AdvancedRewardShaper instance
        render_mode: str | None = None,
    ):
        super().__init__()

        self.task_name = task_name
        self.embedder = embedder
        self.reward_shaper = reward_shaper
        self.render_mode = render_mode

        # Core environment
        self.env = SentinelEnvironment()

        # Action space: 16 threat categories
        self.action_space = spaces.Discrete(len(ThreatCategory))

        # Observation space: embedding dimension (384)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(384,), dtype=np.float32)

        # Episode tracking
        self.current_obs: SentinelObservation | None = None
        self.ground_truth: str | None = None
        self.step_count = 0

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset environment and return initial observation."""
        super().reset(seed=seed)

        # Reset core environment
        obs = self.env.reset(task_name=self.task_name, seed=seed or 42)

        # Embed observation
        if self.embedder:
            embedding = self.embedder.encode(obs.user_prompt)
            if hasattr(embedding, "squeeze"):
                embedding = embedding.squeeze(0)
            if hasattr(embedding, "detach"):
                embedding = embedding.detach().cpu().numpy()
        else:
            # Fallback: random embedding
            embedding = np.random.randn(384).astype(np.float32)

        # Store ground truth for reward shaping
        self.ground_truth = obs.attack_metadata.ground_truth
        self.current_obs = obs
        self.step_count = 0

        info = {
            "ground_truth": self.ground_truth,
            "attack_type": obs.attack_metadata.attack_type,
            "is_attack": not obs.is_safe_prompt,
        }

        return embedding.astype(np.float32), info

    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute action and return (observation, reward, terminated, truncated, info)."""
        # Convert action index to SentinelAction
        classification = list(ThreatCategory)[action_idx]

        assert self.current_obs is not None, "Must call reset() before step()"
        is_safe = self.current_obs.is_safe_prompt

        if classification == ThreatCategory.SAFE or is_safe:
            recommended_action = RecommendedAction.ALLOW
            reasoning = "This prompt appears safe based on current analysis"
        else:
            recommended_action = RecommendedAction.BLOCK
            reasoning = f"Detected potential {classification.value} pattern requiring investigation"

        action = SentinelAction(
            classification=classification,
            reasoning=reasoning,
            recommended_action=recommended_action,
            safe_alternative=None,
        )

        # Step environment
        next_obs, reward, done, info = self.env.step(action)
        self.step_count += 1

        # Apply reward shaping if available
        if self.reward_shaper:
            shaped_reward = self.reward_shaper.shape_reward(
                info.get("step_result", {}),
                action,
                self.ground_truth,
                confidence=0.5,  # Default confidence
            )
            reward = shaped_reward

        # Embed next observation
        if self.embedder:
            next_embedding = self.embedder.encode(next_obs.user_prompt)
            if hasattr(next_embedding, "squeeze"):
                next_embedding = next_embedding.squeeze(0)
            if hasattr(next_embedding, "detach"):
                next_embedding = next_embedding.detach().cpu().numpy()
        else:
            next_embedding = np.random.randn(384).astype(np.float32)

        # Update state
        self.current_obs = next_obs
        self.ground_truth = next_obs.attack_metadata.ground_truth

        # Prepare info dict
        info.update(
            {
                "ground_truth": self.ground_truth,
                "attack_type": next_obs.attack_metadata.attack_type,
                "step_number": self.step_count,
            }
        )

        # Gymnasium requires both terminated and truncated
        terminated = done
        truncated = False  # Episode doesn't truncate, only terminates naturally

        return next_embedding.astype(np.float32), reward, terminated, truncated, info

    def render(self):
        """Render environment (not implemented)."""
        if self.render_mode == "human":
            print(f"Step {self.step_count}")
            print(f"Ground truth: {self.ground_truth}")
            print(f"Observation: {self.current_obs.user_prompt[:100]}...")

    def close(self):
        """Clean up resources."""
        pass
