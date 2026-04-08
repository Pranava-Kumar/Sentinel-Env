"""Tests for the core environment logic."""

import pytest
from models import SentinelAction, ThreatCategory
from server.sentinel_environment import SentinelEnvironment


class TestSentinelEnvironment:
    def test_reset_returns_observation(self):
        env = SentinelEnvironment()
        obs = env.reset(task_name="basic-injection", seed=42)
        assert obs.user_prompt is not None
        assert obs.step_number == 1
        assert obs.max_steps > 0

    def test_step_returns_tuple(self):
        env = SentinelEnvironment()
        env.reset(task_name="basic-injection", seed=42)
        action = SentinelAction(
            classification=ThreatCategory.INJECTION,
            reasoning="Test reasoning",
            recommended_action="block",
        )
        result = env.step(action)
        obs, reward, done, info = result
        assert isinstance(obs, type(env.reset(task_name="basic-injection", seed=42)))
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_state_returns_state(self):
        env = SentinelEnvironment()
        env.reset(task_name="basic-injection", seed=42)
        state = env.state()
        assert state.episode_id is not None
        assert state.task_name == "basic-injection"

    def test_different_seeds_different_attacks(self):
        env1 = SentinelEnvironment()
        env2 = SentinelEnvironment()
        obs1 = env1.reset(task_name="basic-injection", seed=42)
        obs2 = env2.reset(task_name="basic-injection", seed=99)
        assert obs1.user_prompt != obs2.user_prompt

    def test_same_seed_same_attacks(self):
        env1 = SentinelEnvironment()
        env2 = SentinelEnvironment()
        obs1 = env1.reset(task_name="basic-injection", seed=42)
        obs2 = env2.reset(task_name="basic-injection", seed=42)
        assert obs1.user_prompt == obs2.user_prompt

    def test_all_tasks_work(self):
        env = SentinelEnvironment()
        for task in ["basic-injection", "social-engineering", "stealth-exfiltration"]:
            obs = env.reset(task_name=task, seed=42)
            assert obs.user_prompt is not None
            assert obs.attack_metadata.difficulty in ("easy", "medium", "hard")

    def test_episode_ends(self):
        env = SentinelEnvironment()
        env.reset(task_name="basic-injection", seed=42)
        done = False
        for _ in range(20):  # Increased to ensure episode ends
            action = SentinelAction(
                classification=ThreatCategory.SAFE,
                reasoning="This prompt appears to be safe and benign",
                recommended_action="allow",
            )
            obs, reward, done, info = env.step(action)
            if done:
                break
        assert done is True

    def test_grading_works(self):
        env = SentinelEnvironment()
        env.reset(task_name="basic-injection", seed=42)
        for _ in range(12):
            action = SentinelAction(
                classification=ThreatCategory.INJECTION,
                reasoning="Detected as attack for testing",
                recommended_action="block",
            )
            obs, reward, done, info = env.step(action)
            if done:
                break
        grade = env.get_episode_grade()
        assert "score" in grade
        assert 0.0 <= grade["score"] <= 1.0
