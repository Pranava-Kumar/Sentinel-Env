"""Tests for the reward shaper module."""

from models import SentinelAction, ThreatCategory
from server.reward_shaper import compute_reward


class TestComputeReward:
    def test_correct_classification_reward(self):
        action = SentinelAction(
            classification=ThreatCategory.INJECTION,
            reasoning="Direct override attempt detected",
            recommended_action="block",
        )
        reward = compute_reward(action, "injection", 1, 12)
        assert reward >= 0.5

    def test_missed_attack_penalty(self):
        action = SentinelAction(
            classification=ThreatCategory.SAFE,
            reasoning="Looks normal to me",
            recommended_action="allow",
        )
        reward = compute_reward(action, "injection", 1, 12)
        assert reward < 0.3

    def test_false_positive_penalty(self):
        action = SentinelAction(
            classification=ThreatCategory.INJECTION,
            reasoning="Seems suspicious",
            recommended_action="block",
        )
        reward = compute_reward(action, "safe", 1, 12)
        assert reward < 0.5

    def test_reward_bounds(self):
        """All rewards must be in [0.0, 1.0]."""
        for pred in ThreatCategory:
            for actual in [
                "safe",
                "injection",
                "jailbreak",
                "command_injection",
                "authority_impersonation",
                "encoded_payload",
            ]:
                action = SentinelAction(
                    classification=pred,
                    reasoning="Test reasoning with enough words to get a decent score here",
                    recommended_action="block",
                )
                reward = compute_reward(action, actual, 1, 12)
                assert 0.0 <= reward <= 1.0, f"Reward {reward} out of range for {pred} vs {actual}"
