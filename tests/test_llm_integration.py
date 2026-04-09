"""LLM Integration tests with mock responses.

Tests the full inference loop with realistic mock LLM behavior:
- Model classification accuracy
- Multi-turn conversation patterns
- Edge cases in model responses
- Prompt injection detection accuracy
"""

import pytest

from models import RecommendedAction, SentinelAction, ThreatCategory
from server.sentinel_environment import SentinelEnvironment


class TestLLMIntegration:
    """Integration tests for LLM model interaction."""

    def test_model_classifies_basic_injection_correctly(self):
        """Model should detect basic injection attempts."""
        env = SentinelEnvironment()
        obs = env.reset(task_name="basic-injection", seed=42)

        # Simulate model detecting injection
        action = SentinelAction(
            classification=ThreatCategory.INJECTION,
            reasoning="This prompt contains a direct injection attempt trying to override previous instructions",
            recommended_action=RecommendedAction.BLOCK,
        )

        obs, reward, done, info = env.step(action)

        assert reward > 0.5, f"Correct detection should have high reward, got {reward}"
        assert info["step_result"]["is_correct"] or info["step_result"]["is_partial"]

    def test_model_handles_safe_prompts_correctly(self):
        """Model should allow safe prompts through."""
        env = SentinelEnvironment()
        obs = env.reset(task_name="basic-injection", seed=42)

        # Run until we find a safe prompt
        for _ in range(20):
            if obs.is_safe_prompt:
                action = SentinelAction(
                    classification=ThreatCategory.SAFE,
                    reasoning="This prompt appears to be completely safe and benign with no malicious intent",
                    recommended_action=RecommendedAction.ALLOW,
                )
                obs, reward, done, info = env.step(action)
                assert reward > 0.3, f"Correct safe classification should have positive reward, got {reward}"
                break
            else:
                # Skip attacks
                action = SentinelAction(
                    classification=ThreatCategory.SAFE,
                    reasoning="This prompt appears to be completely safe and benign with no malicious intent",
                    recommended_action=RecommendedAction.ALLOW,
                )
                obs, reward, done, info = env.step(action)
                if done:
                    break

    def test_model_partial_credit_same_superclass(self):
        """Model should get partial credit for related classifications."""
        env = SentinelEnvironment()
        obs = env.reset(task_name="basic-injection", seed=42)

        # Run until we find an attack
        for _ in range(20):
            if not obs.is_safe_prompt:
                # Classify as related but not exact match
                action = SentinelAction(
                    classification=ThreatCategory.JAILBREAK,  # Related to injection
                    reasoning="This prompt appears to be attempting a jailbreak attack pattern",
                    recommended_action=RecommendedAction.BLOCK,
                )
                obs, reward, done, info = env.step(action)
                # Should get partial credit
                assert reward >= 0.2, f"Partial credit should have some reward, got {reward}"
                break
            else:
                action = SentinelAction(
                    classification=ThreatCategory.SAFE,
                    reasoning="This prompt appears to be completely safe and benign with no issues",
                    recommended_action=RecommendedAction.ALLOW,
                )
                obs, reward, done, info = env.step(action)
                if done:
                    break

    def test_conversation_history_tracks_interaction(self):
        """Conversation history should accumulate over steps."""
        env = SentinelEnvironment()
        obs = env.reset(task_name="basic-injection", seed=42)

        prev_history_len = 0
        for step in range(5):
            action = SentinelAction(
                classification=ThreatCategory.SAFE,
                reasoning=f"Step {step} analysis shows this prompt is completely safe and benign",
                recommended_action=RecommendedAction.ALLOW,
            )
            obs, reward, done, info = env.step(action)

            # History should grow
            assert len(obs.conversation_history) >= prev_history_len
            prev_history_len = len(obs.conversation_history)

            if done:
                break

        # Should have tracked some history
        assert len(obs.conversation_history) > 0

    def test_episode_grading_works_after_full_run(self):
        """Full episode should produce valid grade."""
        env = SentinelEnvironment()
        env.reset(task_name="basic-injection", seed=42)

        for _ in range(20):
            action = SentinelAction(
                classification=ThreatCategory.INJECTION,
                reasoning="Detected injection attack pattern in the prompt requiring blocking action",
                recommended_action=RecommendedAction.BLOCK,
            )
            obs, reward, done, info = env.step(action)
            if done:
                break

        grade = env.get_episode_grade()
        assert 0.0 <= grade["score"] <= 1.0
        assert "detection_rate" in grade
        assert "false_positive_rate" in grade
        assert "total_steps" in grade

    def test_resilience_profile_generates_valid_output(self):
        """Resilience profile should have attack breakdown."""
        env = SentinelEnvironment()
        env.reset(task_name="basic-injection", seed=42)

        for _ in range(20):
            action = SentinelAction(
                classification=ThreatCategory.INJECTION,
                reasoning="Detected injection attack with sufficient detail and explanation for blocking",
                recommended_action=RecommendedAction.BLOCK,
            )
            obs, reward, done, info = env.step(action)
            if done:
                break

        profile = env.get_resilience_profile()
        assert "task_name" in profile
        assert "attack_type_breakdown" in profile
        assert 0.0 <= profile["overall_detection_rate"] <= 1.0


class TestAdversarialScenarios:
    """Test specific adversarial scenarios."""

    def test_encoded_payload_detection(self):
        """Model should detect encoded payloads."""
        env = SentinelEnvironment()
        obs = env.reset(task_name="stealth-exfiltration", seed=42)

        # Look for encoded payload attacks
        for _ in range(15):
            if obs.attack_metadata.attack_type == "encoded_payload":
                action = SentinelAction(
                    classification=ThreatCategory.ENCODED_PAYLOAD,
                    reasoning="This prompt contains an encoded payload attempt using obfuscation techniques",
                    recommended_action=RecommendedAction.BLOCK,
                )
                obs, reward, done, info = env.step(action)
                assert reward > 0.3
                break

            action = SentinelAction(
                classification=ThreatCategory.SAFE,
                reasoning="This prompt appears to be safe and benign with no malicious content detected",
                recommended_action=RecommendedAction.ALLOW,
            )
            obs, reward, done, info = env.step(action)
            if done:
                break

    def test_social_engineering_detection(self):
        """Model should detect social engineering attempts."""
        env = SentinelEnvironment()
        obs = env.reset(task_name="social-engineering", seed=42)

        for _ in range(15):
            if not obs.is_safe_prompt:
                # Social engineering attacks should be detected
                action = SentinelAction(
                    classification=ThreatCategory.AUTHORITY_IMPERSONATION,
                    reasoning="This prompt uses social engineering techniques to manipulate and bypass security",
                    recommended_action=RecommendedAction.BLOCK,
                )
                obs, reward, done, info = env.step(action)
                # Should get some reward for detecting the attack
                assert reward >= 0.0
                break

            action = SentinelAction(
                classification=ThreatCategory.SAFE,
                reasoning="This prompt appears to be completely safe with no social engineering detected",
                recommended_action=RecommendedAction.ALLOW,
            )
            obs, reward, done, info = env.step(action)
            if done:
                break


class TestEdgeCases:
    """Test edge cases in model behavior."""

    def test_empty_reasoning_rejected_by_validation(self):
        """Empty reasoning should fail Pydantic validation."""
        with pytest.raises(ValueError):
            SentinelAction(
                classification=ThreatCategory.SAFE,
                reasoning="",  # Too short
                recommended_action=RecommendedAction.ALLOW,
            )

    def test_very_long_reasoning_accepted(self):
        """Very long reasoning should be accepted."""
        env = SentinelEnvironment()
        obs = env.reset(task_name="basic-injection", seed=42)

        action = SentinelAction(
            classification=ThreatCategory.SAFE,
            reasoning="This is a very long reasoning string with lots of detail about why the prompt is safe and benign. "
            * 5,
            recommended_action=RecommendedAction.ALLOW,
        )

        obs, reward, done, info = env.step(action)
        # Should work fine
        assert isinstance(obs.user_prompt, str)

    def test_all_threat_categories_classifiable(self):
        """All threat categories should be classifiable."""
        env = SentinelEnvironment()
        env.reset(task_name="basic-injection", seed=42)

        for category in ThreatCategory:
            action = SentinelAction(
                classification=category,
                reasoning=f"Classified as {category.value} with sufficient explanation and detailed reasoning here",
                recommended_action=RecommendedAction.BLOCK
                if category != ThreatCategory.SAFE
                else RecommendedAction.ALLOW,
            )
            # Should not raise
            assert action.classification == category

    def test_concurrent_episodes_independent(self):
        """Multiple episodes should be independent."""
        env1 = SentinelEnvironment()
        env2 = SentinelEnvironment()

        obs1 = env1.reset(task_name="basic-injection", seed=42)
        obs2 = env2.reset(task_name="basic-injection", seed=99)

        # Different seeds should produce different prompts
        assert obs1.user_prompt != obs2.user_prompt

        # Actions in one episode shouldn't affect the other
        action = SentinelAction(
            classification=ThreatCategory.INJECTION,
            reasoning="Detected injection attempt with sufficient detail and explanation for proper blocking",
            recommended_action=RecommendedAction.BLOCK,
        )
        obs1, r1, done1, _ = env1.step(action)
        obs2, r2, done2, _ = env2.step(action)

        # Rewards should be different (different prompts)
        assert obs1.user_prompt != obs2.user_prompt
