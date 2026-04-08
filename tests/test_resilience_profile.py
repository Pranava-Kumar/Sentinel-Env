"""Tests for the resilience profile generator."""
import pytest
from server.resilience_profile import generate_resilience_profile


class TestGenerateResilienceProfile:
    def test_empty_results(self):
        """Test profile with no results."""
        profile = generate_resilience_profile([], "basic-injection", 42)
        assert profile["task_name"] == "basic-injection"
        assert profile["seed"] == 42
        assert profile["attack_type_breakdown"] == {}
        assert profile["overall_detection_rate"] == 0.0

    def test_single_attack_detected(self):
        """Test profile with one correctly detected attack."""
        results = [
            {
                "attack_type": "direct_override",
                "is_correct": True,
                "is_partial": False,
                "is_false_positive": False,
                "is_safe_prompt": False,
            }
        ]
        profile = generate_resilience_profile(results, "basic-injection", 42)
        assert profile["attack_type_breakdown"]["direct_override"]["detected"] == 1
        assert profile["attack_type_breakdown"]["direct_override"]["detection_rate"] == 1.0
        assert profile["overall_detection_rate"] == 1.0

    def test_single_attack_missed(self):
        """Test profile with one missed attack."""
        results = [
            {
                "attack_type": "direct_override",
                "is_correct": False,
                "is_partial": False,
                "is_false_positive": False,
                "is_safe_prompt": False,
            }
        ]
        profile = generate_resilience_profile(results, "basic-injection", 42)
        assert profile["attack_type_breakdown"]["direct_override"]["missed"] == 1
        assert profile["attack_type_breakdown"]["direct_override"]["detection_rate"] == 0.0

    def test_mixed_results(self):
        """Test profile with mixed correct/incorrect/partial."""
        results = [
            {"attack_type": "injection", "is_correct": True, "is_partial": False, "is_false_positive": False, "is_safe_prompt": False},
            {"attack_type": "injection", "is_correct": False, "is_partial": True, "is_false_positive": False, "is_safe_prompt": False},
            {"attack_type": "jailbreak", "is_correct": False, "is_partial": False, "is_false_positive": False, "is_safe_prompt": False},
        ]
        profile = generate_resilience_profile(results, "basic-injection", 42)
        assert profile["attack_type_breakdown"]["injection"]["detected"] == 1
        assert profile["attack_type_breakdown"]["injection"]["partial"] == 1
        assert profile["attack_type_breakdown"]["jailbreak"]["missed"] == 1

    def test_false_positives(self):
        """Test profile correctly tracks false positives."""
        results = [
            {"attack_type": "none", "is_correct": False, "is_partial": False, "is_false_positive": True, "is_safe_prompt": True},
            {"attack_type": "injection", "is_correct": True, "is_partial": False, "is_false_positive": False, "is_safe_prompt": False},
        ]
        profile = generate_resilience_profile(results, "basic-injection", 42)
        assert profile["overall_false_positive_rate"] > 0.0

    def test_resilience_score_calculation(self):
        """Test resilience score is in valid range."""
        results = [
            {"attack_type": "injection", "is_correct": True, "is_partial": False, "is_false_positive": False, "is_safe_prompt": False},
            {"attack_type": "jailbreak", "is_correct": True, "is_partial": False, "is_false_positive": False, "is_safe_prompt": False},
        ]
        profile = generate_resilience_profile(results, "basic-injection", 42)
        assert 0.0 <= profile["resilience_score"] <= 1.0

    def test_multiple_attack_types(self):
        """Test breakdown by multiple attack types."""
        results = [
            {"attack_type": "direct_override", "is_correct": True, "is_partial": False, "is_false_positive": False, "is_safe_prompt": False},
            {"attack_type": "dan_style", "is_correct": False, "is_partial": False, "is_false_positive": False, "is_safe_prompt": False},
            {"attack_type": "shell_command", "is_correct": True, "is_partial": False, "is_false_positive": False, "is_safe_prompt": False},
        ]
        profile = generate_resilience_profile(results, "basic-injection", 42)
        assert len(profile["attack_type_breakdown"]) == 3

    def test_ignores_none_attack_type(self):
        """Test that 'none' attack type is excluded from breakdown."""
        results = [
            {"attack_type": "none", "is_correct": True, "is_partial": False, "is_false_positive": False, "is_safe_prompt": True},
        ]
        profile = generate_resilience_profile(results, "basic-injection", 42)
        assert "none" not in profile["attack_type_breakdown"]
