"""Tests for the resilience profile generator."""

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
            {
                "attack_type": "injection",
                "is_correct": True,
                "is_partial": False,
                "is_false_positive": False,
                "is_safe_prompt": False,
            },
            {
                "attack_type": "injection",
                "is_correct": False,
                "is_partial": True,
                "is_false_positive": False,
                "is_safe_prompt": False,
            },
            {
                "attack_type": "jailbreak",
                "is_correct": False,
                "is_partial": False,
                "is_false_positive": False,
                "is_safe_prompt": False,
            },
        ]
        profile = generate_resilience_profile(results, "basic-injection", 42)
        assert profile["attack_type_breakdown"]["injection"]["detected"] == 1
        assert profile["attack_type_breakdown"]["injection"]["partial"] == 1
        assert profile["attack_type_breakdown"]["jailbreak"]["missed"] == 1

    def test_false_positives(self):
        """Test profile correctly tracks false positives."""
        results = [
            {
                "attack_type": "none",
                "is_correct": False,
                "is_partial": False,
                "is_false_positive": True,
                "is_safe_prompt": True,
            },
            {
                "attack_type": "injection",
                "is_correct": True,
                "is_partial": False,
                "is_false_positive": False,
                "is_safe_prompt": False,
            },
        ]
        profile = generate_resilience_profile(results, "basic-injection", 42)
        assert profile["overall_false_positive_rate"] > 0.0

    def test_resilience_score_calculation(self):
        """Test resilience score is in valid range."""
        results = [
            {
                "attack_type": "injection",
                "is_correct": True,
                "is_partial": False,
                "is_false_positive": False,
                "is_safe_prompt": False,
            },
            {
                "attack_type": "jailbreak",
                "is_correct": True,
                "is_partial": False,
                "is_false_positive": False,
                "is_safe_prompt": False,
            },
        ]
        profile = generate_resilience_profile(results, "basic-injection", 42)
        assert 0.0 <= profile["resilience_score"] <= 1.0

    def test_multiple_attack_types(self):
        """Test breakdown by multiple attack types."""
        results = [
            {
                "attack_type": "direct_override",
                "is_correct": True,
                "is_partial": False,
                "is_false_positive": False,
                "is_safe_prompt": False,
            },
            {
                "attack_type": "dan_style",
                "is_correct": False,
                "is_partial": False,
                "is_false_positive": False,
                "is_safe_prompt": False,
            },
            {
                "attack_type": "shell_command",
                "is_correct": True,
                "is_partial": False,
                "is_false_positive": False,
                "is_safe_prompt": False,
            },
        ]
        profile = generate_resilience_profile(results, "basic-injection", 42)
        assert len(profile["attack_type_breakdown"]) == 3

    def test_ignores_none_attack_type(self):
        """Test that 'none' attack type is excluded from breakdown."""
        results = [
            {
                "attack_type": "none",
                "is_correct": True,
                "is_partial": False,
                "is_false_positive": False,
                "is_safe_prompt": True,
            },
        ]
        profile = generate_resilience_profile(results, "basic-injection", 42)
        assert "none" not in profile["attack_type_breakdown"]


class TestResilienceProfileEdgeCases:
    def test_empty_results_zeroed_profile(self):
        """Empty results should return zeroed profile."""
        profile = generate_resilience_profile([], "basic-injection", 42)
        assert profile["overall_detection_rate"] == 0.0
        assert profile["overall_false_positive_rate"] == 0.0
        assert profile["resilience_score"] == 0.4  # 0.6*0 + 0.4*(1-0) = 0.4
        assert profile["task_name"] == "basic-injection"
        assert profile["seed"] == 42

    def test_all_correct_detections(self):
        """All correct detections should yield high score."""
        results = [
            {
                "is_correct": True,
                "is_partial": False,
                "is_false_positive": False,
                "is_safe_prompt": False,
                "attack_type": "direct_override",
            }
            for _ in range(5)
        ]
        profile = generate_resilience_profile(results, "basic-injection", 42)
        assert profile["overall_detection_rate"] == 1.0
        assert profile["resilience_score"] > 0.8

    def test_all_missed_attacks(self):
        """All missed attacks should yield low score."""
        results = [
            {
                "is_correct": False,
                "is_partial": False,
                "is_false_positive": False,
                "is_safe_prompt": False,
                "attack_type": "direct_override",
            }
            for _ in range(5)
        ]
        profile = generate_resilience_profile(results, "basic-injection", 42)
        assert profile["overall_detection_rate"] == 0.0

    def test_mixed_attack_types(self):
        """Mixed attack types should be tracked separately."""
        results = [
            {
                "is_correct": True,
                "is_partial": False,
                "is_false_positive": False,
                "is_safe_prompt": False,
                "attack_type": "direct_override",
            },
            {
                "is_correct": False,
                "is_partial": True,
                "is_false_positive": False,
                "is_safe_prompt": False,
                "attack_type": "jailbreak",
            },
            {
                "is_correct": False,
                "is_partial": False,
                "is_false_positive": True,
                "is_safe_prompt": True,
                "attack_type": "none",
            },
        ]
        profile = generate_resilience_profile(results, "basic-injection", 42)
        assert "direct_override" in profile["attack_type_breakdown"]
        assert "jailbreak" in profile["attack_type_breakdown"]

    def test_profile_contains_required_fields(self):
        """Profile should contain all required fields."""
        results = [
            {
                "is_correct": True,
                "is_partial": False,
                "is_false_positive": False,
                "is_safe_prompt": False,
                "attack_type": "direct_override",
            }
        ]
        profile = generate_resilience_profile(results, "basic-injection", 42)
        required_fields = [
            "task_name",
            "seed",
            "attack_type_breakdown",
            "overall_detection_rate",
            "overall_false_positive_rate",
            "resilience_score",
        ]
        assert all(field in profile for field in required_fields)

    def test_per_attack_type_stats(self):
        """Each attack type should have correct stats."""
        results = [
            {
                "is_correct": True,
                "is_partial": False,
                "is_false_positive": False,
                "is_safe_prompt": False,
                "attack_type": "direct_override",
            },
            {
                "is_correct": False,
                "is_partial": False,
                "is_false_positive": False,
                "is_safe_prompt": False,
                "attack_type": "direct_override",
            },
            {
                "is_correct": True,
                "is_partial": False,
                "is_false_positive": False,
                "is_safe_prompt": False,
                "attack_type": "jailbreak",
            },
        ]
        profile = generate_resilience_profile(results, "basic-injection", 42)

        direct_override = profile["attack_type_breakdown"]["direct_override"]
        assert direct_override["detected"] == 1
        assert direct_override["missed"] == 1
        assert direct_override["total"] == 2
        assert direct_override["detection_rate"] == 0.5
