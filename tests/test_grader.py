"""Tests for the grader module."""

import pytest
from models import ThreatCategory
from server.grader import grade_step, grade_episode


class TestGradeStep:
    def test_correct_detection(self):
        result = grade_step(
            ThreatCategory.INJECTION,
            "injection",
            "This is a direct prompt injection attempt.",
        )
        assert result["is_correct"] is True
        assert result["is_missed"] is False
        assert result["is_false_positive"] is False
        assert result["reward"] >= 0.6

    def test_missed_attack(self):
        result = grade_step(
            ThreatCategory.SAFE,
            "injection",
            "Seems fine to me.",
        )
        assert result["is_missed"] is True
        assert result["reward"] < 0.3

    def test_false_positive(self):
        result = grade_step(
            ThreatCategory.INJECTION,
            "safe",
            "This looks suspicious.",
        )
        assert result["is_false_positive"] is True
        assert result["reward"] < 0.5

    def test_safe_correct(self):
        result = grade_step(
            ThreatCategory.SAFE,
            "safe",
            "This is a normal request.",
        )
        assert result["is_correct"] is True
        assert result["is_false_positive"] is False

    def test_partial_credit_same_superclass(self):
        result = grade_step(
            ThreatCategory.JAILBREAK,
            "injection",
            "Some kind of attack detected.",
        )
        assert result["is_partial"] is True
        assert result["reward"] >= 0.3

    def test_reward_range(self):
        """All rewards must be in [0.0, 1.0]."""
        for pred in ThreatCategory:
            for actual in ["safe", "injection", "jailbreak", "command_injection"]:
                result = grade_step(pred, actual, "Reasoning here with some detail.")
                assert 0.0 <= result["reward"] <= 1.0


class TestGradeEpisode:
    def test_perfect_episode(self):
        results = [
            {"is_correct": True, "is_partial": False, "is_missed": False,
             "is_false_positive": False, "is_safe_prompt": False, "reasoning_score": 0.8}
            for _ in range(5)
        ]
        grade = grade_episode(results)
        assert grade["score"] >= 0.8
        assert grade["detection_rate"] == 1.0

    def test_all_missed(self):
        results = [
            {"is_correct": False, "is_partial": False, "is_missed": True,
             "is_false_positive": False, "is_safe_prompt": False, "reasoning_score": 0.0}
            for _ in range(5)
        ]
        grade = grade_episode(results)
        assert grade["score"] < 0.3
        assert grade["detection_rate"] == 0.0

    def test_empty_episode(self):
        grade = grade_episode([])
        assert grade["score"] == 0.0

    def test_score_range(self):
        """All scores must be in [0.0, 1.0]."""
        results = [
            {"is_correct": True, "is_partial": False, "is_missed": False,
             "is_false_positive": False, "is_safe_prompt": False, "reasoning_score": 1.0}
            for _ in range(10)
        ]
        grade = grade_episode(results)
        assert 0.0 <= grade["score"] <= 1.0
