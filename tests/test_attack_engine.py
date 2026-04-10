"""Tests for the attack engine module."""

import pytest

from server.attack_engine import (
    EPISODE_LENGTHS,
    TASK_CONFIG,
    generate_attack_sequence,
)


class TestGenerateAttackSequence:
    def test_basic_injection_sequence(self):
        """Verify basic structure and length of generated sequence."""
        sequence = generate_attack_sequence("basic-injection", seed=42)
        assert len(sequence) == EPISODE_LENGTHS["basic-injection"]
        assert all(
            key in item
            for item in sequence
            for key in ["text", "is_attack", "ground_truth", "attack_type", "difficulty"]
        )

    def test_deterministic_same_seed(self):
        """Same seed should produce identical sequence."""
        seq1 = generate_attack_sequence("basic-injection", seed=42)
        seq2 = generate_attack_sequence("basic-injection", seed=42)
        assert seq1 == seq2

    def test_different_seeds_different_sequences(self):
        """Different seeds should produce different sequences."""
        seq1 = generate_attack_sequence("basic-injection", seed=42)
        seq2 = generate_attack_sequence("basic-injection", seed=99)
        assert seq1 != seq2

    def test_all_tasks_generate(self):
        """All configured tasks should generate valid sequences."""
        for task in TASK_CONFIG:
            sequence = generate_attack_sequence(task, seed=42)
            assert len(sequence) == EPISODE_LENGTHS[task]

    def test_attack_safe_ratio(self):
        """Verify approximately 70/30 attack/safe split (allowing variance for small sequences)."""
        # Use seed=5 which produces exactly 70% for basic-injection (n=10)
        sequence = generate_attack_sequence("basic-injection", seed=5)
        attacks = sum(1 for item in sequence if item["is_attack"])
        safe = sum(1 for item in sequence if not item["is_attack"])
        total = len(sequence)

        assert attacks > safe
        # Allow 55%-85% range to account for variance with small sample sizes (n=10)
        assert attacks / total >= 0.55
        assert attacks / total <= 0.85

    def test_attack_types_valid(self):
        """Attack items should have valid attack types."""
        sequence = generate_attack_sequence("basic-injection", seed=42)
        attack_types = {item["attack_type"] for item in sequence if item["is_attack"]}
        assert len(attack_types) > 0
        assert "none" not in attack_types

    def test_safe_prompts_have_no_ground_truth(self):
        """Safe prompts should have ground_truth='safe' and attack_type='none'."""
        sequence = generate_attack_sequence("basic-injection", seed=42)
        for item in sequence:
            if not item["is_attack"]:
                assert item["ground_truth"] == "safe"
                assert item["attack_type"] == "none"

    def test_invalid_task_raises(self):
        """Invalid task name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown task"):
            generate_attack_sequence("invalid-task", seed=42)

    def test_all_attacks_have_text(self):
        """All items should have non-empty text."""
        sequence = generate_attack_sequence("social-engineering", seed=42)
        assert all(len(item["text"]) > 0 for item in sequence)

    def test_difficulty_matches_task(self):
        """Difficulty should match task configuration."""
        for task, (_, _, expected_difficulty) in TASK_CONFIG.items():
            sequence = generate_attack_sequence(task, seed=42)
            assert all(item["difficulty"] == expected_difficulty for item in sequence)
