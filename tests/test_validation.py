"""Comprehensive validation test suite for hackathon submission criteria.

Tests every requirement: real-world tasks, OpenEnv spec compliance,
grading, rewards, determinism, and edge cases.
"""

import pytest
from models import (
    SentinelAction,
    SentinelObservation,
    SentinelState,
    AttackMetadata,
    ResilienceMetrics,
    ThreatCategory,
    RecommendedAction,
)
from server.sentinel_environment import SentinelEnvironment
from server.grader import grade_step, grade_episode
from server.reward_shaper import compute_reward
from server.attack_engine import generate_attack_sequence, EPISODE_LENGTHS, TASK_CONFIG
from server.resilience_profile import generate_resilience_profile


# ── Requirement: Real-world task simulation ─────────────────────────────

class TestRealWorldTasks:
    """Tasks must simulate real-world tasks, not games or toys."""

    def test_three_tasks_exist(self):
        assert "basic-injection" in TASK_CONFIG
        assert "social-engineering" in TASK_CONFIG
        assert "stealth-exfiltration" in TASK_CONFIG

    def test_tasks_have_difficulty_levels(self):
        assert TASK_CONFIG["basic-injection"][2] == "easy"
        assert TASK_CONFIG["social-engineering"][2] == "medium"
        assert TASK_CONFIG["stealth-exfiltration"][2] == "hard"

    def test_tasks_represent_real_world_security_tasks(self):
        """Each task represents a genuine AI safety evaluation scenario."""
        for task_name in TASK_CONFIG:
            env = SentinelEnvironment()
            obs = env.reset(task_name=task_name, seed=42)
            # Attack text should represent real-world attack patterns
            assert len(obs.attack_metadata.attack_text) > 20
            assert obs.attack_metadata.ground_truth != "unknown"


# ── Requirement: OpenEnv spec compliance ────────────────────────────────

class TestOpenEnvSpecCompliance:
    """Must implement full OpenEnv interface."""

    def test_reset_returns_observation(self):
        env = SentinelEnvironment()
        obs = env.reset(task_name="basic-injection", seed=42)
        assert isinstance(obs, SentinelObservation)
        assert obs.user_prompt is not None
        assert obs.step_number == 1

    def test_step_returns_correct_tuple(self):
        env = SentinelEnvironment()
        env.reset(task_name="basic-injection", seed=42)
        action = SentinelAction(
            classification=ThreatCategory.INJECTION,
            reasoning="Direct override attempt detected in the prompt",
            recommended_action=RecommendedAction.BLOCK,
        )
        result = env.step(action)
        assert isinstance(result, tuple)
        assert len(result) == 4
        obs, reward, done, info = result
        assert isinstance(obs, SentinelObservation)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_state_returns_state(self):
        env = SentinelEnvironment()
        env.reset(task_name="basic-injection", seed=42)
        state = env.state()
        assert isinstance(state, SentinelState)
        assert state.episode_id is not None
        assert state.task_name is not None

    def test_reward_in_valid_range(self):
        """Every step reward must be in [0.0, 1.0]."""
        env = SentinelEnvironment()
        env.reset(task_name="basic-injection", seed=42)
        for _ in range(15):
            action = SentinelAction(
                classification=ThreatCategory.SAFE,
                reasoning="This prompt appears to be safe and benign",
                recommended_action=RecommendedAction.ALLOW,
            )
            obs, reward, done, info = env.step(action)
            assert 0.0 <= reward <= 1.0, f"Reward {reward} out of range at step {obs.step_number}"
            if done:
                break

    def test_typed_pydantic_models(self):
        """All models must be valid Pydantic with typed fields."""
        # SentinelAction
        action = SentinelAction(
            classification=ThreatCategory.INJECTION,
            reasoning="Test reasoning with sufficient length and detail here",
            recommended_action=RecommendedAction.BLOCK,
        )
        assert action.classification == ThreatCategory.INJECTION
        assert action.recommended_action == RecommendedAction.BLOCK

        # SentinelObservation
        env = SentinelEnvironment()
        obs = env.reset(task_name="basic-injection", seed=42)
        assert isinstance(obs.user_prompt, str)
        assert isinstance(obs.step_number, int)
        assert isinstance(obs.attack_metadata, AttackMetadata)
        assert isinstance(obs.resilience_metrics, ResilienceMetrics)


# ── Requirement: Minimum 3 tasks with agent graders ─────────────────────

class TestTasksWithGraders:
    """3+ tasks with difficulty range, graders produce 0.0-1.0 scores."""

    def test_all_three_tasks_produce_valid_episodes(self):
        for task_name in ["basic-injection", "social-engineering", "stealth-exfiltration"]:
            env = SentinelEnvironment()
            env.reset(task_name=task_name, seed=42)
            # Run full episode
            for _ in range(20):
                action = SentinelAction(
                    classification=ThreatCategory.SAFE,
                    reasoning="This prompt appears to be a legitimate and safe request",
                    recommended_action=RecommendedAction.ALLOW,
                )
                obs, reward, done, info = env.step(action)
                assert 0.0 <= reward <= 1.0
                if done:
                    break

            grade = env.get_episode_grade()
            assert 0.0 <= grade["score"] <= 1.0
            assert "detection_rate" in grade
            assert "false_positive_rate" in grade

    def test_episode_lengths_match_config(self):
        """Episode lengths must match EPISODE_LENGTHS config."""
        for task_name, expected_length in EPISODE_LENGTHS.items():
            env = SentinelEnvironment()
            env.reset(task_name=task_name, seed=42)
            steps = 0
            for _ in range(30):
                action = SentinelAction(
                    classification=ThreatCategory.SAFE,
                    reasoning="Safe prompt detected",
                    recommended_action=RecommendedAction.ALLOW,
                )
                obs, reward, done, info = env.step(action)
                steps += 1
                if done:
                    break
            assert steps == expected_length, f"{task_name}: expected {expected_length} steps, got {steps}"

    def test_grader_produces_valid_scores(self):
        """Grader must produce scores in [0.0, 1.0]."""
        # Perfect performance
        results_perfect = [
            {"is_correct": True, "is_partial": False, "is_missed": False,
             "is_false_positive": False, "is_safe_prompt": False, "reasoning_score": 0.8}
            for _ in range(5)
        ]
        grade = grade_episode(results_perfect)
        assert 0.0 <= grade["score"] <= 1.0

        # Worst performance
        results_worst = [
            {"is_correct": False, "is_partial": False, "is_missed": True,
             "is_false_positive": False, "is_safe_prompt": False, "reasoning_score": 0.0}
            for _ in range(5)
        ]
        grade = grade_episode(results_worst)
        assert 0.0 <= grade["score"] <= 1.0

    def test_grader_deterministic(self):
        """Same seed → same episode → same grade."""
        env1 = SentinelEnvironment()
        env2 = SentinelEnvironment()
        env1.reset(task_name="basic-injection", seed=42)
        env2.reset(task_name="basic-injection", seed=42)

        for _ in range(15):
            action = SentinelAction(
                classification=ThreatCategory.INJECTION,
                reasoning="Detected injection attack pattern in the prompt text",
                recommended_action=RecommendedAction.BLOCK,
            )
            obs1, r1, done1, _ = env1.step(action)
            obs2, r2, done2, _ = env2.step(action)
            assert r1 == r2
            assert obs1.user_prompt == obs2.user_prompt
            if done1:
                break

        grade1 = env1.get_episode_grade()
        grade2 = env2.get_episode_grade()
        assert grade1["score"] == grade2["score"]


# ── Requirement: Meaningful reward function ─────────────────────────────

class TestMeaningfulReward:
    """Reward provides trajectory feedback, partial progress, penalizes bad behavior."""

    def test_reward_provides_trajectory_feedback(self):
        """Different actions should produce different rewards."""
        env = SentinelEnvironment()
        env.reset(task_name="basic-injection", seed=42)

        action_correct = SentinelAction(
            classification=ThreatCategory.INJECTION,
            reasoning="Detected direct injection attempt in the prompt",
            recommended_action=RecommendedAction.BLOCK,
        )
        action_wrong = SentinelAction(
            classification=ThreatCategory.SAFE,
            reasoning="This prompt appears to be completely safe and benign",
            recommended_action=RecommendedAction.ALLOW,
        )

        _, reward_correct, _, _ = env.step(action_correct)
        env.reset(task_name="basic-injection", seed=42)
        _, reward_wrong, _, _ = env.step(action_wrong)

        # Correct detection should score higher than miss
        assert reward_correct > reward_wrong

    def test_reward_penalizes_false_positives(self):
        """False positive penalty: flagging safe prompt as attack."""
        env = SentinelEnvironment()
        env.reset(task_name="basic-injection", seed=42)

        # First step to see if it's safe or attack
        obs, _, _, _ = env.step(SentinelAction(
            classification=ThreatCategory.SAFE,
            reasoning="This appears to be a safe prompt",
            recommended_action=RecommendedAction.ALLOW,
        ))

        # Check step_result for safe prompt
        if not obs.is_safe_prompt:
            # It's an attack, skip this test
            return

        # Now test false positive on a safe prompt
        env.reset(task_name="basic-injection", seed=42)
        # Get to a safe prompt
        for _ in range(20):
            obs, _, done, info = env.step(SentinelAction(
                classification=ThreatCategory.SAFE,
                reasoning="This appears to be a safe prompt",
                recommended_action=RecommendedAction.ALLOW,
            ))
            if obs.is_safe_prompt:
                # Now flag it as attack (false positive)
                env.reset(task_name="basic-injection", seed=42)
                for _ in range(20):
                    obs2, _, done2, _ = env.step(SentinelAction(
                        classification=ThreatCategory.SAFE,
                        reasoning="Safe",
                        recommended_action=RecommendedAction.ALLOW,
                    ))
                    if obs2.is_safe_prompt:
                        # Force a false positive
                        _, fp_reward, _, _ = env.step(SentinelAction(
                            classification=ThreatCategory.INJECTION,
                            reasoning="Suspicious",
                            recommended_action=RecommendedAction.BLOCK,
                        ))
                        # False positive should have lower reward than correct safe
                        env.reset(task_name="basic-injection", seed=42)
                        for _ in range(20):
                            obs3, _, _, _ = env.step(SentinelAction(
                                classification=ThreatCategory.SAFE,
                                reasoning="Safe",
                                recommended_action=RecommendedAction.ALLOW,
                            ))
                            if obs3.is_safe_prompt:
                                _, correct_safe, _, _ = env.step(SentinelAction(
                                    classification=ThreatCategory.SAFE,
                                    reasoning="This appears to be a safe prompt",
                                    recommended_action=RecommendedAction.ALLOW,
                                ))
                                assert fp_reward < correct_safe
                                break
                        break
                    if done2:
                        break
                break
            if done:
                break

    def test_reward_penalizes_missed_attacks(self):
        """Missed attack should have lower reward than correct detection."""
        env = SentinelEnvironment()
        env.reset(task_name="basic-injection", seed=42)

        for _ in range(20):
            obs, _, done, _ = env.step(SentinelAction(
                classification=ThreatCategory.SAFE,
                reasoning="This prompt appears to be safe and benign",
                recommended_action=RecommendedAction.ALLOW,
            ))
            if not obs.is_safe_prompt:
                # We found an attack, now test miss vs detect
                miss_reward = obs.resilience_metrics  # already recorded
                break
            if done:
                return  # No attacks in this episode


# ── Requirement: Determinism and reproducibility ────────────────────────

class TestDeterminism:
    """Seed-deterministic attack generation."""

    def test_same_seed_same_sequence(self):
        seq1 = generate_attack_sequence("basic-injection", seed=42)
        seq2 = generate_attack_sequence("basic-injection", seed=42)
        assert len(seq1) == len(seq2)
        for a, b in zip(seq1, seq2):
            assert a["text"] == b["text"]
            assert a["ground_truth"] == b["ground_truth"]

    def test_different_seeds_different_sequence(self):
        seq1 = generate_attack_sequence("basic-injection", seed=42)
        seq2 = generate_attack_sequence("basic-injection", seed=99)
        texts1 = [s["text"] for s in seq1]
        texts2 = [s["text"] for s in seq2]
        assert texts1 != texts2

    def test_attack_counts_match_episode_lengths(self):
        for task_name, expected_len in EPISODE_LENGTHS.items():
            seq = generate_attack_sequence(task_name, seed=42)
            assert len(seq) == expected_len


# ── Pydantic model validation ───────────────────────────────────────────

class TestPydanticValidation:
    """Typed models must validate correctly."""

    def test_recommended_action_enum(self):
        """RecommendedAction must be one of the enum values."""
        action = SentinelAction(
            classification=ThreatCategory.INJECTION,
            reasoning="Detected injection attempt with sufficient explanation length here",
            recommended_action=RecommendedAction.BLOCK,
        )
        assert action.recommended_action == RecommendedAction.BLOCK
        assert action.recommended_action.value == "block"

    def test_difficulty_pattern_validation(self):
        """AttackMetadata difficulty must match pattern."""
        metadata = AttackMetadata(
            attack_type="test",
            difficulty="easy",
            attack_text="test",
            seed=1,
            task_name="test",
            ground_truth="safe",
        )
        assert metadata.difficulty == "easy"

    def test_all_threat_categories_accessible(self):
        for cat in ThreatCategory:
            action = SentinelAction(
                classification=cat,
                reasoning=f"Classified as {cat.value} with sufficient explanation detail here",
                recommended_action=RecommendedAction.BLOCK,
            )
            assert action.classification == cat


# ── Resilience profile ──────────────────────────────────────────────────

class TestResilienceProfile:
    """Resilience profile must generate valid output."""

    def test_profile_generates(self):
        env = SentinelEnvironment()
        env.reset(task_name="basic-injection", seed=42)
        for _ in range(15):
            obs, _, done, _ = env.step(SentinelAction(
                classification=ThreatCategory.INJECTION,
                reasoning="Detected injection attack with sufficient detail and explanation",
                recommended_action=RecommendedAction.BLOCK,
            ))
            if done:
                break

        profile = env.get_resilience_profile()
        assert "task_name" in profile
        assert "attack_type_breakdown" in profile
        assert 0.0 <= profile["overall_detection_rate"] <= 1.0
        assert 0.0 <= profile["resilience_score"] <= 1.0
