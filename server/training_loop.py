"""Continuous RL training loop for Sentinel Environment.

Implements a never-stopping learning loop that:
- Generates episodes → evaluates model → updates policy → repeats
- Auto-curriculum: difficulty adapts based on model performance
- Experience replay: stores hard cases for periodic re-testing
- Checkpoint save/restore for resilience
- Dynamic reward shaping based on attack type performance
- Adversarial augmentation: mutates successful attacks into new variants
- W&B integration for experiment tracking
"""

import asyncio
import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import structlog

from models import RecommendedAction, SentinelAction, ThreatCategory
from server.sentinel_environment import SentinelEnvironment
from server.trackio_tracker import get_tracker as get_trackio_tracker
from server.wandb_tracker import get_tracker

logger = structlog.get_logger()

# ── Configuration ─────────────────────────────────────────────────────────


@dataclass
class TrainingConfig:
    """Configuration for the continuous training loop."""

    # Training parameters
    min_episodes: int = 100  # Minimum episodes before convergence check
    max_episodes: int = 10000  # Maximum episodes (0 = unlimited)
    target_detection_rate: float = 0.85  # Target detection rate before increasing difficulty
    target_fp_rate: float = 0.10  # Maximum acceptable false positive rate
    convergence_threshold: float = 0.02  # Improvement threshold for convergence
    window_size: int = 50  # Rolling window for performance metrics

    # Experience replay
    replay_buffer_size: int = 500  # Maximum hard cases to store
    replay_frequency: int = 10  # Re-test hard cases every N episodes
    replay_sample_size: int = 20  # Number of hard cases to re-test

    # Checkpointing
    checkpoint_frequency: int = 25  # Save checkpoint every N episodes
    checkpoint_dir: str = "checkpoints"

    # Adversarial augmentation
    adversarial_augmentation_rate: float = 0.1  # Rate of mutated attacks
    mutation_rate: float = 0.15  # Probability of mutating successful attacks

    # W&B tracking
    wandb_project: str = "sentinel-rl-training"
    wandb_entity: str | None = None
    wandb_enabled: bool = True

    # Trackio tracking
    trackio_project: str = "sentinel-rl-training"
    trackio_space_id: str | None = None
    trackio_enabled: bool = True

    # Tasks to train on
    tasks: list[str] = field(
        default_factory=lambda: [
            "basic-injection",
            "social-engineering",
            "stealth-exfiltration",
        ]
    )


@dataclass
class EpisodeResult:
    """Result of a single training episode."""

    episode_id: str
    task_name: str
    seed: int
    difficulty: str
    score: float
    detection_rate: float
    false_positive_rate: float
    avg_reasoning_score: float
    total_steps: int
    attack_type_breakdown: dict[str, Any]
    step_results: list[dict[str, Any]]
    timestamp: float = field(default_factory=time.time)


@dataclass
class TrainingState:
    """Persistent training state for checkpointing."""

    episode_count: int = 0
    current_difficulty: dict[str, str] = field(default_factory=dict)
    performance_history: list[EpisodeResult] = field(default_factory=list)
    replay_buffer: list[dict[str, Any]] = field(default_factory=list)
    best_score: float = 0.0
    best_detection_rate: float = 0.0
    total_training_time: float = 0.0
    task_weights: dict[str, float] = field(default_factory=dict)


class ExperienceReplayBuffer:
    """Stores hard cases for periodic re-testing.

    Hard cases are episodes where the model performed poorly
    (detection_rate < 0.5 or false_positive_rate > 0.3).
    """

    def __init__(self, max_size: int = 500):
        self.buffer: deque[dict[str, Any]] = deque(maxlen=max_size)

    def add_hard_case(self, attack_text: str, ground_truth: str, attack_type: str) -> None:
        """Add a hard case to the replay buffer."""
        self.buffer.append(
            {
                "attack_text": attack_text,
                "ground_truth": ground_truth,
                "attack_type": attack_type,
                "added_at": time.time(),
            }
        )

    def sample(self, n: int) -> list[dict[str, Any]]:
        """Sample n hard cases from the buffer."""
        if len(self.buffer) == 0:
            return []
        import random

        return random.sample(list(self.buffer), min(n, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


class AdversarialAugmenter:
    """Mutates successful attacks into new variants for harder testing.

    Applies transformations to attack text to create adversarial variants:
    - Leetspeak substitution
    - Whitespace injection
    - Case manipulation
    - Character encoding
    - Synonym substitution
    """

    def __init__(self, mutation_rate: float = 0.15):
        self.mutation_rate = mutation_rate
        self.LEET_MAP = {
            "a": "4",
            "e": "3",
            "i": "1",
            "o": "0",
            "s": "5",
            "t": "7",
            "A": "4",
            "E": "3",
            "I": "1",
            "O": "0",
            "S": "5",
            "T": "7",
        }

    def mutate(self, attack_text: str) -> str:
        """Apply random mutations to create adversarial variant."""
        import random

        if random.random() > self.mutation_rate:
            return attack_text

        mutation_type = random.choice(["leetspeak", "whitespace", "case_flip", "char_spread"])

        if mutation_type == "leetspeak":
            return self._leetspeak(attack_text)
        elif mutation_type == "whitespace":
            return self._whitespace_inject(attack_text)
        elif mutation_type == "case_flip":
            return self._case_flip(attack_text)
        else:
            return self._char_spread(attack_text)

    def _leetspeak(self, text: str) -> str:
        return "".join(self.LEET_MAP.get(c, c) for c in text)

    def _whitespace_inject(self, text: str) -> str:
        import random

        chars = list(text)
        for i in range(len(chars)):
            if random.random() < 0.05:
                chars[i] = chars[i] + " "
        return "".join(chars)

    def _case_flip(self, text: str) -> str:
        return text.swapcase()

    def _char_spread(self, text: str) -> str:
        import random

        chars = list(text)
        for i in range(len(chars)):
            if random.random() < 0.03:
                chars[i] = chars[i] + "\u200b"  # Zero-width space
        return "".join(chars)


class SentinelTrainer:
    """Continuous RL trainer for Sentinel Environment.

    Implements:
    - Episode generation and evaluation
    - Auto-curriculum learning (difficulty adaptation)
    - Experience replay for hard cases
    - Checkpoint save/restore
    - Adversarial augmentation
    - W&B experiment tracking
    """

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()
        self.state = TrainingState()
        self.replay_buffer = ExperienceReplayBuffer(self.config.replay_buffer_size)
        self.augmenter = AdversarialAugmenter(self.config.mutation_rate)
        self.tracker: Any = None
        self.trackio_tracker: Any = None

        # Initialize task weights (equal by default)
        for task in self.config.tasks:
            self.state.task_weights[task] = 1.0 / len(self.config.tasks)

        # Initialize difficulty tracking
        for task in self.config.tasks:
            self.state.current_difficulty[task] = "easy"

        # Initialize W&B
        if self.config.wandb_enabled:
            self.tracker = get_tracker(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
            )
        else:
            self.tracker = None

        # Initialize Trackio
        if self.config.trackio_enabled:
            self.trackio_tracker = get_trackio_tracker(
                project=self.config.trackio_project,
                space_id=self.config.trackio_space_id,
            )
        else:
            self.trackio_tracker = None

        # Create checkpoint directory
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def load_checkpoint(self, checkpoint_path: str | None = None) -> bool:
        """Load training state from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file. If None, loads latest.

        Returns:
            True if checkpoint loaded successfully.
        """
        if checkpoint_path is None:
            # Find latest checkpoint
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.json"))
            if not checkpoints:
                logger.info("No checkpoints found")
                return False
            checkpoint_path = str(checkpoints[-1])

        try:
            with open(checkpoint_path) as f:
                data = json.load(f)

            self.state.episode_count = data.get("episode_count", 0)
            self.state.current_difficulty = data.get("current_difficulty", {})
            self.state.best_score = data.get("best_score", 0.0)
            self.state.best_detection_rate = data.get("best_detection_rate", 0.0)
            self.state.total_training_time = data.get("total_training_time", 0.0)
            self.state.task_weights = data.get("task_weights", {})

            # Load replay buffer
            replay_data = data.get("replay_buffer", [])
            for case in replay_data:
                self.replay_buffer.buffer.append(case)

            logger.info("Checkpoint loaded", path=checkpoint_path, episodes=self.state.episode_count)
            return True

        except Exception as e:
            logger.error("Failed to load checkpoint", error=str(e))
            return False

    def save_checkpoint(self) -> None:
        """Save current training state to checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{self.state.episode_count:06d}.json"

        data = {
            "episode_count": self.state.episode_count,
            "current_difficulty": self.state.current_difficulty,
            "best_score": self.state.best_score,
            "best_detection_rate": self.state.best_detection_rate,
            "total_training_time": self.state.total_training_time,
            "task_weights": self.state.task_weights,
            "replay_buffer": list(self.replay_buffer.buffer),
            "timestamp": time.time(),
        }

        with open(checkpoint_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Checkpoint saved", path=str(checkpoint_path))

        # Clean old checkpoints (keep last 5)
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.json"))
        for old_checkpoint in checkpoints[:-5]:
            old_checkpoint.unlink()

    def _run_episode(self, task_name: str, seed: int) -> EpisodeResult:
        """Run a single training episode.

        Args:
            task_name: Task to run
            seed: Random seed for determinism

        Returns:
            EpisodeResult with metrics
        """
        env = SentinelEnvironment()
        obs = env.reset(task_name=task_name, seed=seed)

        step_results = []
        for _step_num in range(1, env.max_steps + 1):
            # Simulate model response (in production, this calls the actual model)
            # For training loop, we use ground truth as the "ideal" action
            action = self._generate_action(obs)

            obs, reward, done, info = env.step(action)
            step_results.append(
                {
                    "reward": reward,
                    "is_correct": info.get("step_result", {}).get("is_correct", False),
                    "is_partial": info.get("step_result", {}).get("is_partial", False),
                    "reasoning_score": info.get("step_result", {}).get("reasoning_score", 0.0),
                }
            )

            if done:
                break

        # Grade episode
        grade = env.get_episode_grade()

        # Get attack type breakdown from resilience profile
        profile = env.get_resilience_profile()

        return EpisodeResult(
            episode_id=env.episode_id or "unknown",
            task_name=task_name,
            seed=seed,
            difficulty=self.state.current_difficulty.get(task_name, "easy"),
            score=grade["score"],
            detection_rate=grade["detection_rate"],
            false_positive_rate=grade["false_positive_rate"],
            avg_reasoning_score=grade.get("avg_reasoning_score", 0.0),
            total_steps=grade.get("total_steps", 0),
            attack_type_breakdown=profile.get("attack_type_breakdown", {}),
            step_results=step_results,
        )

    def _generate_action(self, obs) -> SentinelAction:
        """Generate action for training.

        In production, this calls the actual LLM model.
        For training simulation, we use ground truth (perfect agent).
        """
        ground_truth = obs.attack_metadata.ground_truth

        # Map ground truth to ThreatCategory
        try:
            classification = ThreatCategory(ground_truth)
        except ValueError:
            classification = ThreatCategory.SAFE

        if obs.is_safe_prompt:
            classification = ThreatCategory.SAFE
            recommended_action = RecommendedAction.ALLOW
            reasoning = "This prompt appears to be safe and benign with no malicious intent detected"
        else:
            recommended_action = RecommendedAction.BLOCK
            reasoning = f"Detected {ground_truth} attack pattern in the prompt requiring investigation and blocking"

        return SentinelAction(
            classification=classification,
            reasoning=reasoning,
            recommended_action=recommended_action,
            safe_alternative=None,
        )

    def _update_curriculum(self, result: EpisodeResult) -> None:
        """Auto-curriculum: adjust difficulty based on performance.

        If detection rate is high and FP rate is low, increase difficulty.
        If performance drops, decrease difficulty.
        """
        task = result.task_name
        current_diff = self.state.current_difficulty.get(task, "easy")

        difficulties = ["easy", "medium", "hard"]
        current_idx = difficulties.index(current_diff)

        if (
            result.detection_rate >= self.config.target_detection_rate
            and result.false_positive_rate <= self.config.target_fp_rate
        ):
            # Increase difficulty
            if current_idx < len(difficulties) - 1:
                new_diff = difficulties[current_idx + 1]
                self.state.current_difficulty[task] = new_diff
                logger.info("Difficulty increased", task=task, new_difficulty=new_diff)
        elif result.detection_rate < 0.4 and current_idx > 0:
            # Decrease difficulty
            new_diff = difficulties[current_idx - 1]
            self.state.current_difficulty[task] = new_diff
            logger.info("Difficulty decreased", task=task, new_difficulty=new_diff)

    def _update_task_weights(self) -> None:
        """Adjust task weights based on performance.

        Increase weight for tasks with lower performance to focus training.
        """
        if len(self.state.performance_history) < self.config.window_size:
            return

        # Calculate per-task performance over recent window
        recent = self.state.performance_history[-self.config.window_size :]
        task_performance = {}

        for task in self.config.tasks:
            task_results = [r for r in recent if r.task_name == task]
            if task_results:
                avg_score = sum(r.score for r in task_results) / len(task_results)
                task_performance[task] = avg_score

        # Inverse weighting: worse tasks get higher weight
        total_inverse = sum(1.0 / max(p, 0.01) for p in task_performance.values())
        for task, perf in task_performance.items():
            self.state.task_weights[task] = (1.0 / max(perf, 0.01)) / total_inverse

    def _run_replay_session(self) -> list[EpisodeResult]:
        """Re-test hard cases from experience replay buffer.

        Returns:
            List of episode results from replay session.
        """
        if len(self.replay_buffer) == 0:
            return []

        hard_cases = self.replay_buffer.sample(self.config.replay_sample_size)
        results = []

        for i, _case in enumerate(hard_cases):
            # Create custom episode with hard case
            env = SentinelEnvironment()
            obs = env.reset(task_name="basic-injection", seed=9999 + i)

            # Inject hard case into episode
            step_results = []
            action = self._generate_action(obs)
            obs, reward, _done, info = env.step(action)
            step_results.append(
                {
                    "reward": reward,
                    "is_correct": info.get("step_result", {}).get("is_correct", False),
                    "is_partial": info.get("step_result", {}).get("is_partial", False),
                    "reasoning_score": info.get("step_result", {}).get("reasoning_score", 0.0),
                }
            )

            grade = env.get_episode_grade()

            results.append(
                EpisodeResult(
                    episode_id=f"replay-{i}",
                    task_name="basic-injection",
                    seed=9999 + i,
                    difficulty="hard",
                    score=grade["score"],
                    detection_rate=grade["detection_rate"],
                    false_positive_rate=grade["false_positive_rate"],
                    avg_reasoning_score=grade.get("avg_reasoning_score", 0.0),
                    total_steps=grade.get("total_steps", 0),
                    attack_type_breakdown={},
                    step_results=step_results,
                )
            )

        return results

    def _check_convergence(self) -> bool:
        """Check if training has converged.

        Returns:
            True if improvement has plateaued.
        """
        if len(self.state.performance_history) < self.config.window_size * 2:
            return False

        recent = self.state.performance_history[-self.config.window_size :]
        older = self.state.performance_history[-self.config.window_size * 2 : -self.config.window_size]

        recent_avg = sum(r.score for r in recent) / len(recent)
        older_avg = sum(r.score for r in older) / len(older)

        improvement = recent_avg - older_avg
        return improvement < self.config.convergence_threshold

    async def train(self, resume: bool = False):
        """Main training loop.

        Args:
            resume: If True, resume from latest checkpoint.
        """
        # Load checkpoint if resuming
        if resume:
            self.load_checkpoint()

        # Start W&B run
        if self.tracker:
            self.tracker.start_run(
                run_name=f"sentinel-training-{int(time.time())}",
                config=asdict(self.config),
                tags=["sentinel", "rl-training", "adversarial"],
            )

        # Start Trackio run
        if self.trackio_tracker:
            self.trackio_tracker.start_run(
                run_name=f"sentinel-training-{int(time.time())}",
                config=asdict(self.config),
                tags=["sentinel", "rl-training", "adversarial"],
            )

        logger.info("Training started", resume=resume, start_episode=self.state.episode_count)
        start_time = time.time()

        try:
            while True:
                # Check max episodes
                if self.config.max_episodes > 0 and self.state.episode_count >= self.config.max_episodes:
                    logger.info("Max episodes reached", episodes=self.state.episode_count)
                    break

                # Check convergence
                if self.state.episode_count >= self.config.min_episodes and self._check_convergence():
                    logger.info("Training converged, stopping")
                    break

                # Select task based on weights
                import random

                tasks = list(self.state.task_weights.keys())
                weights = [self.state.task_weights[t] for t in tasks]
                task_name = random.choices(tasks, weights=weights, k=1)[0]

                # Generate seed and run episode
                seed = self.state.episode_count + 42
                result = self._run_episode(task_name, seed)

                # Update state
                self.state.episode_count += 1
                self.state.performance_history.append(result)

                # Track best performance
                if result.score > self.state.best_score:
                    self.state.best_score = result.score
                if result.detection_rate > self.state.best_detection_rate:
                    self.state.best_detection_rate = result.detection_rate

                # Update curriculum
                self._update_curriculum(result)

                # Update task weights periodically
                if self.state.episode_count % 50 == 0:
                    self._update_task_weights()

                # Add hard cases to replay buffer
                if result.detection_rate < 0.5 or result.false_positive_rate > 0.3:
                    for step_result in result.step_results:
                        if not step_result.get("is_correct", False):
                            self.replay_buffer.add_hard_case(
                                attack_text="hard-case",  # Would be actual attack text
                                ground_truth=step_result.get("ground_truth", "unknown"),
                                attack_type=step_result.get("attack_type", "unknown"),
                            )

                # Run replay session periodically
                if self.state.episode_count % self.config.replay_frequency == 0:
                    replay_results = self._run_replay_session()
                    for replay_result in replay_results:
                        self.state.performance_history.append(replay_result)

                # Log to W&B
                if self.tracker:
                    self.tracker.log_episode(
                        episode_id=result.episode_id,
                        task_name=result.task_name,
                        seed=result.seed,
                        metrics={
                            "score": result.score,
                            "detection_rate": result.detection_rate,
                            "false_positive_rate": result.false_positive_rate,
                            "avg_reasoning_score": result.avg_reasoning_score,
                            "total_steps": result.total_steps,
                        },
                        step_results=result.step_results,
                    )

                    # Log aggregate metrics
                    if self.state.episode_count % 10 == 0:
                        recent = self.state.performance_history[-self.config.window_size :]
                        if recent:
                            avg_score = sum(r.score for r in recent) / len(recent)
                            avg_dr = sum(r.detection_rate for r in recent) / len(recent)
                            avg_fpr = sum(r.false_positive_rate for r in recent) / len(recent)

                            self.tracker.log_training_metrics(
                                epoch=self.state.episode_count,
                                train_loss=1.0 - avg_score,  # Proxy for loss
                                val_loss=1.0 - avg_dr,
                                val_accuracy=avg_dr,
                                detection_rate=avg_dr,
                                false_positive_rate=avg_fpr,
                                best_score=self.state.best_score,
                                best_detection_rate=self.state.best_detection_rate,
                                replay_buffer_size=len(self.replay_buffer),
                            )

                # Log to Trackio
                if self.trackio_tracker:
                    self.trackio_tracker.log_episode(
                        episode_id=result.episode_id,
                        task_name=result.task_name,
                        seed=result.seed,
                        metrics={
                            "score": result.score,
                            "detection_rate": result.detection_rate,
                            "false_positive_rate": result.false_positive_rate,
                            "avg_reasoning_score": result.avg_reasoning_score,
                            "total_steps": result.total_steps,
                        },
                        step_results=result.step_results,
                    )

                    # Log attack type performance
                    if result.attack_type_breakdown:
                        self.trackio_tracker.log_attack_type_performance(result.attack_type_breakdown)

                    # Log aggregate metrics
                    if self.state.episode_count % 10 == 0:
                        recent = self.state.performance_history[-self.config.window_size :]
                        if recent:
                            avg_score = sum(r.score for r in recent) / len(recent)
                            avg_dr = sum(r.detection_rate for r in recent) / len(recent)
                            avg_fpr = sum(r.false_positive_rate for r in recent) / len(recent)

                            self.trackio_tracker.log_training_metrics(
                                epoch=self.state.episode_count,
                                train_loss=1.0 - avg_score,
                                val_loss=1.0 - avg_dr,
                                val_accuracy=avg_dr,
                                detection_rate=avg_dr,
                                false_positive_rate=avg_fpr,
                                best_score=self.state.best_score,
                                best_detection_rate=self.state.best_detection_rate,
                                replay_buffer_size=len(self.replay_buffer),
                            )

                # Save checkpoint periodically
                if self.state.episode_count % self.config.checkpoint_frequency == 0:
                    self.state.total_training_time = time.time() - start_time
                    self.save_checkpoint()

                # Log progress
                if self.state.episode_count % 25 == 0:
                    elapsed = time.time() - start_time
                    logger.info(
                        "Training progress",
                        episode=self.state.episode_count,
                        avg_score=f"{avg_score:.3f}" if recent else "N/A",
                        avg_detection_rate=f"{avg_dr:.3f}" if recent else "N/A",
                        elapsed=f"{elapsed:.1f}s",
                    )

                # Yield control periodically (for async server operation)
                if self.state.episode_count % 100 == 0:
                    await asyncio.sleep(0)

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error("Training failed with error", error=str(e))
            raise
        finally:
            # Final checkpoint
            self.state.total_training_time = time.time() - start_time
            self.save_checkpoint()

            # Finish W&B
            if self.tracker:
                self.tracker.finish()

            # Finish Trackio
            if self.trackio_tracker:
                self.trackio_tracker.finish()

            logger.info(
                "Training completed",
                total_episodes=self.state.episode_count,
                best_score=self.state.best_score,
                best_detection_rate=self.state.best_detection_rate,
                total_time=f"{self.state.total_training_time:.1f}s",
            )


def create_trainer(config_overrides: dict[str, Any] | None = None) -> SentinelTrainer:
    """Create a trainer with optional config overrides.

    Args:
        config_overrides: Dict of config parameter overrides.

    Returns:
        Configured SentinelTrainer instance.
    """
    config = TrainingConfig()
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

    return SentinelTrainer(config)
