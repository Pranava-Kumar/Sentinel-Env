"""Weights & Biases experiment tracking for Sentinel RL episodes.

Provides comprehensive experiment tracking for:
- Episode-level metrics (detection rate, false positive rate, reward curves)
- Per-step metrics (reasoning quality, classification accuracy)
- Model performance across tasks and seeds
- Adversarial training progress
- Continuous learning metrics
"""

import os
from typing import Any

import structlog

logger = structlog.get_logger()

# Lazy import wandb only when needed to avoid hard dependency
_wandb = None


def _get_wandb():
    """Lazy import wandb."""
    global _wandb
    if _wandb is None:
        try:
            import wandb as _wandb
        except ImportError:
            logger.warning("wandb not installed, tracking disabled")
            return None
    return _wandb


class WandBTracker:
    """Tracks Sentinel RL episodes to Weights & Biases."""

    def __init__(self, project: str = "sentinel-env", entity: str | None = None):
        self.project = project
        self.entity = entity
        self.run = None
        self._enabled = False
        self._episode_buffer: list[dict[str, Any]] = []

    def start_run(
        self,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Start a new W&B run.

        Args:
            run_name: Name for the run
            config: Configuration dict (model name, task, hyperparams)
            tags: List of tags for filtering
        """
        wb = _get_wandb()
        if wb is None:
            logger.info("W&B tracking disabled - package not available")
            return

        api_key = os.getenv("WANDB_API_KEY")
        if not api_key:
            logger.info("WANDB_API_KEY not set, tracking disabled")
            return

        wb.login(key=api_key)

        self.run = wb.init(
            project=self.project,
            entity=self.entity,
            name=run_name,
            config=config or {},
            tags=tags or [],
        )
        self._enabled = True
        self._episode_buffer = []
        logger.info("W&B run started", run_name=run_name)

    def log_episode(
        self,
        episode_id: str,
        task_name: str,
        seed: int,
        metrics: dict[str, Any],
        step_results: list[dict[str, Any]] | None = None,
    ) -> None:
        """Log a completed episode to W&B.

        Args:
            episode_id: Unique episode identifier
            task_name: Task name (basic-injection, social-engineering, stealth-exfiltration)
            seed: Random seed used
            metrics: Episode-level metrics from grader
            step_results: Per-step results for detailed logging
        """
        if not self._enabled or self.run is None:
            # Store for later if W&B becomes available
            self._episode_buffer.append(
                {
                    "episode_id": episode_id,
                    "task_name": task_name,
                    "seed": seed,
                    "metrics": metrics,
                    "step_results": step_results or [],
                }
            )
            return

        wb = _get_wandb()
        if wb is None:
            return

        # Log episode-level metrics
        self.run.log(
            {
                "episode/score": metrics.get("score", 0.0),
                "episode/detection_rate": metrics.get("detection_rate", 0.0),
                "episode/false_positive_rate": metrics.get("false_positive_rate", 0.0),
                "episode/correct_detections": metrics.get("correct_detections", 0),
                "episode/missed_attacks": metrics.get("missed_attacks", 0),
                "episode/false_positives": metrics.get("false_positives", 0),
                "episode/avg_reasoning_score": metrics.get("avg_reasoning_score", 0.0),
                "episode/total_steps": metrics.get("total_steps", 0),
                "episode/task": task_name,
                "episode/seed": seed,
            }
        )

        # Log per-step metrics if available
        if step_results:
            for step_idx, step_result in enumerate(step_results):
                self.run.log(
                    {
                        "step/reward": step_result.get("reward", 0.0),
                        "step/is_correct": 1 if step_result.get("is_correct", False) else 0,
                        "step/is_partial": 1 if step_result.get("is_partial", False) else 0,
                        "step/reasoning_score": step_result.get("reasoning_score", 0.0),
                        "step/step_number": step_idx + 1,
                    }
                )

    def log_training_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_accuracy: float,
        **kwargs: Any,
    ) -> None:
        """Log training loop metrics.

        Args:
            epoch: Training epoch number
            train_loss: Training loss
            val_loss: Validation loss
            val_accuracy: Validation accuracy
            **kwargs: Additional metrics to log
        """
        if not self._enabled or self.run is None:
            return

        wb = _get_wandb()
        if wb is None:
            return

        metrics = {
            "training/epoch": epoch,
            "training/train_loss": train_loss,
            "training/val_loss": val_loss,
            "training/val_accuracy": val_accuracy,
        }
        metrics.update({f"training/{k}": v for k, v in kwargs.items()})
        self.run.log(metrics)

    def log_attack_type_performance(
        self,
        attack_type_breakdown: dict[str, dict[str, Any]],
    ) -> None:
        """Log performance breakdown by attack type.

        Args:
            attack_type_breakdown: Dict mapping attack type to performance metrics
        """
        if not self._enabled or self.run is None:
            return

        wb = _get_wandb()
        if wb is None:
            return

        for attack_type, metrics in attack_type_breakdown.items():
            self.run.log(
                {
                    f"attack_performance/{attack_type}/detection_rate": metrics.get("detection_rate", 0.0),
                    f"attack_performance/{attack_type}/detected": metrics.get("detected", 0),
                    f"attack_performance/{attack_type}/missed": metrics.get("missed", 0),
                    f"attack_performance/{attack_type}/total": metrics.get("total", 0),
                }
            )

    def finish(self) -> None:
        """Finish the W&B run."""
        if self.run is not None:
            self.run.finish()
            self.run = None
            self._enabled = False
            logger.info("W&B run finished")

    def __del__(self):
        """Ensure run is finished on garbage collection."""
        if self.run is not None:
            import contextlib

            with contextlib.suppress(Exception):
                self.finish()


# Global tracker instance
_tracker: WandBTracker | None = None


def get_tracker(project: str = "sentinel-env", entity: str | None = None) -> WandBTracker:
    """Get or create the global W&B tracker.

    Args:
        project: W&B project name
        entity: W&B entity (user or team)

    Returns:
        WandBTracker instance
    """
    global _tracker
    if _tracker is None:
        _tracker = WandBTracker(project=project, entity=entity)
    return _tracker
