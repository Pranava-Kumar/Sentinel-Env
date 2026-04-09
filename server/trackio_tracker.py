"""Trackio experiment tracking for Sentinel RL episodes.

Provides comprehensive experiment tracking using Hugging Face's Trackio library:
- Episode-level metrics (detection rate, false positive rate, reward curves)
- Per-step metrics (reasoning quality, classification accuracy)
- Model performance across tasks and seeds
- Adversarial training progress
- Continuous learning metrics
- Training curves and model comparisons

Trackio is API-compatible with wandb.init/log/finish and stores data locally
or on Hugging Face Spaces with a Gradio dashboard.
"""

from typing import Any

import structlog

logger = structlog.get_logger()

# Lazy import trackio only when needed to avoid hard dependency
_trackio = None


def _get_trackio():
    """Lazy import trackio."""
    global _trackio
    if _trackio is None:
        try:
            import trackio as _trackio
        except ImportError:
            logger.warning("trackio not installed, tracking disabled")
            return None
    return _trackio


class TrackioTracker:
    """Tracks Sentinel RL episodes to Trackio."""

    def __init__(
        self,
        project: str = "sentinel-env",
        space_id: str | None = None,
    ):
        """Initialize the Trackio tracker.

        Args:
            project: Trackio project name.
            space_id: Optional Hugging Face Space ID for remote storage
                (e.g. "username/space-name"). If None, stores locally.
        """
        self.project = project
        self.space_id = space_id
        self.run = None
        self._enabled = False
        self._episode_buffer: list[dict[str, Any]] = []

    def start_run(
        self,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        group: str | None = None,
    ) -> None:
        """Start a new Trackio run.

        Args:
            run_name: Name for the run.
            config: Configuration dict (model name, task, hyperparams).
            tags: List of tags for filtering (stored in config for Trackio).
            group: Group name for organizing related runs.
        """
        tr = _get_trackio()
        if tr is None:
            logger.info("Trackio tracking disabled - package not available")
            return

        # Build config with tags and grouping info
        run_config = dict(config or {})
        if tags:
            run_config["tags"] = ", ".join(tags)
        if group:
            run_config["group"] = group

        init_kwargs: dict[str, Any] = {
            "project": self.project,
            "config": run_config,
        }
        if run_name:
            init_kwargs["name"] = run_name
        if group:
            init_kwargs["group"] = group
        if self.space_id:
            init_kwargs["space_id"] = self.space_id

        self.run = tr.init(**init_kwargs)
        self._enabled = True
        self._episode_buffer = []
        logger.info("Trackio run started", run_name=run_name, project=self.project)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log arbitrary metrics to Trackio.

        Args:
            metrics: Dictionary of metric name -> value pairs.
            step: Optional step number. If None, Trackio auto-increments.
        """
        if not self._enabled or self.run is None:
            self._episode_buffer.append({"metrics": metrics, "step": step})
            return

        tr = _get_trackio()
        if tr is None:
            return

        log_kwargs: dict[str, Any] = {"metrics": metrics}
        if step is not None:
            log_kwargs["step"] = step
        tr.log(**log_kwargs)

    def log_episode(
        self,
        episode_id: str,
        task_name: str,
        seed: int,
        metrics: dict[str, Any],
        step_results: list[dict[str, Any]] | None = None,
    ) -> None:
        """Log a completed episode to Trackio.

        Args:
            episode_id: Unique episode identifier.
            task_name: Task name (basic-injection, social-engineering, stealth-exfiltration).
            seed: Random seed used.
            metrics: Episode-level metrics from grader.
            step_results: Per-step results for detailed logging.
        """
        if not self._enabled or self.run is None:
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

        tr = _get_trackio()
        if tr is None:
            return

        # Log episode-level metrics
        tr.log(
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
                tr.log(
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
            epoch: Training epoch number.
            train_loss: Training loss.
            val_loss: Validation loss.
            val_accuracy: Validation accuracy.
            **kwargs: Additional metrics to log.
        """
        if not self._enabled or self.run is None:
            return

        tr = _get_trackio()
        if tr is None:
            return

        metrics = {
            "training/epoch": epoch,
            "training/train_loss": train_loss,
            "training/val_loss": val_loss,
            "training/val_accuracy": val_accuracy,
        }
        metrics.update({f"training/{k}": v for k, v in kwargs.items()})
        tr.log(metrics)

    def log_attack_type_performance(
        self,
        attack_type_breakdown: dict[str, dict[str, Any]],
    ) -> None:
        """Log performance breakdown by attack type.

        Args:
            attack_type_breakdown: Dict mapping attack type to performance metrics.
        """
        if not self._enabled or self.run is None:
            return

        tr = _get_trackio()
        if tr is None:
            return

        for attack_type, metrics in attack_type_breakdown.items():
            tr.log(
                {
                    f"attack_performance/{attack_type}/detection_rate": metrics.get("detection_rate", 0.0),
                    f"attack_performance/{attack_type}/detected": metrics.get("detected", 0),
                    f"attack_performance/{attack_type}/missed": metrics.get("missed", 0),
                    f"attack_performance/{attack_type}/total": metrics.get("total", 0),
                }
            )

    def log_model_comparison(
        self,
        model_name: str,
        metrics: dict[str, float],
    ) -> None:
        """Log model comparison metrics.

        Use this to compare different models or configurations by logging
        aggregated performance under a model name.

        Args:
            model_name: Name/identifier of the model being evaluated.
            metrics: Dictionary of aggregated metrics (e.g. avg_score,
                avg_detection_rate, avg_fp_rate).
        """
        if not self._enabled or self.run is None:
            return

        tr = _get_trackio()
        if tr is None:
            return

        comparison_metrics = {f"model_comparison/{model_name}/{k}": v for k, v in metrics.items()}
        comparison_metrics["model_comparison/model_name"] = model_name
        tr.log(comparison_metrics)

    def flush_buffered_episodes(self) -> None:
        """Flush any episodes that were buffered before the run started."""
        if not self._episode_buffer:
            return

        tr = _get_trackio()
        if tr is None:
            return

        for entry in self._episode_buffer:
            if "metrics" in entry and "episode_id" not in entry:
                # Simple metrics-only entry
                tr.log(entry["metrics"])
            elif "episode_id" in entry:
                self.log_episode(
                    episode_id=entry["episode_id"],
                    task_name=entry.get("task_name", "unknown"),
                    seed=entry.get("seed", 0),
                    metrics=entry.get("metrics", {}),
                    step_results=entry.get("step_results"),
                )

        self._episode_buffer.clear()
        logger.info("Flushed buffered episodes", count=len(self._episode_buffer))

    def finish(self) -> None:
        """Finish the Trackio run."""
        # Flush any remaining buffered episodes first
        self.flush_buffered_episodes()

        tr = _get_trackio()
        if tr is not None:
            tr.finish()

        self.run = None
        self._enabled = False
        logger.info("Trackio run finished")

    def __del__(self):
        """Ensure run is finished on garbage collection."""
        if self.run is not None:
            import contextlib

            with contextlib.suppress(Exception):
                self.finish()


# Global tracker instance
_tracker: TrackioTracker | None = None


def get_tracker(
    project: str = "sentinel-env",
    space_id: str | None = None,
) -> TrackioTracker:
    """Get or create the global Trackio tracker.

    Args:
        project: Trackio project name.
        space_id: Optional HF Space ID for remote storage.

    Returns:
        TrackioTracker instance.
    """
    global _tracker
    if _tracker is None:
        _tracker = TrackioTracker(project=project, space_id=space_id)
    return _tracker


def reset_tracker() -> None:
    """Reset the global tracker (useful for testing or switching projects)."""
    global _tracker
    if _tracker is not None:
        _tracker.finish()
    _tracker = None
