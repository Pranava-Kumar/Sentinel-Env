"""Training runner script for Sentinel RL Environment.

Usage:
    # Basic training
    uv run python train.py

    # Resume from checkpoint
    uv run python train.py --resume

    # Custom configuration
    uv run python train.py --max-episodes 500 --target-detection-rate 0.90

    # With W&B disabled
    uv run python train.py --no-wandb

    # Run W&B sweep
    uv run python train.py --sweep
"""

import argparse
import asyncio
import os
import sys

import structlog

from server.training_loop import SentinelTrainer, TrainingConfig, create_trainer

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
)

logger = structlog.get_logger()


def parse_args():
    parser = argparse.ArgumentParser(description="Train Sentinel RL agent")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--max-episodes", type=int, default=1000, help="Maximum episodes to train")
    parser.add_argument("--target-detection-rate", type=float, default=0.85, help="Target detection rate")
    parser.add_argument("--target-fp-rate", type=float, default=0.10, help="Maximum false positive rate")
    parser.add_argument("--window-size", type=int, default=50, help="Rolling window for metrics")
    parser.add_argument("--replay-buffer-size", type=int, default=500, help="Experience replay buffer size")
    parser.add_argument("--checkpoint-freq", type=int, default=25, help="Checkpoint frequency")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B tracking")
    parser.add_argument("--wandb-project", type=str, default="sentinel-rl-training", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity (user/team)")
    parser.add_argument("--sweep", action="store_true", help="Run W&B hyperparameter sweep")
    return parser.parse_args()


def run_sweep():
    """Run W&B hyperparameter sweep."""
    try:
        import wandb
    except ImportError:
        logger.error("wandb not installed, cannot run sweep")
        sys.exit(1)

    sweep_config = {
        "method": "bayes",
        "metric": {
            "name": "training/val_accuracy",
            "goal": "maximize",
        },
        "parameters": {
            "target_detection_rate": {"min": 0.7, "max": 0.95},
            "target_fp_rate": {"min": 0.05, "max": 0.20},
            "window_size": {"values": [25, 50, 100]},
            "replay_buffer_size": {"values": [200, 500, 1000]},
            "mutation_rate": {"min": 0.05, "max": 0.30},
            "convergence_threshold": {"values": [0.01, 0.02, 0.05]},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project="sentinel-rl-sweep")

    def train_func():
        run = wandb.init()
        config = TrainingConfig(
            target_detection_rate=run.config.target_detection_rate,
            target_fp_rate=run.config.target_fp_rate,
            window_size=run.config.window_size,
            replay_buffer_size=run.config.replay_buffer_size,
            mutation_rate=run.config.mutation_rate,
            convergence_threshold=run.config.convergence_threshold,
            wandb_enabled=True,
        )
        trainer = SentinelTrainer(config)
        asyncio.run(trainer.train())

    wandb.agent(sweep_id, train_func, count=20)


async def main():
    args = parse_args()

    # Set W&B API key if available
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key and not args.no_wandb:
        logger.info("W&B API key found")

    # Run sweep if requested
    if args.sweep:
        run_sweep()
        return

    # Create trainer with config
    trainer = create_trainer(
        {
            "max_episodes": args.max_episodes,
            "target_detection_rate": args.target_detection_rate,
            "target_fp_rate": args.target_fp_rate,
            "window_size": args.window_size,
            "replay_buffer_size": args.replay_buffer_size,
            "checkpoint_frequency": args.checkpoint_freq,
            "wandb_project": args.wandb_project,
            "wandb_entity": args.wandb_entity,
            "wandb_enabled": not args.no_wandb,
        }
    )

    # Train
    await trainer.train(resume=args.resume)


if __name__ == "__main__":
    asyncio.run(main())
