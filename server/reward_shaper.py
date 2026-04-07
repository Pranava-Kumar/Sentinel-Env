"""Per-step reward computation.

Called by the environment on each step to compute the reward signal.
Delegates to grader.grade_step() for consistency.
"""

from typing import Dict, Any
from models import SentinelAction
from server.grader import grade_step


def compute_reward(
    action: SentinelAction,
    ground_truth: str,
    step_number: int,
    max_steps: int,
) -> float:
    """Compute reward for a single step.

    Args:
        action: Agent's action (classification + reasoning)
        ground_truth: True category ("safe" or attack subtype)
        step_number: Current step (1-indexed)
        max_steps: Maximum steps in episode

    Returns:
        Reward value in [0.0, 1.0]
    """
    result = grade_step(action.classification, ground_truth, action.reasoning)
    reward = result["reward"]

    # Small penalty for dragging on when episode should end
    if step_number >= max_steps - 1:
        reward -= 0.05

    return round(max(min(reward, 1.0), 0.0), 2)
