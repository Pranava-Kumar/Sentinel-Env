"""Seed-deterministic attack sequence generator.

Given a task name and seed, produces a reproducible sequence of attacks
interleaved with safe prompts (30% safe, 70% attacks).
"""

import random
from typing import List, Tuple, Dict, Any

from server.attacks.basic_injections import BASIC_INJECTION_ATTACKS, SAFE_PROMPTS_BASIC
from server.attacks.social_engineering import SOCIAL_ENGINEERING_ATTACKS, SAFE_PROMPTS_SOCIAL
from server.attacks.stealth_exfiltration import STEALTH_EXFILTRATION_ATTACKS, SAFE_PROMPTS_STEALTH


# Task configuration: (attacks, safe_prompts, difficulty)
TASK_CONFIG: Dict[str, Tuple[List, List, str]] = {
    "basic-injection": (BASIC_INJECTION_ATTACKS, SAFE_PROMPTS_BASIC, "easy"),
    "social-engineering": (SOCIAL_ENGINEERING_ATTACKS, SAFE_PROMPTS_SOCIAL, "medium"),
    "stealth-exfiltration": (STEALTH_EXFILTRATION_ATTACKS, SAFE_PROMPTS_STEALTH, "hard"),
}

# Episode length: number of prompts (attacks + safe) per episode
EPISODE_LENGTHS = {
    "basic-injection": 12,      # 8 attacks + 4 safe
    "social-engineering": 10,   # 7 attacks + 3 safe
    "stealth-exfiltration": 8,  # 5 attacks + 3 safe
}


def generate_attack_sequence(
    task_name: str,
    seed: int,
) -> List[Dict[str, Any]]:
    """Generate a deterministic sequence of attacks and safe prompts.

    Args:
        task_name: One of 'basic-injection', 'social-engineering', 'stealth-exfiltration'
        seed: Random seed for reproducibility

    Returns:
        List of dicts with keys: text, is_attack, ground_truth, attack_type, difficulty
    """
    if task_name not in TASK_CONFIG:
        raise ValueError(f"Unknown task: {task_name}. Must be one of {list(TASK_CONFIG.keys())}")

    attacks, safe_prompts, difficulty = TASK_CONFIG[task_name]
    episode_length = EPISODE_LENGTHS.get(task_name, 10)

    rng = random.Random(seed)

    # Determine how many attacks vs safe prompts (70/30 split)
    num_attacks = int(episode_length * 0.7)
    num_safe = episode_length - num_attacks

    # Clamp to available prompts
    num_attacks = min(num_attacks, len(attacks))
    num_safe = min(num_safe, len(safe_prompts))

    # Select attacks and safe prompts deterministically
    selected_attacks = rng.sample(attacks, num_attacks)
    selected_safe = rng.sample(safe_prompts, num_safe)

    # Build sequence: shuffle positions but maintain determinism
    sequence_items: List[Dict[str, Any]] = []

    for attack_text, ground_truth, attack_type in selected_attacks:
        sequence_items.append({
            "text": attack_text,
            "is_attack": True,
            "ground_truth": ground_truth,
            "attack_type": attack_type,
            "difficulty": difficulty,
        })

    for safe_text in selected_safe:
        sequence_items.append({
            "text": safe_text,
            "is_attack": False,
            "ground_truth": "safe",
            "attack_type": "none",
            "difficulty": difficulty,
        })

    # Shuffle the combined sequence deterministically
    rng.shuffle(sequence_items)

    return sequence_items
