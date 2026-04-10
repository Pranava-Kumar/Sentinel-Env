"""Backward-compatibility shim for attack_engine.

Re-exports from attack_provider for existing imports.
Provides TASK_CONFIG for tests that expect the old interface.
"""

from server.attack_provider import EPISODE_LENGTHS, generate_attack_sequence

# Task configuration for backward compatibility
# Structure: task_name -> (description, episode_length, difficulty)
TASK_CONFIG = {
    "basic-injection": (
        "Test basic prompt injection detection",
        EPISODE_LENGTHS["basic-injection"],
        "easy",
    ),
    "social-engineering": (
        "Test social engineering attack detection",
        EPISODE_LENGTHS["social-engineering"],
        "medium",
    ),
    "stealth-exfiltration": (
        "Test stealth data exfiltration detection",
        EPISODE_LENGTHS["stealth-exfiltration"],
        "hard",
    ),
}

__all__ = ["EPISODE_LENGTHS", "TASK_CONFIG", "generate_attack_sequence"]
