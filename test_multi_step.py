"""Multi-step episode test"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from client import SentinelEnv
from models import SentinelAction, ThreatCategory


async def test_multi_step():
    """Test a complete multi-step episode"""
    print("\n" + "=" * 70)
    print("  MULTI-STEP EPISODE TEST")
    print("=" * 70)

    env = SentinelEnv(base_url="https://PranavaKumar09-sentinel-env.hf.space")
    await env.__aenter__()

    try:
        # Reset
        obs = await env.reset(task_name="basic-injection", seed=42)
        print("\n✓ Episode started")
        print(f"  Task: {obs.attack_metadata.task_name}")
        print(f"  Attack: {obs.attack_metadata.attack_type}")
        print(f"  Max steps: {obs.max_steps}")

        # Run multiple steps
        rewards = []
        for step in range(1, 6):
            action = SentinelAction(
                classification=ThreatCategory.JAILBREAK,
                reasoning=f"Jailbreak detected on step {step}",
                recommended_action="block",
                safe_alternative="Alternative response",
            )
            obs, reward, done, info = await env.step(action)
            rewards.append(reward)

            print(f"\n  Step {step}:")
            print(f"    Reward: {reward:.2f}")
            print(f"    Done: {done}")
            print(f"    Detection rate: {obs.resilience_metrics.detection_rate:.2f}")
            print(f"    Correct: {info['step_result']['is_correct']}")

            if done:
                print(f"\n  ✓ Episode completed at step {step}")
                break

        # Get final grade
        resp = await env.client.get("/grade")
        grade = resp.json()

        print(f"\n{'─' * 70}")
        print("  FINAL RESULTS")
        print(f"{'─' * 70}")
        print(f"  Score: {grade['score']:.2f}")
        print(f"  Detection Rate: {grade['detection_rate']:.2f}")
        print(f"  False Positive Rate: {grade['false_positive_rate']:.2f}")
        print(f"  Correct Detections: {grade['correct_detections']}")
        print(f"  Missed Attacks: {grade['missed_attacks']}")
        print(f"  Total Steps: {grade['total_steps']}")
        print(f"  Avg Reasoning Score: {grade['avg_reasoning_score']:.2f}")
        print(f"\n  Status: {'✅ PASS' if grade['score'] >= 0.3 else '❌ FAIL'}")
        print(f"{'=' * 70}\n")

    finally:
        await env.close()


if __name__ == "__main__":
    asyncio.run(test_multi_step())
