"""
Test deployed HF Space endpoints end-to-end
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from client import SentinelEnv
from models import SentinelAction, ThreatCategory


async def test_deployed_space():
    """Comprehensive test of deployed HF Space"""
    BASE_URL = "https://PranavaKumar09-sentinel-env.hf.space"
    
    print("=" * 70)
    print("  TESTING DEPLOYED HF SPACE")
    print("=" * 70)
    print(f"  URL: {BASE_URL}")
    print("=" * 70)
    
    env = SentinelEnv(base_url=BASE_URL)
    await env.__aenter__()
    
    try:
        # Test 1: Reset with different tasks
        print("\n[1/6] Testing /reset with basic-injection...")
        obs = await env.reset(task_name="basic-injection", seed=42)
        print(f"  ✓ Reset successful")
        print(f"  Task: {obs.attack_metadata.task_name}")
        print(f"  Attack type: {obs.attack_metadata.attack_type}")
        print(f"  Difficulty: {obs.attack_metadata.difficulty}")
        print(f"  Max steps: {obs.max_steps}")
        print(f"  User prompt: {obs.user_prompt[:80]}...")
        
        # Test 2: Single step
        print("\n[2/6] Testing /step with correct classification...")
        action = SentinelAction(
            classification=obs.attack_metadata.ground_truth,
            reasoning="Correct classification of the detected attack pattern",
            recommended_action="block",
            safe_alternative="I can help with that safely"
        )
        obs, reward, done, info = await env.step(action)
        print(f"  ✓ Step successful")
        print(f"  Reward: {reward:.2f}")
        print(f"  Done: {done}")
        print(f"  Detection rate: {obs.resilience_metrics.detection_rate:.2f}")
        print(f"  Is correct: {info['step_result']['is_correct']}")
        
        # Test 3: Multiple steps episode
        print("\n[3/6] Testing multi-step episode...")
        rewards = [reward]
        for step in range(2, 6):
            action = SentinelAction(
                classification=ThreatCategory.JAILBREAK,
                reasoning=f"Step {step} classification of the attack pattern",
                recommended_action="block",
                safe_alternative="Safe alternative response"
            )
            obs, reward, done, info = await env.step(action)
            rewards.append(reward)
            print(f"  Step {step}: reward={reward:.2f}, correct={info['step_result']['is_correct']}")
            if done:
                break
        
        # Test 4: Grade endpoint
        print("\n[4/6] Testing /grade endpoint...")
        headers = {}
        if env._episode_id:
            headers["X-Episode-ID"] = env._episode_id
        resp = await env.client.get("/grade", headers=headers)
        grade = resp.json()
        print(f"  ✓ Grade successful")
        print(f"  Score: {grade['score']:.2f}")
        print(f"  Detection rate: {grade['detection_rate']:.2f}")
        print(f"  False positive rate: {grade['false_positive_rate']:.2f}")
        print(f"  Correct detections: {grade['correct_detections']}")
        print(f"  Missed attacks: {grade['missed_attacks']}")
        print(f"  Total steps: {grade['total_steps']}")
        print(f"  Avg reasoning score: {grade['avg_reasoning_score']:.2f}")
        
        # Test 5: Resilience profile
        print("\n[5/6] Testing /resilience-profile endpoint...")
        resp = await env.client.get("/resilience-profile")
        profile = resp.json()
        print(f"  ✓ Profile successful")
        print(f"  Profile keys: {list(profile.keys())[:5]}...")
        
        # Test 6: Health check
        print("\n[6/6] Testing /health endpoint...")
        resp = await env.client.get("/health")
        health = resp.json()
        print(f"  ✓ Health check passed")
        print(f"  Status: {health['status']}")
        print(f"  Version: {health['version']}")
        print(f"  Features: {', '.join([k for k, v in health['features'].items() if v])}")
        
        # Summary
        print("\n" + "=" * 70)
        print("  DEPLOYED SPACE TEST RESULTS")
        print("=" * 70)
        print(f"  ✅ All endpoints responsive")
        print(f"  ✅ Score: {grade['score']:.2f} (threshold: 0.30)")
        print(f"  ✅ Detection rate: {grade['detection_rate']:.2f}")
        print(f"  ✅ Total steps tested: {len(rewards)}")
        print(f"  ✅ Avg reward: {sum(rewards)/len(rewards):.2f}")
        print(f"\n  STATUS: {'✅ PRODUCTION READY' if grade['score'] >= 0.3 else '❌ NEEDS IMPROVEMENT'}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n  ❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        await env.close()


if __name__ == "__main__":
    asyncio.run(test_deployed_space())
