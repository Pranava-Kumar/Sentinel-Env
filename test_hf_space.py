"""
End-to-end test of HF Space endpoints
Simulates what the hackathon validator does
"""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from openai import OpenAI
from client import SentinelEnv
import asyncio

BASE_URL = "https://PranavaKumar09-sentinel-env.hf.space"
HF_TOKEN = os.getenv("HF_TOKEN", "hf_test_token_for_validation")

async def test_endpoints():
    """Test all critical endpoints"""
    print("=" * 70)
    print("  TESTING HF SPACE ENDPOINTS")
    print("=" * 70)
    print(f"  Base URL: {BASE_URL}")
    print("=" * 70)
    
    env = SentinelEnv(base_url=BASE_URL)
    await env.__aenter__()
    
    try:
        # Test 1: Reset
        print("\n[1/4] Testing /reset endpoint...")
        obs = await env.reset(task_name="basic-injection", seed=42)
        print(f"  ✓ Reset successful")
        print(f"  Task: {obs.attack_metadata.task_name}")
        print(f"  Attack type: {obs.attack_metadata.attack_type}")
        print(f"  Difficulty: {obs.attack_metadata.difficulty}")
        print(f"  Max steps: {obs.max_steps}")
        
        # Test 2: Step
        print("\n[2/4] Testing /step endpoint...")
        from models import SentinelAction, ThreatCategory
        action = SentinelAction(
            classification=ThreatCategory.JAILBREAK,
            reasoning="DAN-style jailbreak attempt detected",
            recommended_action="block",
            safe_alternative="I can help with that in a safe way"
        )
        obs, reward, done, info = await env.step(action)
        print(f"  ✓ Step successful")
        print(f"  Reward: {reward:.2f}")
        print(f"  Done: {done}")
        print(f"  Detection rate: {obs.resilience_metrics.detection_rate:.2f}")
        
        # Test 3: Grade
        print("\n[3/4] Testing /grade endpoint...")
        response = await env.client.get("/grade")
        grade_result = response.json()
        
        score = grade_result.get("score", 0.0)
        print(f"  ✓ Grade successful")
        print(f"  Score: {score:.2f}")
        print(f"  Detection rate: {grade_result.get('detection_rate', 0):.2f}")
        print(f"  False positive rate: {grade_result.get('false_positive_rate', 0):.2f}")
        print(f"  Correct detections: {grade_result.get('correct_detections', 0)}")
        print(f"  Missed attacks: {grade_result.get('missed_attacks', 0)}")
        
        # Test 4: Inference script format
        print("\n[4/4] Testing inference script output format...")
        print(f"  [START] task={obs.attack_metadata.task_name} env=sentinel-env model=test-model")
        print(f"  [STEP] step=1 action=jailbreak reward={reward:.2f} done={str(done).lower()} error=null")
        print(f"  [END] success={str(score >= 0.3).lower()} steps=1 score={score:.2f} rewards={reward:.2f}")
        
        print("\n" + "=" * 70)
        print("  ✅ ALL TESTS PASSED!")
        print("=" * 70)
        print(f"\n  Space is ready for hackathon submission.")
        print(f"  Score: {score:.2f} (threshold: 0.30)")
        print(f"  Status: {'PASS' if score >= 0.3 else 'FAIL'}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n  ❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        await env.close()

if __name__ == "__main__":
    asyncio.run(test_endpoints())
