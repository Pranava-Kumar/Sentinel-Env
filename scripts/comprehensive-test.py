"""
COMPREHENSIVE END-TO-END TEST SUITE
Tests EVERY aspect of the deployed Sentinel Environment against the live HF Space.
No assumptions - every output is validated against expected values.
"""

import json
import urllib.request
import urllib.error
import time
import sys

BASE_URL = "https://PranavaKumar09-sentinel-env.hf.space"
PASS = 0
FAIL = 0
WARN = 0

def check(name, condition, detail=""):
    global PASS, FAIL, WARN
    if condition:
        PASS += 1
        print(f"  ✅ PASS: {name}")
    else:
        FAIL += 1
        msg = f"  ❌ FAIL: {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)

def warn(name, detail=""):
    global WARN
    WARN += 1
    msg = f"  ⚠️  WARN: {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)

def api(endpoint, method="GET", data=None):
    url = f"{BASE_URL}{endpoint}"
    headers = {"Content-Type": "application/json"}
    if method == "POST":
        req = urllib.request.Request(url, method="POST", headers=headers)
        if data:
            req.data = json.dumps(data).encode()
    else:
        req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"_error": str(e)}

# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("  SENTINEL ENVIRONMENT — COMPREHENSIVE END-TO-END TEST")
print(f"  Space: {BASE_URL}")
print(f"  Time:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# ───────────────────────────────────────────────────────────────────
# TEST 1: Health endpoint
# ───────────────────────────────────────────────────────────────────
print("\n[TEST 1] Health Check Endpoint")
health = api("/health")
check("Returns JSON", "_error" not in health, str(health.get("_error", "")))
check("Status is healthy", health.get("status") == "healthy", f"Got: {health.get('status')}")
check("Service name correct", health.get("service") == "sentinel-env", f"Got: {health.get('service')}")
check("Version present", "version" in health, f"Keys: {list(health.keys())}")

# ───────────────────────────────────────────────────────────────────
# TEST 2: All 3 tasks reset correctly
# ───────────────────────────────────────────────────────────────────
print("\n[TEST 2] Task Reset Endpoints")
tasks = {
    "basic-injection": {"difficulty": "easy", "expected_steps": 16},
    "social-engineering": {"difficulty": "medium", "expected_steps": 13},
    "stealth-exfiltration": {"difficulty": "hard", "expected_steps": 11},
}

for task_name, cfg in tasks.items():
    obs = api(f"/reset?task_name={task_name}&seed=42", "POST")
    check(f"{task_name} returns data", "_error" not in obs, str(obs.get("_error", "")))
    check(f"{task_name} has user_prompt", "user_prompt" in obs and len(obs.get("user_prompt", "")) > 0)
    check(f"{task_name} difficulty={cfg['difficulty']}",
          obs.get("attack_metadata", {}).get("difficulty") == cfg["difficulty"],
          f"Got: {obs.get('attack_metadata', {}).get('difficulty')}")
    check(f"{task_name} max_steps={cfg['expected_steps']}",
          obs.get("max_steps") == cfg["expected_steps"],
          f"Got: {obs.get('max_steps')} expected: {cfg['expected_steps']}")
    check(f"{task_name} step_number=1", obs.get("step_number") == 1, f"Got: {obs.get('step_number')}")
    check(f"{task_name} has attack_metadata", "attack_metadata" in obs)
    check(f"{task_name} has resilience_metrics", "resilience_metrics" in obs)
    check(f"{task_name} has ground_truth",
          obs.get("attack_metadata", {}).get("ground_truth") in
          ["safe", "injection", "jailbreak", "command_injection",
           "authority_impersonation", "urgency_manipulation", "emotional_manipulation",
           "roleplay_attack", "context_reframe", "encoded_payload", "context_manipulation",
           "cot_hijack", "split_query", "format_injection", "prompt_extraction", "tool_abuse"],
          f"Got: {obs.get('attack_metadata', {}).get('ground_truth')}")

# ───────────────────────────────────────────────────────────────────
# TEST 3: Step endpoint — correct classification
# ───────────────────────────────────────────────────────────────────
print("\n[TEST 3] Step Endpoint — Correct Attack Detection")

for task_name in tasks:
    obs = api(f"/reset?task_name={task_name}&seed=42", "POST")

    # If first prompt is an attack, classify it correctly
    if not obs.get("is_safe_prompt", True):
        gt = obs["attack_metadata"]["ground_truth"]
        action = {
            "classification": gt,
            "reasoning": f"Correctly identified as {gt} based on analysis of the prompt content and intent",
            "recommended_action": "block",
        }
        result = api("/step", "POST", action)
        check(f"{task_name} step returns data", "_error" not in result)
        check(f"{task_name} reward >= 0.5 for correct detection",
              result.get("reward", 0) >= 0.5,
              f"Got: {result.get('reward')}")
        check(f"{task_name} done is boolean", isinstance(result.get("done"), bool))
        check(f"{task_name} has observation", "observation" in result)
        check(f"{task_name} has info", "info" in result)
    else:
        # First prompt is safe - classify as safe
        action = {
            "classification": "safe",
            "reasoning": "This appears to be a legitimate and benign request with no malicious intent",
            "recommended_action": "allow",
        }
        result = api("/step", "POST", action)
        check(f"{task_name} safe prompt returns data", "_error" not in result)
        # Safe correctly classified should give good reward
        check(f"{task_name} reward >= 0.5 for correct safe classification",
              result.get("reward", 0) >= 0.5,
              f"Got: {result.get('reward')}")

# ───────────────────────────────────────────────────────────────────
# TEST 4: Step endpoint — wrong classification (should get low reward)
# ───────────────────────────────────────────────────────────────────
print("\n[TEST 4] Step Endpoint — Wrong Classification Penalty")

for task_name in tasks:
    obs = api(f"/reset?task_name={task_name}&seed=99", "POST")

    # Find an attack prompt
    for _ in range(20):
        if not obs.get("is_safe_prompt", True):
            gt = obs["attack_metadata"]["ground_truth"]
            # Classify as SAFE (wrong answer)
            action = {
                "classification": "safe",
                "reasoning": "I think this is fine",
                "recommended_action": "allow",
            }
            result = api("/step", "POST", action)
            check(f"{task_name} missed attack gets low reward",
                  result.get("reward", 1) < 0.5,
                  f"Got: {result.get('reward')} (should be < 0.5)")
            break
        else:
            result = api("/step", "POST", {
                "classification": "safe",
                "reasoning": "Safe prompt correctly identified",
                "recommended_action": "allow",
            })
            if result.get("done"):
                warn(f"{task_name} no attack found in episode")
                break
            obs = result.get("observation", {})

# ───────────────────────────────────────────────────────────────────
# TEST 5: False positive penalty
# ───────────────────────────────────────────────────────────────────
print("\n[TEST 5] False Positive Penalty")

for task_name in tasks:
    obs = api(f"/reset?task_name={task_name}&seed=77", "POST")
    found_safe = False
    for _ in range(20):
        if obs.get("is_safe_prompt", False):
            found_safe = True
            # Classify safe prompt as attack (false positive)
            action = {
                "classification": "injection",
                "reasoning": "This looks suspicious to me",
                "recommended_action": "block",
            }
            result = api("/step", "POST", action)
            check(f"{task_name} FP gets penalized",
                  result.get("reward", 1) < 0.5,
                  f"Got: {result.get('reward')}")
            break
        else:
            result = api("/step", "POST", {
                "classification": "safe",
                "reasoning": "Correctly identified as safe",
                "recommended_action": "allow",
            })
            if result.get("done"):
                break
            obs = result.get("observation", {})

    if not found_safe:
        warn(f"{task_name} no safe prompt found to test FP")

# ───────────────────────────────────────────────────────────────────
# TEST 6: Full episode — all steps until done
# ───────────────────────────────────────────────────────────────────
print("\n[TEST 6] Full Episode Completion")

for task_name, cfg in tasks.items():
    obs = api(f"/reset?task_name={task_name}&seed=42", "POST")
    steps = 0
    rewards = []
    for _ in range(30):  # Safety limit
        gt = obs.get("attack_metadata", {}).get("ground_truth", "safe")
        if obs.get("is_safe_prompt", True):
            classification = "safe"
        else:
            classification = gt

        action = {
            "classification": classification,
            "reasoning": f"Classified as {classification} based on prompt analysis with sufficient detail",
            "recommended_action": "block" if classification != "safe" else "allow",
        }

        result = api("/step", "POST", action)
        steps += 1
        rewards.append(result.get("reward", 0))

        if result.get("done"):
            break
        obs = result.get("observation", {})

    check(f"{task_name} episode completes", steps <= 30, f"Steps: {steps}")
    check(f"{task_name} correct step count", steps == cfg["expected_steps"],
          f"Got: {steps} expected: {cfg['expected_steps']}")
    check(f"{task_name} has rewards", len(rewards) > 0, f"Count: {len(rewards)}")
    check(f"{task_name} all rewards in [0,1]",
          all(0 <= r <= 1 for r in rewards),
          f"Rewards: {rewards}")

    # Grade the episode
    grade = api("/grade")
    check(f"{task_name} grade returns score", "score" in grade)
    check(f"{task_name} score in [0,1]",
          0 <= grade.get("score", -1) <= 1,
          f"Got: {grade.get('score')}")
    check(f"{task_name} has detection_rate", "detection_rate" in grade)
    check(f"{task_name} has false_positive_rate", "false_positive_rate" in grade)

    print(f"    Score: {grade.get('score', 'N/A'):.2f} | "
          f"Detection: {grade.get('detection_rate', 0):.2f} | "
          f"FP: {grade.get('false_positive_rate', 0):.2f} | "
          f"Correct: {grade.get('correct_detections', 0)} | "
          f"Missed: {grade.get('missed_attacks', 0)}")

# ───────────────────────────────────────────────────────────────────
# TEST 7: State endpoint
# ───────────────────────────────────────────────────────────────────
print("\n[TEST 7] State Endpoint")

api("/reset?task_name=basic-injection&seed=42", "POST")
api("/step", "POST", {
    "classification": "injection",
    "reasoning": "Detected injection attempt with sufficient explanation and detail",
    "recommended_action": "block",
})
state = api("/state")
check("State returns data", "_error" not in state)
check("State has episode_id", "episode_id" in state and state["episode_id"] != "none")
check("State has task_name", state.get("task_name") == "basic-injection")
check("State has step_count", state.get("step_count", 0) >= 1)
check("State has resilience_score", 0 <= state.get("current_resilience_score", -1) <= 1)

# ───────────────────────────────────────────────────────────────────
# TEST 8: Resilience profile
# ───────────────────────────────────────────────────────────────────
print("\n[TEST 8] Resilience Profile Endpoint")

api("/reset?task_name=basic-injection&seed=42", "POST")
for _ in range(20):
    result = api("/step", "POST", {
        "classification": "injection",
        "reasoning": "Detected attack with sufficient reasoning and explanation detail here",
        "recommended_action": "block",
    })
    if result.get("done"):
        break

profile = api("/resilience-profile")
check("Profile returns data", "_error" not in profile)
check("Profile has task_name", "task_name" in profile)
check("Profile has detection_rate", 0 <= profile.get("overall_detection_rate", -1) <= 1)
check("Profile has fp_rate", 0 <= profile.get("overall_false_positive_rate", -1) <= 1)
check("Profile has resilience_score", 0 <= profile.get("resilience_score", -1) <= 1)
check("Profile has attack breakdown", "attack_type_breakdown" in profile)

# ───────────────────────────────────────────────────────────────────
# TEST 9: Seed determinism
# ───────────────────────────────────────────────────────────────────
print("\n[TEST 9] Seed Determinism")

obs_a = api("/reset?task_name=basic-injection&seed=42", "POST")
obs_b = api("/reset?task_name=basic-injection&seed=42", "POST")
obs_c = api("/reset?task_name=basic-injection&seed=99", "POST")

check("Same seed → same attack text",
      obs_a["user_prompt"] == obs_b["user_prompt"],
      f"A: {obs_a['user_prompt'][:40]}... B: {obs_b['user_prompt'][:40]}...")
check("Different seed → different attack",
      obs_a["user_prompt"] != obs_c["user_prompt"],
      f"A: {obs_a['user_prompt'][:40]}... C: {obs_c['user_prompt'][:40]}...")

# ───────────────────────────────────────────────────────────────────
# TEST 10: L1B3RT4S/G0DM0D3 jailbreak detection
# ───────────────────────────────────────────────────────────────────
print("\n[TEST 10] L1B3RT4S/G0DM0D3 Jailbreak Detection")

# Test with various seeds to try to hit the new jailbreak attacks
pliny_found = False
for seed in [42, 99, 77, 123, 200, 300, 400, 500]:
    obs = api(f"/reset?task_name=basic-injection&seed={seed}", "POST")
    # Step through the episode looking for Pliny/GODMODE attacks
    for _ in range(20):
        atype = obs.get("attack_metadata", {}).get("attack_type", "")
        if atype.startswith("pliny_") or atype.startswith("godmode_") or atype.startswith("unicode_"):
            pliny_found = True
            gt = obs["attack_metadata"]["ground_truth"]
            action = {
                "classification": gt,
                "reasoning": f"Detected {atype} jailbreak pattern from L1B3RT4S/G0DM0D3 research",
                "recommended_action": "block",
            }
            result = api("/step", "POST", action)
            check(f"L1B3RT4S attack '{atype}' detected correctly",
                  result.get("reward", 0) >= 0.4,
                  f"Got: {result.get('reward')}")
            print(f"    Attack: {atype} | GT: {gt} | Reward: {result.get('reward', 0):.2f}")
            break
        else:
            result = api("/step", "POST", {
                "classification": "safe" if obs.get("is_safe_prompt") else "injection",
                "reasoning": "Classification with sufficient reasoning detail and explanation",
                "recommended_action": "allow" if obs.get("is_safe_prompt") else "block",
            })
            if result.get("done"):
                break
            obs = result.get("observation", {})

    if pliny_found:
        break

if not pliny_found:
    warn("L1B3RT4S/G0DM0D3 attacks not encountered in tested seeds",
         "They exist in catalog (202 total attacks) but weren't sampled")

# ───────────────────────────────────────────────────────────────────
# TEST 11: Pydantic model validation
# ───────────────────────────────────────────────────────────────────
print("\n[TEST 11] Action Model Validation")

# Test valid action
valid_action = {
    "classification": "injection",
    "reasoning": "Valid reasoning with sufficient length and detail about the detected attack pattern here",
    "recommended_action": "block",
}
result = api("/reset?task_name=basic-injection&seed=42", "POST")
result = api("/step", "POST", valid_action)
check("Valid action accepted", "_error" not in result)

# Test invalid action (missing required field)
invalid_action = {"classification": "injection"}  # missing reasoning, recommended_action
result = api("/step", "POST", invalid_action)
check("Invalid action rejected", "_error" in result or "detail" in result,
      f"Response keys: {list(result.keys())}")

# ───────────────────────────────────────────────────────────────────
# SUMMARY
# ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"  RESULTS: {PASS} passed, {FAIL} failed, {WARN} warnings")
print(f"  Total checks: {PASS + FAIL + WARN}")
print("=" * 70)

if FAIL == 0:
    print("  🎉 ALL TESTS PASSED — Space is fully functional!")
elif FAIL <= 3:
    print("  ⚠️  Minor failures — acceptable for submission")
else:
    print(f"  ❌ {FAIL} failures — needs attention before submission")

print("=" * 70)
sys.exit(0 if FAIL == 0 else 1)
