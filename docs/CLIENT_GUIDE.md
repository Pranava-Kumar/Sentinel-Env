# Client Library Guide

Complete guide for using the Sentinel Environment Python client.

## Installation

The client is included in the project repository. Import it directly:

```python
from client import SentinelEnv
from models import SentinelAction, SentinelObservation, SentinelState, ThreatCategory, RecommendedAction
```

## Quick Start

```python
import asyncio
from client import SentinelEnv
from models import SentinelAction, ThreatCategory, RecommendedAction

async def main():
    # Connect to server
    async with SentinelEnv(base_url="http://localhost:7860") as env:
        # Start episode
        observation = await env.reset(task_name="basic-injection", seed=42)
        print(f"Episode ID: {env.episode_id}")
        print(f"Prompt: {observation.user_prompt}")
        
        # Submit classification
        action = SentinelAction(
            classification=ThreatCategory.INJECTION,
            reasoning="Prompt attempts to override system instructions",
            recommended_action=RecommendedAction.BLOCK
        )
        
        # Execute step
        obs, reward, done, info = await env.step(action)
        print(f"Reward: {reward}, Done: {done}")
        
        # Get episode grade when complete
        if done:
            grade = await env.grade()
            print(f"Final Score: {grade['score']}")

asyncio.run(main())
```

## Client Methods

### `__init__(base_url, api_key)`

Initialize the client with server connection details.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | string | `"http://localhost:7860"` | Server URL |
| `api_key` | string or None | `None` | API key for authentication |

**Example:**
```python
# Development (no auth)
env = SentinelEnv(base_url="http://localhost:7860")

# Production (with auth)
env = SentinelEnv(
    base_url="https://pranavakumar09-sentinel-env.hf.space",
    api_key="your-api-key"
)
```

---

### `async reset(task_name, seed)`

Start a new evaluation episode.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_name` | string | `"basic-injection"` | Task type: `basic-injection`, `social-engineering`, or `stealth-exfiltration` |
| `seed` | int | `42` | Random seed for reproducible attack sequences |

**Returns:** `SentinelObservation`

**Example:**
```python
# Start basic injection episode
obs = await env.reset(task_name="basic-injection", seed=42)

# Start social engineering episode  
obs = await env.reset(task_name="social-engineering", seed=123)

# Start stealth exfiltration episode
obs = await env.reset(task_name="stealth-exfiltration", seed=456)
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `user_prompt` | string | The prompt to analyze |
| `conversation_history` | list[dict] | Prior conversation turns |
| `attack_metadata` | AttackMetadata | Attack type, difficulty, ground truth |
| `resilience_metrics` | ResilienceMetrics | Running performance metrics |
| `step_number` | int | Current step (1-indexed) |
| `max_steps` | int | Total steps in episode |
| `is_safe_prompt` | bool | Whether this is a benign prompt |

**Side Effects:**
- Stores episode ID internally for subsequent `step()` calls
- Accessible via `env.episode_id` property

---

### `async step(action)`

Submit the agent's classification for the current prompt.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `action` | SentinelAction | Yes | Agent's response |

**Returns:** `tuple[SentinelObservation, float, bool, dict[str, Any]]`

Return value is a tuple of:
- `observation`: Next prompt to classify
- `reward`: Step reward (0.0 to 1.0)
- `done`: Whether episode is complete
- `info`: Additional metadata

**SentinelAction Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `classification` | ThreatCategory | Yes | How you classified the prompt |
| `reasoning` | string | Yes | Explanation (10-500 chars) |
| `recommended_action` | RecommendedAction | Yes | Action to take |
| `safe_alternative` | string or None | No | Suggested alternative if blocking |

**Example:**
```python
action = SentinelAction(
    classification=ThreatCategory.JAILBREAK,
    reasoning="Prompt uses hypothetical scenario to bypass safety filters",
    recommended_action=RecommendedAction.BLOCK,
    safe_alternative="Answer the underlying question directly without roleplay"
)

obs, reward, done, info = await env.step(action)
print(f"Correct: {info.get('was_correct')}")
print(f"Reward: {reward}")

if not done:
    # Continue episode
    next_action = SentinelAction(...)
    obs, reward, done, info = await env.step(next_action)
```

**Error Handling:**
```python
try:
    obs, reward, done, info = await env.step(action)
except RuntimeError as e:
    print(f"Episode error: {e}")  # e.g., episode already done
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

### `async state()`

Get current episode state without submitting an action.

**Returns:** `SentinelState`

**Example:**
```python
state = await env.state()
print(f"Episode: {state.episode_id}")
print(f"Task: {state.task_name}")
print(f"Progress: {state.step_count}/{state.max_steps}")
print(f"Detection Rate: {state.current_resilience_score:.2%}")
```

**State Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `episode_id` | string | Episode identifier |
| `task_name` | string | Task type |
| `step_count` | int | Steps completed |
| `total_attacks_presented` | int | Attacks shown (excludes safe) |
| `attacks_correctly_detected` | int | Correct classifications |
| `false_positives` | int | Safe prompts incorrectly flagged |
| `current_resilience_score` | float | Running score (0.0-1.0) |
| `done` | bool | Episode completion status |

---

### `async grade()`

Get final grade for completed episode.

**Returns:** `dict[str, Any]`

**Example:**
```python
grade = await env.grade()
print(f"Score: {grade['score']:.2%}")
print(f"Detection Rate: {grade['detection_rate']:.2%}")
print(f"False Positive Rate: {grade['false_positive_rate']:.2%}")
print(f"Correct: {grade['correct_detections']}/{grade['total_attacks']}")
print(f"Avg Reasoning: {grade['avg_reasoning_score']:.2%}")
```

**Grade Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `score` | float | Composite score (0.0-1.0) |
| `detection_rate` | float | Attack detection rate |
| `false_positive_rate` | float | False alarm rate |
| `correct_detections` | int | Correct attack classifications |
| `missed_attacks` | int | Missed attacks |
| `false_positives` | int | False alarms |
| `total_attacks` | int | Total attack prompts |
| `total_safe` | int | Total safe prompts |
| `total_steps` | int | Total steps taken |
| `avg_reasoning_score` | float | Reasoning quality (0.0-1.0) |

---

### `async close()`

Close the HTTP client and clean up resources.

**Example:**
```python
env = SentinelEnv()
try:
    # ... use client ...
finally:
    await env.close()
```

**Note:** When using the async context manager (`async with`), `close()` is called automatically.

---

### `async from_docker_image(image_name, port)` — Class Method

Create a client connected to a Docker-based environment.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_name` | string or None | `None` | Docker image name (unused, kept for API compatibility) |
| `port` | int | `7860` | Server port |

**Returns:** `SentinelEnv`

**Example:**
```python
# Connect to local Docker container
env = await SentinelEnv.from_docker_image(port=7860)
```

---

## Connection Patterns

### Context Manager (Recommended)

Automatically manages connection lifecycle:

```python
async with SentinelEnv(base_url="http://localhost:7860") as env:
    obs = await env.reset()
    # ... episode loop ...
# Connection closed automatically
```

### Manual Management

For advanced use cases:

```python
env = SentinelEnv(base_url="http://localhost:7860")
try:
    await env.client.__aenter__()  # Initialize client
    obs = await env.reset()
    # ... episode loop ...
finally:
    await env.close()
```

---

## Complete Examples

### Single Episode Run

```python
import asyncio
from client import SentinelEnv
from models import SentinelAction, ThreatCategory, RecommendedAction

async def run_episode():
    async with SentinelEnv() as env:
        # Start episode
        obs = await env.reset(task_name="basic-injection", seed=42)
        print(f"Episode: {env.episode_id}")
        print(f"Task: {obs.attack_metadata.task_name}")
        print(f"Steps: {obs.step_number}/{obs.max_steps}\n")
        
        # Episode loop
        while True:
            # Analyze prompt
            is_attack = obs.attack_metadata.attack_type != "safe"
            
            # Classify (replace with your model)
            if is_attack:
                action = SentinelAction(
                    classification=ThreatCategory[obs.attack_metadata.attack_type.upper()],
                    reasoning="Attack pattern detected in prompt",
                    recommended_action=RecommendedAction.BLOCK
                )
            else:
                action = SentinelAction(
                    classification=ThreatCategory.SAFE,
                    reasoning="No attack patterns detected",
                    recommended_action=RecommendedAction.ALLOW
                )
            
            # Submit
            obs, reward, done, info = await env.step(action)
            correct = info.get("was_correct", False)
            print(f"Step {obs.step_number}: {'✓' if correct else '✗'} (reward: {reward:.2f})")
            
            if done:
                break
        
        # Grade episode
        grade = await env.grade()
        print(f"\nFinal Results:")
        print(f"  Score: {grade['score']:.2%}")
        print(f"  Detection Rate: {grade['detection_rate']:.2%}")
        print(f"  False Positives: {grade['false_positives']}")

asyncio.run(run_episode())
```

### Multiple Episodes with Different Tasks

```python
async def run_all_tasks():
    tasks = ["basic-injection", "social-engineering", "stealth-exfiltration"]
    results = {}
    
    async with SentinelEnv() as env:
        for task in tasks:
            print(f"\n{'='*60}")
            print(f"Running: {task}")
            print(f"{'='*60}")
            
            obs = await env.reset(task_name=task, seed=42)
            
            while True:
                # Your classification logic here
                action = your_model_classify(obs)
                obs, reward, done, info = await env.step(action)
                
                if done:
                    break
            
            grade = await env.grade()
            results[task] = grade
            print(f"Score: {grade['score']:.2%}")
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for task, grade in results.items():
        print(f"{task:25s} {grade['score']:.2%}")

asyncio.run(run_all_tasks())
```

### Batch Episode Evaluation

```python
async def run_parallel_episodes():
    """Run multiple episodes in parallel."""
    import asyncio
    
    configs = [
        ("basic-injection", 42),
        ("basic-injection", 123),
        ("social-engineering", 42),
    ]
    
    async def run_single(task_name, seed):
        async with SentinelEnv() as env:
            obs = await env.reset(task_name=task_name, seed=seed)
            
            while True:
                action = your_model_classify(obs)
                obs, reward, done, info = await env.step(action)
                if done:
                    break
            
            return await env.grade()
    
    # Run all episodes concurrently
    tasks = [run_single(task, seed) for task, seed in configs]
    results = await asyncio.gather(*tasks)
    
    # Average score
    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"Average Score: {avg_score:.2%}")

asyncio.run(run_parallel_episodes())
```

### Integration with LLM (OpenAI API)

```python
from openai import AsyncOpenAI
import json

async def llm_episode():
    client = AsyncOpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key="your-hf-token"
    )
    
    async with SentinelEnv() as env:
        obs = await env.reset(task_name="basic-injection")
        
        system_prompt = """You are an AI safety analyst. Classify prompts as attacks or safe.
Respond with JSON: {"classification": "...", "reasoning": "...", "recommended_action": "..."}"""
        
        while True:
            # Get LLM classification
            response = await client.chat.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": obs.user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            # Parse response
            llm_response = json.loads(response.choices[0].message.content)
            action = SentinelAction(
                classification=ThreatCategory[llm_response["classification"].upper()],
                reasoning=llm_response["reasoning"],
                recommended_action=RecommendedAction[llm_response["recommended_action"].upper()]
            )
            
            obs, reward, done, info = await env.step(action)
            
            if done:
                break
        
        grade = await env.grade()
        print(f"LLM Score: {grade['score']:.2%}")

asyncio.run(llm_episode())
```

### Monitoring with State Checks

```python
async def monitored_episode():
    async with SentinelEnv() as env:
        obs = await env.reset(task_name="stealth-exfiltration")
        
        while True:
            action = your_model_classify(obs)
            obs, reward, done, info = await env.step(action)
            
            # Check state periodically
            if obs.step_number % 5 == 0:
                state = await env.state()
                print(f"\nState Check:")
                print(f"  Step: {state.step_count}")
                print(f"  Attacks Detected: {state.attacks_correctly_detected}")
                print(f"  False Positives: {state.false_positives}")
                print(f"  Score: {state.current_resilience_score:.2%}")
            
            if done:
                break
        
        # Final grade
        grade = await env.grade()
        print(f"\nFinal: {grade['score']:.2%}")

asyncio.run(monitored_episode())
```

---

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `RuntimeError: Client not initialized` | Called methods outside context manager | Use `async with` or manually initialize client |
| `httpx.HTTPStatusError: 401` | Invalid/missing API key | Check `SENTINEL_API_KEY` configuration |
| `httpx.HTTPStatusError: 404` | Episode not found | Episode may have expired (1hr TTL) or invalid ID |
| `httpx.HTTPStatusError: 429` | Rate limit exceeded | Wait and retry, implement backoff |
| `httpx.HTTPStatusError: 400` | Missing `X-Episode-ID` or invalid action | Check headers and action format |

### Retry Logic

```python
import asyncio
from httpx import HTTPStatusError

async def step_with_retry(env, action, max_retries=3):
    """Submit step with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            return await env.step(action)
        except HTTPStatusError as e:
            if e.response.status_code == 429:  # Rate limited
                wait_time = 2 ** attempt
                print(f"Rate limited, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                raise
    raise Exception("Max retries exceeded")
```

---

## Performance Considerations

1. **Connection Reuse:** The client reuses the HTTP connection across multiple requests. Avoid creating new client instances per episode.

2. **Async Context Manager:** Always prefer `async with` pattern for automatic cleanup and connection management.

3. **Episode Timeout:** Episodes expire after 3600 seconds (1 hour). Long-running evaluations should checkpoint state.

4. **Batch Operations:** For high-throughput scenarios, use `/api/v1/batch/*` endpoints to reduce HTTP overhead.

5. **Concurrent Episodes:** The server supports up to 1000 concurrent episodes. Use `asyncio.gather()` for parallel execution.

---

## Migration from Direct HTTP

If you're currently using raw HTTP calls, migrating to the client:

**Before (raw HTTP):**
```python
async with httpx.AsyncClient(base_url=BASE_URL) as client:
    # Manual episode ID tracking
    resp = await client.post("/reset", params={"task_name": "basic-injection"})
    episode_id = resp.json()["episode_id"]
    
    headers = {"X-Episode-ID": episode_id}
    resp = await client.post("/step", json=action, headers=headers)
```

**After (client):**
```python
async with SentinelEnv() as env:
    # Automatic episode ID tracking
    obs = await env.reset(task_name="basic-injection")
    obs, reward, done, info = await env.step(action)
```

Benefits:
- Automatic `X-Episode-ID` header management
- Type-safe responses (Pydantic models)
- Built-in error handling
- Cleaner API

---

## Testing

### Mock Client for Unit Tests

```python
from unittest.mock import AsyncMock

async def test_your_model():
    env = SentinelEnv()
    env.client = AsyncMock()
    env._episode_id = "test_ep_001"
    
    # Mock reset response
    env.client.post.return_value.json.return_value = {
        "episode_id": "test_ep_001",
        "user_prompt": "Test attack prompt",
        "attack_metadata": {"attack_type": "injection"},
        # ... other fields ...
    }
    
    # Test your classification logic
    obs = await env.reset()
    action = your_model_classify(obs)
    
    assert action.classification == ThreatCategory.INJECTION
```

---

## Best Practices

1. **Always use context managers:** Ensures proper cleanup
2. **Track episode IDs:** Use `env.episode_id` property
3. **Handle errors gracefully:** Implement retries for transient failures
4. **Log request IDs:** Available in response headers for debugging
5. **Monitor metrics:** Use `state()` for progress tracking
6. **Batch when possible:** Use `/api/v1/batch/*` for throughput
7. **Set appropriate timeouts:** Default is 30s, adjust for your use case
8. **Use seeds for reproducibility:** Critical for testing and benchmarking
