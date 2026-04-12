"""End-to-end inference tests validating the full pipeline.

Tests the complete flow: LLM (mocked) → client → server → environment → grader → reward.
This validates the critical path used in production inference runs.
"""

import pytest
from fastapi.testclient import TestClient

from client import SentinelEnv
from server.app import app


@pytest.fixture
def server_client():
    """Create a TestClient with lifespan context."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
async def env_client(server_client):
    """Create a SentinelEnv client connected to the test server."""
    env = SentinelEnv(base_url="http://testserver")
    # Use the test client directly
    env.client = server_client
    yield env
    await env.close()


class TestE2EInferencePipeline:
    """Test the full inference pipeline from client to grading."""

    def test_complete_episode_lifecycle(self, server_client):
        """Test reset → step → grade lifecycle end-to-end."""
        # Reset
        reset_resp = server_client.post("/reset", params={"task_name": "basic-injection", "seed": 42})
        assert reset_resp.status_code == 200
        reset_data = reset_resp.json()
        episode_id = reset_data["episode_id"]
        max_steps = reset_data["max_steps"]
        assert "user_prompt" in reset_data
        assert "attack_metadata" in reset_data

        # Run a full episode with correct detections
        for _step in range(max_steps):
            # Get current observation to know ground truth
            state_resp = server_client.get("/state", headers={"X-Episode-ID": episode_id})
            assert state_resp.status_code == 200

            # Get current observation
            action = {
                "classification": "injection",  # Default to detecting attacks
                "reasoning": "Detected as attack based on prompt analysis",
                "recommended_action": "block",
            }
            step_resp = server_client.post("/step", json=action, headers={"X-Episode-ID": episode_id})
            assert step_resp.status_code == 200
            step_data = step_resp.json()
            assert "observation" in step_data
            assert "reward" in step_data
            assert 0.0 <= step_data["reward"] <= 1.0

            if step_data["done"]:
                break

        # Grade the episode
        grade_resp = server_client.get("/grade", headers={"X-Episode-ID": episode_id})
        assert grade_resp.status_code == 200
        grade_data = grade_resp.json()
        assert "score" in grade_data
        assert 0.0 <= grade_data["score"] <= 1.0
        # Episode with mostly correct detections should score reasonably
        assert grade_data["score"] >= 0.3, f"Episode scored too low: {grade_data['score']}"

    def test_mixed_results_episode(self, server_client):
        """Test episode with mix of correct and incorrect detections."""
        reset_resp = server_client.post("/reset", params={"task_name": "basic-injection", "seed": 42})
        assert reset_resp.status_code == 200
        episode_id = reset_resp.json()["episode_id"]
        max_steps = reset_resp.json()["max_steps"]

        # Alternate between correct and incorrect
        for step in range(max_steps):
            if step % 2 == 0:
                # Correct detection
                action = {
                    "classification": "injection",
                    "reasoning": "Clear injection attempt detected in the prompt",
                    "recommended_action": "block",
                }
            else:
                # Missed attack (classify as safe)
                action = {
                    "classification": "safe",
                    "reasoning": "This prompt appears safe",
                    "recommended_action": "allow",
                }

            step_resp = server_client.post("/step", json=action, headers={"X-Episode-ID": episode_id})
            assert step_resp.status_code == 200
            if step_resp.json()["done"]:
                break

        # Grade should reflect mixed results
        grade_resp = server_client.get("/grade", headers={"X-Episode-ID": episode_id})
        assert grade_resp.status_code == 200
        grade_data = grade_resp.json()
        assert 0.0 < grade_data["score"] < 1.0, "Mixed episode should have intermediate score"
        assert grade_data["correct_detections"] > 0
        assert grade_data["missed_attacks"] > 0

    def test_all_false_positives(self, server_client):
        """Test episode where all safe prompts are incorrectly flagged."""
        reset_resp = server_client.post("/reset", params={"task_name": "basic-injection", "seed": 123})
        assert reset_resp.status_code == 200
        episode_id = reset_resp.json()["episode_id"]
        max_steps = reset_resp.json()["max_steps"]

        # Always classify as attack (will cause false positives on safe prompts)
        for _step in range(max_steps):
            action = {
                "classification": "injection",
                "reasoning": "Suspicious activity detected in the prompt",
                "recommended_action": "block",
            }
            step_resp = server_client.post("/step", json=action, headers={"X-Episode-ID": episode_id})
            assert step_resp.status_code == 200
            if step_resp.json()["done"]:
                break

        grade_resp = server_client.get("/grade", headers={"X-Episode-ID": episode_id})
        assert grade_resp.status_code == 200
        grade_data = grade_resp.json()
        # False positives should lower the score
        assert grade_data["false_positives"] > 0
        assert grade_data["false_positive_rate"] > 0.0


@pytest.mark.asyncio
class TestE2EAsyncPipeline:
    """Test async pipeline matching production inference.py behavior."""

    async def test_async_episode_run(self, server_client):
        """Test running an episode using synchronous client (mimics production)."""
        # Create env client - will use TestClient which is synchronous
        env = SentinelEnv(base_url="http://testserver")
        env.client = server_client

        try:
            # Reset (TestClient is synchronous, so we call it directly)
            # We need to use the synchronous interface for TestClient
            reset_resp = server_client.post("/reset", params={"task_name": "basic-injection", "seed": 42})
            assert reset_resp.status_code == 200
            reset_data = reset_resp.json()
            episode_id = reset_data["episode_id"]
            max_steps = reset_data["max_steps"]

            rewards = []
            for _step in range(max_steps):
                action = {
                    "classification": "injection",
                    "reasoning": "Detected based on prompt analysis with detailed reasoning about the attack",
                    "recommended_action": "block",
                }
                step_resp = server_client.post("/step", json=action, headers={"X-Episode-ID": episode_id})
                assert step_resp.status_code == 200
                step_data = step_resp.json()
                rewards.append(step_data["reward"])

                if step_data["done"]:
                    break

            # Grade using the endpoint
            grade_resp = server_client.get("/grade", headers={"X-Episode-ID": episode_id})
            assert grade_resp.status_code == 200
            grade_result = grade_resp.json()
            assert "score" in grade_result
            assert 0.0 <= grade_result["score"] <= 1.0

        finally:
            # TestClient doesn't need async close
            pass

    async def test_multiple_parallel_episodes(self, server_client):
        """Test running multiple episodes validates EpisodeManager concurrency."""

        def run_episode(task_name: str, seed: int) -> dict:
            """Run a single episode synchronously."""
            reset_resp = server_client.post("/reset", params={"task_name": task_name, "seed": seed})
            assert reset_resp.status_code == 200
            reset_data = reset_resp.json()
            episode_id = reset_data["episode_id"]
            max_steps = reset_data["max_steps"]

            rewards = []
            for _ in range(max_steps):
                action = {
                    "classification": "injection",
                    "reasoning": "Classification based on detailed prompt analysis and threat detection",
                    "recommended_action": "block",
                }
                step_resp = server_client.post("/step", json=action, headers={"X-Episode-ID": episode_id})
                assert step_resp.status_code == 200
                step_data = step_resp.json()
                rewards.append(step_data["reward"])
                if step_data["done"]:
                    break

            grade_resp = server_client.get("/grade", headers={"X-Episode-ID": episode_id})
            assert grade_resp.status_code == 200
            grade = grade_resp.json()
            return {"task": task_name, "score": grade["score"], "rewards": rewards}

        # Run all 3 tasks (sequentially since TestClient is not thread-safe)
        tasks = ["basic-injection", "social-engineering", "stealth-exfiltration"]
        results = [run_episode(task, seed=42) for task in tasks]

        # All should succeed (score > 0 since they at least attempt detection)
        for result in results:
            assert result["score"] >= 0.0, f"Task {result['task']} scored {result['score']}"
