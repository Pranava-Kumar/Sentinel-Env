"""Integration tests for the full client-server-environment pipeline."""

import pytest
from fastapi.testclient import TestClient

from server.app import app, rate_limiter


@pytest.fixture
def client():
    """FastAPI TestClient for the server."""
    # Reset rate limiter before each test to prevent 429 errors
    rate_limiter.requests.clear()
    return TestClient(app)


class TestFullEpisodePipeline:
    """Test complete episode lifecycle end-to-end."""

    def test_basic_injection_episode(self, client):
        """Run a full basic-injection episode through all endpoints."""
        # Reset
        reset_resp = client.post("/reset", params={"task_name": "basic-injection", "seed": 42})
        assert reset_resp.status_code == 200
        reset_data = reset_resp.json()
        assert "episode_id" in reset_data
        assert "user_prompt" in reset_data
        episode_id = reset_data["episode_id"]

        # Run steps
        for _ in range(5):
            action = {
                "classification": "injection",
                "reasoning": "This appears to be an injection attack pattern",
                "recommended_action": "block",
            }
            step_resp = client.post(
                "/step",
                json=action,
                headers={"X-Episode-ID": episode_id},
            )
            assert step_resp.status_code == 200
            data = step_resp.json()
            assert "observation" in data
            assert "reward" in data
            assert "done" in data
            assert "info" in data

        # Grade
        grade_resp = client.get("/grade", headers={"X-Episode-ID": episode_id})
        assert grade_resp.status_code == 200
        grade_data = grade_resp.json()
        assert "score" in grade_data
        assert 0.0 <= grade_data["score"] <= 1.0
        assert "detection_rate" in grade_data
        assert "false_positive_rate" in grade_data

    def test_social_engineering_episode(self, client):
        """Run a full social-engineering episode."""
        reset_resp = client.post("/reset", params={"task_name": "social-engineering", "seed": 123})
        assert reset_resp.status_code == 200
        episode_id = reset_resp.json()["episode_id"]

        # Step with safe classification
        for _ in range(3):
            action = {
                "classification": "safe",
                "reasoning": "This prompt appears to be benign with no attack detected",
                "recommended_action": "allow",
            }
            step_resp = client.post(
                "/step",
                json=action,
                headers={"X-Episode-ID": episode_id},
            )
            assert step_resp.status_code == 200

        # Grade
        grade_resp = client.get("/grade", headers={"X-Episode-ID": episode_id})
        assert grade_resp.status_code == 200
        assert 0.0 <= grade_resp.json()["score"] <= 1.0

    def test_stealth_exfiltration_episode(self, client):
        """Run a full stealth-exfiltration episode."""
        reset_resp = client.post("/reset", params={"task_name": "stealth-exfiltration", "seed": 456})
        assert reset_resp.status_code == 200
        episode_id = reset_resp.json()["episode_id"]

        # Mix of correct and incorrect classifications
        classifications = ["encoded_payload", "safe", "cot_hijack", "safe", "format_injection"]
        for classification in classifications:
            action = {
                "classification": classification,
                "reasoning": f"Detected {classification} attack pattern in the prompt requiring analysis and review",
                "recommended_action": "block" if classification != "safe" else "allow",
            }
            step_resp = client.post(
                "/step",
                json=action,
                headers={"X-Episode-ID": episode_id},
            )
            assert step_resp.status_code == 200

        grade_resp = client.get("/grade", headers={"X-Episode-ID": episode_id})
        assert grade_resp.status_code == 200


class TestStateEndpoint:
    """Test state endpoint returns correct episode state."""

    def test_state_returns_correct_fields(self, client):
        reset_resp = client.post("/reset", params={"task_name": "basic-injection", "seed": 42})
        episode_id = reset_resp.json()["episode_id"]

        state_resp = client.get("/state", headers={"X-Episode-ID": episode_id})
        assert state_resp.status_code == 200
        state_data = state_resp.json()

        assert "episode_id" in state_data
        assert "task_name" in state_data
        assert "step_count" in state_data
        assert "total_attacks_presented" in state_data
        assert "attacks_correctly_detected" in state_data
        assert "false_positives" in state_data
        assert "current_resilience_score" in state_data
        assert "done" in state_data


class TestResilienceProfile:
    """Test resilience profile endpoint."""

    def test_resilience_profile_after_steps(self, client):
        reset_resp = client.post("/reset", params={"task_name": "basic-injection", "seed": 42})
        episode_id = reset_resp.json()["episode_id"]

        # Run a few steps
        for _ in range(3):
            action = {
                "classification": "injection",
                "reasoning": "Injection attack detected in the incoming prompt text",
                "recommended_action": "block",
            }
            client.post("/step", json=action, headers={"X-Episode-ID": episode_id})

        profile_resp = client.get("/resilience-profile", headers={"X-Episode-ID": episode_id})
        assert profile_resp.status_code == 200
        profile_data = profile_resp.json()
        assert isinstance(profile_data, dict)


class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_invalid_episode_id(self, client):
        """Step with invalid episode ID returns 404."""
        action = {
            "classification": "safe",
            "reasoning": "This prompt appears to be benign with no attack detected",
            "recommended_action": "allow",
        }
        step_resp = client.post(
            "/step",
            json=action,
            headers={"X-Episode-ID": "nonexistent-episode"},
        )
        assert step_resp.status_code == 404

    def test_missing_episode_id(self, client):
        """Step without episode ID returns 400."""
        action = {
            "classification": "safe",
            "reasoning": "This prompt appears to be benign with no attack detected",
            "recommended_action": "allow",
        }
        step_resp = client.post("/step", json=action)
        assert step_resp.status_code == 400  # Handler rejects missing X-Episode-ID

    def test_invalid_classification(self, client):
        """Step with invalid classification returns error."""
        reset_resp = client.post("/reset", params={"task_name": "basic-injection", "seed": 42})
        episode_id = reset_resp.json()["episode_id"]

        action = {
            "classification": "not_a_valid_category",
            "reasoning": "This reasoning text is long enough to meet the minimum length requirement of ten characters",
            "recommended_action": "allow",
        }
        step_resp = client.post(
            "/step",
            json=action,
            headers={"X-Episode-ID": episode_id},
        )
        assert step_resp.status_code == 422  # Pydantic validation


class TestBatchAPI:
    """Test batch API v1 endpoints."""

    def test_batch_evaluate(self, client):
        """Test batch evaluation of multiple prompts."""
        request_body = {
            "prompts": ["Hello world", "Test prompt", "Simple query"],
            "task_name": "basic-injection",
        }
        resp = client.post("/api/v1/batch/evaluate", json=request_body)
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert "total" in data
        assert "processing_time_ms" in data
        assert data["total"] == 3

    def test_batch_episodes(self, client):
        """Test running multiple episodes in batch."""
        action = {
            "classification": "safe",
            "reasoning": "This prompt appears to be benign with no attack detected",
            "recommended_action": "allow",
            "safe_alternative": None,
        }
        request_body = {
            "task_name": "basic-injection",
            "seed": 42,
            "num_episodes": 3,
            "model_action": action,
        }
        resp = client.post("/api/v1/batch/episodes", json=request_body)
        assert resp.status_code == 200
        data = resp.json()
        assert "episodes" in data
        assert "aggregate" in data
        assert len(data["episodes"]) == 3

    def test_model_register_and_list(self, client):
        """Test model registration and listing."""
        register_resp = client.post(
            "/api/v1/models/register",
            json={
                "model_id": "test-model-1",
                "model_name": "Test Model",
                "description": "A test model for integration tests",
            },
        )
        assert register_resp.status_code == 200

        list_resp = client.get("/api/v1/models")
        assert list_resp.status_code == 200
        assert "models" in list_resp.json()

    def test_cached_health_v1(self, client):
        """Test v1 health endpoint with caching."""
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "api_version" in data
