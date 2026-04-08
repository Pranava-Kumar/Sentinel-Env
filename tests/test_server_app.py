"""Tests for the FastAPI server endpoints."""
import pytest
from fastapi.testclient import TestClient
from models import ThreatCategory
from server.app import app


@pytest.fixture
def client():
    """Create test client with lifespan."""
    # Use the client with lifespan to initialize env
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "sentinel-env"


class TestResetEndpoint:
    def test_reset_returns_observation(self, client):
        """Test reset endpoint returns valid observation."""
        response = client.post("/reset", params={"task_name": "basic-injection", "seed": 42})
        assert response.status_code == 200
        data = response.json()
        assert "user_prompt" in data
        assert "attack_metadata" in data
        assert "resilience_metrics" in data
        assert "episode_id" in data

    def test_reset_invalid_task(self, client):
        """Test reset with invalid task name."""
        response = client.post("/reset", params={"task_name": "invalid-task", "seed": 42})
        # Should return 500 with error message, not silently succeed
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data

    def test_reset_different_seeds(self, client):
        """Test different seeds produce different observations."""
        resp1 = client.post("/reset", params={"task_name": "basic-injection", "seed": 42})
        resp2 = client.post("/reset", params={"task_name": "basic-injection", "seed": 99})
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        # Verify different episode IDs
        data1 = resp1.json()
        data2 = resp2.json()
        assert data1["episode_id"] != data2["episode_id"]


class TestStepEndpoint:
    def test_step_without_reset(self, client):
        """Test step without reset returns error."""
        action = {
            "classification": "injection",
            "reasoning": "Test reasoning here",
            "recommended_action": "block",
        }
        response = client.post("/step", json=action)
        # Should return 400 (episode_id header required)
        assert response.status_code == 400

    def test_step_after_reset(self, client):
        """Test step works after reset."""
        # Reset first and get episode ID
        reset_response = client.post("/reset", params={"task_name": "basic-injection", "seed": 42})
        assert reset_response.status_code == 200
        episode_id = reset_response.json()["episode_id"]

        action = {
            "classification": "injection",
            "reasoning": "Direct override attempt detected",
            "recommended_action": "block",
        }
        response = client.post("/step", json=action, headers={"X-Episode-ID": episode_id})
        assert response.status_code == 200
        data = response.json()
        assert "observation" in data
        assert "reward" in data
        assert "done" in data
        assert "info" in data

    def test_step_invalid_action(self, client):
        """Test step with invalid action returns 422."""
        response = client.post("/step", json={"invalid": "data"})
        assert response.status_code == 422

    def test_step_with_invalid_episode_id(self, client):
        """Test step with non-existent episode ID returns 404."""
        action = {
            "classification": "injection",
            "reasoning": "Test reasoning here",
            "recommended_action": "block",
        }
        response = client.post("/step", json=action, headers={"X-Episode-ID": "nonexistent-id"})
        assert response.status_code == 404


class TestStateEndpoint:
    def test_state_without_episode_id(self, client):
        """Test state without episode_id returns 400."""
        response = client.get("/state")
        assert response.status_code == 400

    def test_state_returns_valid_state(self, client):
        """Test state endpoint returns valid state."""
        # Reset first to get episode ID
        reset_response = client.post("/reset", params={"task_name": "basic-injection", "seed": 42})
        episode_id = reset_response.json()["episode_id"]

        response = client.get("/state", headers={"X-Episode-ID": episode_id})
        assert response.status_code == 200
        data = response.json()
        assert "episode_id" in data
        assert "task_name" in data

    def test_state_with_invalid_episode_id(self, client):
        """Test state with non-existent episode ID returns 404."""
        response = client.get("/state", headers={"X-Episode-ID": "nonexistent-id"})
        assert response.status_code == 404


class TestGradeEndpoint:
    def test_grade_without_episode_id(self, client):
        """Test grade without episode_id returns 400."""
        response = client.get("/grade")
        assert response.status_code == 400

    def test_grade_without_steps(self, client):
        """Test grade endpoint with no steps."""
        # Reset first to get episode ID
        reset_response = client.post("/reset", params={"task_name": "basic-injection", "seed": 42})
        episode_id = reset_response.json()["episode_id"]

        response = client.get("/grade", headers={"X-Episode-ID": episode_id})
        assert response.status_code == 200
        data = response.json()
        assert "score" in data

    def test_grade_with_invalid_episode_id(self, client):
        """Test grade with non-existent episode ID returns 404."""
        response = client.get("/grade", headers={"X-Episode-ID": "nonexistent-id"})
        assert response.status_code == 404


class TestResilienceProfileEndpoint:
    def test_resilience_profile_without_episode_id(self, client):
        """Test resilience profile without episode_id returns 400."""
        response = client.get("/resilience-profile")
        assert response.status_code == 400

    def test_resilience_profile_returns_valid(self, client):
        """Test resilience profile endpoint."""
        # Reset first to get episode ID
        reset_response = client.post("/reset", params={"task_name": "basic-injection", "seed": 42})
        episode_id = reset_response.json()["episode_id"]

        response = client.get("/resilience-profile", headers={"X-Episode-ID": episode_id})
        assert response.status_code == 200
        data = response.json()
        assert "task_name" in data

    def test_resilience_profile_with_invalid_episode_id(self, client):
        """Test resilience profile with non-existent episode ID returns 404."""
        response = client.get("/resilience-profile", headers={"X-Episode-ID": "nonexistent-id"})
        assert response.status_code == 404
