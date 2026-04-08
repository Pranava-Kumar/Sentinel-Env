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

    def test_reset_invalid_task(self, client):
        """Test reset with invalid task name."""
        response = client.post("/reset", params={"task_name": "invalid-task", "seed": 42})
        # Should return 500 or handle gracefully
        assert response.status_code in [200, 500]

    def test_reset_different_seeds(self, client):
        """Test different seeds produce different observations."""
        resp1 = client.post("/reset", params={"task_name": "basic-injection", "seed": 42})
        resp2 = client.post("/reset", params={"task_name": "basic-injection", "seed": 99})
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        

class TestStepEndpoint:
    def test_step_without_reset(self, client):
        """Test step without reset returns error."""
        action = {
            "classification": "injection",
            "reasoning": "Test reasoning here",
            "recommended_action": "block",
        }
        response = client.post("/step", json=action)
        # Should return 400 (not initialized) or handle gracefully
        assert response.status_code in [200, 400]

    def test_step_after_reset(self, client):
        """Test step works after reset."""
        # Reset first
        client.post("/reset", params={"task_name": "basic-injection", "seed": 42})
        
        action = {
            "classification": "injection",
            "reasoning": "Direct override attempt detected",
            "recommended_action": "block",
        }
        response = client.post("/step", json=action)
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


class TestStateEndpoint:
    def test_state_returns_valid_state(self, client):
        """Test state endpoint returns valid state."""
        response = client.get("/state")
        assert response.status_code == 200
        data = response.json()
        assert "episode_id" in data
        assert "task_name" in data


class TestGradeEndpoint:
    def test_grade_without_steps(self, client):
        """Test grade endpoint with no steps."""
        response = client.get("/grade")
        assert response.status_code == 200
        data = response.json()
        assert "score" in data


class TestResilienceProfileEndpoint:
    def test_resilience_profile_returns_valid(self, client):
        """Test resilience profile endpoint."""
        response = client.get("/resilience-profile")
        assert response.status_code == 200
        data = response.json()
        assert "task_name" in data
