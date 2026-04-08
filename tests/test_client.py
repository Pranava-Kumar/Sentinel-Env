"""Tests for the SentinelEnv client."""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
from models import SentinelAction, SentinelObservation, SentinelState, ThreatCategory
from client import SentinelEnv


class TestSentinelEnv:
    @pytest_asyncio.fixture
    async def env(self):
        """Create an env instance with mocked httpx client."""
        env = SentinelEnv(base_url="http://test:7860")
        env.client = AsyncMock(spec=httpx.AsyncClient)
        return env

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context manager properly initializes client."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance
            async with SentinelEnv(base_url="http://test:7860") as env:
                assert env.client is not None
            mock_instance.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_reset_returns_observation(self, env):
        """Test reset() returns proper observation."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "user_prompt": "test prompt",
            "conversation_history": [],
            "attack_metadata": {
                "attack_type": "direct_override",
                "difficulty": "easy",
                "attack_text": "test prompt",
                "seed": 42,
                "task_name": "basic-injection",
                "ground_truth": "injection",
            },
            "resilience_metrics": {
                "detection_rate": 0.0,
                "false_positive_rate": 0.0,
                "attacks_correctly_detected": 0,
                "attacks_missed": 0,
                "false_positives": 0,
                "total_attacks": 0,
                "total_safe_prompts": 0,
            },
            "step_number": 1,
            "max_steps": 16,
            "is_safe_prompt": False,
        }
        mock_response.raise_for_status = MagicMock()
        env.client.post.return_value = mock_response

        obs = await env.reset(task_name="basic-injection", seed=42)
        assert isinstance(obs, SentinelObservation)
        assert obs.user_prompt == "test prompt"
        assert obs.step_number == 1

    @pytest.mark.asyncio
    async def test_step_returns_tuple(self, env):
        """Test step() returns observation, reward, done, info."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "observation": {
                "user_prompt": "next prompt",
                "conversation_history": [],
                "attack_metadata": {
                    "attack_type": "direct_override",
                    "difficulty": "easy",
                    "attack_text": "next prompt",
                    "seed": 42,
                    "task_name": "basic-injection",
                    "ground_truth": "injection",
                },
                "resilience_metrics": {
                    "detection_rate": 0.5,
                    "false_positive_rate": 0.0,
                    "attacks_correctly_detected": 1,
                    "attacks_missed": 0,
                    "false_positives": 0,
                    "total_attacks": 1,
                    "total_safe_prompts": 0,
                },
                "step_number": 2,
                "max_steps": 16,
                "is_safe_prompt": False,
            },
            "reward": 0.8,
            "done": False,
            "info": {"step_result": {"is_correct": True}},
        }
        mock_response.raise_for_status = MagicMock()
        env.client.post.return_value = mock_response

        action = SentinelAction(
            classification=ThreatCategory.INJECTION,
            reasoning="Test reasoning",
            recommended_action="block",
        )
        obs, reward, done, info = await env.step(action)

        assert isinstance(obs, SentinelObservation)
        assert reward == 0.8
        assert done is False
        assert isinstance(info, dict)

    @pytest.mark.asyncio
    async def test_state_returns_state(self, env):
        """Test state() returns proper state."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "episode_id": "test-42-abc123",
            "task_name": "basic-injection",
            "step_count": 5,
            "total_attacks_presented": 3,
            "attacks_correctly_detected": 2,
            "false_positives": 0,
            "current_resilience_score": 0.75,
            "done": False,
        }
        mock_response.raise_for_status = MagicMock()
        env.client.get.return_value = mock_response

        state = await env.state()
        assert isinstance(state, SentinelState)
        assert state.episode_id == "test-42-abc123"
        assert state.step_count == 5

    @pytest.mark.asyncio
    async def test_close_without_client(self):
        """Test close() works when client is None."""
        env = SentinelEnv()
        env.client = None
        await env.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_reset_without_client_raises(self):
        """Test reset() raises error if client not initialized."""
        env = SentinelEnv()
        with pytest.raises(RuntimeError, match="Client not initialized"):
            await env.reset()

    @pytest.mark.asyncio
    async def test_step_without_client_raises(self):
        """Test step() raises error if client not initialized."""
        env = SentinelEnv()
        action = SentinelAction(
            classification=ThreatCategory.SAFE,
            reasoning="Test reasoning with enough characters",
            recommended_action="allow",
        )
        with pytest.raises(RuntimeError, match="Client not initialized"):
            await env.step(action)

    @pytest.mark.asyncio
    async def test_state_without_client_raises(self):
        """Test state() raises error if client not initialized."""
        env = SentinelEnv()
        with pytest.raises(RuntimeError, match="Client not initialized"):
            await env.state()

    @pytest.mark.asyncio
    async def test_from_docker_image(self):
        """Test from_docker_image factory method."""
        env = await SentinelEnv.from_docker_image(port=8080)
        assert env.base_url == "http://localhost:8080"
        assert env.client is not None
        await env.close()


class TestSentinelEnvContextManager:
    @pytest.mark.asyncio
    async def test_aenter_initializes_client(self):
        """Async context manager should initialize httpx client."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance
            async with SentinelEnv("http://localhost:7860") as env:
                assert env.client is not None
                assert env.client is mock_instance

    @pytest.mark.asyncio
    async def test_aexit_closes_client(self):
        """Exiting context should close the client."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance
            env = SentinelEnv("http://localhost:7860")
            async with env:
                pass
            mock_instance.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_methods_after_close(self):
        """Methods should raise RuntimeError after client is closed."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance
            env = SentinelEnv("http://localhost:7860")
            async with env:
                pass  # Client is now closed

            with pytest.raises(RuntimeError, match="Client not initialized"):
                await env.reset()

    @pytest.mark.asyncio
    async def test_aenter_handles_connection_error(self):
        """Should propagate initialization errors and not leak client."""
        with patch("httpx.AsyncClient", side_effect=ConnectionError("Connection refused")):
            env = SentinelEnv("http://localhost:7860")
            with pytest.raises(ConnectionError, match="Connection refused"):
                async with env:
                    pass
            # Client should be cleaned up even on error
            assert env.client is None


class TestSentinelEnvErrorHandling:
    @pytest.mark.asyncio
    async def test_step_without_client_raises(self):
        """Step should raise RuntimeError if client not initialized."""
        env = SentinelEnv("http://localhost:7860")
        action = SentinelAction(
            classification=ThreatCategory.SAFE,
            reasoning="Test reasoning",
            recommended_action="allow",
        )
        with pytest.raises(RuntimeError, match="Client not initialized"):
            await env.step(action)

    @pytest.mark.asyncio
    async def test_reset_without_client_raises(self):
        """Reset should raise RuntimeError if client not initialized."""
        env = SentinelEnv("http://localhost:7860")
        with pytest.raises(RuntimeError, match="Client not initialized"):
            await env.reset()

    @pytest.mark.asyncio
    async def test_state_without_client_raises(self):
        """State should raise RuntimeError if client not initialized."""
        env = SentinelEnv("http://localhost:7860")
        with pytest.raises(RuntimeError, match="Client not initialized"):
            await env.state()

    @pytest.mark.asyncio
    async def test_from_docker_image_creates_client(self):
        """from_docker_image should create and return a client."""
        env = await SentinelEnv.from_docker_image(port=7860)
        try:
            assert env.client is not None
            assert isinstance(env.client, httpx.AsyncClient)
        finally:
            await env.close()

    @pytest.mark.asyncio
    async def test_from_docker_image_handles_error(self):
        """from_docker_image should clean up on initialization failure."""
        with patch("httpx.AsyncClient", side_effect=ConnectionError("Connection refused")):
            with pytest.raises(ConnectionError, match="Connection refused"):
                await SentinelEnv.from_docker_image(port=7860)

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self):
        """Calling close multiple times should not raise."""
        env = SentinelEnv("http://localhost:7860")
        await env.close()
        await env.close()  # Should not raise
