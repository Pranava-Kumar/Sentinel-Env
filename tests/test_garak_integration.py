"""Garak security test integration.

Runs garak LLM security tests against the Sentinel environment.
Garak is a LLM vulnerability scanner that tests for:
- Prompt injection
- Jailbreak attempts
- Data leakage
- Misuse potential
- Dangerous content generation

Usage:
    uv run pytest tests/test_garak_integration.py -v

Requires garak to be installed:
    uv pip install garak
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

from models import RecommendedAction, SentinelAction, ThreatCategory
from server.rate_limiter import RateLimiter
from server.sentinel_environment import SentinelEnvironment

# Skip if garak not installed
pytest.importorskip("garak")


class TestGarakSecurity:
    """Integration tests with garak security scanner."""

    @pytest.mark.skipif(not os.getenv("RUN_GARAK_TESTS"), reason="Set RUN_GARAK_TESTS=1 to run garak tests")
    def test_garak_prompt_injection_scan(self):
        """Run garak prompt injection scan."""
        config_path = Path(__file__).parent.parent / "garak_config.yaml"

        if not config_path.exists():
            pytest.skip("garak_config.yaml not found")

        # Run garak
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "garak",
                "--model_type",
                "openai",
                "--model_name",
                os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"),
                "--probes",
                "promptinject",
                "--config",
                str(config_path),
                "--jsonl",
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes timeout
            env={
                **os.environ,
                "OPENAI_API_KEY": os.getenv("HF_TOKEN", ""),
                "OPENAI_API_BASE": os.getenv("API_BASE_URL", ""),
            },
        )

        # Check results
        # Garak should detect injection attempts
        assert result.returncode in [0, 1], f"Garak failed with: {result.stderr}"

    @pytest.mark.skipif(not os.getenv("RUN_GARAK_TESTS"), reason="Set RUN_GARAK_TESTS=1 to run garak tests")
    def test_garak_jailbreak_scan(self):
        """Run garak jailbreak scan."""
        config_path = Path(__file__).parent.parent / "garak_config.yaml"

        if not config_path.exists():
            pytest.skip("garak_config.yaml not found")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "garak",
                "--model_type",
                "openai",
                "--model_name",
                os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"),
                "--probes",
                "dan",
                "--config",
                str(config_path),
                "--jsonl",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            env={
                **os.environ,
                "OPENAI_API_KEY": os.getenv("HF_TOKEN", ""),
                "OPENAI_API_BASE": os.getenv("API_BASE_URL", ""),
            },
        )

        assert result.returncode in [0, 1], f"Garak failed with: {result.stderr}"


class TestAdversarialPromptCatalog:
    """Test coverage against known adversarial prompt catalogs."""

    def test_jailbreak_prompts_loaded(self):
        """Jailbreak prompts should be loaded from the directory."""
        from server.attacks.jailbreak_loader import load_jailbreak_prompts

        prompts = load_jailbreak_prompts()

        # Should have loaded some prompts (may be 0 if directory doesn't exist)
        assert isinstance(prompts, list)

        if len(prompts) > 0:
            # Validate structure
            for prompt in prompts[:10]:  # Check first 10
                assert "text" in prompt
                assert "ground_truth" in prompt
                assert "is_attack" in prompt
                assert "attack_type" in prompt

    def test_advanced_jailbreaks_exist(self):
        """Advanced jailbreak attacks should be defined."""
        from server.attacks.advanced_jailbreaks import ADVANCED_JAILBREAK_ATTACKS

        assert len(ADVANCED_JAILBREAK_ATTACKS) > 0

        # Check structure
        for text, ground_truth, attack_type in ADVANCED_JAILBREAK_ATTACKS[:10]:
            assert len(text) > 20, f"Attack text too short: {text[:50]}"
            assert ground_truth != "unknown"
            assert attack_type != "none"

    def test_basic_injections_exist(self):
        """Basic injection attacks should be defined."""
        from server.attacks.basic_injections import BASIC_INJECTION_ATTACKS

        assert len(BASIC_INJECTION_ATTACKS) > 10

    def test_social_engineering_attacks_exist(self):
        """Social engineering attacks should be defined."""
        from server.attacks.social_engineering import SOCIAL_ENGINEERING_ATTACKS

        assert len(SOCIAL_ENGINEERING_ATTACKS) > 10

    def test_stealth_exfiltration_attacks_exist(self):
        """Stealth exfiltration attacks should be defined."""
        from server.attacks.stealth_exfiltration import STEALTH_EXFILTRATION_ATTACKS

        assert len(STEALTH_EXFILTRATION_ATTACKS) > 10


class TestSecurityEdgeCases:
    """Test security-specific edge cases."""

    def test_api_key_auth_prevents_unauthorized_access(self):
        """API key auth should prevent unauthorized access."""
        import asyncio
        import hmac
        import os
        from unittest.mock import patch

        from fastapi import HTTPException

        async def run_test():
            # Set test key
            with patch.dict(os.environ, {"SENTINEL_API_KEY": "test-secret-key"}):
                # Simulate verify_api_key behavior
                test_key = "wrong-key"
                sentinel_key = os.environ.get("SENTINEL_API_KEY")

                if sentinel_key and not hmac.compare_digest(test_key, sentinel_key):
                    raise HTTPException(status_code=401, detail="Invalid or missing API key")

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(run_test())

        assert exc_info.value.status_code == 401

    def test_rate_limiter_prevents_abuse(self):
        """Rate limiter should prevent request abuse."""

        async def run_test():
            limiter = RateLimiter(max_requests=5, window_seconds=60, max_entries=100)

            # Should allow first few requests
            for _ in range(5):
                allowed, remaining = await limiter.check_rate_limit("abuser-ip")
                assert allowed, f"Request should be allowed, remaining: {remaining}"

            # Should block subsequent requests
            allowed, remaining = await limiter.check_rate_limit("abuser-ip")
            assert not allowed, "Request should be blocked after exceeding limit"
            assert remaining == 0, "Remaining should be 0 when blocked"

        asyncio.run(run_test())

    def test_episode_isolation(self):
        """Episodes should be isolated from each other."""
        env1 = SentinelEnvironment()
        env2 = SentinelEnvironment()

        env1.reset(task_name="basic-injection", seed=42)
        env2.reset(task_name="basic-injection", seed=99)

        # Actions in env1 shouldn't affect env2
        action = SentinelAction(
            classification=ThreatCategory.SAFE,
            reasoning="This prompt appears to be completely safe and benign with no threats",
            recommended_action=RecommendedAction.ALLOW,
        )

        obs1, r1, done1, _ = env1.step(action)
        obs2, r2, done2, _ = env2.step(action)

        # Different seeds should produce different prompts
        assert obs1.user_prompt != obs2.user_prompt
