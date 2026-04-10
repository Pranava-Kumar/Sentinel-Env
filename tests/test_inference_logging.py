"""Tests for inference logging format compliance."""

import sys
from io import StringIO

from inference_logging import log_end, log_start, log_step


def capture_print(func, *args, **kwargs):
    """Capture stdout from a print function."""
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    func(*args, **kwargs)
    output = sys.stdout.getvalue()
    sys.stdout = old_stdout
    return output


class TestLogStart:
    def test_format_matches_spec(self):
        """Test [START] line matches OpenENV spec."""
        output = capture_print(log_start, "basic-injection", "sentinel", "Qwen2.5-72B")
        assert output.startswith("[START]")
        assert "task=basic-injection" in output
        assert "env=sentinel" in output
        assert "model=Qwen2.5-72B" in output

    def test_no_newlines(self):
        """Test single line output."""
        output = capture_print(log_start, "task", "env", "model")
        assert output.count("\n") == 1  # Just the trailing newline


class TestLogStep:
    def test_format_matches_spec(self):
        """Test [STEP] line matches OpenENV spec."""
        output = capture_print(log_step, 1, "injection", 0.8, False, None)
        assert output.startswith("[STEP]")
        assert "step=1" in output
        assert "action=injection" in output
        assert "reward=0.80" in output
        assert "done=false" in output
        assert "error=null" in output

    def test_done_true(self):
        """Test done=true formatting."""
        output = capture_print(log_step, 5, "block", 1.0, True, None)
        assert "done=true" in output

    def test_with_error(self):
        """Test error message formatting."""
        output = capture_print(log_step, 3, "safe", 0.0, True, "Connection refused")
        assert "error=Connection refused" in output

    def test_reward_precision(self):
        """Test reward is formatted to 2 decimal places."""
        output = capture_print(log_step, 1, "action", 0.12345, False, None)
        assert "reward=0.12" in output


class TestLogEnd:
    def test_format_matches_spec(self):
        """Test [END] line matches OpenENV spec."""
        output = capture_print(log_end, True, 10, 0.85, [0.0, 0.8, 1.0])
        assert output.startswith("[END]")
        assert "success=true" in output
        assert "steps=10" in output
        assert "score=0.85" in output
        assert "rewards=0.00,0.80,1.00" in output

    def test_success_false(self):
        """Test success=false formatting."""
        output = capture_print(log_end, False, 5, 0.2, [0.0, 0.0])
        assert "success=false" in output

    def test_empty_rewards(self):
        """Test empty rewards list."""
        output = capture_print(log_end, False, 0, 0.0, [])
        assert "rewards=" in output
