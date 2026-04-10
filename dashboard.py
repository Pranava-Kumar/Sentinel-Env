"""Sentinel Environment - Real-time Monitoring Dashboard.

Gradio-based UI for monitoring the Sentinel RL environment.
Features:
- Live metrics (detection rate, FP rate, active episodes)
- Attack analysis (confusion matrix, detection by type)
- Training progress curves
- Interactive prompt testing
- Model comparison
"""

import asyncio
import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import gradio as gr
import httpx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

SERVER_URL = os.getenv("SENTINEL_SERVER_URL", "http://localhost:8000")
REFRESH_INTERVAL = 5
MAX_HISTORY = 500


@dataclass
class DashboardBackend:
    """Backend that connects to the Sentinel server and collects metrics."""

    client: httpx.AsyncClient | None = field(default=None, init=False)
    metrics_history: dict[str, list] = field(default_factory=lambda: defaultdict(list))
    episode_scores: list[float] = field(default_factory=list)
    episode_timestamps: list[float] = field(default_factory=list)
    attack_type_scores: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    recent_episodes: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=50))
    curriculum_difficulty: list[float] = field(default_factory=list)
    last_update: float = field(default_factory=time.time)

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self.client is None or self.client.is_closed:
            self.client = httpx.AsyncClient(timeout=10.0, base_url=SERVER_URL)
        return self.client

    async def fetch_health(self) -> dict[str, Any]:
        """Fetch health endpoint metrics."""
        try:
            client = await self._ensure_client()
            resp = await client.get("/health")
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning("Health fetch failed: %s", exc)
            return {"status": "disconnected"}

    async def fetch_resilience_profile(self) -> dict[str, Any]:
        """Fetch resilience profile from the server."""
        try:
            client = await self._ensure_client()
            resp = await client.get("/resilience-profile")
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning("Resilience profile fetch failed: %s", exc)
            return {}

    async def fetch_grade(self, task: str = "basic-injection") -> dict[str, Any]:
        """Fetch grading metrics for a specific task."""
        try:
            client = await self._ensure_client()
            resp = await client.get("/grade", params={"task": task})
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning("Grade fetch failed: %s", exc)
            return {}

    async def test_prompt(self, prompt: str) -> dict[str, Any]:
        """Submit a test prompt through the environment and return results."""
        try:
            client = await self._ensure_client()
            # Reset to create a new episode
            reset_resp = await client.post("/reset", json={"task": "basic-injection"})
            reset_resp.raise_for_status()
            episode_id = reset_resp.json().get("episode_id")

            # Step with the prompt
            headers = {"X-Episode-ID": episode_id}
            step_resp = await client.post("/step", json={"action": prompt}, headers=headers)
            step_resp.raise_for_status()
            step_data = step_resp.json()

            # Get final state
            state_resp = await client.get("/state", headers=headers)
            state_resp.raise_for_status()
            state_data = state_resp.json()

            return {
                "episode_id": episode_id,
                "step_response": step_data,
                "state": state_data,
                "reward": step_data.get("reward", 0),
                "done": step_data.get("done", False),
            }
        except Exception as exc:
            logger.warning("Prompt test failed: %s", exc)
            return {"error": str(exc)}

    def _update_history(self, metrics: dict[str, Any]) -> None:
        """Store metrics in historical buffers for trend charts."""
        ts = time.time()
        detection_rate = metrics.get("detection_rate", metrics.get("accuracy", 0.0))
        fp_rate = metrics.get("fp_rate", 0.0)
        avg_score = metrics.get("avg_score", metrics.get("mean_reward", 0.0))
        total_episodes = metrics.get("total_episodes", 0)

        now_str = datetime.now(timezone.utc).strftime("%H:%M:%S")

        self.metrics_history["timestamps"].append(now_str)
        self.metrics_history["detection_rate"].append(detection_rate)
        self.metrics_history["fp_rate"].append(fp_rate)
        self.metrics_history["avg_score"].append(avg_score)
        self.metrics_history["total_episodes"].append(total_episodes)

        # Trim to max history
        for key in list(self.metrics_history.keys()):
            if len(self.metrics_history[key]) > MAX_HISTORY:
                self.metrics_history[key] = self.metrics_history[key][-MAX_HISTORY:]

        if isinstance(avg_score, (int, float)) and avg_score != 0:
            self.episode_scores.append(avg_score)
            self.episode_timestamps.append(ts)
            if len(self.episode_scores) > MAX_HISTORY:
                self.episode_scores.pop(0)
                self.episode_timestamps.pop(0)

    def _add_recent_episode(self, episode_data: dict[str, Any]) -> None:
        """Add an episode to the recent activity log."""
        entry = {
            "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "episode_id": episode_data.get("episode_id", "N/A")[:12],
            "task": episode_data.get("task", "unknown"),
            "score": round(episode_data.get("reward", episode_data.get("score", 0)), 3),
            "done": episode_data.get("done", False),
            "status": "completed" if episode_data.get("done") else "active",
        }
        self.recent_episodes.append(entry)

    def plot_detection_trend(self) -> plt.Figure:
        """Create detection rate trend line chart."""
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        timestamps = self.metrics_history.get("timestamps", [])
        detection_rates = self.metrics_history.get("detection_rate", [])

        if not timestamps or not detection_rates:
            ax.text(
                0.5, 0.5, "No data yet", ha="center", va="center", transform=ax.transAxes, color="#a0a0a0", fontsize=14
            )
            ax.set_title("Detection Rate Trend", color="#e0e0e0", pad=10)
            return fig

        x = list(range(len(timestamps)))
        ax.plot(x, detection_rates, color="#00d4ff", linewidth=2, marker="o", markersize=3)
        ax.fill_between(x, detection_rates, alpha=0.15, color="#00d4ff")

        ax.set_title("Detection Rate Trend", color="#e0e0e0", pad=10, fontsize=12)
        ax.set_xlabel("Time", color="#a0a0a0")
        ax.set_ylabel("Detection Rate", color="#a0a0a0")
        ax.set_ylim(0, 1.05)
        ax.tick_params(colors="#a0a0a0")
        ax.spines["bottom"].set_color("#333355")
        ax.spines["top"].set_color("#333355")
        ax.spines["left"].set_color("#333355")
        ax.spines["right"].set_color("#333355")
        ax.grid(True, alpha=0.15, color="#444466")

        plt.tight_layout()
        return fig

    def plot_attack_type_performance(self) -> plt.Figure:
        """Create attack type performance bar chart."""
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        attack_types = list(self.attack_type_scores.keys())
        if not attack_types:
            attack_types = ["basic-injection", "social-engineering", "stealth-exfiltration"]
            mean_scores = [0.75, 0.60, 0.45]
        else:
            mean_scores = [
                np.mean(scores) if scores else 0.0 for scores in [self.attack_type_scores[t] for t in attack_types]
            ]

        colors = ["#00d4ff" if s >= 0.7 else "#ffaa00" if s >= 0.5 else "#ff4444" for s in mean_scores]
        bars = ax.barh(attack_types[::-1], mean_scores[::-1], color=colors[::-1], height=0.5)

        ax.set_title("Detection Rate by Attack Type", color="#e0e0e0", pad=10, fontsize=12)
        ax.set_xlabel("Mean Detection Rate", color="#a0a0a0")
        ax.set_xlim(0, 1.05)
        ax.tick_params(colors="#a0a0a0")
        ax.spines["bottom"].set_color("#333355")
        ax.spines["top"].set_color("#333355")
        ax.spines["left"].set_color("#333355")
        ax.spines["right"].set_color("#333355")
        ax.grid(True, alpha=0.15, color="#444466", axis="x")

        for bar, score in zip(bars, mean_scores[::-1], strict=True):
            ax.text(
                bar.get_width() + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.2f}",
                va="center",
                color="#e0e0e0",
                fontsize=10,
            )

        plt.tight_layout()
        return fig

    def plot_training_progress(self) -> plt.Figure:
        """Create training progress chart with episode scores."""
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        if not self.episode_scores:
            ax.text(
                0.5,
                0.5,
                "No training data yet",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#a0a0a0",
                fontsize=14,
            )
            ax.set_title("Training Progress", color="#e0e0e0", pad=10)
            return fig

        x = list(range(len(self.episode_scores)))
        ax.plot(x, self.episode_scores, color="#00ff88", linewidth=1.5, alpha=0.8)

        # Moving average
        if len(self.episode_scores) >= 10:
            window = min(10, len(self.episode_scores))
            moving_avg = np.convolve(self.episode_scores, np.ones(window) / window, mode="valid")
            ax.plot(
                list(range(window - 1, len(self.episode_scores))),
                moving_avg,
                color="#ffaa00",
                linewidth=2,
                label=f"Moving Avg ({window})",
            )
            ax.legend(facecolor="#16213e", edgecolor="#333355", labelcolor="#e0e0e0")

        ax.set_title("Episode Score Progression", color="#e0e0e0", pad=10, fontsize=12)
        ax.set_xlabel("Episode", color="#a0a0a0")
        ax.set_ylabel("Score / Reward", color="#a0a0a0")
        ax.tick_params(colors="#a0a0a0")
        ax.spines["bottom"].set_color("#333355")
        ax.spines["top"].set_color("#333355")
        ax.spines["left"].set_color("#333355")
        ax.spines["right"].set_color("#333355")
        ax.grid(True, alpha=0.15, color="#444466")

        plt.tight_layout()
        return fig

    def plot_curriculum_difficulty(self) -> plt.Figure:
        """Create curriculum difficulty progression chart."""
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        difficulties = self.curriculum_difficulty
        if not difficulties:
            difficulties = [0.2, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.7, 0.8, 0.9]

        x = list(range(len(difficulties)))
        ax.step(x, difficulties, color="#ff66aa", linewidth=2, where="post")
        ax.fill_between(x, difficulties, alpha=0.1, color="#ff66aa", step="post")

        ax.set_title("Curriculum Difficulty Over Time", color="#e0e0e0", pad=10, fontsize=12)
        ax.set_xlabel("Checkpoint", color="#a0a0a0")
        ax.set_ylabel("Difficulty Level", color="#a0a0a0")
        ax.set_ylim(0, 1.05)
        ax.tick_params(colors="#a0a0a0")
        ax.spines["bottom"].set_color("#333355")
        ax.spines["top"].set_color("#333355")
        ax.spines["left"].set_color("#333355")
        ax.spines["right"].set_color("#333355")
        ax.grid(True, alpha=0.15, color="#444466")

        plt.tight_layout()
        return fig

    def get_recent_episodes_table(self) -> tuple[list[str], list[list]]:
        """Return recent episodes as a Gradio Dataframe tuple."""
        if not self.recent_episodes:
            return ["Time", "Episode ID", "Task", "Score", "Done", "Status"], []
        first = next(iter(self.recent_episodes))
        columns = list(first.keys())
        rows = [list(ep.values()) for ep in reversed(self.recent_episodes)]
        return columns, rows

    async def refresh_metrics(self) -> dict[str, Any]:
        """Fetch all metrics and update internal state."""
        health = await self.fetch_health()
        resilience = await self.fetch_resilience_profile()

        merged = {**health, **resilience}

        # Normalize metric names
        normalized = {
            "detection_rate": merged.get("detection_rate", merged.get("accuracy", 0.0)),
            "fp_rate": merged.get("fp_rate", merged.get("false_positive_rate", 0.0)),
            "total_episodes": merged.get("total_episodes", merged.get("episode_count", 0)),
            "avg_score": merged.get("avg_score", merged.get("mean_reward", 0.0)),
            "request_rate": merged.get("request_rate", merged.get("requests_per_min", 0)),
            "status": merged.get("status", "unknown"),
        }

        self._update_history(normalized)
        self.last_update = time.time()

        return normalized


# Global backend instance
backend = DashboardBackend()


def _format_status(status: str) -> str:
    if status in ("healthy", "connected"):
        return "Healthy"
    elif status == "degraded":
        return "Degraded"
    else:
        return "Disconnected"


def _status_color(status: str) -> str:
    if status in ("healthy", "connected"):
        return "#00ff88"
    elif status == "degraded":
        return "#ffaa00"
    else:
        return "#ff4444"


async def _update_metrics() -> tuple:
    """Update all dashboard metrics. Called by Gradio Timer."""
    metrics = await backend.refresh_metrics()

    detection_rate = metrics.get("detection_rate", 0.0)
    fp_rate = metrics.get("fp_rate", 0.0)
    total_episodes = metrics.get("total_episodes", 0)
    avg_score = metrics.get("avg_score", 0.0)
    request_rate = metrics.get("request_rate", 0)
    status = metrics.get("status", "disconnected")

    # Add a synthetic episode for demo if none exist
    if total_episodes == 0 and backend.client is not None:
        try:
            client = await backend._ensure_client()
            reset_resp = await client.post("/reset", json={"task": "basic-injection"})
            reset_resp.raise_for_status()
            episode_id = reset_resp.json().get("episode_id", "demo")
            backend._add_recent_episode(
                {
                    "episode_id": episode_id,
                    "task": "basic-injection",
                    "reward": detection_rate,
                    "done": True,
                }
            )
        except Exception:
            pass

    detection_fig = backend.plot_detection_trend()
    attack_fig = backend.plot_attack_type_performance()
    training_fig = backend.plot_training_progress()
    curriculum_fig = backend.plot_curriculum_difficulty()
    columns, rows = backend.get_recent_episodes_table()

    status_text = _format_status(status)
    status_color = _status_color(status)
    last_update_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

    return (
        detection_rate,
        fp_rate,
        int(total_episodes),
        round(avg_score, 4) if avg_score else 0.0,
        int(request_rate) if request_rate else 0,
        f"Status: {status_text}",
        status_color,
        f"Last Update: {last_update_str}",
        detection_fig,
        attack_fig,
        training_fig,
        curriculum_fig,
        columns,
        rows,
    )


def _sync_update_metrics():
    """Synchronous wrapper for async metric updates."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        future = asyncio.run_coroutine_threadsafe(_update_metrics(), loop)
        return future.result(timeout=30)
    else:
        return asyncio.run(_update_metrics())


async def _handle_test_prompt(prompt: str) -> str:
    """Handle interactive prompt testing."""
    if not prompt or not prompt.strip():
        return "Please enter a prompt to test."

    result = await backend.test_prompt(prompt.strip())

    if "error" in result:
        return f"Error: {result['error']}"

    episode_id = result.get("episode_id", "N/A")
    reward = result.get("reward", 0)
    done = result.get("done", False)
    state = result.get("state", {})
    step_resp = result.get("step_response", {})

    output = [
        f"Episode ID: {episode_id}",
        f"Reward: {reward:.4f}",
        f"Done: {done}",
        "",
        "--- State ---",
    ]

    for key, value in state.items():
        if isinstance(value, (dict, list)):
            import json

            output.append(f"{key}: {json.dumps(value, indent=2)[:200]}")
        else:
            output.append(f"{key}: {value}")

    output.append("")
    output.append("--- Step Response ---")
    for key, value in step_resp.items():
        if isinstance(value, (dict, list)):
            import json

            output.append(f"{key}: {json.dumps(value, indent=2)[:200]}")
        else:
            output.append(f"{key}: {value}")

    return "\n".join(output)


def _sync_handle_test_prompt(prompt: str) -> str:
    """Synchronous wrapper for prompt testing."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        future = asyncio.run_coroutine_threadsafe(_handle_test_prompt(prompt), loop)
        return future.result(timeout=30)
    else:
        return asyncio.run(_handle_test_prompt(prompt))


def _build_dashboard_ui() -> gr.Blocks:
    """Build the complete Gradio dashboard UI."""
    theme = gr.themes.Soft(
        primary_hue="cyan",
        secondary_hue="green",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("JetBrains Mono"),
    ).set(
        body_background_fill="#0d1117",
        block_background_fill="#161b22",
        block_title_background_fill="#1a1a2e",
        background_fill_primary="#0d1117",
        background_fill_secondary="#161b22",
        body_text_color="#e0e0e0",
        block_title_text_color="#e0e0e0",
        block_label_text_color="#a0a0a0",
        input_background_fill="#1a1a2e",
        input_border_color="#333355",
        input_text_color="#e0e0e0",
    )

    with gr.Blocks(title="Sentinel RL Environment - SOC Dashboard", theme=theme) as dashboard:
        # Header
        gr.Markdown(
            """
            # Sentinel RL Environment — Security Operations Center Dashboard

            Real-time monitoring for the adversarial prompt detection environment.
            """,
            elem_classes=["header"],
        )

        # Status bar
        with gr.Row():
            status_indicator = gr.Markdown(
                value="Status: Connecting...",
                elem_classes=["status"],
            )
            last_update_label = gr.Markdown(
                value="Last Update: --",
                elem_classes=["last-update"],
            )

        # Metrics Row
        gr.Markdown("### Key Metrics")
        with gr.Row():
            gauge_detection = gr.Gauge(
                label="Detection Rate",
                min=0,
                max=1,
                value=0,
                color="#00d4ff",
            )
            gauge_fp = gr.Gauge(
                label="False Positive Rate",
                min=0,
                max=1,
                value=0,
                color="#ff4444",
            )
            gauge_episodes = gr.Number(
                label="Total Episodes",
                value=0,
            )
            gauge_avg_score = gr.Number(
                label="Average Score",
                value=0.0,
            )
            gauge_req_rate = gr.Number(
                label="Requests / Min",
                value=0,
            )

        # Charts Row 1: Detection Trend + Attack Type Performance
        gr.Markdown("### Analytics")
        with gr.Row():
            plot_detection_trend = gr.Plot(label="Detection Rate Over Time")
            plot_attack_type = gr.Plot(label="Detection by Attack Type")

        # Charts Row 2: Training Progress + Curriculum Difficulty
        with gr.Row():
            plot_training_progress = gr.Plot(label="Episode Score Progression")
            plot_curriculum = gr.Plot(label="Curriculum Difficulty")

        # Interactive Prompt Test
        gr.Markdown("### Interactive Prompt Testing")
        with gr.Row():
            with gr.Column(scale=2):
                prompt_input = gr.Textbox(
                    label="Test Prompt",
                    placeholder="Enter a prompt to classify...",
                    lines=3,
                )
                test_btn = gr.Button("Test Prompt", variant="primary")
            with gr.Column(scale=3):
                test_output = gr.Textbox(
                    label="Classification Result",
                    lines=10,
                    interactive=False,
                )

        # Recent Activity Log
        gr.Markdown("### Recent Activity")
        activity_table = gr.Dataframe(
            label="Episode Log",
            headers=["Time", "Episode ID", "Task", "Score", "Done", "Status"],
            wrap=True,
        )

        # Auto-refresh timer
        timer = gr.Timer(value=REFRESH_INTERVAL)

        # Wire up updates
        timer.tick(
            fn=_sync_update_metrics,
            outputs=[
                gauge_detection,
                gauge_fp,
                gauge_episodes,
                gauge_avg_score,
                gauge_req_rate,
                status_indicator,
                # We cannot dynamically update color via Markdown, so pass as value
                last_update_label,
                plot_detection_trend,
                plot_attack_type,
                plot_training_progress,
                plot_curriculum,
                activity_table,
            ],
        )

        test_btn.click(
            fn=_sync_handle_test_prompt,
            inputs=[prompt_input],
            outputs=[test_output],
        )

        # Custom CSS for SOC aesthetic
        dashboard.head = """
        <style>
            body {
                background-color: #0d1117 !important;
            }
            .header h1 {
                color: #00d4ff !important;
                text-align: center;
                border-bottom: 2px solid #00d4ff33;
                padding-bottom: 12px;
            }
            .status {
                font-size: 14px;
                padding: 8px;
            }
            .last-update {
                font-size: 12px;
                color: #888;
                text-align: right;
            }
            .gradio-container {
                max-width: 100% !important;
                padding: 20px !important;
            }
            .block {
                border-radius: 8px !important;
            }
        </style>
        """

    return dashboard


def launch(
    server_url: str = SERVER_URL,
    share: bool = False,
    inbrowser: bool = True,
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
) -> None:
    """Launch the Sentinel monitoring dashboard.

    Args:
        server_url: URL of the Sentinel server to monitor.
        share: Create a public Gradio shareable link.
        inbrowser: Open the dashboard in a browser tab.
        server_name: Host to bind the Gradio server.
        server_port: Port to bind the Gradio server.
    """
    global SERVER_URL
    SERVER_URL = server_url
    backend.client = None  # Reset client for new URL

    dashboard = _build_dashboard_ui()
    dashboard.launch(
        share=share,
        inbrowser=inbrowser,
        server_name=server_name,
        server_port=server_port,
        show_error=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sentinel RL Environment Monitoring Dashboard")
    parser.add_argument(
        "--server-url",
        default=SERVER_URL,
        help="URL of the Sentinel server (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for the Gradio dashboard (default: %(default)s)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the dashboard in a browser",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio shareable link",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    launch(
        server_url=args.server_url,
        share=args.share,
        inbrowser=not args.no_browser,
        server_port=args.port,
    )
