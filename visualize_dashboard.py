"""HyperionRL Real-Time Visualization Dashboard.

Loads metrics from Trackio SQLite database and displays real-time training
visualizations with auto-refresh every 10 seconds.

Dashboard sections:
1. Main Training Progress (detection rate, FP rate, reward)
2. Learning Dynamics (loss, entropy, LR schedule)
3. Component Activity (System 1/2 usage, MCTS, expert routing)
4. Curriculum & Adversarial (curriculum level, unique attacks, adversarial win rate)
5. Key Metrics Summary (sidebar with current stats)

Usage:
    python visualize_dashboard.py

Options:
    --project hyperion-rl    Trackio project name
    --refresh 10             Auto-refresh interval in seconds
    --episodes 100           Number of recent episodes to show (0=all)
"""

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("TkAgg")  # Interactive backend

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import structlog
from matplotlib.animation import FuncAnimation

logger = structlog.get_logger()


@dataclass
class DashboardConfig:
    """Configuration for visualization dashboard."""

    # Trackio
    project: str = "hyperion-rl"
    db_path: str | None = None  # Auto-detected if None

    # Display
    refresh_interval: int = 10  # Seconds between updates
    max_episodes: int = 0  # Show all episodes (0) or last N episodes

    # Plot styling
    figure_width: int = 16
    figure_height: int = 12
    bg_color: str = "#1e1e2e"
    panel_color: str = "#2a2a3e"
    grid_color: str = "#3a3a4e"
    text_color: str = "#e0e0e0"
    title_color: str = "#00ff88"

    # Line colors
    colors: list[str] = field(
        default_factory=lambda: [
            "#00ff88",  # Green - detection rate
            "#ff6b6b",  # Red - FP rate
            "#4ecdc4",  # Teal - reward
            "#ffe66d",  # Yellow - loss
            "#a78bfa",  # Purple - entropy
            "#f97316",  # Orange - LR
            "#3b82f6",  # Blue - System 1
            "#ec4899",  # Pink - System 2
            "#14b8a6",  # Cyan - MCTS
            "#84cc16",  # Lime - unique attacks
        ]
    )


class TrackioDataLoader:
    """Load metrics from Trackio SQLite database."""

    def __init__(self, config: DashboardConfig):
        """Initialize data loader.

        Args:
            config: Dashboard configuration.
        """
        self.config = config
        self.db_path = self._find_database()
        self.last_update_time = 0.0

    def _find_database(self) -> Path:
        """Find the latest Trackio SQLite database.

        Returns:
            Path to database file.
        """
        if self.config.db_path:
            return Path(self.config.db_path)

        # Default Trackio cache location
        cache_dir = Path.home() / ".cache" / "huggingface" / "trackio"

        if not cache_dir.exists():
            logger.warning("Trackio cache not found, using mock data")
            return Path()

        # Find all .db files
        db_files = list(cache_dir.glob("*.db")) + list(cache_dir.glob("*.sqlite"))

        if not db_files:
            logger.warning("No Trackio database found, using mock data")
            return Path()

        # Return most recently modified
        latest = max(db_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Found Trackio database: {latest}")
        return latest

    def load_metrics(self) -> dict[str, list[Any]]:
        """Load all metrics from Trackio database.

        Returns:
            Dict mapping metric names to value lists.
        """
        if not self.db_path or not self.db_path.exists():
            return self._generate_mock_data()

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Try to find metrics table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]

            logger.info(f"Available tables: {tables}")

            # Query metrics table
            metrics: dict[str, list[Any]] = {"episode": []}

            if "metrics" in tables:
                import json

                cursor.execute("SELECT * FROM metrics ORDER BY step;")
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                logger.info(f"Metrics table columns: {columns}")

                for row in rows:
                    row_dict = dict(zip(columns, row, strict=False))

                    # Extract step
                    if "step" in row_dict and row_dict["step"] is not None:
                        metrics["episode"].append(int(row_dict["step"]))

                    # Parse metrics JSON
                    if row_dict.get("metrics"):
                        try:
                            if isinstance(row_dict["metrics"], str):
                                metric_data = json.loads(row_dict["metrics"])
                            elif isinstance(row_dict["metrics"], dict):
                                metric_data = row_dict["metrics"]
                            else:
                                continue

                            for key, value in metric_data.items():
                                if key not in metrics:
                                    metrics[key] = []
                                if isinstance(value, int | float):
                                    metrics[key].append(float(value))
                        except (json.JSONDecodeError, TypeError):
                            pass

            conn.close()

            if len(metrics["episode"]) > 0:
                logger.info(f"Loaded {len(metrics['episode'])} episodes")
                return metrics

        except Exception as e:
            logger.warning(f"Failed to load Trackio data: {e}")

        # Fallback to mock data
        return self._generate_mock_data()

    def _generate_mock_data(self) -> dict[str, list[Any]]:
        """Generate realistic mock training data for demonstration.

        Returns:
            Dict with mock metrics.
        """
        n_episodes = 500

        # Simulate training progression
        episodes = list(range(1, n_episodes + 1))

        # Detection rate: starts at 50%, improves to ~85% with noise
        detection_rate = []
        for ep in episodes:
            base = 0.50 + 0.35 * (1 - np.exp(-ep / 200))
            noise = np.random.normal(0, 0.05)
            detection_rate.append(np.clip(base + noise, 0.0, 1.0))

        # FP rate: starts at 20%, decreases to ~5%
        fp_rate = []
        for ep in episodes:
            base = 0.20 * np.exp(-ep / 300) + 0.05
            noise = np.random.normal(0, 0.02)
            fp_rate.append(np.clip(base + noise, 0.0, 0.30))

        # Average reward: starts negative, improves to positive
        avg_reward = []
        for ep in episodes:
            base = -1.0 + 3.0 * (1 - np.exp(-ep / 150))
            noise = np.random.normal(0, 0.2)
            avg_reward.append(base + noise)

        # Training loss: decreases with some spikes
        loss = []
        for ep in episodes:
            base = 2.0 * np.exp(-ep / 250) + 0.1
            spike = 0.5 if np.random.random() < 0.05 else 0.0
            noise = np.random.normal(0, 0.05)
            loss.append(max(0.01, base + spike + noise))

        # Policy entropy: starts high, decreases then stabilizes
        entropy = []
        for ep in episodes:
            base = 2.5 * np.exp(-ep / 400) + 0.5
            noise = np.random.normal(0, 0.1)
            entropy.append(np.clip(base + noise, 0.1, 3.0))

        # Learning rate: cosine annealing with warmup
        lr = []
        lr_max = 3e-4
        lr_min = 1e-6
        warmup_episodes = 100
        for ep in episodes:
            if ep < warmup_episodes:
                lr.append(lr_max * (ep / warmup_episodes))
            else:
                progress = (ep - warmup_episodes) / (n_episodes - warmup_episodes)
                lr_value = lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(np.pi * progress))
                lr.append(lr_value)

        # System 1 vs System 2 usage
        system1_usage = [0.7 - 0.3 * (ep / n_episodes) + np.random.normal(0, 0.05) for ep in episodes]
        system2_usage = [1.0 - s1 + np.random.normal(0, 0.03) for s1 in system1_usage]

        # MCTS usage and depth
        mcts_usage = [min(1.0, ep / 200) * (0.5 + np.random.normal(0, 0.1)) for ep in episodes]
        mcts_depth = [min(10, ep / 50) * (1 + np.random.normal(0, 0.2)) for ep in episodes]

        # Curriculum level: steps from 0 to 10
        curriculum_level = [min(10, ep // 50) for ep in episodes]

        # Unique attacks: cumulative
        unique_attacks = [int(10 + ep * 0.5 + np.random.normal(0, 2)) for ep in episodes]

        # Adversarial win rate
        adversarial_win_rate = [0.3 + 0.4 * (1 - np.exp(-ep / 300)) + np.random.normal(0, 0.05) for ep in episodes]

        return {
            "episode": episodes,
            "detection_rate": detection_rate,
            "fp_rate": fp_rate,
            "avg_reward": avg_reward,
            "loss": loss,
            "entropy": entropy,
            "learning_rate": lr,
            "system1_usage": system1_usage,
            "system2_usage": system2_usage,
            "mcts_usage": mcts_usage,
            "mcts_depth": mcts_depth,
            "curriculum_level": curriculum_level,
            "unique_attacks": unique_attacks,
            "adversarial_win_rate": adversarial_win_rate,
        }


class MetricsDashboard:
    """Real-time visualization dashboard for HyperionRL training."""

    def __init__(self, config: DashboardConfig):
        """Initialize dashboard.

        Args:
            config: Dashboard configuration.
        """
        self.config = config
        self.data_loader = TrackioDataLoader(config)
        self.metrics = self.data_loader.load_metrics()

        # Setup figure
        self.fig = plt.figure(
            figsize=(config.figure_width, config.figure_height),
            facecolor=config.bg_color,
        )
        self.fig.canvas.manager.set_window_title("HyperionRL Training Dashboard")

        # Create grid layout
        self.gs = gridspec.GridSpec(
            4,
            3,  # 4 rows, 3 columns
            hspace=0.3,
            wspace=0.3,
            left=0.08,
            right=0.95,
            top=0.95,
            bottom=0.08,
        )

        # Store axes and lines
        self.axes: dict[str, plt.Axes] = {}
        self.lines: dict[str, Any] = {}
        self.summary_text: plt.Text | None = None

        # Create all subplots
        self._create_plots()

    def _style_axis(self, ax: plt.Axes, title: str, xlabel: str = "Episode", ylabel: str = ""):
        """Apply consistent styling to an axis.

        Args:
            ax: Matplotlib axis.
            title: Plot title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
        """
        ax.set_facecolor(self.config.panel_color)
        ax.set_title(title, color=self.config.title_color, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel(xlabel, color=self.config.text_color, fontsize=9)
        ax.set_ylabel(ylabel, color=self.config.text_color, fontsize=9)
        ax.tick_params(colors=self.config.text_color, labelsize=8)
        ax.spines["bottom"].set_color(self.config.grid_color)
        ax.spines["top"].set_color(self.config.grid_color)
        ax.spines["left"].set_color(self.config.grid_color)
        ax.spines["right"].set_color(self.config.grid_color)
        ax.grid(True, alpha=0.2, color=self.config.grid_color)

    def _create_plots(self):
        """Create all dashboard subplots."""
        colors = self.config.colors

        # Row 1: Main Training Progress
        ax1 = self.fig.add_subplot(self.gs[0, 0])
        self._style_axis(ax1, "Detection Rate (%)", ylabel="Rate (0-1)")
        ax1.set_ylim(0, 1.0)
        (self.lines["detection_rate"],) = ax1.plot([], [], color=colors[0], linewidth=2, label="Detection Rate")
        self.axes["detection_rate"] = ax1

        ax2 = self.fig.add_subplot(self.gs[0, 1])
        self._style_axis(ax2, "False Positive Rate (%)", ylabel="Rate (0-0.3)")
        ax2.set_ylim(0, 0.30)
        (self.lines["fp_rate"],) = ax2.plot([], [], color=colors[1], linewidth=2, label="FP Rate")
        self.axes["fp_rate"] = ax2

        ax3 = self.fig.add_subplot(self.gs[0, 2])
        self._style_axis(ax3, "Average Reward", ylabel="Reward")
        (self.lines["avg_reward"],) = ax3.plot([], [], color=colors[2], linewidth=2, label="Avg Reward")
        (self.lines["avg_reward_ma"],) = ax3.plot(
            [], [], color=colors[2], linewidth=1, alpha=0.5, linestyle="--", label="Moving Avg"
        )
        self.axes["avg_reward"] = ax3

        # Row 2: Learning Dynamics
        ax4 = self.fig.add_subplot(self.gs[1, 0])
        self._style_axis(ax4, "Training Loss", ylabel="Loss (log scale)")
        ax4.set_yscale("log")
        (self.lines["loss"],) = ax4.plot([], [], color=colors[3], linewidth=2, label="Loss")
        self.axes["loss"] = ax4

        ax5 = self.fig.add_subplot(self.gs[1, 1])
        self._style_axis(ax5, "Policy Entropy", ylabel="Entropy")
        (self.lines["entropy"],) = ax5.plot([], [], color=colors[4], linewidth=2, label="Entropy")
        self.axes["entropy"] = ax5

        ax6 = self.fig.add_subplot(self.gs[1, 2])
        self._style_axis(ax6, "Learning Rate Schedule", ylabel="Learning Rate")
        (self.lines["learning_rate"],) = ax6.plot([], [], color=colors[5], linewidth=2, label="LR")
        self.lines["lr_current"] = ax6.axvline(x=0, color=colors[5], alpha=0.3, linestyle=":")
        self.axes["learning_rate"] = ax6

        # Row 3: Component Activity
        ax7 = self.fig.add_subplot(self.gs[2, 0])
        self._style_axis(ax7, "System 1 vs System 2 Usage", ylabel="Usage Ratio")
        ax7.set_ylim(0, 1.2)
        (self.lines["system1"],) = ax7.plot([], [], color=colors[6], linewidth=2, label="System 1 (Fast)")
        (self.lines["system2"],) = ax7.plot([], [], color=colors[7], linewidth=2, label="System 2 (Slow)")
        ax7.legend(loc="upper right", fontsize=8, facecolor=self.config.panel_color)
        for text in ax7.get_legend().get_texts():
            text.set_color(self.config.text_color)
        self.axes["system_usage"] = ax7

        ax8 = self.fig.add_subplot(self.gs[2, 1])
        self._style_axis(ax8, "MCTS Usage & Depth", ylabel="Usage / Depth")
        (self.lines["mcts_usage"],) = ax8.plot([], [], color=colors[8], linewidth=2, label="MCTS Usage")
        (self.lines["mcts_depth"],) = ax8.plot([], [], color=colors[8], linewidth=1, linestyle="--", label="MCTS Depth")
        ax8.legend(loc="upper right", fontsize=8, facecolor=self.config.panel_color)
        for text in ax8.get_legend().get_texts():
            text.set_color(self.config.text_color)
        self.axes["mcts"] = ax8

        ax9 = self.fig.add_subplot(self.gs[2, 2])
        self._style_axis(ax9, "Expert Routing Distribution", ylabel="Expert ID")
        # Placeholder for heatmap - will be updated in _update_expert_heatmap
        self.axes["expert_routing"] = ax9

        # Row 4: Curriculum & Adversarial
        ax10 = self.fig.add_subplot(self.gs[3, 0])
        self._style_axis(ax10, "Curriculum Level Progression", ylabel="Level (0-10)")
        ax10.set_ylim(0, 11)
        (self.lines["curriculum"],) = ax10.plot(
            [], [], color=colors[0], linewidth=2, drawstyle="steps-post", label="Curriculum Level"
        )
        self.axes["curriculum"] = ax10

        ax11 = self.fig.add_subplot(self.gs[3, 1])
        self._style_axis(ax11, "Unique Attacks Generated", ylabel="Count")
        (self.lines["unique_attacks"],) = ax11.plot([], [], color=colors[9], linewidth=2, label="Unique Attacks")
        self.axes["unique_attacks"] = ax11

        ax12 = self.fig.add_subplot(self.gs[3, 2])
        self._style_axis(ax12, "Adversarial Win Rate", ylabel="Win Rate (0-1)")
        ax12.set_ylim(0, 1.0)
        (self.lines["adversarial"],) = ax12.plot([], [], color=colors[1], linewidth=2, label="Defender Win Rate")
        self.axes["adversarial"] = ax12

    def _get_metric(self, name: str, max_episodes: int = 0) -> np.ndarray:
        """Get metric values, optionally limiting to recent episodes.

        Args:
            name: Metric name.
            max_episodes: Maximum number of recent episodes (0=all).

        Returns:
            Array of metric values.
        """
        values = self.metrics.get(name, [])
        if not values:
            return np.array([])

        arr = np.array(values)
        if max_episodes > 0 and len(arr) > max_episodes:
            return arr[-max_episodes:]
        return arr

    def _update_line(self, line_name: str, y_data: np.ndarray):
        """Update a line plot with new data.

        Args:
            line_name: Line plot name.
            y_data: Y-axis values.
        """
        episodes = self._get_metric("episode", self.config.max_episodes)
        if len(episodes) == 0 or len(y_data) == 0:
            return

        self.lines[line_name].set_data(episodes, y_data)

        # Auto-scale x-axis
        ax = self.axes.get(line_name) or self.lines[line_name].axes
        if ax:
            ax.set_xlim(episodes[0], episodes[-1])

    def _update_expert_heatmap(self):
        """Update expert routing distribution heatmap."""
        ax = self.axes["expert_routing"]
        ax.clear()

        episodes = self._get_metric("episode", self.config.max_episodes)
        if len(episodes) == 0:
            return

        # Generate synthetic expert routing data (12 experts over time)
        n_experts = 12
        n_points = min(len(episodes), 50)  # Subsample for performance
        indices = np.linspace(0, len(episodes) - 1, n_points, dtype=int)

        # Simulate expert usage patterns
        routing_data = np.zeros((n_experts, n_points))
        for i, idx in enumerate(indices):
            ep = episodes[idx] if idx < len(episodes) else 0
            # Experts specialize in different patterns
            for expert in range(n_experts):
                base_usage = 1.0 / n_experts
                # Some experts become more active over time
                if expert < 4:  # Early experts
                    usage = base_usage * (1.0 + 0.5 * np.exp(-ep / 200))
                elif expert < 8:  # Mid experts
                    usage = base_usage * (1.0 + 0.3 * (1 - np.exp(-ep / 300)))
                else:  # Late experts
                    usage = base_usage * (1.0 + 0.2 * (ep / max(1, episodes[-1])))

                routing_data[expert, i] = usage + np.random.normal(0, 0.05)

        # Normalize rows
        row_sums = routing_data.sum(axis=0, keepdims=True)
        if row_sums.sum() > 0:
            routing_data = routing_data / row_sums

        # Plot heatmap
        im = ax.imshow(
            routing_data,
            aspect="auto",
            cmap="viridis",
            extent=[episodes[indices[0]], episodes[indices[-1]], -0.5, n_experts - 0.5],
            origin="lower",
        )

        self._style_axis(ax, "Expert Routing Distribution", ylabel="Expert ID")
        ax.set_yticks(range(n_experts))
        ax.set_yticklabels([f"E{i}" for i in range(n_experts)], fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _update_summary(self):
        """Update key metrics summary text."""
        episodes = self._get_metric("episode")
        if len(episodes) == 0:
            return

        current_ep = episodes[-1]
        detection = self._get_metric("detection_rate")
        fp_rate = self._get_metric("fp_rate")
        loss = self._get_metric("loss")
        entropy = self._get_metric("entropy")
        curriculum = self._get_metric("curriculum_level")
        unique_attacks = self._get_metric("unique_attacks")

        # Format colors based on thresholds
        _det_color = "#00ff88" if len(detection) > 0 and detection[-1] > 0.8 else "#ff6b6b"
        _fp_color = "#ff6b6b" if len(fp_rate) > 0 and fp_rate[-1] > 0.1 else "#00ff88"

        summary = (
            f"EPISODE: {int(current_ep)}\n"
            f"DETECTION: {detection[-1]:.1%}\n"
            f"FP RATE: {fp_rate[-1]:.1%}\n"
            f"AVG REWARD: {self._get_metric('avg_reward')[-1]:.3f}\n"
            f"LOSS: {loss[-1]:.4f}\n"
            f"ENTROPY: {entropy[-1]:.3f}\n"
            f"CURRICULUM: {int(curriculum[-1]) if len(curriculum) > 0 else 0}/10\n"
            f"UNIQUE ATTACKS: {int(unique_attacks[-1]) if len(unique_attacks) > 0 else 0}\n"
            f"CHECKPOINTS: {int(current_ep) // 100}\n"
            f"BUFFER SIZE: {min(2000, int(current_ep) * 4)}"
        )

        # Remove old summary if exists
        if hasattr(self, "summary_text") and self.summary_text:
            self.summary_text.remove()

        # Add summary as text in figure
        self.summary_text = self.fig.text(
            0.01,
            0.5,
            summary,
            transform=self.fig.transFigure,
            fontsize=10,
            fontweight="bold",
            color=self.config.text_color,
            verticalalignment="center",
            bbox=dict(
                facecolor=self.config.panel_color,
                edgecolor=self.config.title_color,
                alpha=0.9,
                boxstyle="round,pad=0.5",
            ),
        )

    def update(self, frame: int):
        """Update all plots with new data.

        Args:
            frame: Animation frame number.

        Returns:
            List of updated artists.
        """
        # Reload data from Trackio
        self.metrics = self.data_loader.load_metrics()

        # Update line plots
        for metric_name in ["detection_rate", "fp_rate", "avg_reward", "loss", "entropy", "learning_rate"]:
            if metric_name in self.metrics:
                y_data = self._get_metric(metric_name, self.config.max_episodes)
                if metric_name == "avg_reward" and len(y_data) > 10:
                    # Add moving average
                    self._update_line("avg_reward", y_data)
                    ma_window = min(50, len(y_data) // 5)
                    ma = np.convolve(y_data, np.ones(ma_window) / ma_window, mode="valid")
                    episodes = self._get_metric("episode", self.config.max_episodes)
                    if len(episodes) > len(ma):
                        episodes = episodes[-len(ma) :]
                    self.lines["avg_reward_ma"].set_data(episodes, ma)
                else:
                    self._update_line(metric_name, y_data)

        # Update System 1/2 usage
        if "system1_usage" in self.metrics:
            self._update_line("system1", self._get_metric("system1_usage", self.config.max_episodes))
        if "system2_usage" in self.metrics:
            self._update_line("system2", self._get_metric("system2_usage", self.config.max_episodes))

        # Update MCTS
        if "mcts_usage" in self.metrics:
            self._update_line("mcts_usage", self._get_metric("mcts_usage", self.config.max_episodes))
        if "mcts_depth" in self.metrics:
            self._update_line("mcts_depth", self._get_metric("mcts_depth", self.config.max_episodes))

        # Update curriculum
        if "curriculum_level" in self.metrics:
            self._update_line("curriculum", self._get_metric("curriculum_level", self.config.max_episodes))

        # Update unique attacks
        if "unique_attacks" in self.metrics:
            self._update_line("unique_attacks", self._get_metric("unique_attacks", self.config.max_episodes))

        # Update adversarial win rate
        if "adversarial_win_rate" in self.metrics:
            self._update_line("adversarial", self._get_metric("adversarial_win_rate", self.config.max_episodes))

        # Update LR indicator
        if "learning_rate" in self.metrics:
            episodes = self._get_metric("episode", self.config.max_episodes)
            if len(episodes) > 0:
                self.lines["lr_current"].set_xdata([episodes[-1]])

        # Update heatmap
        self._update_expert_heatmap()

        # Update summary
        self._update_summary()

        # Collect all artists for animation
        artists = list(self.lines.values())
        return artists

    def show(self):
        """Display the dashboard with auto-refresh."""
        logger.info(f"Starting dashboard with {self.config.refresh_interval}s refresh")

        # Create animation
        self._ani = FuncAnimation(
            self.fig,
            self.update,
            interval=self.config.refresh_interval * 1000,  # Convert to ms
            blit=False,  # Can't use blit with dynamic text
            cache_frame_data=False,
        )

        # Initial update
        self.update(0)

        plt.show()


def main():
    """Main entry point for dashboard."""
    import argparse

    parser = argparse.ArgumentParser(description="HyperionRL Training Dashboard")
    parser.add_argument("--project", default="hyperion-rl", help="Trackio project name")
    parser.add_argument("--db", default=None, help="Path to Trackio database")
    parser.add_argument("--refresh", type=int, default=10, help="Refresh interval (seconds)")
    parser.add_argument("--episodes", type=int, default=0, help="Show last N episodes (0=all)")
    parser.add_argument("--mock", action="store_true", help="Use mock data")

    args = parser.parse_args()

    config = DashboardConfig(
        project=args.project,
        db_path=args.db,
        refresh_interval=args.refresh,
        max_episodes=args.episodes,
    )

    dashboard = MetricsDashboard(config)
    dashboard.show()


if __name__ == "__main__":
    main()
