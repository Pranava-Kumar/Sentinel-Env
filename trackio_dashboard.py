#!/usr/bin/env python
"""Trackio Dashboard Launcher for Sentinel Environment.

Launches the Trackio dashboard to visualize experiment tracking data
for Sentinel RL training runs.

Usage:
    # Launch dashboard for all projects
    python trackio_dashboard.py

    # Launch for a specific project
    python trackio_dashboard.py --project sentinel-rl-training

    # Launch on a specific port and host
    python trackio_dashboard.py --project sentinel-rl-training --host 0.0.0.0 --port 7861

    # Launch without opening browser
    python trackio_dashboard.py --no-browser

    # List available projects
    python trackio_dashboard.py --list-projects
"""

import argparse
import sys


def list_projects() -> None:
    """List all available Trackio projects."""
    try:
        from trackio.local import LocalStorage

        storage = LocalStorage()
        projects = storage.list_projects()
        if not projects:
            print("No Trackio projects found.")
            return
        print("Available Trackio projects:")
        for project in projects:
            print(f"  - {project}")
    except ImportError:
        print("Error: trackio is not installed. Install it with: pip install trackio")
        sys.exit(1)
    except Exception as e:
        print(f"Could not list projects: {e}")
        print("Trackio projects are stored in ~/.cache/trackio/ by default.")


def launch_dashboard(
    project: str | None = None,
    host: str = "127.0.0.1",
    port: int = 7860,
    open_browser: bool = True,
) -> None:
    """Launch the Trackio dashboard.

    Args:
        project: Project name to show. If None, shows all projects.
        host: Host to bind the server.
        port: Port to bind the server.
        open_browser: Whether to open a browser tab.
    """
    try:
        import trackio
    except ImportError:
        print("Error: trackio is not installed. Install it with: pip install trackio")
        sys.exit(1)

    print("Launching Trackio dashboard...")
    if project:
        print(f"  Project: {project}")
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  Open browser: {open_browser}")
    print()

    result = trackio.show(
        project=project,
        host=host,
        open_browser=open_browser,
    )

    if result is not None:
        url = result.get("url", "unknown")
        print(f"\nDashboard available at: {url}")
        share_url = result.get("share_url")
        if share_url:
            print(f"Public share URL: {share_url}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Sentinel Environment - Trackio Dashboard Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Launch for all projects
  %(prog)s --project sentinel-rl-training     # Show specific project
  %(prog)s --host 0.0.0.0 --port 7861         # Custom host/port
  %(prog)s --no-browser                       # Don't open browser
  %(prog)s --list-projects                    # List available projects
        """,
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Project name to display (default: show all projects)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server (default: %(default)s). Use 0.0.0.0 for remote access.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for the Trackio dashboard (default: %(default)s)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the dashboard in a browser",
    )
    parser.add_argument(
        "--list-projects",
        action="store_true",
        help="List available Trackio projects and exit",
    )
    args = parser.parse_args()

    if args.list_projects:
        list_projects()
        return

    launch_dashboard(
        project=args.project,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
