"""Sentinel Environment client.

Wraps HTTP/WebSocket communication with the server following OpenEnv conventions.
"""

from typing import Any

import httpx

from models import SentinelAction, SentinelObservation, SentinelState


class SentinelEnv:
    """Client for the Sentinel Environment server."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self.client: httpx.AsyncClient | None = None
        self._episode_id: str | None = None

    async def __aenter__(self):
        try:
            self.client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
            return self
        except Exception:
            await self.close()
            raise

    async def __aexit__(self, *args):
        await self.close()

    async def close(self):
        if self.client:
            await self.client.aclose()
            self.client = None

    @property
    def episode_id(self) -> str | None:
        """Return the current episode ID."""
        return self._episode_id

    async def reset(self, task_name: str = "basic-injection", seed: int = 42) -> SentinelObservation:
        """Start a new episode."""
        if self.client is None:
            raise RuntimeError("Client not initialized. Use async context manager.")

        response = await self.client.post(
            "/reset",
            params={"task_name": task_name, "seed": seed},
        )
        response.raise_for_status()
        data = response.json()

        # Store episode ID for subsequent step calls
        self._episode_id = data.get("episode_id")

        return SentinelObservation(**data)

    async def step(self, action: SentinelAction) -> tuple[SentinelObservation, float, bool, dict[str, Any]]:
        """Execute one step."""
        if self.client is None:
            raise RuntimeError("Client not initialized.")

        headers = {}
        if self._episode_id:
            headers["X-Episode-ID"] = self._episode_id

        response = await self.client.post("/step", json=action.model_dump(), headers=headers)
        response.raise_for_status()
        data = response.json()

        obs = SentinelObservation(**data["observation"])
        reward = data["reward"]
        done = data["done"]
        info = data.get("info", {})

        return obs, reward, done, info

    async def state(self) -> SentinelState:
        """Get current state."""
        if self.client is None:
            raise RuntimeError("Client not initialized.")

        response = await self.client.get("/state")
        response.raise_for_status()
        return SentinelState(**response.json())

    @classmethod
    async def from_docker_image(
        cls,
        image_name: str | None = None,
        port: int = 7860,
    ) -> "SentinelEnv":
        """Create client connected to a docker-based environment."""
        base_url = f"http://localhost:{port}"
        instance = cls(base_url=base_url)
        try:
            instance.client = httpx.AsyncClient(base_url=base_url, timeout=30.0)
            return instance
        except Exception:
            await instance.close()
            raise
