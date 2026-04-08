"""Sentinel Environment client.

Wraps HTTP/WebSocket communication with the server following OpenEnv conventions.
"""

import asyncio
from typing import Optional, Tuple, Dict, Any
import httpx

from models import SentinelAction, SentinelObservation, SentinelState


class SentinelEnv:
    """Client for the Sentinel Environment server."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def close(self):
        if self.client:
            await self.client.aclose()
            self.client = None

    async def reset(self, task_name: str = "basic-injection", seed: int = 42) -> SentinelObservation:
        """Start a new episode."""
        if self.client is None:
            raise RuntimeError("Client not initialized. Use async context manager.")

        response = await self.client.post(
            "/reset",
            params={"task_name": task_name, "seed": seed},
        )
        response.raise_for_status()
        return SentinelObservation(**response.json())

    async def step(self, action: SentinelAction) -> Tuple[SentinelObservation, float, bool, Dict[str, Any]]:
        """Execute one step."""
        if self.client is None:
            raise RuntimeError("Client not initialized.")

        response = await self.client.post("/step", json=action.model_dump())
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
        image_name: Optional[str] = None,
        port: int = 7860,
    ) -> "SentinelEnv":
        """Create client connected to a docker-based environment."""
        base_url = f"http://localhost:{port}"
        instance = cls(base_url=base_url)
        instance.client = httpx.AsyncClient(base_url=base_url, timeout=30.0)
        return instance
