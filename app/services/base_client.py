"""
Base HTTP client for communicating with AI inference services.
Provides async request handling, retry logic, and health checks.
"""

import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


class BaseServiceClient:
    """Base client for AI inference service communication."""

    def __init__(self, base_url: str, timeout: float = 0.5):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout, connect=5.0),
            )
        return self._client

    async def post(
        self,
        endpoint: str,
        json_data: Optional[dict] = None,
        content: Optional[bytes] = None,
        headers: Optional[dict] = None,
    ) -> httpx.Response:
        """Send a POST request to the inference service."""
        client = await self._get_client()
        try:
            if json_data is not None:
                response = await client.post(
                    endpoint, json=json_data, headers=headers
                )
            else:
                response = await client.post(
                    endpoint, content=content, headers=headers
                )
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            logger.error(
                "Service %s%s returned %d: %s",
                self.base_url,
                endpoint,
                e.response.status_code,
                e.response.text[:200],
            )
            raise
        except httpx.RequestError as e:
            logger.error(
                "Connection error to %s%s: %s", self.base_url, endpoint, e
            )
            raise

    async def get(self, endpoint: str) -> httpx.Response:
        """Send a GET request to the inference service."""
        client = await self._get_client()
        try:
            response = await client.get(endpoint)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            logger.error(
                "Service %s%s returned %d: %s",
                self.base_url, endpoint, e.response.status_code,
                e.response.text[:200],
            )
            raise
        except httpx.RequestError as e:
            logger.error("Connection error to %s%s: %s", self.base_url, endpoint, e)
            raise

    async def health_check(self) -> bool:
        """Check if the service is reachable."""
        try:
            client = await self._get_client()
            response = await client.get("/health")
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
