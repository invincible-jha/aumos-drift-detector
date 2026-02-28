"""HTTP client for embedding text using an external embedding service.

Used by LLMOutputMonitor (GAP-168) to convert LLM output strings into
dense vector representations for MMD-based drift detection.

GAP-168: LLM Output Monitoring
"""

from __future__ import annotations

from typing import Any

import httpx

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class EmbeddingClient:
    """Async HTTP client for a compatible text embedding REST API.

    Sends text batches to an embedding endpoint and returns dense vectors.

    Args:
        base_url: Base URL of the embedding service.
        api_key: Bearer token for authentication.
        model: Embedding model name to request.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "text-embedding-3-small",
        timeout: float = 30.0,
    ) -> None:
        """Initialise the embedding client.

        Args:
            base_url: Embedding service base URL.
            api_key: Bearer token for authentication.
            model: Name of the embedding model.
            timeout: HTTP timeout in seconds.
        """
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into dense vectors.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            RuntimeError: If the embedding request fails.
        """
        url = f"{self._base_url}/v1/embeddings"
        payload: dict[str, Any] = {"input": texts, "model": self._model}
        headers = {"Authorization": f"Bearer {self._api_key}"}

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                embeddings = [item["embedding"] for item in data["data"]]
                logger.debug("embed_success", n_texts=len(texts), model=self._model)
                return embeddings
        except Exception as exc:
            logger.error("embed_failed", error=str(exc), n_texts=len(texts))
            raise RuntimeError(f"Embedding request failed: {exc}") from exc
