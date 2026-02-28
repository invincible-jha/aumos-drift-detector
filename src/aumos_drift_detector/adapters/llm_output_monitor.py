"""LLM output monitoring via embedding-based MMD drift detection.

Detects semantic drift in LLM outputs by computing embeddings of
production responses and comparing them to a reference distribution
using Maximum Mean Discrepancy (MMD).

GAP-168: LLM Output Monitoring
"""

from __future__ import annotations

from typing import Any

from aumos_common.observability import get_logger

from aumos_drift_detector.adapters.mmd import mmd_test

logger = get_logger(__name__)


class LLMOutputMonitor:
    """Monitors LLM output quality by detecting semantic distribution drift.

    Computes text embeddings for reference and production outputs, then applies
    MMD with an RBF kernel to detect distributional shift. Optionally delegates
    embedding computation to an external embedding service.

    Args:
        embedding_client: Client implementing `embed(texts: list[str]) -> list[list[float]]`.
        mmd_threshold: MMD^2 score threshold above which drift is flagged.
    """

    def __init__(
        self,
        embedding_client: Any,
        mmd_threshold: float = 0.05,
    ) -> None:
        """Initialise the LLM output monitor.

        Args:
            embedding_client: Service that converts text to embeddings.
            mmd_threshold: Threshold for MMD^2-based drift detection.
        """
        self._embedding_client = embedding_client
        self._mmd_threshold = mmd_threshold

    async def detect(
        self,
        reference_outputs: list[str],
        production_outputs: list[str],
    ) -> dict[str, Any]:
        """Detect semantic drift between reference and production LLM outputs.

        Embeds both sets of outputs and computes MMD^2. Returns the full
        test result including drift status and the raw MMD score.

        Args:
            reference_outputs: Baseline LLM responses to compare against.
            production_outputs: Recent LLM responses from production.

        Returns:
            Dictionary with mmd_squared, drift_detected, and summary statistics.
        """
        if not reference_outputs or not production_outputs:
            logger.warning("llm_output_monitor_empty_inputs")
            return {
                "test": "llm_output_mmd",
                "drift_detected": False,
                "mmd_squared": 0.0,
                "reason": "empty_inputs",
            }

        try:
            reference_embeddings = await self._embedding_client.embed(reference_outputs)
            production_embeddings = await self._embedding_client.embed(production_outputs)
        except Exception as exc:
            logger.error("llm_output_monitor_embed_failed", error=str(exc))
            return {
                "test": "llm_output_mmd",
                "drift_detected": False,
                "mmd_squared": 0.0,
                "error": str(exc),
            }

        result = mmd_test(
            reference_embeddings=reference_embeddings,
            production_embeddings=production_embeddings,
            threshold=self._mmd_threshold,
        )
        result["test"] = "llm_output_mmd"
        result["n_reference_texts"] = len(reference_outputs)
        result["n_production_texts"] = len(production_outputs)

        logger.info(
            "llm_output_drift_check",
            mmd_squared=result["mmd_squared"],
            drift_detected=result["drift_detected"],
        )
        return result
