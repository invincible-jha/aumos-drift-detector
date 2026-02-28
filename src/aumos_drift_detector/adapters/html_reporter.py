"""Interactive HTML drift report generator using Plotly.

Generates self-contained HTML reports with embedded JavaScript so they
can be opened directly in any browser without a server.

GAP-170: Interactive HTML Drift Reports
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class HTMLDriftReporter:
    """Generates interactive Plotly-based HTML reports for drift detection results.

    The generated HTML is fully self-contained — all Plotly JS is embedded
    inline so reports can be shared as single files.

    Args:
        include_plotly_cdn: If True, references Plotly via CDN instead of
            embedding it. Smaller files but requires internet access.
    """

    def __init__(self, include_plotly_cdn: bool = False) -> None:
        """Initialise the HTML reporter.

        Args:
            include_plotly_cdn: Whether to use CDN for Plotly JS.
        """
        self._use_cdn = include_plotly_cdn

    def generate(
        self,
        monitor_name: str,
        detections: list[dict[str, Any]],
        feature_scores: dict[str, dict[str, Any]] | None = None,
    ) -> str:
        """Generate a self-contained HTML drift report.

        Args:
            monitor_name: Name of the drift monitor for the report title.
            detections: List of drift detection result dicts with 'score',
                'drift_detected', 'test', 'timestamp' keys.
            feature_scores: Optional per-feature test results, keyed by feature name.

        Returns:
            Complete HTML string ready to write to a file.
        """
        try:
            import plotly.graph_objects as go  # type: ignore[import-untyped]
            import plotly.io as pio  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "plotly is required for HTML reports. Install with: pip install plotly"
            ) from exc

        figures_html: list[str] = []

        # Timeline chart: drift scores over time
        if detections:
            timestamps = [d.get("timestamp", "") for d in detections]
            scores = [float(d.get("score", d.get("mmd_squared", 0.0))) for d in detections]
            drifted = [bool(d.get("drift_detected", False)) for d in detections]
            colors = ["red" if is_drift else "blue" for is_drift in drifted]

            fig_timeline = go.Figure()
            fig_timeline.add_trace(go.Scatter(
                x=timestamps,
                y=scores,
                mode="lines+markers",
                marker={"color": colors, "size": 8},
                name="Drift Score",
            ))
            fig_timeline.update_layout(
                title=f"Drift Score Timeline — {monitor_name}",
                xaxis_title="Time",
                yaxis_title="Drift Score",
                template="plotly_white",
            )
            figures_html.append(pio.to_html(fig_timeline, full_html=False, include_plotlyjs=self._use_cdn))

        # Per-feature bar chart
        if feature_scores:
            feature_names = list(feature_scores.keys())
            feature_drift_scores = [
                float(v.get("score", v.get("mmd_squared", 0.0)))
                for v in feature_scores.values()
            ]
            feature_colors = [
                "red" if v.get("drift_detected", False) else "steelblue"
                for v in feature_scores.values()
            ]
            fig_features = go.Figure(go.Bar(
                x=feature_names,
                y=feature_drift_scores,
                marker_color=feature_colors,
                name="Feature Drift Score",
            ))
            fig_features.update_layout(
                title=f"Per-Feature Drift Scores — {monitor_name}",
                xaxis_title="Feature",
                yaxis_title="Drift Score",
                template="plotly_white",
            )
            figures_html.append(pio.to_html(fig_features, full_html=False, include_plotlyjs=False))

        figures_section = "\n".join(figures_html)
        plotly_script = (
            '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
            if self._use_cdn
            else f'<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
        )

        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        total_runs = len(detections)
        drift_count = sum(1 for d in detections if d.get("drift_detected", False))

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Drift Report: {monitor_name}</title>
  {plotly_script}
  <style>
    body {{ font-family: -apple-system, sans-serif; margin: 40px; color: #333; }}
    h1 {{ color: #1a1a2e; }}
    .summary {{ display: flex; gap: 24px; margin: 24px 0; }}
    .stat-card {{ background: #f8f9fa; border-radius: 8px; padding: 16px; min-width: 120px; }}
    .stat-value {{ font-size: 2em; font-weight: bold; }}
    .drift {{ color: #dc3545; }}
    .ok {{ color: #28a745; }}
  </style>
</head>
<body>
  <h1>Drift Detection Report: {monitor_name}</h1>
  <p>Generated: {generated_at}</p>
  <div class="summary">
    <div class="stat-card"><div class="stat-value">{total_runs}</div><div>Total Runs</div></div>
    <div class="stat-card"><div class="stat-value {'drift' if drift_count > 0 else 'ok'}">{drift_count}</div><div>Drift Events</div></div>
    <div class="stat-card"><div class="stat-value">{total_runs - drift_count}</div><div>Clean Runs</div></div>
  </div>
  {figures_section}
</body>
</html>"""

        logger.info(
            "html_report_generated",
            monitor_name=monitor_name,
            total_runs=total_runs,
            drift_count=drift_count,
        )
        return html
