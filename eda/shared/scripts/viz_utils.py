"""Visualization helpers for EDA outputs."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go


def save_plotly(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".html":
        fig.write_html(path, include_plotlyjs="cdn")
    else:
        fig.write_image(path)
