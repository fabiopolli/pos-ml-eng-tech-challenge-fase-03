"""Render a static preview of the Grafana observability dashboard.

This script produces two artefacts under ``reports/figures/``:

* ``triage_ml_dashboard.json`` — verbatim copy of
  ``monitoring/grafana/dashboards/triage_ml.json`` so reviewers can
  inspect the provisioned panels without having the full monitoring
  stack running.
* ``triage_ml_dashboard.png`` — synthetic matplotlib preview that
  reproduces the 4-panel layout of the Grafana dashboard (requests
  by route/status, latency p95, prediction error rate and the
  baseline-vs-optimised table). The preview is a faithful
  representation of the layout — PromQL expressions are listed as
  text inside each panel.

The script is intentionally lightweight: it only depends on
``matplotlib`` and the Python standard library. It does not start the
Prometheus/Grafana stack or call any external service.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_SOURCE = REPO_ROOT / "monitoring" / "grafana" / "dashboards" / "triage_ml.json"
OUTPUT_DIR = REPO_ROOT / "reports" / "figures"
JSON_OUTPUT = OUTPUT_DIR / "triage_ml_dashboard.json"
PNG_OUTPUT = OUTPUT_DIR / "triage_ml_dashboard.png"


def _panel_rect(panel: dict[str, Any], total_width: int = 24) -> tuple[int, int, int, int]:
    grid = panel["gridPos"]
    x = grid["x"]
    y = grid["y"]
    w = grid["w"]
    h = grid["h"]
    return x, y, w, h


def _panel_queries(panel: dict[str, Any]) -> list[str]:
    queries: list[str] = []
    for target in panel.get("targets", []):
        expr = str(target.get("expr", "")).strip()
        if expr:
            queries.append(expr)
    return queries


def _wrap_promql(query: str, *, panel_width: int) -> list[str]:
    """Wrap a PromQL string so it never overflows the panel width.

    The dashboard uses a 24-column grid; each column maps to ~1 character
    per PromQL symbol at the chosen font size, but we leave a safety
    margin for monospace rendering and clipping.
    """

    width_chars = max(20, int(panel_width * 3.6))
    wrapped: list[str] = []
    line = ""
    for token in query.split():
        if len(token) > width_chars:
            # Hard-split a long token
            chunks = [token[i : i + width_chars] for i in range(0, len(token), width_chars)]
            for chunk in chunks:
                if line:
                    wrapped.append(line)
                    line = ""
                wrapped.append(chunk)
            continue
        candidate = token if not line else f"{line} {token}"
        if len(candidate) <= width_chars:
            line = candidate
        else:
            wrapped.append(line)
            line = token
    if line:
        wrapped.append(line)
    return wrapped


def render_png(dashboard: dict[str, Any], dest: Path) -> None:
    """Render a 24-col × 16-row preview matching the Grafana layout."""

    fig, ax = plt.subplots(figsize=(14.5, 8.0), dpi=140)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 16)
    ax.invert_yaxis()  # Grafana top-left origin
    ax.set_aspect("equal")
    ax.axis("off")

    # Title bar
    ax.add_patch(patches.Rectangle((0, 0), 24, 1, facecolor="#1f1f1f", edgecolor="none"))
    ax.text(
        0.3,
        0.5,
        str(dashboard.get("title", "Triage ML Dashboard")),
        color="white",
        fontsize=13,
        fontweight="bold",
        verticalalignment="center",
    )
    ax.text(
        23.7,
        0.5,
        f"refresh={dashboard.get('refresh', '10s')} · uid={dashboard.get('uid', '?')}",
        color="#bbbbbb",
        fontsize=8,
        ha="right",
        verticalalignment="center",
    )

    # Panels
    for idx, panel in enumerate(dashboard.get("panels", []), start=1):
        x, y, w, h = _panel_rect(panel)
        # Panel background
        ax.add_patch(
            patches.Rectangle(
                (x, y + 1),  # leave room for title bar
                w,
                h,
                facecolor="#111217",
                edgecolor="#3a3a3a",
                linewidth=0.8,
            )
        )
        # Panel title strip
        ax.add_patch(patches.Rectangle((x, y + 1), w, 0.7, facecolor="#26262b", edgecolor="none"))
        ax.text(
            x + 0.3,
            y + 1 + 0.35,
            f"{idx}. {panel.get('title', '(untitled)')}",
            color="#e8e8e8",
            fontsize=10,
            fontweight="bold",
            verticalalignment="center",
        )
        # queries inside the panel
        queries = _panel_queries(panel)
        body_y = y + 1 + 0.85
        for q in queries:
            wrapped = _wrap_promql(q, panel_width=w)
            for line_text in wrapped:
                ax.text(
                    x + 0.4,
                    body_y,
                    line_text,
                    family="monospace",
                    color="#9cdcfe",
                    fontsize=7.0,
                    verticalalignment="top",
                )
                body_y += 0.42
        if not queries:
            ax.text(
                x + w / 2,
                y + 1 + h / 2,
                "(no queries)",
                color="#666666",
                fontsize=9,
                ha="center",
                va="center",
            )

    # Footer
    ax.text(
        0.3,
        15.85,
        "Source: monitoring/grafana/dashboards/triage_ml.json · Fase 2 / Etapa 6",
        color="#666666",
        fontsize=8,
    )

    fig.tight_layout(pad=0.5)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dashboard",
        type=Path,
        default=DASHBOARD_SOURCE,
        help="Path to the source dashboard JSON",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=JSON_OUTPUT,
        help="Destination for the JSON copy",
    )
    parser.add_argument(
        "--png-output",
        type=Path,
        default=PNG_OUTPUT,
        help="Destination for the PNG preview",
    )
    args = parser.parse_args()

    if not args.dashboard.is_file():
        raise FileNotFoundError(f"dashboard JSON not found: {args.dashboard}")

    dashboard = json.loads(args.dashboard.read_text(encoding="utf-8"))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.dashboard, args.json_output)
    render_png(dashboard, args.png_output)
    print(f"wrote {args.json_output.relative_to(REPO_ROOT)}")
    print(f"wrote {args.png_output.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
