"""Deterministic visual timelines for replayable Silver runs."""

from html import escape
import math
from typing import Any, Sequence


def run_timeline_svg(run: Any, *, title: str = "Training run timeline") -> str:
    """Render state transitions, epoch metrics, and timing from a StoredRun."""
    events = tuple(run.events)
    width, height = 1080, 460
    left, right = 65, 1015
    baseline = 145
    times = [float(event.timestamp) for event in events]
    start = min(times, default=0.0)
    duration = max(max(times, default=start) - start, 1e-9)
    colors = {
        "run_started": "#43d8ff", "epoch": "#9b7dff",
        "run_completed": "#4de0a3", "run_failed": "#ff6b7d",
        "checkpoint_saved": "#ffbd5c",
    }
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="{escape(title)}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Inter,ui-sans-serif,system-ui,sans-serif}"
        ".title{fill:#f5f7fa;font-size:25px;font-weight:700}.meta{fill:#9fb0c3;font-size:12px}"
        ".label{fill:#dce7f3;font-size:11px}.line{stroke:#30475c;stroke-width:3}"
        ".loss{fill:none;stroke:#43d8ff;stroke-width:3}.val{fill:none;stroke:#9b7dff;stroke-width:3}</style>",
        '<rect width="1080" height="460" rx="22" fill="#0a1017"/>',
        f'<text class="title" x="34" y="44">{escape(title)}</text>',
        f'<text class="meta" x="34" y="68">{escape(str(run.id))} · {escape(str(run.state))} · '
        f'{len(events)} events · {_duration(getattr(run, "duration", None))}</text>',
        f'<line class="line" x1="{left}" y1="{baseline}" x2="{right}" y2="{baseline}"/>',
    ]
    for index, event in enumerate(events):
        x = left + (float(event.timestamp) - start) * (right - left) / duration
        color = colors.get(str(event.kind), "#8da2b7")
        radius = 7 if event.kind != "epoch" else 4
        parts.append(f'<circle cx="{x:.1f}" cy="{baseline}" r="{radius}" fill="{color}"/>')
        if event.kind != "epoch" or index in (0, len(events) - 1):
            parts.append(f'<text class="label" x="{x:.1f}" y="{baseline - 15}" text-anchor="middle">{escape(str(event.kind))}</text>')
    history = [event.data.get("metrics", {}) for event in events
               if isinstance(event.data.get("metrics"), dict)]
    loss = _series(history, "loss")
    validation = _series(history, "val_loss")
    all_values = loss + validation
    chart_x, chart_y, chart_w, chart_h = 65, 235, 950, 145
    for step in range(5):
        y = chart_y + step * chart_h / 4
        parts.append(f'<line x1="{chart_x}" y1="{y}" x2="{chart_x + chart_w}" y2="{y}" stroke="#223242"/>')
    parts.append(f'<polyline class="loss" points="{_points(loss, all_values, chart_x, chart_y, chart_w, chart_h)}"/>')
    parts.append(f'<polyline class="val" points="{_points(validation, all_values, chart_x, chart_y, chart_w, chart_h)}"/>')
    parts.extend([
        f'<text class="meta" x="{chart_x}" y="{chart_y - 18}">METRIC HISTORY · {len(history)} epochs</text>',
        f'<text x="{chart_x}" y="{chart_y + chart_h + 28}" fill="#43d8ff" font-size="12">training loss</text>',
        f'<text x="{chart_x + 100}" y="{chart_y + chart_h + 28}" fill="#9b7dff" font-size="12">validation loss</text>',
        "</svg>",
    ])
    return "".join(parts)


def _series(history: Sequence[dict], name: str) -> list[float]:
    return [float(item[name]) for item in history if name in item and _finite(item[name])]


def _points(values: Sequence[float], scale: Sequence[float], x: float, y: float,
            width: float, height: float) -> str:
    if not values:
        return ""
    minimum = min(scale, default=min(values))
    maximum = max(scale, default=max(values))
    span = maximum - minimum or 1.0
    return " ".join("%.1f,%.1f" % (
        x + index * width / max(1, len(values) - 1),
        y + height - (value - minimum) * height / span,
    ) for index, value in enumerate(values))


def _finite(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _duration(value: Any) -> str:
    return "duration unavailable" if value is None else f"{float(value):.3f}s"

