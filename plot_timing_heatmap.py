from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT = "artifacts/timing/fmsr_utterance_plan_execute.json"
DEFAULT_OUTPUT = "artifacts/timing/fmsr_utterance_cached_heatmap.svg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a one-column SVG heatmap from a timing JSON artifact."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Timing JSON produced by time_fmsr_utterance.py.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="SVG file to write.",
    )
    parser.add_argument(
        "--metric",
        default="wall_time_seconds",
        help="Scenario-level metric to plot.",
    )
    parser.add_argument(
        "--column-label",
        default="Cached",
        help="Label for the single heatmap column.",
    )
    parser.add_argument(
        "--title",
        default="Wall time heatmap with caching (lower = faster)",
        help="Chart title.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(Path(args.input).read_text())
    if _is_cache_comparison(data):
        rows = _extract_cache_comparison_rows(data)
        svg = render_multi_column_heatmap(
            rows,
            title="Mean wall time by cache mode (lower = faster)",
            column_labels=["No cache", "Cache"],
            metric_label="Mean wall time (s)",
        )
    else:
        rows = _extract_rows(data, args.metric)
        svg = render_one_column_heatmap(
            rows,
            title=args.title,
            column_label=args.column_label,
            metric_label=_metric_label(args.metric),
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg)
    print(f"Wrote {output_path}")


def _extract_rows(data: dict[str, Any], metric: str) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    for scenario in data.get("scenarios", []):
        if metric not in scenario:
            continue
        rows.append((int(scenario["id"]), float(scenario[metric])))
    if not rows:
        raise ValueError(f"No scenario rows found for metric {metric!r}")
    return sorted(rows)


def _is_cache_comparison(data: dict[str, Any]) -> bool:
    return bool(data.get("scenarios")) and all(
        "no_cache" in scenario and "cache" in scenario
        for scenario in data.get("scenarios", [])
    )


def _extract_cache_comparison_rows(
    data: dict[str, Any],
) -> list[tuple[int, list[float | None]]]:
    rows: list[tuple[int, list[float | None]]] = []
    for scenario in data.get("scenarios", []):
        rows.append(
            (
                int(scenario["id"]),
                [
                    scenario["no_cache"].get("average_wall_time_seconds"),
                    scenario["cache"].get("average_wall_time_seconds"),
                ],
            )
        )
    if not rows:
        raise ValueError("No cache comparison rows found")
    return sorted(rows)


def render_one_column_heatmap(
    rows: list[tuple[int, float]],
    *,
    title: str,
    column_label: str,
    metric_label: str,
) -> str:
    return render_multi_column_heatmap(
        [(scenario_id, [value]) for scenario_id, value in rows],
        title=title,
        column_labels=[column_label],
        metric_label=metric_label,
    )


def render_multi_column_heatmap(
    rows: list[tuple[int, list[float | None]]],
    *,
    title: str,
    column_labels: list[str],
    metric_label: str,
) -> str:
    margin_left = 92
    margin_top = 72
    cell_width = 155
    cell_height = 34
    colorbar_gap = 46
    colorbar_width = 30
    colorbar_height = max(cell_height * len(rows), 160)
    title_height = 32
    x_label_height = 54
    right_margin = 130
    bottom_margin = 56

    heatmap_width = cell_width * len(column_labels)
    heatmap_height = cell_height * len(rows)
    width = margin_left + heatmap_width + colorbar_gap + colorbar_width + right_margin
    height = margin_top + title_height + heatmap_height + x_label_height + bottom_margin
    grid_top = margin_top + title_height
    colorbar_left = margin_left + heatmap_width + colorbar_gap

    values = [
        value
        for _, row_values in rows
        for value in row_values
        if value is not None
    ]
    min_value = min(values)
    max_value = max(values)

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, Helvetica, sans-serif; fill: #111; }",
        ".title { font-size: 22px; font-weight: 500; }",
        ".axis { font-size: 17px; }",
        ".tick { font-size: 16px; }",
        ".cell-label { font-size: 15px; font-weight: 700; }",
        ".small { font-size: 14px; }",
        "</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text class="title" x="{width / 2:.1f}" y="34" text-anchor="middle">{html.escape(title)}</text>',
    ]

    for column_index, column_label in enumerate(column_labels):
        x = margin_left + column_index * cell_width + cell_width / 2
        parts.append(
            f'<text class="axis" x="{x:.1f}" y="{grid_top - 14}" text-anchor="middle">{html.escape(column_label)}</text>'
        )

    for row_index, (scenario_id, row_values) in enumerate(rows):
        y = grid_top + row_index * cell_height
        parts.append(
            f'<text class="tick" x="{margin_left - 18}" y="{y + cell_height * 0.67:.1f}" text-anchor="end">{scenario_id}</text>'
        )
        for column_index, value in enumerate(row_values):
            x = margin_left + column_index * cell_width
            if value is None:
                fill = "#d9d9d9"
                text = "n/a"
                text_fill = "#111"
            else:
                fill = _color_for_value(value, min_value, max_value)
                text = f"{value:.1f}s"
                text_fill = "#111" if _relative_luminance(fill) > 0.48 else "#f8f8f8"
            parts.extend(
                [
                    f'<rect x="{x}" y="{y}" width="{cell_width}" height="{cell_height}" fill="{fill}"/>',
                    f'<text class="cell-label" x="{x + cell_width / 2:.1f}" y="{y + cell_height * 0.65:.1f}" text-anchor="middle" fill="{text_fill}">{text}</text>',
                ]
            )

    parts.extend(
        [
            f'<rect x="{margin_left}" y="{grid_top}" width="{heatmap_width}" height="{heatmap_height}" fill="none" stroke="#111" stroke-width="1.2"/>',
            f'<text class="axis" transform="translate(24 {grid_top + heatmap_height / 2:.1f}) rotate(-90)" text-anchor="middle">Scenario ID</text>',
            f'<text class="axis" x="{margin_left + heatmap_width / 2:.1f}" y="{grid_top + heatmap_height + 42}" text-anchor="middle">Cache mode</text>',
        ]
    )

    parts.extend(
        _render_colorbar(
            left=colorbar_left,
            top=grid_top,
            width=colorbar_width,
            height=colorbar_height,
            min_value=min_value,
            max_value=max_value,
            label=metric_label,
        )
    )
    parts.append("</svg>")
    return "\n".join(parts)


def _render_colorbar(
    *,
    left: int,
    top: int,
    width: int,
    height: int,
    min_value: float,
    max_value: float,
    label: str,
) -> list[str]:
    parts: list[str] = []
    steps = 80
    for index in range(steps):
        ratio = index / (steps - 1)
        value = max_value - ratio * (max_value - min_value)
        y = top + ratio * height
        fill = _color_for_value(value, min_value, max_value)
        parts.append(
            f'<rect x="{left}" y="{y:.2f}" width="{width}" height="{height / steps + 1:.2f}" fill="{fill}"/>'
        )

    tick_values = _tick_values(min_value, max_value)
    for value in tick_values:
        ratio = 0.0 if max_value == min_value else (max_value - value) / (max_value - min_value)
        y = top + ratio * height
        parts.extend(
            [
                f'<line x1="{left + width}" x2="{left + width + 7}" y1="{y:.1f}" y2="{y:.1f}" stroke="#111"/>',
                f'<text class="small" x="{left + width + 12}" y="{y + 5:.1f}">{value:.0f}</text>',
            ]
        )

    parts.extend(
        [
            f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="none" stroke="#111" stroke-width="1"/>',
            f'<text class="axis" transform="translate({left + width + 72} {top + height / 2:.1f}) rotate(-90)" text-anchor="middle">{html.escape(label)}</text>',
        ]
    )
    return parts


def _color_for_value(value: float, min_value: float, max_value: float) -> str:
    if max_value == min_value:
        ratio = 0.0
    else:
        ratio = (value - min_value) / (max_value - min_value)
    ratio = max(0.0, min(1.0, ratio))

    green = (0, 104, 55)
    yellow = (255, 255, 191)
    red = (165, 0, 38)
    if ratio < 0.5:
        return _interpolate_color(green, yellow, ratio * 2)
    return _interpolate_color(yellow, red, (ratio - 0.5) * 2)


def _interpolate_color(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    ratio: float,
) -> str:
    red = round(start[0] + (end[0] - start[0]) * ratio)
    green = round(start[1] + (end[1] - start[1]) * ratio)
    blue = round(start[2] + (end[2] - start[2]) * ratio)
    return f"#{red:02x}{green:02x}{blue:02x}"


def _relative_luminance(hex_color: str) -> float:
    red = int(hex_color[1:3], 16) / 255
    green = int(hex_color[3:5], 16) / 255
    blue = int(hex_color[5:7], 16) / 255
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _tick_values(min_value: float, max_value: float) -> list[float]:
    if max_value == min_value:
        return [min_value]
    return [min_value, (min_value + max_value) / 2, max_value]


def _metric_label(metric: str) -> str:
    labels = {
        "wall_time_seconds": "Wall time (s)",
    }
    return labels.get(metric, metric.replace("_", " "))


if __name__ == "__main__":
    main()
