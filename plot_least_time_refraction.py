#!/usr/bin/env python3
"""
plot_least_time_refraction.py

Generate the least-time refraction illustration for The Harlequin Paradox.

The figure treats attention as a ray moving through two interpretive regions:

- a rigid frame with higher interpretive resistance
- a playful frame with lower interpretive resistance

The crossing point is chosen by minimizing an optical-path-like cost,

    T(x) = n_rigid * d_rigid(x) + n_play * d_play(x),

which produces a Snell-style stationary condition,

    n_rigid sin(theta_rigid)
        = n_play sin(theta_play).

This is a conceptual analogy for frame selection. It is not a claim that
human attention literally obeys geometrical optics.

Author: Don Merrow
Project: The Harlequin Paradox
Organization: IcreateCrypto Research
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RefractionConfig:
    """Geometry and rendering controls."""

    source_x: float = -3.4
    source_y: float = 2.6
    target_x: float = 3.8
    target_y: float = -2.5

    rigid_index: float = 1.75
    play_index: float = 0.82

    search_min: float = -4.5
    search_max: float = 4.5
    search_points: int = 20_001

    output: Path = Path("figures/least_time_refraction.png")
    dpi: int = 180
    show: bool = False


@dataclass(frozen=True)
class RefractionResult:
    """Calculated least-cost path and angle relationship."""

    crossing_x: float
    rigid_distance: float
    play_distance: float
    path_cost: float
    rigid_angle_degrees: float
    play_angle_degrees: float
    snell_left: float
    snell_right: float
    snell_error: float


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def distance(
    x1: float,
    y1: float,
    x2: FloatArray | float,
    y2: float,
) -> FloatArray:
    """Euclidean distance supporting scalar or array x2."""
    return np.sqrt((np.asarray(x2) - x1) ** 2 + (y2 - y1) ** 2)


def optical_path_cost(
    crossing_x: FloatArray,
    config: RefractionConfig,
) -> FloatArray:
    """
    Calculate the interpretive travel cost for every crossing point.

    The interface is y = 0.
    """
    upper_distance = distance(
        config.source_x,
        config.source_y,
        crossing_x,
        0.0,
    )
    lower_distance = distance(
        config.target_x,
        config.target_y,
        crossing_x,
        0.0,
    )

    return (
        config.rigid_index * upper_distance
        + config.play_index * lower_distance
    )


def solve_least_time_path(
    config: RefractionConfig,
) -> RefractionResult:
    """Find the minimum-cost crossing point by dense deterministic search."""
    crossing_grid = np.linspace(
        config.search_min,
        config.search_max,
        config.search_points,
        dtype=np.float64,
    )

    costs = optical_path_cost(crossing_grid, config)
    minimum_index = int(np.argmin(costs))
    crossing_x = float(crossing_grid[minimum_index])

    rigid_distance = float(
        distance(
            config.source_x,
            config.source_y,
            crossing_x,
            0.0,
        )
    )
    play_distance = float(
        distance(
            crossing_x,
            0.0,
            config.target_x,
            config.target_y,
        )
    )

    rigid_horizontal = abs(crossing_x - config.source_x)
    play_horizontal = abs(config.target_x - crossing_x)

    rigid_angle = float(
        np.arctan2(rigid_horizontal, abs(config.source_y))
    )
    play_angle = float(
        np.arctan2(play_horizontal, abs(config.target_y))
    )

    snell_left = config.rigid_index * np.sin(rigid_angle)
    snell_right = config.play_index * np.sin(play_angle)

    return RefractionResult(
        crossing_x=crossing_x,
        rigid_distance=rigid_distance,
        play_distance=play_distance,
        path_cost=float(costs[minimum_index]),
        rigid_angle_degrees=float(np.degrees(rigid_angle)),
        play_angle_degrees=float(np.degrees(play_angle)),
        snell_left=float(snell_left),
        snell_right=float(snell_right),
        snell_error=float(abs(snell_left - snell_right)),
    )


def draw_angle_arc(
    axis: plt.Axes,
    center_x: float,
    center_y: float,
    radius: float,
    start_degrees: float,
    end_degrees: float,
    label: str,
    label_x: float,
    label_y: float,
) -> None:
    """Draw a simple angle arc without extra plotting dependencies."""
    angles = np.radians(
        np.linspace(start_degrees, end_degrees, 80)
    )
    axis.plot(
        center_x + radius * np.cos(angles),
        center_y + radius * np.sin(angles),
        linewidth=1.2,
    )
    axis.text(
        label_x,
        label_y,
        label,
        fontsize=10,
        ha="center",
        va="center",
    )


def plot_refraction(
    config: RefractionConfig,
    result: RefractionResult,
) -> Path:
    """Render the least-time path and interpretive regions."""
    output_path = config.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(11, 7))

    x_min = min(
        config.search_min,
        config.source_x - 0.8,
        config.target_x - 0.8,
    )
    x_max = max(
        config.search_max,
        config.source_x + 0.8,
        config.target_x + 0.8,
    )
    y_min = min(config.target_y - 0.9, -3.4)
    y_max = max(config.source_y + 0.9, 3.5)

    axis.axhspan(
        0.0,
        y_max,
        alpha=0.10,
        label=(
            "Rigid frame: higher interpretive resistance "
            rf"$n={config.rigid_index:.2f}$"
        ),
    )
    axis.axhspan(
        y_min,
        0.0,
        alpha=0.08,
        label=(
            "Playful frame: lower interpretive resistance "
            rf"$n={config.play_index:.2f}$"
        ),
    )

    axis.axhline(
        0.0,
        linestyle="-",
        linewidth=1.5,
        label="Frame boundary",
    )

    axis.plot(
        [result.crossing_x, result.crossing_x],
        [-1.2, 1.2],
        linestyle=":",
        linewidth=1.2,
        label="Boundary normal",
    )

    axis.plot(
        [
            config.source_x,
            result.crossing_x,
            config.target_x,
        ],
        [
            config.source_y,
            0.0,
            config.target_y,
        ],
        linewidth=2.8,
        marker="o",
        label="Least-cost attention path",
    )

    direct_x = np.array(
        [config.source_x, config.target_x],
        dtype=np.float64,
    )
    direct_y = np.array(
        [config.source_y, config.target_y],
        dtype=np.float64,
    )
    axis.plot(
        direct_x,
        direct_y,
        linestyle="--",
        linewidth=1.1,
        alpha=0.65,
        label="Unrefracted geometric path",
    )

    axis.scatter(
        [config.source_x],
        [config.source_y],
        s=80,
        zorder=5,
    )
    axis.scatter(
        [config.target_x],
        [config.target_y],
        s=80,
        zorder=5,
    )
    axis.scatter(
        [result.crossing_x],
        [0.0],
        s=90,
        zorder=6,
    )

    axis.annotate(
        "Aggressive frame",
        xy=(config.source_x, config.source_y),
        xytext=(config.source_x - 0.15, config.source_y + 0.48),
        ha="center",
        arrowprops={"arrowstyle": "->", "linewidth": 1.0},
    )

    axis.annotate(
        "Reframed interpretation",
        xy=(config.target_x, config.target_y),
        xytext=(config.target_x - 0.2, config.target_y - 0.62),
        ha="center",
        arrowprops={"arrowstyle": "->", "linewidth": 1.0},
    )

    axis.annotate(
        "Humorous disruption",
        xy=(result.crossing_x, 0.0),
        xytext=(result.crossing_x + 1.2, 0.72),
        ha="center",
        arrowprops={"arrowstyle": "->", "linewidth": 1.0},
    )

    rigid_angle = result.rigid_angle_degrees
    play_angle = result.play_angle_degrees

    draw_angle_arc(
        axis=axis,
        center_x=result.crossing_x,
        center_y=0.0,
        radius=0.62,
        start_degrees=90.0,
        end_degrees=90.0 + rigid_angle,
        label=r"$\theta_{\mathrm{rigid}}$",
        label_x=result.crossing_x - 0.46,
        label_y=0.53,
    )

    draw_angle_arc(
        axis=axis,
        center_x=result.crossing_x,
        center_y=0.0,
        radius=0.72,
        start_degrees=-90.0,
        end_degrees=-90.0 + play_angle,
        label=r"$\theta_{\mathrm{play}}$",
        label_x=result.crossing_x + 0.52,
        label_y=-0.55,
    )

    equation_text = (
        r"$n_{\mathrm{rigid}}\sin\theta_{\mathrm{rigid}}"
        r"="
        r"n_{\mathrm{play}}\sin\theta_{\mathrm{play}}$"
        "\n"
        rf"{result.snell_left:.3f}"
        r"$\;\approx\;$"
        rf"{result.snell_right:.3f}"
    )

    axis.text(
        0.02,
        0.03,
        equation_text,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "alpha": 0.82,
        },
    )

    axis.text(
        0.98,
        0.97,
        (
            "Conceptual analogy only\n"
            "Attention is not claimed to obey optics"
        ),
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )

    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Interpretive position")
    axis.set_ylabel("Frame region")
    axis.set_title(
        "Least-Time Refraction of Attention Between Rigid and Playful Frames"
    )
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=config.dpi, bbox_inches="tight")

    if config.show:
        plt.show()

    plt.close(figure)
    return output_path


def print_summary(
    config: RefractionConfig,
    result: RefractionResult,
    output_path: Path,
) -> None:
    """Print the solved geometry and Snell-style consistency check."""
    print("\nHarlequin least-time refraction")
    print("=" * 32)
    print(f"Rigid-frame index:        {config.rigid_index:.6f}")
    print(f"Play-frame index:         {config.play_index:.6f}")
    print(f"Least-cost crossing x:    {result.crossing_x:.6f}")
    print(f"Rigid-segment distance:   {result.rigid_distance:.6f}")
    print(f"Play-segment distance:    {result.play_distance:.6f}")
    print(f"Total path cost:          {result.path_cost:.6f}")
    print(
        "Rigid angle from normal: "
        f"{result.rigid_angle_degrees:.6f} degrees"
    )
    print(
        "Play angle from normal:  "
        f"{result.play_angle_degrees:.6f} degrees"
    )
    print(f"Snell left side:          {result.snell_left:.6f}")
    print(f"Snell right side:         {result.snell_right:.6f}")
    print(f"Absolute Snell error:     {result.snell_error:.8f}")
    print(f"Generated figure:         {output_path}")
    print(
        "\nInterpretation warning: this is a least-cost analogy for "
        "attention and frame selection, not a physical law of negotiation."
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate the Harlequin Paradox least-time "
            "refraction illustration."
        )
    )
    parser.add_argument(
        "--source-x",
        type=float,
        default=-3.4,
    )
    parser.add_argument(
        "--source-y",
        type=positive_float,
        default=2.6,
        help="positive y-coordinate in the rigid frame",
    )
    parser.add_argument(
        "--target-x",
        type=float,
        default=3.8,
    )
    parser.add_argument(
        "--target-depth",
        type=positive_float,
        default=2.5,
        help="positive depth below the boundary",
    )
    parser.add_argument(
        "--rigid-index",
        type=positive_float,
        default=1.75,
    )
    parser.add_argument(
        "--play-index",
        type=positive_float,
        default=0.82,
    )
    parser.add_argument(
        "--search-min",
        type=float,
        default=-4.5,
    )
    parser.add_argument(
        "--search-max",
        type=float,
        default=4.5,
    )
    parser.add_argument(
        "--search-points",
        type=positive_int,
        default=20_001,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/least_time_refraction.png"),
    )
    parser.add_argument(
        "--dpi",
        type=positive_int,
        default=180,
    )
    parser.add_argument(
        "--show",
        action="store_true",
    )

    return parser.parse_args()


def main() -> int:
    """Program entry point."""
    args = parse_args()

    if args.search_max <= args.search_min:
        raise SystemExit(
            "--search-max must be greater than --search-min"
        )

    config = RefractionConfig(
        source_x=args.source_x,
        source_y=args.source_y,
        target_x=args.target_x,
        target_y=-args.target_depth,
        rigid_index=args.rigid_index,
        play_index=args.play_index,
        search_min=args.search_min,
        search_max=args.search_max,
        search_points=args.search_points,
        output=args.output,
        dpi=args.dpi,
        show=args.show,
    )

    result = solve_least_time_path(config)
    output_path = plot_refraction(config, result)
    print_summary(config, result, output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
