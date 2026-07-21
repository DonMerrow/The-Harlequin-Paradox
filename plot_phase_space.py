#!/usr/bin/env python3
"""
plot_phase_space.py

Generate a phase-space visualization for The Harlequin Paradox.

The graph compares:

    x-axis: playful uncertainty U(H_effect)
    y-axis: aggressor influence Phi_A

A heuristic boundary separates the region where the expanded jester influence
is expected to exceed aggressor influence from the region where aggression is
expected to remain dominant.

This is an exploratory visualization. The boundary is derived from the model
assumptions selected here and is not an empirically established law.

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
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class PhaseSpaceConfig:
    """Controls for sampling and rendering the phase-space graph."""

    samples: int = 2_000
    seed: int = 42
    output: Path = Path("figures/phase_space.png")
    dpi: int = 180
    grid_size: int = 250
    show: bool = False


@dataclass(frozen=True)
class PhaseSpaceData:
    """Sampled variables and calculated phase-space terms."""

    playful_uncertainty: FloatArray
    aggressor_influence: FloatArray
    supporting_disruption: FloatArray
    total_jester_influence: FloatArray
    harlequin_margin: FloatArray
    silly_equilibrium: BoolArray


def positive_int(value: str) -> int:
    """Validate positive integer command-line values."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    """Validate non-negative integer command-line values."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def sample_phase_space(
    config: PhaseSpaceConfig,
) -> PhaseSpaceData:
    """
    Sample the Harlequin phase space.

    Aggressor influence:
        Phi_A = P_aggr * L * S * A_size

    Playful uncertainty:
        U_H = U * H_effect

    Supporting disruption:
        B = T_pain + C * C_size * S_aggr * J_size + X

    Total jester influence:
        Phi_J = U_H + B

    Silly Equilibrium:
        Phi_J > Phi_A
    """
    rng = np.random.default_rng(config.seed)
    n = config.samples

    perceived_dominance = rng.uniform(0.0, 1.0, n)
    aggression_level = rng.uniform(0.0, 1.0, n)
    stance_rigidity = rng.uniform(0.0, 1.0, n)
    aggressor_size = rng.uniform(0.5, 2.0, n)

    uncertainty = rng.uniform(0.0, 0.65, n)
    humor_effectiveness = rng.uniform(0.0, 1.0, n)
    playful_uncertainty = uncertainty * humor_effectiveness

    pain_tolerance = rng.uniform(0.0, 0.20, n)
    crowd_engagement = rng.uniform(0.0, 1.25, n)
    crowd_size = rng.integers(1, 11, n).astype(np.float64)
    shame_susceptibility = rng.uniform(0.0, 0.20, n)
    jester_size = rng.uniform(0.25, 1.0, n)
    chaos = rng.uniform(0.0, 0.55, n)

    aggressor_influence = (
        perceived_dominance
        * aggression_level
        * stance_rigidity
        * aggressor_size
    )

    supporting_disruption = (
        pain_tolerance
        + (
            crowd_engagement
            * crowd_size
            * shame_susceptibility
            * jester_size
        )
        + chaos
    )

    total_jester_influence = (
        playful_uncertainty + supporting_disruption
    )

    harlequin_margin = (
        total_jester_influence - aggressor_influence
    )

    silly_equilibrium = harlequin_margin > 0.0

    return PhaseSpaceData(
        playful_uncertainty=playful_uncertainty,
        aggressor_influence=aggressor_influence,
        supporting_disruption=supporting_disruption,
        total_jester_influence=total_jester_influence,
        harlequin_margin=harlequin_margin,
        silly_equilibrium=silly_equilibrium,
    )


def heuristic_boundary(
    x_values: FloatArray,
    supporting_disruption: FloatArray,
) -> FloatArray:
    """
    Construct a representative phase boundary.

    Since the exact boundary depends on all supporting variables, this plot
    uses their median aggregate contribution:

        Phi_A = U(H_effect) + median(B)

    where:
        B = T_pain + crowd term + X
    """
    baseline = float(np.median(supporting_disruption))
    return x_values + baseline


def save_figure(
    path: Path,
    config: PhaseSpaceConfig,
) -> None:
    """Save and optionally display the current figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=config.dpi, bbox_inches="tight")

    if config.show:
        plt.show()

    plt.close()


def plot_phase_space(
    data: PhaseSpaceData,
    config: PhaseSpaceConfig,
) -> Path:
    """Render the sampled outcome space and heuristic boundary."""
    output_path = config.output

    x_max = max(
        0.70,
        float(data.playful_uncertainty.max()) * 1.05,
    )
    y_max = max(
        1.10,
        float(data.aggressor_influence.max()) * 1.05,
    )

    x_grid = np.linspace(0.0, x_max, config.grid_size)
    boundary = heuristic_boundary(
        x_grid,
        data.supporting_disruption,
    )
    clipped_boundary = np.clip(boundary, 0.0, y_max)

    plt.figure(figsize=(10, 7))

    plt.fill_between(
        x_grid,
        0.0,
        clipped_boundary,
        alpha=0.10,
        label="Heuristic Harlequin-favoured region",
    )

    plt.fill_between(
        x_grid,
        clipped_boundary,
        y_max,
        alpha=0.08,
        label="Heuristic aggressor-favoured region",
    )

    mask = data.silly_equilibrium

    plt.scatter(
        data.playful_uncertainty[mask],
        data.aggressor_influence[mask],
        s=22,
        alpha=0.48,
        label="Simulated Silly Equilibrium",
    )

    plt.scatter(
        data.playful_uncertainty[~mask],
        data.aggressor_influence[~mask],
        s=22,
        alpha=0.48,
        label="Simulated aggressor dominance",
    )

    plt.plot(
        x_grid,
        boundary,
        linestyle="--",
        linewidth=2.0,
        label=(
            r"Heuristic threshold: "
            r"$\Phi_A=U(H_{\mathrm{effect}})+\widetilde{B}$"
        ),
    )

    plt.xlim(0.0, x_max)
    plt.ylim(0.0, y_max)
    plt.xlabel(
        r"Playful uncertainty $U(H_{\mathrm{effect}})$"
    )
    plt.ylabel(
        r"Aggressor influence "
        r"$\Phi_A=P_{\mathrm{aggr}}LSA_{\mathrm{size}}$"
    )
    plt.title(
        "Harlequin Paradox Phase Space"
    )
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)

    median_support = float(
        np.median(data.supporting_disruption)
    )

    annotation = (
        "Boundary uses median supporting disruption\n"
        rf"$\widetilde{{B}}={median_support:.3f}$"
    )

    plt.text(
        0.98,
        0.03,
        annotation,
        transform=plt.gca().transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )

    save_figure(output_path, config)
    return output_path


def print_summary(
    data: PhaseSpaceData,
    config: PhaseSpaceConfig,
    output_path: Path,
) -> None:
    """Print the simulation assumptions and outcome counts."""
    wins = int(np.count_nonzero(data.silly_equilibrium))
    losses = config.samples - wins

    print("\nHarlequin phase-space simulation")
    print("=" * 33)
    print(f"Random seed:                {config.seed}")
    print(f"Samples:                    {config.samples}")
    print(
        "Median supporting term:    "
        f"{np.median(data.supporting_disruption):.6f}"
    )
    print(
        "Mean aggressor influence:  "
        f"{data.aggressor_influence.mean():.6f}"
    )
    print(
        "Mean jester influence:     "
        f"{data.total_jester_influence.mean():.6f}"
    )
    print(
        "Silly Equilibrium samples: "
        f"{wins} ({wins / config.samples:.2%})"
    )
    print(
        "Aggressor-dominant samples:"
        f" {losses} ({losses / config.samples:.2%})"
    )
    print(f"Generated figure:           {output_path}")

    print(
        "\nInterpretation warning: the displayed boundary is heuristic. "
        "It uses the median of the omitted supporting variables and is "
        "not an empirical decision boundary."
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate the Harlequin Paradox phase-space graph."
        )
    )
    parser.add_argument(
        "--samples",
        type=positive_int,
        default=2_000,
        help="number of simulated encounters (default: 2000)",
    )
    parser.add_argument(
        "--seed",
        type=nonnegative_int,
        default=42,
        help="random seed (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/phase_space.png"),
        help=(
            "output image path "
            "(default: figures/phase_space.png)"
        ),
    )
    parser.add_argument(
        "--dpi",
        type=positive_int,
        default=180,
        help="image resolution in dots per inch (default: 180)",
    )
    parser.add_argument(
        "--grid-size",
        type=positive_int,
        default=250,
        help="number of points used for the threshold line",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="display the graph interactively",
    )
    return parser.parse_args()


def main() -> int:
    """Program entry point."""
    args = parse_args()

    config = PhaseSpaceConfig(
        samples=args.samples,
        seed=args.seed,
        output=args.output,
        dpi=args.dpi,
        grid_size=args.grid_size,
        show=args.show,
    )

    data = sample_phase_space(config)
    output_path = plot_phase_space(data, config)
    print_summary(data, config, output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
