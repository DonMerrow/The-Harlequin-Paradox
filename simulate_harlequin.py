#!/usr/bin/env python3
"""
simulate_harlequin.py

Reconstructs the original Harlequin Paradox Monte Carlo graph series:

1. Cumulative probability of Harlequin margin
2. Jester-win and aggressor-win scatter plots
3. Wave comparison across indexed simulations

The simulation is exploratory. Its output reflects the chosen distributions,
weights, and random seed. It is not an empirical real-world success rate.

Author: Don Merrow
Project: The Harlequin Paradox
Organization: IcreateCrypto Research
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for one Harlequin Paradox Monte Carlo run."""

    samples: int = 1_000
    seed: int = 42
    output_dir: Path = Path("figures")
    dpi: int = 180
    show: bool = False


@dataclass(frozen=True)
class SimulationData:
    """Randomized model variables."""

    perceived_dominance: FloatArray
    aggression_level: FloatArray
    stance_rigidity: FloatArray
    aggressor_size: FloatArray

    uncertainty: FloatArray
    pain_tolerance: FloatArray
    crowd_engagement: FloatArray
    crowd_size: FloatArray
    shame_susceptibility: FloatArray
    jester_size: FloatArray
    humor_effectiveness: FloatArray
    chaos: FloatArray


@dataclass(frozen=True)
class SimulationResult:
    """Calculated aggregate terms and outcomes."""

    aggressor_influence: FloatArray
    jester_influence: FloatArray
    harlequin_margin: FloatArray
    silly_equilibrium: BoolArray


DEFAULT_FIGURE_NAMES: Final[dict[str, str]] = {
    "cumulative": "cumulative_probability.png",
    "jester_scatter": "jester_wins_scatter.png",
    "aggressor_scatter": "aggressor_wins_scatter.png",
    "combined_scatter": "combined_outcomes_scatter.png",
    "wave": "wave_comparison.png",
}


def positive_int(value: str) -> int:
    """argparse validator for positive integers."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    """argparse validator for non-negative integers."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def generate_data(config: SimulationConfig) -> SimulationData:
    """
    Generate bounded random variables for the exploratory model.

    Most behavioral variables are normalized to [0, 1].
    Size and crowd variables use wider ranges to preserve the contrast
    present in the original simulations.
    """
    rng = np.random.default_rng(config.seed)
    n = config.samples

    return SimulationData(
        perceived_dominance=rng.uniform(0.0, 1.0, n),
        aggression_level=rng.uniform(0.0, 1.0, n),
        stance_rigidity=rng.uniform(0.0, 1.0, n),
        aggressor_size=rng.uniform(0.5, 2.0, n),
        uncertainty=rng.uniform(0.0, 0.5, n),
        pain_tolerance=rng.uniform(0.0, 0.2, n),
        crowd_engagement=rng.uniform(0.0, 2.0, n),
        crowd_size=rng.integers(1, 11, n).astype(np.float64),
        shame_susceptibility=rng.uniform(0.0, 0.2, n),
        jester_size=rng.uniform(0.25, 1.0, n),
        humor_effectiveness=rng.uniform(0.0, 1.0, n),
        chaos=rng.uniform(0.0, 1.5, n),
    )


def calculate_outcomes(data: SimulationData) -> SimulationResult:
    r"""
    Calculate the aggregate influence terms.

    Aggressor influence:
        Phi_A = P_aggr * L * S * A_size

    Jester influence:
        Phi_J =
            U * H_effect
            + T_pain
            + C * C_size * S_aggr * J_size
            + X

    Harlequin margin:
        Delta_Phi = Phi_J - Phi_A

    Silly Equilibrium holds when:
        Delta_Phi > 0
    """
    aggressor_influence = (
        data.perceived_dominance
        * data.aggression_level
        * data.stance_rigidity
        * data.aggressor_size
    )

    jester_influence = (
        data.uncertainty * data.humor_effectiveness
        + data.pain_tolerance
        + (
            data.crowd_engagement
            * data.crowd_size
            * data.shame_susceptibility
            * data.jester_size
        )
        + data.chaos
    )

    harlequin_margin = jester_influence - aggressor_influence
    silly_equilibrium = harlequin_margin > 0.0

    return SimulationResult(
        aggressor_influence=aggressor_influence,
        jester_influence=jester_influence,
        harlequin_margin=harlequin_margin,
        silly_equilibrium=silly_equilibrium,
    )


def save_figure(path: Path, dpi: int, show: bool) -> None:
    """Save, optionally display, and close the active figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()


def plot_cumulative_probability(
    result: SimulationResult,
    config: SimulationConfig,
) -> Path:
    """
    Plot the empirical cumulative distribution of the Harlequin margin.

    Positive margin means jester influence exceeds aggressor influence.
    The zero line is the Silly Equilibrium threshold.
    """
    sorted_margin = np.sort(result.harlequin_margin)
    cumulative_probability = (
        np.arange(1, sorted_margin.size + 1, dtype=np.float64)
        / sorted_margin.size
    )

    output_path = config.output_dir / DEFAULT_FIGURE_NAMES["cumulative"]

    plt.figure(figsize=(10, 6))
    plt.plot(
        sorted_margin,
        cumulative_probability,
        linewidth=2.0,
        label="Empirical cumulative probability",
    )
    plt.axvline(
        0.0,
        linestyle="--",
        linewidth=1.5,
        label="Silly Equilibrium threshold",
    )
    plt.xlabel(
        r"Harlequin margin $\Delta\Phi = \Phi_J - \Phi_A$"
    )
    plt.ylabel("Cumulative probability")
    plt.title("Cumulative Probability of Humor vs. Aggression Outcomes")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_figure(output_path, config.dpi, config.show)
    return output_path


def plot_separate_scatter_outcomes(
    result: SimulationResult,
    config: SimulationConfig,
) -> tuple[Path, Path]:
    """Create separate scatter plots for jester and aggressor outcomes."""
    mask = result.silly_equilibrium
    maximum = float(
        max(
            result.aggressor_influence.max(),
            result.jester_influence.max(),
        )
    )

    jester_path = config.output_dir / DEFAULT_FIGURE_NAMES["jester_scatter"]
    aggressor_path = (
        config.output_dir / DEFAULT_FIGURE_NAMES["aggressor_scatter"]
    )

    plt.figure(figsize=(8, 6))
    plt.scatter(
        result.aggressor_influence[mask],
        result.jester_influence[mask],
        alpha=0.55,
        s=24,
        label="Silly Equilibrium",
    )
    plt.plot(
        [0.0, maximum],
        [0.0, maximum],
        linestyle="--",
        linewidth=1.2,
        label=r"Equality line: $\Phi_J = \Phi_A$",
    )
    plt.xlabel(r"Aggressor influence $\Phi_A$")
    plt.ylabel(r"Jester influence $\Phi_J$")
    plt.title("Silly Equilibrium: Jester Influence Prevails")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_figure(jester_path, config.dpi, config.show)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        result.aggressor_influence[~mask],
        result.jester_influence[~mask],
        alpha=0.55,
        s=24,
        label="Aggressor dominance",
    )
    plt.plot(
        [0.0, maximum],
        [0.0, maximum],
        linestyle="--",
        linewidth=1.2,
        label=r"Equality line: $\Phi_J = \Phi_A$",
    )
    plt.xlabel(r"Aggressor influence $\Phi_A$")
    plt.ylabel(r"Jester influence $\Phi_J$")
    plt.title("Aggressor Dominance: Humor Fails to Disrupt")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_figure(aggressor_path, config.dpi, config.show)

    return jester_path, aggressor_path


def plot_combined_scatter(
    result: SimulationResult,
    config: SimulationConfig,
) -> Path:
    """Create a single phase-style scatter plot containing both outcomes."""
    mask = result.silly_equilibrium
    maximum = float(
        max(
            result.aggressor_influence.max(),
            result.jester_influence.max(),
        )
    )

    output_path = (
        config.output_dir / DEFAULT_FIGURE_NAMES["combined_scatter"]
    )

    plt.figure(figsize=(9, 7))
    plt.scatter(
        result.aggressor_influence[mask],
        result.jester_influence[mask],
        alpha=0.5,
        s=24,
        label="Jester influence prevails",
    )
    plt.scatter(
        result.aggressor_influence[~mask],
        result.jester_influence[~mask],
        alpha=0.5,
        s=24,
        label="Aggressor influence prevails",
    )
    plt.plot(
        [0.0, maximum],
        [0.0, maximum],
        linestyle="--",
        linewidth=1.2,
        label=r"Threshold: $\Phi_J = \Phi_A$",
    )
    plt.xlabel(r"Aggressor influence $\Phi_A$")
    plt.ylabel(r"Jester influence $\Phi_J$")
    plt.title("Harlequin Paradox Outcome Space")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_figure(output_path, config.dpi, config.show)
    return output_path


def plot_wave_comparison(
    result: SimulationResult,
    config: SimulationConfig,
) -> Path:
    """
    Compare influence terms across indexed Monte Carlo trials.

    The x-axis is an index of independent trials, not physical time.
    """
    x = np.arange(result.aggressor_influence.size)
    mask = result.silly_equilibrium
    output_path = config.output_dir / DEFAULT_FIGURE_NAMES["wave"]

    plt.figure(figsize=(12, 6))
    plt.plot(
        x,
        result.jester_influence,
        alpha=0.72,
        linewidth=1.0,
        label=r"Jester influence $\Phi_J$",
    )
    plt.plot(
        x,
        result.aggressor_influence,
        alpha=0.72,
        linewidth=1.0,
        label=r"Aggressor influence $\Phi_A$",
    )
    plt.fill_between(
        x,
        result.jester_influence,
        result.aggressor_influence,
        where=mask,
        alpha=0.25,
        interpolate=True,
        label="Humor wins",
    )
    plt.fill_between(
        x,
        result.jester_influence,
        result.aggressor_influence,
        where=~mask,
        alpha=0.25,
        interpolate=True,
        label="Aggression wins",
    )
    plt.xlabel("Indexed Monte Carlo trial")
    plt.ylabel("Aggregate influence")
    plt.title("Wave Comparison of Humor vs. Aggression")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_figure(output_path, config.dpi, config.show)
    return output_path


def print_summary(
    result: SimulationResult,
    config: SimulationConfig,
    generated_paths: list[Path],
) -> None:
    """Print a transparent summary of the simulation and generated files."""
    wins = int(np.count_nonzero(result.silly_equilibrium))
    losses = config.samples - wins
    win_rate = wins / config.samples
    loss_rate = losses / config.samples

    print("\nHarlequin Paradox simulation")
    print("=" * 31)
    print(f"Random seed:             {config.seed}")
    print(f"Trials:                  {config.samples}")
    print(f"Jester-influence wins:   {wins} ({win_rate:.2%})")
    print(f"Aggressor-influence wins:{losses} ({loss_rate:.2%})")
    print(
        "Mean Harlequin margin:  "
        f"{result.harlequin_margin.mean():.6f}"
    )
    print(
        "Median Harlequin margin:"
        f" {np.median(result.harlequin_margin):.6f}"
    )
    print("\nGenerated figures:")
    for path in generated_paths:
        print(f"  - {path}")

    print(
        "\nInterpretation warning: these percentages describe this "
        "simulation configuration only. They are not empirical rates."
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate reconstructed Harlequin Paradox Monte Carlo graphs."
        )
    )
    parser.add_argument(
        "--samples",
        type=positive_int,
        default=1_000,
        help="number of Monte Carlo trials (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=nonnegative_int,
        default=42,
        help="random seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="directory for generated figures (default: figures)",
    )
    parser.add_argument(
        "--dpi",
        type=positive_int,
        default=180,
        help="image resolution in dots per inch (default: 180)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="display figures interactively after saving",
    )
    return parser.parse_args()


def main() -> int:
    """Program entry point."""
    args = parse_args()
    config = SimulationConfig(
        samples=args.samples,
        seed=args.seed,
        output_dir=args.output_dir,
        dpi=args.dpi,
        show=args.show,
    )

    data = generate_data(config)
    result = calculate_outcomes(data)

    generated_paths: list[Path] = []
    generated_paths.append(plot_cumulative_probability(result, config))

    jester_path, aggressor_path = plot_separate_scatter_outcomes(
        result,
        config,
    )
    generated_paths.extend([jester_path, aggressor_path])

    generated_paths.append(plot_combined_scatter(result, config))
    generated_paths.append(plot_wave_comparison(result, config))

    print_summary(result, config, generated_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
