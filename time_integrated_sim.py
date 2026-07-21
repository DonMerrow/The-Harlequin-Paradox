#!/usr/bin/env python3
"""
time_integrated_sim.py

Time-integrated simulation for The Harlequin Paradox.

This model extends the original static Silly Equilibrium with:

- time-varying aggressor and jester influence
- Harlequin exhaustion and recovery
- effective-humor damping
- cumulative influence margin
- safety abort conditions
- distance-to-resolution tracking
- Monte Carlo outcome distributions

The simulation is exploratory. Its output depends on the selected equations,
parameter ranges, random seed, and thresholds. It does not establish a
real-world success rate for humor in conflict.

Author: Don Merrow
Project: The Harlequin Paradox
Organization: IcreateCrypto Research
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


class OutcomeClass(str, Enum):
    """Four influence-versus-resolution outcome classes."""

    PRODUCTIVE_HARLEQUIN = "productive_harlequin"
    PERFORMATIVE_VICTORY = "performative_victory"
    AGGRESSOR_LED_CLOSURE = "aggressor_led_closure"
    AGGRESSIVE_FAILURE = "aggressive_failure"
    SAFETY_ABORT = "safety_abort"


@dataclass(frozen=True)
class SimulationConfig:
    """Global simulation controls."""

    duration: float = 30.0
    steps: int = 600
    runs: int = 1_000
    seed: int = 42
    output_dir: Path = Path("figures")
    dpi: int = 180
    show: bool = False
    safety_threshold: float = 1.25


@dataclass(frozen=True)
class TrialParameters:
    """Parameters sampled once for a simulated encounter."""

    perceived_dominance: float
    aggressor_size: float
    jester_size: float
    crowd_size: float
    shame_susceptibility: float

    initial_issue_distance: float
    resolution_floor: float

    exhaustion_growth: float
    recovery_rate: float
    exhaustion_damping: float

    humor_effort_weight: float
    aggression_effort_weight: float

    humor_resolution_coupling: float
    aggression_resolution_coupling: float
    natural_resolution_drift: float
    resolution_noise: float


@dataclass(frozen=True)
class TrialSeries:
    """Time series produced by one simulated encounter."""

    time: FloatArray
    aggression_level: FloatArray
    stance_rigidity: FloatArray
    humor_effectiveness: FloatArray
    effective_humor: FloatArray
    uncertainty: FloatArray
    pain_tolerance: FloatArray
    crowd_engagement: FloatArray
    chaos: FloatArray

    aggressor_influence: FloatArray
    jester_influence: FloatArray
    margin: FloatArray
    cumulative_margin: FloatArray

    exhaustion: FloatArray
    risk: FloatArray
    issue_distance: FloatArray
    resolution_velocity: FloatArray

    aborted: bool
    abort_index: int | None
    outcome: OutcomeClass


@dataclass(frozen=True)
class MonteCarloSummary:
    """Aggregate results across many trials."""

    final_margins: FloatArray
    initial_distances: FloatArray
    final_distances: FloatArray
    progress: FloatArray
    aborted: BoolArray
    outcomes: tuple[OutcomeClass, ...]


FIGURE_NAMES: Final[dict[str, str]] = {
    "dynamics": "time_integrated_dynamics.png",
    "margin": "cumulative_margin.png",
    "distance": "resolution_distance.png",
    "distribution": "outcome_distribution.png",
    "outcome_classes": "outcome_classes.png",
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def bounded_wave(
    time: FloatArray,
    base: float,
    amplitude: float,
    frequency: float,
    phase: float,
    noise_scale: float,
    rng: np.random.Generator,
    lower: float = 0.0,
    upper: float = 1.0,
) -> FloatArray:
    """Create a noisy bounded sinusoidal behavioral signal."""
    values = (
        base
        + amplitude * np.sin((2.0 * np.pi * frequency * time) + phase)
        + rng.normal(0.0, noise_scale, time.size)
    )
    return np.clip(values, lower, upper)


def smooth_signal(values: FloatArray, window: int = 9) -> FloatArray:
    """Apply a simple moving average without changing array length."""
    if window <= 1:
        return values.copy()

    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: values.size]


def sample_trial_parameters(
    rng: np.random.Generator,
) -> TrialParameters:
    """Sample one plausible but explicitly synthetic encounter."""
    return TrialParameters(
        perceived_dominance=float(rng.uniform(0.45, 1.0)),
        aggressor_size=float(rng.uniform(0.75, 1.8)),
        jester_size=float(rng.uniform(0.35, 1.0)),
        crowd_size=float(rng.integers(1, 11)),
        shame_susceptibility=float(rng.uniform(0.02, 0.22)),
        initial_issue_distance=float(rng.uniform(0.75, 1.5)),
        resolution_floor=float(rng.uniform(0.02, 0.10)),
        exhaustion_growth=float(rng.uniform(0.40, 0.75)),
        recovery_rate=float(rng.uniform(0.20, 0.48)),
        exhaustion_damping=float(rng.uniform(0.55, 1.05)),
        humor_effort_weight=float(rng.uniform(0.45, 0.70)),
        aggression_effort_weight=float(rng.uniform(0.30, 0.55)),
        humor_resolution_coupling=float(rng.uniform(0.15, 0.45)),
        aggression_resolution_coupling=float(rng.uniform(-0.12, 0.22)),
        natural_resolution_drift=float(rng.uniform(-0.02, 0.05)),
        resolution_noise=float(rng.uniform(0.004, 0.015)),
    )


def classify_outcome(
    final_margin: float,
    progress: float,
    aborted: bool,
) -> OutcomeClass:
    """
    Classify the encounter using cumulative influence and resolution movement.

    final_margin > 0:
        jester influence prevailed over time

    progress > 0:
        the issue moved closer to resolution
    """
    if aborted:
        return OutcomeClass.SAFETY_ABORT

    if final_margin > 0.0 and progress > 0.0:
        return OutcomeClass.PRODUCTIVE_HARLEQUIN

    if final_margin > 0.0 and progress <= 0.0:
        return OutcomeClass.PERFORMATIVE_VICTORY

    if final_margin <= 0.0 and progress > 0.0:
        return OutcomeClass.AGGRESSOR_LED_CLOSURE

    return OutcomeClass.AGGRESSIVE_FAILURE


def simulate_trial(
    config: SimulationConfig,
    rng: np.random.Generator,
    parameters: TrialParameters | None = None,
) -> TrialSeries:
    """Simulate one time-varying Harlequin encounter."""
    params = parameters or sample_trial_parameters(rng)

    time = np.linspace(0.0, config.duration, config.steps)
    dt = float(time[1] - time[0]) if time.size > 1 else config.duration

    aggression_level = bounded_wave(
        time=time,
        base=float(rng.uniform(0.42, 0.78)),
        amplitude=float(rng.uniform(0.08, 0.24)),
        frequency=float(rng.uniform(0.025, 0.080)),
        phase=float(rng.uniform(0.0, 2.0 * np.pi)),
        noise_scale=float(rng.uniform(0.015, 0.050)),
        rng=rng,
    )

    stance_rigidity = bounded_wave(
        time=time,
        base=float(rng.uniform(0.42, 0.82)),
        amplitude=float(rng.uniform(0.05, 0.18)),
        frequency=float(rng.uniform(0.015, 0.060)),
        phase=float(rng.uniform(0.0, 2.0 * np.pi)),
        noise_scale=float(rng.uniform(0.010, 0.035)),
        rng=rng,
    )

    humor_effectiveness = bounded_wave(
        time=time,
        base=float(rng.uniform(0.35, 0.72)),
        amplitude=float(rng.uniform(0.08, 0.30)),
        frequency=float(rng.uniform(0.030, 0.095)),
        phase=float(rng.uniform(0.0, 2.0 * np.pi)),
        noise_scale=float(rng.uniform(0.020, 0.060)),
        rng=rng,
    )

    uncertainty = bounded_wave(
        time=time,
        base=float(rng.uniform(0.18, 0.48)),
        amplitude=float(rng.uniform(0.05, 0.20)),
        frequency=float(rng.uniform(0.020, 0.090)),
        phase=float(rng.uniform(0.0, 2.0 * np.pi)),
        noise_scale=float(rng.uniform(0.010, 0.045)),
        rng=rng,
        upper=0.65,
    )

    pain_tolerance = bounded_wave(
        time=time,
        base=float(rng.uniform(0.05, 0.16)),
        amplitude=float(rng.uniform(0.01, 0.05)),
        frequency=float(rng.uniform(0.010, 0.050)),
        phase=float(rng.uniform(0.0, 2.0 * np.pi)),
        noise_scale=float(rng.uniform(0.003, 0.012)),
        rng=rng,
        upper=0.25,
    )

    crowd_engagement = bounded_wave(
        time=time,
        base=float(rng.uniform(0.10, 0.65)),
        amplitude=float(rng.uniform(0.05, 0.25)),
        frequency=float(rng.uniform(0.015, 0.070)),
        phase=float(rng.uniform(0.0, 2.0 * np.pi)),
        noise_scale=float(rng.uniform(0.010, 0.050)),
        rng=rng,
    )

    chaos = np.clip(
        smooth_signal(
            rng.gamma(
                shape=float(rng.uniform(1.2, 2.2)),
                scale=float(rng.uniform(0.020, 0.070)),
                size=config.steps,
            ),
            window=7,
        ),
        0.0,
        0.45,
    )

    recovery_signal = bounded_wave(
        time=time,
        base=float(rng.uniform(0.28, 0.62)),
        amplitude=float(rng.uniform(0.04, 0.18)),
        frequency=float(rng.uniform(0.010, 0.045)),
        phase=float(rng.uniform(0.0, 2.0 * np.pi)),
        noise_scale=float(rng.uniform(0.010, 0.030)),
        rng=rng,
    )

    exhaustion = np.zeros(config.steps, dtype=np.float64)
    effective_humor = np.zeros(config.steps, dtype=np.float64)
    aggressor_influence = np.zeros(config.steps, dtype=np.float64)
    jester_influence = np.zeros(config.steps, dtype=np.float64)
    margin = np.zeros(config.steps, dtype=np.float64)
    cumulative_margin = np.zeros(config.steps, dtype=np.float64)
    risk = np.zeros(config.steps, dtype=np.float64)
    issue_distance = np.zeros(config.steps, dtype=np.float64)
    resolution_velocity = np.zeros(config.steps, dtype=np.float64)

    issue_distance[0] = params.initial_issue_distance

    aborted = False
    abort_index: int | None = None

    for index in range(config.steps):
        if index > 0:
            exhaustion_rate = (
                params.exhaustion_growth
                * (
                    params.humor_effort_weight * humor_effectiveness[index - 1]
                    + params.aggression_effort_weight
                    * aggression_level[index - 1]
                    * stance_rigidity[index - 1]
                )
                - params.recovery_rate * recovery_signal[index - 1]
            )

            exhaustion[index] = max(
                0.0,
                exhaustion[index - 1] + (dt * exhaustion_rate),
            )
        else:
            exhaustion[index] = 0.0

        effective_humor[index] = (
            humor_effectiveness[index]
            * np.exp(-params.exhaustion_damping * exhaustion[index])
        )

        aggressor_influence[index] = (
            params.perceived_dominance
            * aggression_level[index]
            * stance_rigidity[index]
            * params.aggressor_size
        )

        jester_influence[index] = (
            uncertainty[index] * effective_humor[index]
            + pain_tolerance[index]
            + (
                crowd_engagement[index]
                * params.crowd_size
                * params.shame_susceptibility
                * params.jester_size
            )
            + chaos[index]
        )

        margin[index] = (
            jester_influence[index] - aggressor_influence[index]
        )

        if index > 0:
            cumulative_margin[index] = (
                cumulative_margin[index - 1]
                + 0.5 * dt * (margin[index - 1] + margin[index])
            )

        risk[index] = aggression_level[index] * params.aggressor_size

        if index > 0:
            resolution_velocity[index] = (
                params.natural_resolution_drift
                + params.humor_resolution_coupling
                * max(margin[index], 0.0)
                + params.aggression_resolution_coupling
                * max(-margin[index], 0.0)
                - 0.08 * exhaustion[index]
                - 0.12 * max(risk[index] - 0.85, 0.0)
                + float(rng.normal(0.0, params.resolution_noise))
            )

            issue_distance[index] = max(
                params.resolution_floor,
                issue_distance[index - 1]
                - (dt * resolution_velocity[index]),
            )

        if risk[index] > config.safety_threshold:
            aborted = True
            abort_index = index

            if index < config.steps - 1:
                exhaustion[index + 1 :] = exhaustion[index]
                effective_humor[index + 1 :] = effective_humor[index]
                aggressor_influence[index + 1 :] = aggressor_influence[index]
                jester_influence[index + 1 :] = jester_influence[index]
                margin[index + 1 :] = 0.0
                cumulative_margin[index + 1 :] = cumulative_margin[index]
                risk[index + 1 :] = risk[index]
                issue_distance[index + 1 :] = issue_distance[index]
                resolution_velocity[index + 1 :] = 0.0

            break

    final_margin = float(cumulative_margin[-1])
    progress = float(issue_distance[0] - issue_distance[-1])
    outcome = classify_outcome(final_margin, progress, aborted)

    return TrialSeries(
        time=time,
        aggression_level=aggression_level,
        stance_rigidity=stance_rigidity,
        humor_effectiveness=humor_effectiveness,
        effective_humor=effective_humor,
        uncertainty=uncertainty,
        pain_tolerance=pain_tolerance,
        crowd_engagement=crowd_engagement,
        chaos=chaos,
        aggressor_influence=aggressor_influence,
        jester_influence=jester_influence,
        margin=margin,
        cumulative_margin=cumulative_margin,
        exhaustion=exhaustion,
        risk=risk,
        issue_distance=issue_distance,
        resolution_velocity=resolution_velocity,
        aborted=aborted,
        abort_index=abort_index,
        outcome=outcome,
    )


def run_monte_carlo(
    config: SimulationConfig,
    rng: np.random.Generator,
) -> MonteCarloSummary:
    """Run many independent encounters."""
    final_margins = np.zeros(config.runs, dtype=np.float64)
    initial_distances = np.zeros(config.runs, dtype=np.float64)
    final_distances = np.zeros(config.runs, dtype=np.float64)
    progress = np.zeros(config.runs, dtype=np.float64)
    aborted = np.zeros(config.runs, dtype=np.bool_)
    outcomes: list[OutcomeClass] = []

    for index in range(config.runs):
        series = simulate_trial(config, rng)

        final_margins[index] = series.cumulative_margin[-1]
        initial_distances[index] = series.issue_distance[0]
        final_distances[index] = series.issue_distance[-1]
        progress[index] = (
            series.issue_distance[0] - series.issue_distance[-1]
        )
        aborted[index] = series.aborted
        outcomes.append(series.outcome)

    return MonteCarloSummary(
        final_margins=final_margins,
        initial_distances=initial_distances,
        final_distances=final_distances,
        progress=progress,
        aborted=aborted,
        outcomes=tuple(outcomes),
    )


def save_figure(
    path: Path,
    config: SimulationConfig,
) -> None:
    """Save and optionally display the active Matplotlib figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=config.dpi, bbox_inches="tight")

    if config.show:
        plt.show()

    plt.close()


def plot_time_integrated_dynamics(
    series: TrialSeries,
    config: SimulationConfig,
) -> Path:
    """Plot aggressor and jester influence over time."""
    output_path = config.output_dir / FIGURE_NAMES["dynamics"]

    plt.figure(figsize=(11, 6))
    plt.plot(
        series.time,
        series.aggressor_influence,
        linewidth=1.8,
        label=r"Aggressor influence $\Phi_A(t)$",
    )
    plt.plot(
        series.time,
        series.jester_influence,
        linewidth=1.8,
        label=r"Jester influence $\Phi_J(t)$",
    )
    plt.plot(
        series.time,
        series.effective_humor,
        linewidth=1.2,
        alpha=0.8,
        label=r"Effective humor $H^\star_{\mathrm{effect}}(t)$",
    )

    if series.aborted and series.abort_index is not None:
        abort_time = series.time[series.abort_index]
        plt.axvline(
            abort_time,
            linestyle="--",
            linewidth=1.4,
            label="Safety abort",
        )

    plt.xlabel("Time")
    plt.ylabel("Normalized influence")
    plt.title("Time-Integrated Harlequin Dynamics")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_figure(output_path, config)
    return output_path


def plot_cumulative_margin(
    series: TrialSeries,
    config: SimulationConfig,
) -> Path:
    """Plot the accumulated Harlequin margin."""
    output_path = config.output_dir / FIGURE_NAMES["margin"]

    plt.figure(figsize=(11, 6))
    plt.plot(
        series.time,
        series.cumulative_margin,
        linewidth=2.0,
        label=r"$D(t)=\int_0^t(\Phi_J-\Phi_A)\,d\tau$",
    )
    plt.axhline(
        0.0,
        linestyle="--",
        linewidth=1.2,
        label="Neutral cumulative margin",
    )

    plt.xlabel("Time")
    plt.ylabel("Cumulative Harlequin margin")
    plt.title("Cumulative Influence Margin")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_figure(output_path, config)
    return output_path


def plot_resolution_distance(
    series: TrialSeries,
    config: SimulationConfig,
) -> Path:
    """Plot distance to resolution and Harlequin exhaustion."""
    output_path = config.output_dir / FIGURE_NAMES["distance"]

    fig, primary_axis = plt.subplots(figsize=(11, 6))

    primary_axis.plot(
        series.time,
        series.issue_distance,
        linewidth=2.0,
        label=r"Distance to resolution $d_R(t)$",
    )
    primary_axis.set_xlabel("Time")
    primary_axis.set_ylabel("Distance to resolution")
    primary_axis.grid(True, alpha=0.3)

    secondary_axis = primary_axis.twinx()
    secondary_axis.plot(
        series.time,
        series.exhaustion,
        linewidth=1.5,
        linestyle="--",
        label=r"Harlequin exhaustion $E_{\mathrm{exh}}(t)$",
    )
    secondary_axis.set_ylabel("Exhaustion")

    primary_lines, primary_labels = primary_axis.get_legend_handles_labels()
    secondary_lines, secondary_labels = (
        secondary_axis.get_legend_handles_labels()
    )
    primary_axis.legend(
        primary_lines + secondary_lines,
        primary_labels + secondary_labels,
        loc="best",
    )

    plt.title("Resolution Distance and Harlequin Exhaustion")
    save_figure(output_path, config)
    return output_path


def plot_outcome_distribution(
    summary: MonteCarloSummary,
    config: SimulationConfig,
) -> Path:
    """Plot the distribution of final cumulative margins."""
    output_path = config.output_dir / FIGURE_NAMES["distribution"]

    plt.figure(figsize=(10, 6))
    plt.hist(
        summary.final_margins,
        bins=40,
        alpha=0.75,
        edgecolor="black",
    )
    plt.axvline(
        0.0,
        linestyle="--",
        linewidth=1.5,
        label="Neutral margin",
    )
    plt.xlabel("Final integrated Harlequin margin")
    plt.ylabel("Monte Carlo run count")
    plt.title("Distribution of Final Integrated Margins")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)

    save_figure(output_path, config)
    return output_path


def plot_outcome_classes(
    summary: MonteCarloSummary,
    config: SimulationConfig,
) -> Path:
    """Plot counts for the five outcome classes."""
    output_path = config.output_dir / FIGURE_NAMES["outcome_classes"]

    ordered_classes = [
        OutcomeClass.PRODUCTIVE_HARLEQUIN,
        OutcomeClass.PERFORMATIVE_VICTORY,
        OutcomeClass.AGGRESSOR_LED_CLOSURE,
        OutcomeClass.AGGRESSIVE_FAILURE,
        OutcomeClass.SAFETY_ABORT,
    ]

    labels = [
        "Productive\nHarlequin",
        "Performative\nvictory",
        "Aggressor-led\nclosure",
        "Aggressive\nfailure",
        "Safety\nabort",
    ]

    counts = [
        sum(outcome is outcome_class for outcome in summary.outcomes)
        for outcome_class in ordered_classes
    ]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts)
    plt.ylabel("Monte Carlo run count")
    plt.title("Influence and Resolution Outcome Classes")
    plt.grid(True, axis="y", alpha=0.3)

    for index, count in enumerate(counts):
        plt.text(
            index,
            count,
            str(count),
            ha="center",
            va="bottom",
        )

    save_figure(output_path, config)
    return output_path


def print_single_trial_summary(
    series: TrialSeries,
) -> None:
    """Print summary values for the representative encounter."""
    initial_distance = float(series.issue_distance[0])
    final_distance = float(series.issue_distance[-1])
    progress = initial_distance - final_distance

    print("\nRepresentative encounter")
    print("=" * 24)
    print(f"Outcome:                 {series.outcome.value}")
    print(f"Safety abort:            {series.aborted}")
    print(
        "Final cumulative margin:"
        f" {series.cumulative_margin[-1]:.6f}"
    )
    print(f"Initial issue distance:  {initial_distance:.6f}")
    print(f"Final issue distance:    {final_distance:.6f}")
    print(f"Resolution progress:     {progress:.6f}")
    print(f"Final exhaustion:        {series.exhaustion[-1]:.6f}")


def print_monte_carlo_summary(
    summary: MonteCarloSummary,
    config: SimulationConfig,
    generated_paths: list[Path],
) -> None:
    """Print transparent Monte Carlo results."""
    print("\nMonte Carlo summary")
    print("=" * 19)
    print(f"Random seed:             {config.seed}")
    print(f"Runs:                    {config.runs}")
    print(f"Time steps per run:      {config.steps}")
    print(f"Duration:                {config.duration:.2f}")
    print(f"Safety threshold:        {config.safety_threshold:.3f}")
    print(
        "Mean final margin:      "
        f"{summary.final_margins.mean():.6f}"
    )
    print(
        "Median final margin:    "
        f"{np.median(summary.final_margins):.6f}"
    )
    print(
        "Mean resolution progress:"
        f" {summary.progress.mean():.6f}"
    )
    print(
        "Safety aborts:          "
        f"{np.count_nonzero(summary.aborted)} "
        f"({np.mean(summary.aborted):.2%})"
    )

    print("\nOutcome classes:")
    for outcome_class in OutcomeClass:
        count = sum(
            outcome is outcome_class
            for outcome in summary.outcomes
        )
        print(
            f"  {outcome_class.value:24s}"
            f" {count:5d} ({count / config.runs:.2%})"
        )

    print("\nGenerated figures:")
    for path in generated_paths:
        print(f"  - {path}")

    print(
        "\nInterpretation warning: these results describe this synthetic "
        "model only. They are not empirical rates or safety advice."
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line controls."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the time-integrated Harlequin Paradox simulation."
        )
    )
    parser.add_argument(
        "--duration",
        type=positive_float,
        default=30.0,
        help="simulated encounter duration (default: 30)",
    )
    parser.add_argument(
        "--steps",
        type=positive_int,
        default=600,
        help="time steps per encounter (default: 600)",
    )
    parser.add_argument(
        "--runs",
        type=positive_int,
        default=1_000,
        help="Monte Carlo runs (default: 1000)",
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
        help="figure output directory (default: figures)",
    )
    parser.add_argument(
        "--dpi",
        type=positive_int,
        default=180,
        help="image resolution (default: 180)",
    )
    parser.add_argument(
        "--safety-threshold",
        type=positive_float,
        default=1.25,
        help="risk threshold that triggers abort (default: 1.25)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="display figures interactively",
    )
    return parser.parse_args()


def main() -> int:
    """Program entry point."""
    args = parse_args()

    config = SimulationConfig(
        duration=args.duration,
        steps=args.steps,
        runs=args.runs,
        seed=args.seed,
        output_dir=args.output_dir,
        dpi=args.dpi,
        show=args.show,
        safety_threshold=args.safety_threshold,
    )

    rng = np.random.default_rng(config.seed)

    representative = simulate_trial(config, rng)
    monte_carlo = run_monte_carlo(config, rng)

    generated_paths = [
        plot_time_integrated_dynamics(representative, config),
        plot_cumulative_margin(representative, config),
        plot_resolution_distance(representative, config),
        plot_outcome_distribution(monte_carlo, config),
        plot_outcome_classes(monte_carlo, config),
    ]

    print_single_trial_summary(representative)
    print_monte_carlo_summary(
        monte_carlo,
        config,
        generated_paths,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
