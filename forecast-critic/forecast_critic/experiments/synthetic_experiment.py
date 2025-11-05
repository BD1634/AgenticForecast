from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from forecast_critic.config import Config, PerturbationType
from forecast_critic.critic.llm import CriticResponse, ForecastCritic
from forecast_critic.data.perturbations import (
    PerturbedSample,
    generate_perturbed_dataset,
    make_unperturbed_samples,
)
from forecast_critic.data.synthetic import generate_dataset
from forecast_critic.metrics.evaluation import (
    classification_report,
    per_class_f1,
    weighted_f1,
)
from forecast_critic.prompts.templates import SYNTHETIC_PROMPT
from forecast_critic.visualization.plots import render_synthetic_plot

logger = logging.getLogger(__name__)


def _evaluate_samples(
    samples: list[PerturbedSample],
    critic: ForecastCritic,
    desc: str = "",
) -> list[CriticResponse]:
    """Render plots and evaluate via LLM critic."""
    items: list[tuple[bytes, str]] = []
    for s in tqdm(samples, desc=f"Rendering {desc}", unit="plot"):
        img = render_synthetic_plot(s)
        items.append((img, SYNTHETIC_PROMPT))

    logger.info("Evaluating %d samples via LLM...", len(items))
    return critic.evaluate_batch(items)


def run_single_perturbation(
    perturbation: PerturbationType,
    config: Config,
    critic: ForecastCritic,
    rng: np.random.Generator,
) -> dict:
    """Run experiment for a single perturbation type."""
    logger.info("Running perturbation: %s", perturbation.value)

    # Generate time series for perturbed examples
    perturbed_ts = generate_dataset(
        config.experiment.n_generate_per_type,
        config.synthetic,
        seed=rng.integers(0, 2**31),
    )
    perturbed_samples = generate_perturbed_dataset(
        perturbed_ts,
        perturbation,
        config.perturbation,
        config.synthetic,
        config.experiment.n_perturbed,
        rng,
    )

    # Generate unperturbed examples
    unperturbed_ts = generate_dataset(
        config.experiment.n_unperturbed,
        config.synthetic,
        seed=rng.integers(0, 2**31),
    )
    unperturbed_samples = make_unperturbed_samples(unperturbed_ts)

    # Combine and shuffle
    all_samples = perturbed_samples + unperturbed_samples
    indices = rng.permutation(len(all_samples))
    all_samples = [all_samples[i] for i in indices]

    # Ground truth: perturbed=2 (unreasonable), unperturbed=1 (reasonable)
    y_true = [2 if s.is_perturbed else 1 for s in all_samples]

    # Evaluate
    responses = _evaluate_samples(all_samples, critic, perturbation.value)
    y_pred = [r.label for r in responses]

    # Metrics
    wf1 = weighted_f1(y_true, y_pred)
    cf1 = per_class_f1(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    logger.info("Perturbation: %s | Weighted F1: %.3f", perturbation.value, wf1)
    logger.info("\n%s", report)

    return {
        "perturbation": perturbation.value,
        "n_samples": len(all_samples),
        "weighted_f1": wf1,
        "per_class_f1": cf1,
        "report": report,
        "predictions": [
            {
                "true_label": yt,
                "pred_label": yp,
                "explanation": r.explanation,
            }
            for yt, yp, r in zip(y_true, y_pred, responses)
        ],
    }


def run_mixture(
    config: Config,
    critic: ForecastCritic,
    rng: np.random.Generator,
) -> dict:
    """Run the mixture evaluation (any perturbation may be applied)."""
    logger.info("Running mixture evaluation")

    all_perturbed: list[PerturbedSample] = []
    per_type = config.experiment.n_perturbed // len(PerturbationType)
    generate_per_type = int(per_type / config.experiment.smape_filter_pct) + 1

    for pert in PerturbationType:
        ts_list = generate_dataset(
            generate_per_type,
            config.synthetic,
            seed=rng.integers(0, 2**31),
        )
        samples = generate_perturbed_dataset(
            ts_list, pert, config.perturbation, config.synthetic, per_type, rng,
        )
        all_perturbed.extend(samples)

    # Trim to exact count
    all_perturbed = all_perturbed[: config.experiment.n_perturbed]

    unperturbed_ts = generate_dataset(
        config.experiment.n_unperturbed,
        config.synthetic,
        seed=rng.integers(0, 2**31),
    )
    unperturbed_samples = make_unperturbed_samples(unperturbed_ts)

    all_samples = all_perturbed + unperturbed_samples
    indices = rng.permutation(len(all_samples))
    all_samples = [all_samples[i] for i in indices]

    y_true = [2 if s.is_perturbed else 1 for s in all_samples]
    responses = _evaluate_samples(all_samples, critic, "mixture")
    y_pred = [r.label for r in responses]

    wf1 = weighted_f1(y_true, y_pred)
    cf1 = per_class_f1(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    logger.info("Mixture | Weighted F1: %.3f", wf1)
    logger.info("\n%s", report)

    return {
        "perturbation": "mixture",
        "n_samples": len(all_samples),
        "weighted_f1": wf1,
        "per_class_f1": cf1,
        "report": report,
    }


def run_synthetic_experiment(config: Config) -> dict:
    """Run the full synthetic perturbation experiment (Experiment 1)."""
    rng = np.random.default_rng(config.experiment.seed)
    critic = ForecastCritic(config.critic)

    results: dict = {"per_perturbation": [], "mixture": None}

    # Per-perturbation evaluations
    for pert in PerturbationType:
        result = run_single_perturbation(pert, config, critic, rng)
        results["per_perturbation"].append(result)

    # Mixture evaluation
    results["mixture"] = run_mixture(config, critic, rng)

    # Save results
    output_dir = config.output_dir / "synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"

    # Serialize (skip non-serializable fields)
    serializable = {
        "per_perturbation": [
            {k: v for k, v in r.items() if k != "predictions"}
            for r in results["per_perturbation"]
        ],
        "mixture": {
            k: v
            for k, v in results["mixture"].items()
            if k != "predictions"
        },
    }
    output_path.write_text(json.dumps(serializable, indent=2))
    logger.info("Results saved to %s", output_path)

    # Print summary
    print("\n" + "=" * 60)
    print("SYNTHETIC EXPERIMENT RESULTS")
    print("=" * 60)
    for r in results["per_perturbation"]:
        print(f"  {r['perturbation']:25s} | F1: {r['weighted_f1']:.3f}")
    print(f"  {'mixture':25s} | F1: {results['mixture']['weighted_f1']:.3f}")
    print("=" * 60)

    return results

