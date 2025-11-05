from __future__ import annotations

import json
import logging

import numpy as np
from tqdm import tqdm

from forecast_critic.config import Config, PromotionalScenario
from forecast_critic.critic.llm import CriticResponse, ForecastCritic
from forecast_critic.data.promotions import PromotionalSample, generate_promotional_dataset
from forecast_critic.metrics.evaluation import (
    classification_report,
    per_class_f1,
    weighted_f1,
)
from forecast_critic.prompts.templates import build_promotional_prompt
from forecast_critic.visualization.plots import render_promotional_plot

logger = logging.getLogger(__name__)

# Ground truth labels per scenario
SCENARIO_LABELS = {
    PromotionalScenario.A: 1,  # reasonable
    PromotionalScenario.B: 2,  # unreasonable
    PromotionalScenario.C: 2,  # unreasonable
    PromotionalScenario.D: 1,  # reasonable
}


def _evaluate_scenario(
    samples: list[PromotionalSample],
    critic: ForecastCritic,
    scenario_name: str,
) -> list[CriticResponse]:
    """Render promotional plots and evaluate via LLM."""
    items: list[tuple[bytes, str]] = []
    for s in tqdm(samples, desc=f"Rendering {scenario_name}", unit="plot"):
        img = render_promotional_plot(s)
        prompt = build_promotional_prompt(s.holiday_hist_time, s.holiday_forecast_time)
        items.append((img, prompt))

    logger.info("Evaluating %d samples for scenario %s...", len(items), scenario_name)
    return critic.evaluate_batch(items)


def run_exogenous_experiment(config: Config) -> dict:
    """Run the exogenous feature injection experiment (Experiment 2)."""
    critic = ForecastCritic(config.critic)
    results: dict = {"scenarios": []}

    all_y_true: list[int] = []
    all_y_pred: list[int] = []

    for scenario in PromotionalScenario:
        logger.info("Running scenario: %s", scenario.value)

        samples = generate_promotional_dataset(
            scenario=scenario,
            n_samples=config.experiment.n_promo_per_scenario,
            synth_config=config.synthetic,
            promo_config=config.promotion,
            seed=config.experiment.seed + hash(scenario.value) % 1000,
        )

        responses = _evaluate_scenario(samples, critic, scenario.value)

        y_true = [s.label for s in samples]
        y_pred = [r.label for r in responses]

        wf1 = weighted_f1(y_true, y_pred)
        accuracy = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / len(y_true)

        logger.info("Scenario %s | F1: %.3f | Accuracy: %.3f", scenario.value, wf1, accuracy)

        results["scenarios"].append({
            "scenario": scenario.value,
            "expected_label": SCENARIO_LABELS[scenario],
            "n_samples": len(samples),
            "weighted_f1": wf1,
            "accuracy": accuracy,
            "per_class_f1": per_class_f1(y_true, y_pred),
            "report": classification_report(y_true, y_pred),
        })

        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

    # Overall metrics
    overall_f1 = weighted_f1(all_y_true, all_y_pred)
    results["overall_weighted_f1"] = overall_f1
    results["overall_report"] = classification_report(all_y_true, all_y_pred)

    # Save
    output_dir = config.output_dir / "exogenous"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"

    serializable = {
        "scenarios": [
            {k: v for k, v in s.items() if k != "predictions"}
            for s in results["scenarios"]
        ],
        "overall_weighted_f1": results["overall_weighted_f1"],
    }
    output_path.write_text(json.dumps(serializable, indent=2))
    logger.info("Results saved to %s", output_path)

    # Print summary
    print("\n" + "=" * 60)
    print("EXOGENOUS FEATURE INJECTION RESULTS")
    print("=" * 60)
    for s in results["scenarios"]:
        label_str = "reasonable" if s["expected_label"] == 1 else "unreasonable"
        print(
            f"  Scenario {s['scenario']:25s} ({label_str:12s}) | "
            f"F1: {s['weighted_f1']:.3f} | Acc: {s['accuracy']:.3f}"
        )
    print(f"  {'Overall':25s} {'':14s} | F1: {overall_f1:.3f}")
    print("=" * 60)

    return results

