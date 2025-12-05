"""Experiment 4: Self-healing forecast pipeline.

Generates perturbed forecasts, runs the critic→diagnose→correct→verify
loop, and measures how much SMAPE improves after correction.
"""
from __future__ import annotations

import json
import logging

import numpy as np
from tqdm import tqdm

from forecast_critic.config import Config, PerturbationType
from forecast_critic.data.perturbations import (
    PerturbedSample,
    generate_perturbed_dataset,
    make_unperturbed_samples,
    smape,
)
from forecast_critic.data.synthetic import generate_dataset
from forecast_critic.surgeon.pipeline import SurgeonResult, heal_forecast

logger = logging.getLogger(__name__)


def run_surgeon_experiment(config: Config) -> dict:
    """Run the self-healing experiment.

    For each perturbation type:
    1. Generate perturbed forecasts
    2. Run the surgeon pipeline on each
    3. Measure: how many were healed, SMAPE improvement, iteration stats
    """
    rng = np.random.default_rng(config.experiment.seed)
    results: dict = {"per_perturbation": [], "aggregate": {}}

    all_original_smapes: list[float] = []
    all_corrected_smapes: list[float] = []
    all_healed = 0
    all_total = 0

    for pert in PerturbationType:
        logger.info("=" * 50)
        logger.info("Surgeon experiment: %s", pert.value)
        logger.info("=" * 50)

        # Generate perturbed samples
        ts_list = generate_dataset(
            config.experiment.n_generate_per_type,
            config.synthetic,
            seed=rng.integers(0, 2**31),
        )
        perturbed_samples = generate_perturbed_dataset(
            ts_list, pert, config.perturbation, config.synthetic,
            config.experiment.n_perturbed, rng,
        )

        # Run surgeon on each
        surgeon_results: list[SurgeonResult] = []
        for sample in tqdm(perturbed_samples, desc=f"Healing {pert.value}", unit="ts"):
            result = heal_forecast(sample, config)
            surgeon_results.append(result)

        # Collect metrics
        healed_count = sum(1 for r in surgeon_results if r.final_verdict == 1)
        original_smapes = [r.original_smape for r in surgeon_results]
        corrected_smapes = [r.corrected_smape for r in surgeon_results]
        avg_iterations = np.mean([r.n_iterations for r in surgeon_results])

        # Only count improvement on forecasts that were actually corrected
        corrected_results = [r for r in surgeon_results if r.was_corrected]
        if corrected_results:
            smape_improvements = [
                r.original_smape - r.corrected_smape
                for r in corrected_results
            ]
            avg_improvement = float(np.mean(smape_improvements))
            pct_improved = sum(1 for x in smape_improvements if x > 0) / len(smape_improvements)
        else:
            avg_improvement = 0.0
            pct_improved = 0.0

        # Method breakdown
        methods_used: dict[str, int] = {}
        for r in surgeon_results:
            for step in r.steps:
                m = step.method
                methods_used[m] = methods_used.get(m, 0) + 1

        pert_result = {
            "perturbation": pert.value,
            "n_samples": len(perturbed_samples),
            "n_healed": healed_count,
            "heal_rate": healed_count / len(perturbed_samples) if perturbed_samples else 0,
            "avg_original_smape": float(np.mean(original_smapes)),
            "avg_corrected_smape": float(np.mean(corrected_smapes)),
            "avg_smape_improvement": avg_improvement,
            "pct_improved": pct_improved,
            "avg_iterations": float(avg_iterations),
            "methods_used": methods_used,
        }
        results["per_perturbation"].append(pert_result)

        all_original_smapes.extend(original_smapes)
        all_corrected_smapes.extend(corrected_smapes)
        all_healed += healed_count
        all_total += len(perturbed_samples)

        logger.info(
            "%s: healed %d/%d (%.1f%%), SMAPE %.4f → %.4f (avg improvement: %.4f)",
            pert.value, healed_count, len(perturbed_samples),
            100 * pert_result["heal_rate"],
            pert_result["avg_original_smape"],
            pert_result["avg_corrected_smape"],
            avg_improvement,
        )

    # Aggregate
    results["aggregate"] = {
        "total_samples": all_total,
        "total_healed": all_healed,
        "overall_heal_rate": all_healed / all_total if all_total > 0 else 0,
        "avg_original_smape": float(np.mean(all_original_smapes)),
        "avg_corrected_smape": float(np.mean(all_corrected_smapes)),
    }

    # Save
    output_dir = config.output_dir / "surgeon"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"
    output_path.write_text(json.dumps(results, indent=2))
    logger.info("Results saved to %s", output_path)

    # Print summary
    print("\n" + "=" * 70)
    print("SURGEON EXPERIMENT RESULTS (Self-Healing Forecasts)")
    print("=" * 70)
    print(f"  {'Perturbation':<25s} {'Healed':>8s} {'Rate':>8s} {'SMAPE Before':>14s} {'SMAPE After':>13s} {'Improvement':>13s}")
    print(f"  {'-'*82}")
    for r in results["per_perturbation"]:
        print(
            f"  {r['perturbation']:<25s} "
            f"{r['n_healed']:>5d}/{r['n_samples']:<3d}"
            f"{r['heal_rate']:>7.1%} "
            f"{r['avg_original_smape']:>13.4f} "
            f"{r['avg_corrected_smape']:>13.4f} "
            f"{r['avg_smape_improvement']:>+12.4f}"
        )
    agg = results["aggregate"]
    print(f"  {'-'*82}")
    print(
        f"  {'TOTAL':<25s} "
        f"{agg['total_healed']:>5d}/{agg['total_samples']:<3d}"
        f"{agg['overall_heal_rate']:>7.1%} "
        f"{agg['avg_original_smape']:>13.4f} "
        f"{agg['avg_corrected_smape']:>13.4f}"
    )
    print("=" * 70)

    return results

