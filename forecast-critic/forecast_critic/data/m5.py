"""Experiment 5: Committee of foundation models.

Runs multiple forecasters on M5 time series, uses the critic to
select/blend, and compares sCRPS of committee output vs individual models.
"""
from __future__ import annotations

import json
import logging

import numpy as np
from tqdm import tqdm

from forecast_critic.config import Config
from forecast_critic.committee.forecasters import build_committee, ForecastResult
from forecast_critic.committee.pipeline import run_committee, CommitteeResult
from forecast_critic.data.m5 import M5TimeSeries, prepare_m5_time_series
from forecast_critic.metrics.evaluation import scrps

logger = logging.getLogger(__name__)


def _compute_point_scrps(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute sCRPS for a point forecast (treated as a degenerate distribution)."""
    # For point forecasts, use a single quantile at 0.5
    quantile_forecasts = y_pred.reshape(-1, 1)
    quantiles = np.array([0.5])
    return scrps(y_true, quantile_forecasts, quantiles)


def run_committee_experiment(config: Config) -> dict:
    """Run the committee experiment on M5 data."""
    # Load M5 time series
    logger.info("Loading M5 dataset...")
    time_series = prepare_m5_time_series(config.m5, seed=config.experiment.seed)

    # Build committee
    logger.info("Building committee: %s", config.committee.forecasters)
    forecasters = build_committee(
        config.committee.forecasters,
        config.committee,
        config.committee.device,
    )
    model_names = [f.name for f in forecasters]
    logger.info("Active committee members: %s", model_names)

    # Run on each time series
    individual_scrps: dict[str, list[float]] = {name: [] for name in model_names}
    committee_scrps: list[float] = []
    committee_weights: list[dict[str, float]] = []

    for ts in tqdm(time_series, desc="Committee evaluation", unit="ts"):
        try:
            result = run_committee(
                history=ts.history,
                horizon=config.m5.forecast_days,
                config=config,
                forecasters=forecasters,
            )

            # Committee sCRPS
            c_scrps = _compute_point_scrps(ts.future, result.selection.final_forecast)
            committee_scrps.append(c_scrps)
            committee_weights.append(result.selection.model_weights)

            # Individual model sCRPS
            for fc_result in result.individual_forecasts:
                s = _compute_point_scrps(ts.future, fc_result.forecast)
                individual_scrps[fc_result.model_name].append(s)

        except Exception as e:
            logger.warning("Failed on %s: %s", ts.item_id, e)

    # Aggregate results
    results: dict = {
        "strategy": config.committee.strategy.value,
        "n_samples": len(committee_scrps),
        "models": model_names,
        "committee": {
            "median_scrps": float(np.median(committee_scrps)) if committee_scrps else 0,
            "mean_scrps": float(np.mean(committee_scrps)) if committee_scrps else 0,
            "std_scrps": float(np.std(committee_scrps)) if committee_scrps else 0,
        },
        "individual": {},
        "avg_weights": {},
    }

    for name in model_names:
        scores = individual_scrps[name]
        if scores:
            results["individual"][name] = {
                "median_scrps": float(np.median(scores)),
                "mean_scrps": float(np.mean(scores)),
                "std_scrps": float(np.std(scores)),
            }

    # Average weights across all samples
    if committee_weights:
        for name in model_names:
            w = [cw.get(name, 0.0) for cw in committee_weights]
            results["avg_weights"][name] = float(np.mean(w))

    # Compute improvement over individual models
    results["improvement"] = {}
    if committee_scrps:
        committee_mean = results["committee"]["mean_scrps"]
        for name, stats in results["individual"].items():
            ind_mean = stats["mean_scrps"]
            if ind_mean > 0:
                pct = (ind_mean - committee_mean) / ind_mean * 100
                results["improvement"][name] = round(pct, 2)

    # Save
    output_dir = config.output_dir / "committee"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"
    output_path.write_text(json.dumps(results, indent=2))
    logger.info("Results saved to %s", output_path)

    # Print summary
    print("\n" + "=" * 70)
    print(f"COMMITTEE EXPERIMENT RESULTS (Strategy: {config.committee.strategy.value})")
    print("=" * 70)
    print(f"\n  {'Model':<20s} {'Median sCRPS':>14s} {'Mean sCRPS':>12s} {'Std':>10s}")
    print(f"  {'-'*57}")
    for name, stats in results["individual"].items():
        print(f"  {name:<20s} {stats['median_scrps']:>14.4f} {stats['mean_scrps']:>12.4f} {stats['std_scrps']:>10.4f}")

    c = results["committee"]
    print(f"  {'-'*57}")
    print(f"  {'COMMITTEE':<20s} {c['median_scrps']:>14.4f} {c['mean_scrps']:>12.4f} {c['std_scrps']:>10.4f}")

    if results["improvement"]:
        print(f"\n  Improvement over individual models:")
        for name, pct in results["improvement"].items():
            symbol = "+" if pct > 0 else ""
            print(f"    vs {name:<15s}: {symbol}{pct:.1f}%")

    if results["avg_weights"]:
        print(f"\n  Average blend weights:")
        for name, w in results["avg_weights"].items():
            print(f"    {name:<15s}: {w:.3f}")

    print("=" * 70)

    return results

