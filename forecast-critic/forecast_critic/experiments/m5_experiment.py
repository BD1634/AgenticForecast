from __future__ import annotations

import json
import logging

import numpy as np
from tqdm import tqdm

from forecast_critic.config import Config
from forecast_critic.critic.llm import CriticResponse, ForecastCritic
from forecast_critic.data.m5 import M5Forecast, prepare_m5_time_series, run_chronos_forecasts
from forecast_critic.metrics.evaluation import mann_whitney_test, scrps
from forecast_critic.prompts.templates import M5_PROMPT
from forecast_critic.visualization.plots import render_m5_plot

logger = logging.getLogger(__name__)


def _compute_scrps_for_forecast(forecast: M5Forecast) -> float:
    """Compute sCRPS for a single M5 forecast."""
    quantiles = np.array(sorted(forecast.quantiles.keys()))
    quantile_values = np.column_stack(
        [forecast.quantiles[q] for q in quantiles]
    )  # shape: (T, n_quantiles)
    return scrps(forecast.future_actual, quantile_values, quantiles)


def run_m5_experiment(config: Config) -> dict:
    """Run the M5 real-world experiment (Experiment 3)."""
    critic = ForecastCritic(config.critic)

    # Load data
    logger.info("Loading M5 dataset...")
    time_series = prepare_m5_time_series(config.m5, seed=config.experiment.seed)

    # Generate Chronos forecasts
    logger.info("Running Chronos forecasts...")
    forecasts = run_chronos_forecasts(time_series, config.m5)

    # Render plots and evaluate
    items: list[tuple[bytes, str]] = []
    for f in tqdm(forecasts, desc="Rendering M5 plots", unit="plot"):
        img = render_m5_plot(f, show_future=False)
        items.append((img, M5_PROMPT))

    logger.info("Evaluating %d M5 forecasts via LLM...", len(items))
    responses = critic.evaluate_batch(items)

    # Compute sCRPS for each forecast
    scrps_scores: list[float] = []
    for f in tqdm(forecasts, desc="Computing sCRPS", unit="ts"):
        scrps_scores.append(_compute_scrps_for_forecast(f))

    # Split into reasonable / unreasonable
    reasonable_scrps: list[float] = []
    unreasonable_scrps: list[float] = []
    reasonable_count = 0
    unreasonable_count = 0

    for score, resp in zip(scrps_scores, responses):
        if resp.label == 1:
            reasonable_scrps.append(score)
            reasonable_count += 1
        elif resp.label == 2:
            unreasonable_scrps.append(score)
            unreasonable_count += 1

    total = reasonable_count + unreasonable_count
    logger.info(
        "Reasonable: %d (%.1f%%), Unreasonable: %d (%.1f%%)",
        reasonable_count,
        100 * reasonable_count / total if total > 0 else 0,
        unreasonable_count,
        100 * unreasonable_count / total if total > 0 else 0,
    )

    # Mann-Whitney U test
    results: dict = {
        "n_total": total,
        "n_reasonable": reasonable_count,
        "n_unreasonable": unreasonable_count,
        "pct_reasonable": 100 * reasonable_count / total if total > 0 else 0,
    }

    if len(reasonable_scrps) > 0 and len(unreasonable_scrps) > 0:
        mw = mann_whitney_test(
            np.array(reasonable_scrps),
            np.array(unreasonable_scrps),
        )
        results["mann_whitney"] = {
            "u_stat": mw.u_stat,
            "p_value": mw.p_value,
            "reasonable_median": mw.reasonable_median,
            "reasonable_mean": mw.reasonable_mean,
            "reasonable_std": mw.reasonable_std,
            "unreasonable_median": mw.unreasonable_median,
            "unreasonable_mean": mw.unreasonable_mean,
            "unreasonable_std": mw.unreasonable_std,
            "median_pct_diff": mw.median_pct_diff,
            "mean_pct_diff": mw.mean_pct_diff,
        }
    else:
        results["mann_whitney"] = None
        logger.warning("Cannot compute Mann-Whitney: one group is empty")

    # Per-forecast details
    results["forecasts"] = [
        {
            "item_id": f.item_id,
            "scrps": s,
            "label": r.label,
            "explanation": r.explanation,
        }
        for f, s, r in zip(forecasts, scrps_scores, responses)
    ]

    # Save
    output_dir = config.output_dir / "m5"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"
    output_path.write_text(json.dumps(
        {k: v for k, v in results.items() if k != "forecasts"},
        indent=2,
    ))
    logger.info("Results saved to %s", output_path)

    # Print summary (Table 1 style)
    print("\n" + "=" * 60)
    print("M5 EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"  Total forecasts:    {total}")
    print(f"  Reasonable:         {reasonable_count} ({results['pct_reasonable']:.1f}%)")
    print(f"  Unreasonable:       {unreasonable_count} ({100 - results['pct_reasonable']:.1f}%)")

    if results["mann_whitney"]:
        mw_data = results["mann_whitney"]
        print(f"\n  {'Metric':<30s} {'Reasonable':>12s} {'Unreasonable':>12s}")
        print(f"  {'-'*55}")
        print(f"  {'Median sCRPS':<30s} {mw_data['reasonable_median']:>12.4f} {mw_data['unreasonable_median']:>12.4f}")
        print(f"  {'Mean sCRPS':<30s} {mw_data['reasonable_mean']:>12.4f} {mw_data['unreasonable_mean']:>12.4f}")
        print(f"  {'Std sCRPS':<30s} {mw_data['reasonable_std']:>12.4f} {mw_data['unreasonable_std']:>12.4f}")
        print(f"\n  Median % Difference: {mw_data['median_pct_diff']:.1f}%")
        print(f"  Mean % Difference:   {mw_data['mean_pct_diff']:.1f}%")
        print(f"  Mann-Whitney U:      {mw_data['u_stat']:.0f}")
        print(f"  p-value:             {mw_data['p_value']:.2e}")
    print("=" * 60)

    return results

