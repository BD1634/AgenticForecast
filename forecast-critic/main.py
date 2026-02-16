"""The Forecast Critic - CLI entry point.

Usage:
    python main.py --experiment synthetic     # Experiment 1: perturbation detection
    python main.py --experiment exogenous     # Experiment 2: promotional context
    python main.py --experiment m5            # Experiment 3: M5 real-world
    python main.py --experiment surgeon       # Experiment 4: self-healing forecasts
    python main.py --experiment committee     # Experiment 5: committee of foundation models
    python main.py --experiment all           # Run all experiments (1-3)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from forecast_critic.config import BlendStrategy, Config, CriticConfig, ExperimentConfig, M5Config
from forecast_critic.llm_provider import DEFAULT_MODELS, get_default_model


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="The Forecast Critic: LLM-based forecast monitoring",
    )
    parser.add_argument(
        "--experiment",
        choices=["synthetic", "exogenous", "m5", "surgeon", "committee", "all"],
        required=True,
        help="Which experiment to run",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "gemini", "anthropic"],
        default="ollama",
        help="LLM provider: ollama (local/free), gemini (free API), anthropic (paid). Default: ollama",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID (default: auto per provider — llama3.2-vision / gemini-2.0-flash / claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Override number of samples (for quick testing)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max concurrent API calls (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory for results (default: outputs/)",
    )
    parser.add_argument(
        "--m5-data-dir",
        type=Path,
        default=Path("data/m5"),
        help="Path to M5 dataset directory",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Device for Chronos model: auto detects MPS on Apple Silicon (default: auto)",
    )
    parser.add_argument(
        "--strategy",
        choices=["pick_best", "weighted_avg", "segment_blend"],
        default="weighted_avg",
        help="Committee blend strategy (default: weighted_avg)",
    )
    parser.add_argument(
        "--forecasters",
        type=str,
        nargs="+",
        default=None,
        help="Committee forecasters to use (e.g., --forecasters chronos naive)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    config = Config()

    provider = args.provider
    model = args.model or get_default_model(provider)

    config.critic = CriticConfig(
        provider=provider,
        model=model,
        concurrency=args.concurrency,
    )
    # Surgeon codegen uses the same provider/model
    config.surgeon.codegen_provider = provider
    config.surgeon.codegen_model = model

    config.output_dir = args.output_dir
    config.experiment.seed = args.seed
    config.m5.data_dir = args.m5_data_dir
    config.m5.device = args.device
    config.committee.device = args.device
    config.committee.strategy = BlendStrategy(args.strategy)
    if args.forecasters:
        config.committee.forecasters = args.forecasters

    if args.n_samples is not None:
        n = args.n_samples
        config.experiment.n_perturbed = n
        config.experiment.n_unperturbed = n
        config.experiment.n_generate_per_type = int(n / 0.75) + 1
        config.experiment.n_promo_per_scenario = n
        config.m5.n_samples = n

    return config


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    config = build_config(args)

    experiments_to_run = (
        ["synthetic", "exogenous", "m5"] if args.experiment == "all"
        else [args.experiment]
    )

    for exp in experiments_to_run:
        if exp == "synthetic":
            from forecast_critic.experiments.synthetic_experiment import run_synthetic_experiment
            run_synthetic_experiment(config)

        elif exp == "exogenous":
            from forecast_critic.experiments.exogenous_experiment import run_exogenous_experiment
            run_exogenous_experiment(config)

        elif exp == "m5":
            from forecast_critic.experiments.m5_experiment import run_m5_experiment
            run_m5_experiment(config)

        elif exp == "surgeon":
            from forecast_critic.experiments.surgeon_experiment import run_surgeon_experiment
            run_surgeon_experiment(config)

        elif exp == "committee":
            from forecast_critic.experiments.committee_experiment import run_committee_experiment
            run_committee_experiment(config)


if __name__ == "__main__":
    main()

