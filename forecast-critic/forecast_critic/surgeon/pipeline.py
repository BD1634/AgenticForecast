"""Self-healing forecast pipeline.

Orchestrates: Critic → Diagnose → Correct → Re-evaluate loop.
The critic only sees images. Corrections operate on numpy arrays.
Loop continues until the critic approves or max iterations are hit.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from forecast_critic.config import Config, CriticConfig, FailureMode, SurgeonConfig
from forecast_critic.critic.llm import CriticResponse, ForecastCritic, _parse_response
from forecast_critic.data.perturbations import PerturbedSample
from forecast_critic.prompts.templates import DIAGNOSIS_PROMPT, SYNTHETIC_PROMPT
from forecast_critic.surgeon.corrections import apply_known_correction, has_known_correction
from forecast_critic.surgeon.codegen import generate_and_apply_correction
from forecast_critic.surgeon.diagnosis import Diagnosis, diagnose
from forecast_critic.visualization.plots import render_synthetic_plot

logger = logging.getLogger(__name__)


@dataclass
class CorrectionStep:
    iteration: int
    diagnosis: Diagnosis | None
    failure_modes_applied: list[str]
    method: str  # "hardcoded", "codegen", or "none"
    critic_verdict: int  # 1=reasonable, 2=unreasonable
    critic_explanation: str


@dataclass
class SurgeonResult:
    original_forecast: NDArray[np.float64]
    corrected_forecast: NDArray[np.float64]
    was_corrected: bool
    n_iterations: int
    final_verdict: int  # 1=reasonable, 2=unreasonable
    steps: list[CorrectionStep] = field(default_factory=list)
    original_smape: float = 0.0
    corrected_smape: float = 0.0


def _compute_relative_change(y_old: NDArray, y_new: NDArray) -> float:
    """Compute the relative change between two arrays."""
    denom = np.max(np.abs(y_old))
    if denom < 1e-10:
        denom = 1.0
    return float(np.max(np.abs(y_new - y_old)) / denom)


def heal_forecast(
    sample: PerturbedSample,
    config: Config,
) -> SurgeonResult:
    """Run the self-healing loop on a single forecast.

    1. Critic evaluates the forecast plot
    2. If unreasonable: diagnose failure modes
    3. Apply corrections (hardcoded for known types, codegen for unknown)
    4. Re-render and re-evaluate
    5. Repeat until reasonable or max iterations
    """
    critic = ForecastCritic(config.critic)
    ts = sample.ts
    y_current = sample.y_perturbed.copy()
    y_original = sample.y_perturbed.copy()
    steps: list[CorrectionStep] = []

    for iteration in range(config.surgeon.max_iterations + 1):
        # Build current sample for rendering
        current_sample = PerturbedSample(
            ts=ts,
            y_perturbed=y_current,
            perturbation=sample.perturbation,
            is_perturbed=sample.is_perturbed,
            smape=sample.smape,
        )

        # Render and evaluate
        img = render_synthetic_plot(current_sample)
        critic_response = critic.evaluate(img, SYNTHETIC_PROMPT)

        if critic_response.label == 1 or iteration == config.surgeon.max_iterations:
            # Approved or out of iterations
            steps.append(CorrectionStep(
                iteration=iteration,
                diagnosis=None,
                failure_modes_applied=[],
                method="none",
                critic_verdict=critic_response.label,
                critic_explanation=critic_response.explanation,
            ))
            break

        # Diagnose
        logger.info("Iteration %d: forecast flagged, diagnosing...", iteration)
        diag = diagnose(img, DIAGNOSIS_PROMPT, config.critic)
        logger.info(
            "Diagnosis: %s (failure modes: %s)",
            diag.overall_description,
            [fm.failure_type.value for fm in diag.failure_modes],
        )

        # Apply corrections — highest severity first
        sorted_failures = sorted(diag.failure_modes, key=lambda f: f.severity, reverse=True)
        applied_modes: list[str] = []
        method = "none"
        y_before = y_current.copy()

        for fm in sorted_failures:
            if has_known_correction(fm.failure_type):
                # Hardcoded correction
                logger.info("Applying hardcoded fix: %s", fm.failure_type.value)
                y_current = apply_known_correction(
                    fm.failure_type,
                    ts.y_hist,
                    y_current,
                    ts.t_hist,
                    ts.t_forecast,
                )
                applied_modes.append(fm.failure_type.value)
                method = "hardcoded"
            elif fm.failure_type == FailureMode.UNKNOWN:
                # LLM code generation fallback
                logger.info("Unknown failure mode, trying codegen: %s", fm.description)
                y_codegen = generate_and_apply_correction(
                    diagnosis_text=fm.description,
                    y_history=ts.y_hist,
                    y_forecast=y_current,
                    t_history=ts.t_hist,
                    t_forecast=ts.t_forecast,
                    critic_config=config.critic,
                    surgeon_config=config.surgeon,
                )
                if y_codegen is not None:
                    y_current = y_codegen
                    applied_modes.append("codegen")
                    method = "codegen"
                else:
                    logger.warning("Codegen correction failed, skipping")
                    applied_modes.append("codegen_failed")

        steps.append(CorrectionStep(
            iteration=iteration,
            diagnosis=diag,
            failure_modes_applied=applied_modes,
            method=method,
            critic_verdict=critic_response.label,
            critic_explanation=critic_response.explanation,
        ))

        # Check convergence
        change = _compute_relative_change(y_before, y_current)
        if change < config.surgeon.convergence_threshold:
            logger.info("Converged (change=%.4f < threshold=%.4f), stopping",
                        change, config.surgeon.convergence_threshold)
            break

    # Compute SMAPE for original and corrected
    from forecast_critic.data.perturbations import smape
    original_smape = smape(ts.y_forecast, y_original)
    corrected_smape = smape(ts.y_forecast, y_current)

    was_corrected = not np.allclose(y_original, y_current)

    return SurgeonResult(
        original_forecast=y_original,
        corrected_forecast=y_current,
        was_corrected=was_corrected,
        n_iterations=len(steps),
        final_verdict=steps[-1].critic_verdict if steps else -1,
        steps=steps,
        original_smape=original_smape,
        corrected_smape=corrected_smape,
    )

