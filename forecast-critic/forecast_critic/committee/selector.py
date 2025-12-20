"""LLM-guided forecast selection from a committee of foundation models.

Given multiple candidate forecasts, the LLM visually inspects each one
overlaid on the historical data and picks the best, or assigns weights
for blending.
"""
from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass

import anthropic
import numpy as np
from numpy.typing import NDArray

from forecast_critic.config import BlendStrategy, CommitteeConfig, CriticConfig
from forecast_critic.prompts.templates import (
    build_selection_prompt,
    build_ranking_prompt,
)
from forecast_critic.visualization.plots import render_committee_plot

logger = logging.getLogger(__name__)


@dataclass
class SelectionResult:
    chosen_index: int | None
    weights: NDArray[np.float64] | None
    blended_forecast: NDArray[np.float64]
    raw_response: str
    strategy: BlendStrategy


def _parse_pick_best(response: str, n_candidates: int) -> int:
    match = re.search(r"<answer>\s*(\d+)\s*</answer>", response)
    if match:
        idx = int(match.group(1)) - 1
        if 0 <= idx < n_candidates:
            return idx
    logger.warning("Could not parse selection from response, defaulting to 0")
    return 0


def _parse_weights(response: str, n_candidates: int) -> NDArray[np.float64]:
    match = re.search(r"<weights>\s*([\d.,\s]+)\s*</weights>", response)
    if match:
        try:
            raw = match.group(1).strip()
            weights = np.array([float(w.strip()) for w in raw.split(",")])
            if len(weights) == n_candidates and np.all(weights >= 0):
                total = weights.sum()
                if total > 0:
                    return weights / total
        except (ValueError, IndexError):
            pass
    logger.warning("Could not parse weights, using uniform")
    return np.ones(n_candidates) / n_candidates


def _call_llm(image_bytes: bytes, prompt: str, config: CriticConfig) -> str:
    client = anthropic.Anthropic()
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    message = client.messages.create(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return message.content[0].text


def select_forecast(
    y_history: NDArray,
    t_history: NDArray,
    t_forecast: NDArray,
    candidate_forecasts: list[NDArray],
    model_names: list[str],
    committee_config: CommitteeConfig,
    critic_config: CriticConfig,
) -> SelectionResult:
    n = len(candidate_forecasts)
    strategy = committee_config.blend_strategy

    image_bytes = render_committee_plot(
        t_history=t_history,
        y_history=y_history,
        t_forecast=t_forecast,
        forecasts=candidate_forecasts,
        model_names=model_names,
    )

    if strategy == BlendStrategy.PICK_BEST:
        prompt = build_selection_prompt(model_names=model_names)
        response = _call_llm(image_bytes, prompt, critic_config)
        idx = _parse_pick_best(response, n)
        return SelectionResult(
            chosen_index=idx, weights=None,
            blended_forecast=candidate_forecasts[idx],
            raw_response=response, strategy=strategy,
        )

    elif strategy == BlendStrategy.WEIGHTED_AVERAGE:
        prompt = build_ranking_prompt(model_names=model_names)
        response = _call_llm(image_bytes, prompt, critic_config)
        weights = _parse_weights(response, n)
        blended = np.zeros_like(candidate_forecasts[0], dtype=np.float64)
        for w, fc in zip(weights, candidate_forecasts):
            blended += w * fc
        return SelectionResult(
            chosen_index=None, weights=weights,
            blended_forecast=blended,
            raw_response=response, strategy=strategy,
        )

    elif strategy == BlendStrategy.SEGMENT_BLEND:
        seg_len = max(1, len(t_forecast) // n)
        blended = np.empty_like(candidate_forecasts[0], dtype=np.float64)
        responses = []
        for seg_i in range(n):
            start = seg_i * seg_len
            end = min(start + seg_len, len(t_forecast)) if seg_i < n - 1 else len(t_forecast)
            prompt = build_selection_prompt(
                model_names=model_names,
                segment_info=f"Focus on the forecast segment from index {start} to {end}.",
            )
            response = _call_llm(image_bytes, prompt, critic_config)
            responses.append(response)
            idx = _parse_pick_best(response, n)
            blended[start:end] = candidate_forecasts[idx][start:end]
        return SelectionResult(
            chosen_index=None, weights=None,
            blended_forecast=blended,
            raw_response="\n---\n".join(responses), strategy=strategy,
        )
    else:
        raise ValueError(f"Unknown blend strategy: {strategy}")
