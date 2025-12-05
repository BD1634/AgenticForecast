from __future__ import annotations

import base64
import logging

import anthropic
import numpy as np
from numpy.typing import NDArray

from forecast_critic.config import SurgeonConfig, CriticConfig
from forecast_critic.prompts.templates import build_codegen_prompt

logger = logging.getLogger(__name__)


def _strip_code_fences(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)
    return code.strip()


def _validate_code(code: str) -> list[str]:
    violations = []
    dangerous_tokens = [
        "import ", "open(", "exec(", "eval(", "__", "os.", "sys.",
        "subprocess", "shutil", "pathlib", "socket", "http",
        "requests", "urllib",
    ]
    for token in dangerous_tokens:
        if token in code:
            violations.append(f"Disallowed token: {token!r}")
    return violations


def _execute_sandboxed(code, y_history, y_forecast, t_history, t_forecast):
    y_fc_copy = y_forecast.copy()
    namespace = {
        "np": np,
        "y_history": y_history.copy(),
        "y_forecast": y_fc_copy,
        "t_history": t_history.copy(),
        "t_forecast": t_forecast.copy(),
    }
    safe_builtins = {
        "range": range, "len": len, "int": int, "float": float,
        "abs": abs, "min": min, "max": max, "sum": sum,
        "enumerate": enumerate, "zip": zip,
        "True": True, "False": False, "None": None,
    }
    exec(code, {"__builtins__": safe_builtins}, namespace)
    return namespace["y_forecast"]


def _validate_result(y_original, y_corrected, y_history, config):
    if len(y_corrected) != len(y_original):
        return False
    if not np.all(np.isfinite(y_corrected)):
        return False
    all_vals = np.concatenate([y_history, y_original])
    val_range = np.ptp(all_vals)
    val_min = np.min(all_vals) - config.safety_range_factor * val_range
    val_max = np.max(all_vals) + config.safety_range_factor * val_range
    if np.any(y_corrected < val_min) or np.any(y_corrected > val_max):
        return False
    return True


def generate_and_apply_correction(
    diagnosis_text, y_history, y_forecast, t_history, t_forecast,
    critic_config, surgeon_config,
):
    prompt = build_codegen_prompt(
        diagnosis=diagnosis_text,
        hist_len=len(y_history),
        fc_len=len(y_forecast),
    )
    client = anthropic.Anthropic()
    try:
        message = client.messages.create(
            model=surgeon_config.codegen_model,
            max_tokens=surgeon_config.codegen_max_tokens,
            temperature=surgeon_config.codegen_temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        code = message.content[0].text
    except Exception as e:
        logger.error("Code generation API call failed: %s", e)
        return None

    code = _strip_code_fences(code)
    logger.debug("Generated correction code:\n%s", code)

    violations = _validate_code(code)
    if violations:
        logger.warning("Generated code failed validation: %s", violations)
        return None

    try:
        y_corrected = _execute_sandboxed(code, y_history, y_forecast, t_history, t_forecast)
    except Exception as e:
        logger.warning("Generated code execution failed: %s", e)
        return None

    if not _validate_result(y_forecast, y_corrected, y_history, surgeon_config):
        return None

    return y_corrected
