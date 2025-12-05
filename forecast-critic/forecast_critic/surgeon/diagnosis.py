from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass

import anthropic

from forecast_critic.config import CriticConfig, FailureMode

logger = logging.getLogger(__name__)


@dataclass
class FailureDiagnosis:
    failure_type: FailureMode
    severity: float
    description: str
    affected_range: tuple[int | None, int | None]


@dataclass
class Diagnosis:
    failure_modes: list[FailureDiagnosis]
    overall_description: str
    raw_json: dict


def _parse_failure_type(type_str: str) -> FailureMode:
    type_str = type_str.lower().strip()
    for mode in FailureMode:
        if mode.value == type_str:
            return mode
    if "trend" in type_str:
        return FailureMode.TREND_MISMATCH
    if "shift" in type_str or "translat" in type_str or "level" in type_str:
        return FailureMode.VERTICAL_SHIFT
    if "volatil" in type_str or "variance" in type_str or "smooth" in type_str:
        return FailureMode.VOLATILITY_COLLAPSE
    if "spurious" in type_str or "extra" in type_str or "false" in type_str:
        return FailureMode.SPURIOUS_SPIKE
    if "missing" in type_str or "absent" in type_str:
        return FailureMode.MISSING_SPIKE
    if "period" in type_str or "frequen" in type_str or "stretch" in type_str:
        return FailureMode.PERIODICITY_MISMATCH
    return FailureMode.UNKNOWN


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1:
        try:
            return json.loads(text[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass
    logger.warning("Could not parse diagnosis JSON from response: %s", text[:200])
    return {"failure_modes": [], "overall_description": "Parse failure"}


def diagnose(image_bytes: bytes, prompt: str, config: CriticConfig) -> Diagnosis:
    client = anthropic.Anthropic()
    b64_image = base64.standard_b64encode(image_bytes).decode("utf-8")
    message = client.messages.create(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64_image}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    raw_text = message.content[0].text
    data = _extract_json(raw_text)

    failure_modes = []
    for fm in data.get("failure_modes", []):
        affected = fm.get("affected_range", [None, None])
        if not isinstance(affected, (list, tuple)) or len(affected) < 2:
            affected = [None, None]
        failure_modes.append(FailureDiagnosis(
            failure_type=_parse_failure_type(fm.get("type", "unknown")),
            severity=float(fm.get("severity", 0.5)),
            description=fm.get("description", ""),
            affected_range=(affected[0], affected[1]),
        ))

    if not failure_modes:
        failure_modes.append(FailureDiagnosis(
            failure_type=FailureMode.UNKNOWN,
            severity=0.5,
            description=data.get("overall_description", "Unspecified issue"),
            affected_range=(None, None),
        ))

    return Diagnosis(
        failure_modes=failure_modes,
        overall_description=data.get("overall_description", ""),
        raw_json=data,
    )
