from __future__ import annotations

# ---------------------------------------------------------------------------
# Experiment 1: Synthetic perturbation detection (Figure 2)
# ---------------------------------------------------------------------------
SYNTHETIC_PROMPT = """\
You are shown an image of historical data (in black) and a forecast \
(in blue). Based on the historical trend, assess whether the forecast \
is reasonable. A reasonable forecast should generally follow the same \
direction and capture any seasonal trends if there are any.

Please provide a brief explanation (1–2 sentences) justifying your \
decision. Then present your final answer using one of the following \
options, wrapped in <answer> tags:
<answer> 1 </answer> — The forecast is reasonable.
<answer> 2 </answer> — The forecast is unreasonable."""

# ---------------------------------------------------------------------------
# Experiment 2: Exogenous promotional feature injection (Figure 13)
# ---------------------------------------------------------------------------
PROMOTIONAL_PROMPT_TEMPLATE = """\
You are shown an image of historical data (in black) and a forecast \
(in blue). Based on the historical trend, assess whether the forecast \
is reasonable.

A reasonable forecast should generally follow the same direction and \
capture any seasonal trends if there are any. Note, there is a holiday \
at t = {holiday_hist_time} in the historical and a second holiday at \
time t = {holiday_forecast_time} in the forecast that may affect the demand.

Please provide a brief explanation (1–2 sentences) justifying your \
decision. Then present your final answer using one of the following \
options, wrapped in <answer> tags:
<answer> 1 </answer> — The forecast is reasonable.
<answer> 2 </answer> — The forecast is unreasonable."""

# ---------------------------------------------------------------------------
# Experiment 3: M5 real-world dataset (Figure 18)
# ---------------------------------------------------------------------------
M5_PROMPT = """\
You are shown an image of historical data (in black) and a forecast \
(in blue). Your task is to assess whether the forecast appears visually \
reasonable.

A reasonable forecast should generally follow the same direction as the \
historical trend and reflect any clear seasonal patterns, if present.

IMPORTANT: Only label a forecast as unreasonable if there is an obvious \
and significant mismatch — for example, if the forecast goes in the \
opposite direction of the trend, ignores strong seasonal patterns, or \
shows extreme jumps that are not supported by the historical data.

Minor deviations or slight over/underestimates are acceptable and should \
still be considered reasonable.

Please provide a brief explanation (1–2 sentences) justifying your \
decision. Then present your final answer using one of the following \
options, wrapped in <answer> tags:
<answer> 1 </answer> — The forecast is reasonable.
<answer> 2 </answer> — The forecast is unreasonable.

If you find the forecast unreasonable, clearly explain what makes it \
obviously inconsistent."""


# ---------------------------------------------------------------------------
# Surgeon: Structured diagnosis prompt
# ---------------------------------------------------------------------------
DIAGNOSIS_PROMPT = """\
You are shown an image of historical data (in black) and a forecast \
(in blue). The forecast has been flagged as unreasonable.

Analyze the image and produce a structured diagnosis. Identify ALL \
failure modes present. For each failure mode, classify it as one of:
- trend_mismatch: forecast trend direction or magnitude doesn't match history
- vertical_shift: forecast is shifted up or down relative to expected level
- volatility_collapse: forecast is much smoother or more volatile than history
- spurious_spike: forecast contains spikes not justified by history
- missing_spike: forecast is missing expected spikes visible in history
- periodicity_mismatch: forecast oscillation frequency differs from history
- unknown: any other mismatch not covered above

Respond with ONLY a JSON object (no markdown fencing), using this schema:
{
  "failure_modes": [
    {
      "type": "<failure_mode_type>",
      "severity": <float 0.0 to 1.0>,
      "description": "<what is wrong, 1 sentence>",
      "affected_range": [<start_index_or_null>, <end_index_or_null>]
    }
  ],
  "overall_description": "<1-2 sentence summary of all issues>"
}"""

# ---------------------------------------------------------------------------
# Surgeon: LLM code generation prompt for novel failures
# ---------------------------------------------------------------------------
CODEGEN_PROMPT_TEMPLATE = """\
You are a time series correction expert. A forecast has been diagnosed \
with the following issue that does not match any standard correction:

DIAGNOSIS:
{diagnosis}

You are given:
- y_history: numpy array of historical values, shape ({hist_len},)
- y_forecast: numpy array of forecast values to correct, shape ({fc_len},)
- t_history: numpy array of historical time points, shape ({hist_len},)
- t_forecast: numpy array of forecast time points, shape ({fc_len},)
- np: the numpy module

Write a Python code snippet that modifies y_forecast IN-PLACE to fix \
the diagnosed issue. The correction should:
1. Use patterns from y_history to guide the fix
2. Maintain continuity at the history/forecast boundary
3. Keep values within a reasonable range

Respond with ONLY the Python code (no markdown fencing, no explanation). \
The code will be executed in a sandboxed environment with only numpy and \
the arrays above available."""


def build_promotional_prompt(
    holiday_hist_time: float,
    holiday_forecast_time: float,
) -> str:
    """Build the promotional prompt with specific holiday times."""
    return PROMOTIONAL_PROMPT_TEMPLATE.format(
        holiday_hist_time=holiday_hist_time,
        holiday_forecast_time=holiday_forecast_time,
    )


def build_codegen_prompt(
    diagnosis: str,
    hist_len: int,
    fc_len: int,
) -> str:
    """Build the code generation prompt with array dimensions."""
    return CODEGEN_PROMPT_TEMPLATE.format(
        diagnosis=diagnosis,
        hist_len=hist_len,
        fc_len=fc_len,
    )

