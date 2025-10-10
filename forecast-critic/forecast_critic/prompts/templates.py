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
