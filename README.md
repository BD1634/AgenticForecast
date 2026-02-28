# AgenticForecast

**LLM-based automated forecast monitoring and self-healing.** Uses vision-language models to evaluate time-series forecasts, blend committee predictions, and iteratively correct flawed forecasts via diagnosis → correction → re-evaluation loops.

## Overview

AgenticForecast explores the intersection of **agentic AI** and **time-series forecasting**:

| Component | Description |
|-----------|-------------|
| **Critic** | Vision LLM evaluates forecast plots (reasonable vs unreasonable) and provides explanations |
| **Committee** | Ensemble of forecasters (Chronos, TimesFM, naive) with LLM-driven blending strategies |
| **Surgeon** | Self-healing pipeline: diagnose failure modes → apply corrections (hardcoded or codegen) → re-evaluate until approved |

Supports **Ollama** (local), **Google Gemini** (free tier), and **Anthropic** (paid).

## Installation

```bash
cd forecast-critic
pip install -e .
```

**Optional dependencies** (install as needed):

```bash
# For Ollama (local, free) - requires Ollama running with llama3.2-vision
pip install -e ".[ollama]"

# For Google Gemini (free API)
pip install -e ".[gemini]"

# For Anthropic Claude (paid)
pip install -e ".[anthropic]"

# For M5 experiment (Chronos foundation model)
pip install -e ".[m5]"
```

## Quick Start

```bash
# Run synthetic perturbation detection (default: Ollama)
python main.py --experiment synthetic --provider ollama

# Run committee experiment with Gemini
python main.py --experiment committee --provider gemini --strategy weighted_avg
```

## Experiments

| Experiment | Description |
|------------|-------------|
| `synthetic` | Perturbation detection on synthetic time series |
| `exogenous` | Promotional context and exogenous variable impact |
| `m5` | Real-world M5 competition dataset (requires Chronos) |
| `surgeon` | Self-healing forecasts: Critic → Diagnose → Correct loop |
| `committee` | Committee of foundation models with LLM-driven blending |
| `all` | Run synthetic, exogenous, and m5 sequentially |

## Usage

```bash
python main.py --experiment <synthetic|exogenous|m5|surgeon|committee|all> [options]
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--provider` | `ollama` | `ollama`, `gemini`, or `anthropic` |
| `--model` | auto | Override model ID (e.g. `llama3.2-vision`, `gemini-2.0-flash`) |
| `--strategy` | `weighted_avg` | Committee blend: `pick_best`, `weighted_avg`, `segment_blend` |
| `--forecasters` | chronos, naive, … | Committee members to use |
| `--n-samples` | config | Override sample count for quick testing |
| `--concurrency` | 5 | Max concurrent API calls |
| `--output-dir` | `outputs/` | Where to save results |
| `--m5-data-dir` | `data/m5` | Path to M5 dataset |
| `--device` | `auto` | `auto`, `cpu`, `mps`, or `cuda` for Chronos |
| `--seed` | 42 | Random seed |

### Examples

```bash
# Self-healing with Gemini
python main.py --experiment surgeon --provider gemini

# Committee with pick-best strategy
python main.py --experiment committee --strategy pick_best --forecasters chronos naive

# Quick run (fewer samples)
python main.py --experiment synthetic --n-samples 10
```

## Project Structure

```
forecast-critic/
├── main.py                 # CLI entry point
├── forecast_critic/
│   ├── critic/             # Vision LLM forecast evaluator
│   ├── committee/          # Multi-forecaster ensemble + blending
│   ├── surgeon/            # Diagnosis, corrections, self-healing loop
│   ├── data/               # Synthetic, M5, perturbations, promotions
│   ├── experiments/        # Experiment runners
│   ├── metrics/            # SMAPE, evaluation
│   ├── prompts/            # LLM prompt templates
│   ├── visualization/      # Plot rendering
│   └── llm_provider.py     # Ollama / Gemini / Anthropic abstraction
└── pyproject.toml
```

## Requirements

- Python ≥ 3.11
- LLM access: Ollama (local), Gemini (API key), or Anthropic (API key)

For M5/Chronos: `torch`, `chronos-forecasting`

## License

MIT
