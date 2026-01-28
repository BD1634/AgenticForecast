from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class PerturbationType(Enum):
    VERTICAL_SHIFT = "vertical_shift"
    TREND_MODIFICATION = "trend_modification"
    TIME_STRETCH = "time_stretch"
    RANDOM_SPIKES = "random_spikes"


class PromotionalScenario(Enum):
    A = "no_dep_no_spike"      # No historical lift, no forecast spike (reasonable)
    B = "no_dep_false_spike"   # No historical lift, spurious forecast spike (unreasonable)
    C = "dep_missing_spike"    # Historical lift, missing forecast spike (unreasonable)
    D = "dep_correct_spike"    # Historical lift, correct forecast spike (reasonable)


@dataclass
class SyntheticConfig:
    dt: float = 0.1
    t_max: float = 10.0
    split_time: float = 8.0
    n_basis_min: int = 1
    n_basis_max: int = 4
    weight_min: float = 0.5
    weight_max: float = 2.0
    input_scale_min: float = 0.5
    input_scale_max: float = 2.0
    input_shift_min: float = 0.0
    input_shift_max: float = 4.0
    n_basis_functions: int = 14
    eps_threshold: float = 1e-10

    @property
    def n_points(self) -> int:
        return int(self.t_max / self.dt) + 1

    @property
    def split_index(self) -> int:
        return int(self.split_time / self.dt)


@dataclass
class PerturbationConfig:
    omega: float = 0.5         # vertical shift scale
    beta: float = -3.0         # trend slope scaling
    alpha: float = 3.0         # time stretch factor
    gamma: float = 0.5         # spike magnitude scale
    n_spikes_max: int = 3      # max number of random spikes


@dataclass
class PromotionConfig:
    spike_magnitude_min: float = 2.0
    spike_magnitude_max: float = 5.0
    spike_width: int = 1  # number of points affected by spike


@dataclass
class M5Config:
    data_dir: Path = field(default_factory=lambda: Path("data/m5"))
    n_samples: int = 1000
    history_days: int = 120
    forecast_days: int = 28
    quantiles: list[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )
    chronos_model: str = "amazon/chronos-t5-small"
    device: str = "auto"  # "auto", "mps", "cpu", "cuda"


@dataclass
class ExperimentConfig:
    n_perturbed: int = 250
    n_unperturbed: int = 250
    n_generate_per_type: int = 334   # generate more, filter to worst 75%
    smape_filter_pct: float = 0.75
    n_promo_per_scenario: int = 500
    seed: int = 42


@dataclass
class CriticConfig:
    provider: str = "anthropic"  # "anthropic" or "gemini"
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024
    temperature: float = 0.0
    concurrency: int = 5
    max_retries: int = 3
    retry_base_delay: float = 1.0


class FailureMode(Enum):
    TREND_MISMATCH = "trend_mismatch"
    VERTICAL_SHIFT = "vertical_shift"
    VOLATILITY_COLLAPSE = "volatility_collapse"
    SPURIOUS_SPIKE = "spurious_spike"
    MISSING_SPIKE = "missing_spike"
    PERIODICITY_MISMATCH = "periodicity_mismatch"
    UNKNOWN = "unknown"


@dataclass
class SurgeonConfig:
    max_iterations: int = 3
    codegen_provider: str = "anthropic"  # "anthropic" or "gemini"
    codegen_model: str = "claude-sonnet-4-20250514"
    codegen_max_tokens: int = 2048
    codegen_temperature: float = 0.0
    # Safety bounds: corrected forecast must stay within this factor of original range
    safety_range_factor: float = 3.0
    # If corrected forecast changes by less than this fraction, stop iterating
    convergence_threshold: float = 0.01


class BlendStrategy(Enum):
    PICK_BEST = "pick_best"             # Critic picks the single best forecast
    WEIGHTED_AVERAGE = "weighted_avg"    # Weighted average by critic confidence
    SEGMENT_BLEND = "segment_blend"     # Different models for different segments


@dataclass
class CommitteeConfig:
    forecasters: list[str] = field(default_factory=lambda: [
        "chronos",
        "timesfm",
        "lagllama",
    ])
    strategy: BlendStrategy = BlendStrategy.WEIGHTED_AVERAGE
    device: str = "auto"
    # Model IDs for each forecaster
    chronos_model: str = "amazon/chronos-t5-small"
    timesfm_model: str = "google/timesfm-1.0-200m-pytorch"
    lagllama_model: str = "time-series-foundation-models/Lag-Llama"


@dataclass
class Config:
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    perturbation: PerturbationConfig = field(default_factory=PerturbationConfig)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    m5: M5Config = field(default_factory=M5Config)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    surgeon: SurgeonConfig = field(default_factory=SurgeonConfig)
    committee: CommitteeConfig = field(default_factory=CommitteeConfig)
    output_dir: Path = field(default_factory=lambda: Path("outputs"))

