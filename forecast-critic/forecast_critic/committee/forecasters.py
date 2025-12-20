"""Pluggable forecaster interface for the committee.

Each forecaster takes a history array and returns a point forecast.
Adding a new model = subclass BaseForecaster and implement `predict`.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    model_name: str
    forecast: NDArray[np.float64]
    # Optional quantiles for probabilistic models
    quantiles: dict[float, NDArray[np.float64]] | None = None


class BaseForecaster(ABC):
    """Base class for all forecasters in the committee."""

    name: str

    @abstractmethod
    def predict(
        self,
        history: NDArray[np.float64],
        horizon: int,
    ) -> ForecastResult:
        """Generate a forecast from historical data.

        Args:
            history: historical values, shape (T,)
            horizon: number of future steps to forecast

        Returns:
            ForecastResult with at minimum a point forecast
        """
        ...

    @abstractmethod
    def load(self, device: str = "cpu") -> None:
        """Load model weights. Called once before predictions."""
        ...


class ChronosForecaster(BaseForecaster):
    name = "chronos"

    def __init__(self, model_id: str = "amazon/chronos-t5-small"):
        self.model_id = model_id
        self._pipeline = None

    def load(self, device: str = "cpu") -> None:
        import torch
        from chronos import ChronosPipeline

        logger.info("Loading Chronos: %s on %s", self.model_id, device)
        self._pipeline = ChronosPipeline.from_pretrained(
            self.model_id,
            device_map=device,
            torch_dtype=torch.float32,
        )

    def predict(self, history: NDArray, horizon: int) -> ForecastResult:
        import torch

        context = torch.tensor(history, dtype=torch.float32).unsqueeze(0)
        samples = self._pipeline.predict(context, prediction_length=horizon, num_samples=100)
        samples_np = samples.squeeze(0).numpy()

        median = np.median(samples_np, axis=0)
        quantiles = {
            q: np.quantile(samples_np, q, axis=0)
            for q in [0.1, 0.5, 0.9]
        }

        return ForecastResult(
            model_name=self.name,
            forecast=median,
            quantiles=quantiles,
        )


class TimesFMForecaster(BaseForecaster):
    name = "timesfm"

    def __init__(self, model_id: str = "google/timesfm-1.0-200m-pytorch"):
        self.model_id = model_id
        self._model = None

    def load(self, device: str = "cpu") -> None:
        try:
            import timesfm
        except ImportError:
            raise ImportError("Install timesfm: uv pip install timesfm")

        logger.info("Loading TimesFM: %s", self.model_id)
        backend = "gpu" if device in ("cuda", "mps") else "cpu"
        self._model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=backend,
                per_core_batch_size=32,
                horizon_len=128,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=self.model_id,
            ),
        )

    def predict(self, history: NDArray, horizon: int) -> ForecastResult:
        forecast, _ = self._model.forecast(
            [history.tolist()],
            freq=[0],  # 0 = unknown frequency
        )

        point_forecast = np.array(forecast[0][:horizon], dtype=np.float64)

        return ForecastResult(
            model_name=self.name,
            forecast=point_forecast,
        )


class LagLlamaForecaster(BaseForecaster):
    name = "lagllama"

    def __init__(self, model_id: str = "time-series-foundation-models/Lag-Llama"):
        self.model_id = model_id
        self._pipeline = None

    def load(self, device: str = "cpu") -> None:
        try:
            from huggingface_hub import hf_hub_download
            from lag_llama.gluon.estimator import LagLlamaEstimator
        except ImportError:
            raise ImportError(
                "Install lag-llama from GitHub:\n"
                '  uv pip install "lag-llama @ git+https://github.com/time-series-foundation-models/lag-llama.git"'
            )

        logger.info("Loading Lag-Llama: %s on %s", self.model_id, device)
        import torch

        ckpt_path = hf_hub_download(
            repo_id=self.model_id,
            filename="lag-llama.ckpt",
        )

        estimator = LagLlamaEstimator(
            ckpt_path=ckpt_path,
            prediction_length=128,
            context_length=512,
            input_size=1,
            n_layer=8,
            n_embd_per_head=32,
            n_head=4,
            num_samples=100,
            device=torch.device(device),
        )
        self._pipeline = estimator.create_lightning_module()
        self._device = device

    def predict(self, history: NDArray, horizon: int) -> ForecastResult:
        import torch
        from gluonts.dataset.pandas import PandasDataset
        import pandas as pd

        # Wrap in GluonTS format
        index = pd.period_range(start="2020-01-01", periods=len(history), freq="D")
        df = pd.DataFrame({"target": history}, index=index)
        dataset = PandasDataset.from_long_dataframe(df, target="target")

        # Generate forecasts
        from lag_llama.gluon.estimator import LagLlamaEstimator

        forecasts = list(self._pipeline.predict(dataset, num_samples=100))
        samples = forecasts[0].samples[:, :horizon]  # (100, horizon)

        median = np.median(samples, axis=0)
        quantiles = {
            q: np.quantile(samples, q, axis=0)
            for q in [0.1, 0.5, 0.9]
        }

        return ForecastResult(
            model_name=self.name,
            forecast=median,
            quantiles=quantiles,
        )


class NaiveForecaster(BaseForecaster):
    """Simple baseline: repeats the last `horizon` values from history.

    Always available, no dependencies. Useful as a sanity-check member.
    """
    name = "naive"

    def load(self, device: str = "cpu") -> None:
        pass  # no model to load

    def predict(self, history: NDArray, horizon: int) -> ForecastResult:
        # Repeat the last `horizon` points from history
        tail = history[-horizon:]
        return ForecastResult(model_name=self.name, forecast=tail.copy())


FORECASTER_REGISTRY: dict[str, type[BaseForecaster]] = {
    "chronos": ChronosForecaster,
    "timesfm": TimesFMForecaster,
    "lagllama": LagLlamaForecaster,
    "naive": NaiveForecaster,
}


def resolve_device(device: str) -> str:
    """Resolve 'auto' to the best available device."""
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def build_committee(
    forecaster_names: list[str],
    config,
    device: str = "auto",
) -> list[BaseForecaster]:
    """Instantiate and load all forecasters in the committee."""
    device = resolve_device(device)
    committee: list[BaseForecaster] = []

    model_ids = {
        "chronos": config.chronos_model,
        "timesfm": config.timesfm_model,
        "lagllama": config.lagllama_model,
    }

    for name in forecaster_names:
        if name not in FORECASTER_REGISTRY:
            logger.warning("Unknown forecaster '%s', skipping", name)
            continue

        cls = FORECASTER_REGISTRY[name]
        model_id = model_ids.get(name)
        if model_id and name != "naive":
            forecaster = cls(model_id=model_id)
        else:
            forecaster = cls()

        try:
            forecaster.load(device)
            committee.append(forecaster)
            logger.info("Loaded forecaster: %s", name)
        except (ImportError, Exception) as e:
            logger.warning("Failed to load forecaster '%s': %s. Skipping.", name, e)

    if not committee:
        logger.warning("No forecasters loaded! Adding naive baseline.")
        naive = NaiveForecaster()
        naive.load(device)
        committee.append(naive)

    return committee

