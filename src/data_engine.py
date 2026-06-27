"""Synthetic Kronos-style data engine for cross-asset latent alignment."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyntheticKronosConfig:
    """Configuration for synthetic foundation-model token generation."""

    num_samples: int = 1_200
    sequence_length: int = 30
    d_model: int = 256
    lag_steps: int = 2
    deviation_fraction: float = 0.15
    noise_blend: float = 0.08
    innovation_scale: float = 0.035
    idiosyncratic_scale: float = 0.010
    price_return_scale: float = 0.006
    price_noise_scale: float = 0.002
    breakdown_price_shock_scale: float = 0.045
    seed: int = 42

    def validate(self) -> None:
        if self.num_samples <= 10:
            raise ValueError("num_samples must be greater than 10.")
        if self.sequence_length <= self.lag_steps + 2:
            raise ValueError("sequence_length must be at least lag_steps + 3.")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive.")
        if not 0.0 <= self.deviation_fraction < 1.0:
            raise ValueError("deviation_fraction must be in [0, 1).")
        if not 0.0 <= self.noise_blend <= 1.0:
            raise ValueError("noise_blend must be in [0, 1].")
        if self.lag_steps < 0:
            raise ValueError("lag_steps must be non-negative.")


@dataclass(frozen=True)
class SyntheticKronosDataset:
    """Container for rolling latent token windows and synthetic close prices."""

    asset_a_tokens: np.ndarray
    asset_b_tokens: np.ndarray
    labels: np.ndarray
    asset_a_prices: np.ndarray
    asset_b_prices: np.ndarray
    anomaly_indices: np.ndarray

    @property
    def num_samples(self) -> int:
        return int(self.labels.shape[0])

    def train_test_split(
        self,
        train_fraction: float = 0.70,
    ) -> Tuple["SyntheticKronosDataset", "SyntheticKronosDataset"]:
        """Return chronological train and test datasets."""
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be in (0, 1).")
        split_index = int(self.num_samples * train_fraction)
        if split_index <= 0 or split_index >= self.num_samples:
            raise ValueError("train_fraction leaves an empty split.")
        return self.slice(0, split_index), self.slice(split_index, self.num_samples)

    def slice(self, start: int, stop: int) -> "SyntheticKronosDataset":
        """Return a range-preserving dataset slice."""
        labels = self.labels[start:stop].copy()
        return SyntheticKronosDataset(
            asset_a_tokens=self.asset_a_tokens[start:stop].copy(),
            asset_b_tokens=self.asset_b_tokens[start:stop].copy(),
            labels=labels,
            asset_a_prices=self.asset_a_prices[start:stop].copy(),
            asset_b_prices=self.asset_b_prices[start:stop].copy(),
            anomaly_indices=np.flatnonzero(labels == 0.0).astype(np.int64),
        )


class SyntheticKronosDataEngine:
    """Generate leader-lagger latent states with labeled regime breakdowns.

    The engine creates a continuous correlated Gaussian random walk, converts it
    into rolling token windows, builds a lagging asset through a temporal shift
    operator, and replaces exactly 15 percent of default sample windows with
    independent breakdown regimes.
    """

    def __init__(
        self,
        config: SyntheticKronosConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config or SyntheticKronosConfig()
        self.config.validate()
        self.rng = np.random.default_rng(self.config.seed)
        self.logger = logger or LOGGER

    def generate(self) -> SyntheticKronosDataset:
        cfg = self.config
        total_steps = cfg.num_samples + cfg.sequence_length - 1
        factor_count = min(8, cfg.d_model)

        loadings = self._factor_loadings(factor_count, cfg.d_model)
        factor_innovations = self._correlated_factors(total_steps, factor_count)
        idiosyncratic_noise = self.rng.normal(
            loc=0.0,
            scale=cfg.idiosyncratic_scale,
            size=(total_steps, cfg.d_model),
        )
        leader_innovations = (
            cfg.innovation_scale * factor_innovations @ loadings
            + idiosyncratic_noise
        )
        leader_states = np.cumsum(leader_innovations, axis=0)
        leader_states = self._zscore_2d(leader_states)

        shifted_leader_states = self._apply_temporal_shift_2d(
            matrix=leader_states,
            lag_steps=cfg.lag_steps,
        )
        lag_noise = self.rng.normal(
            loc=0.0,
            scale=1.0,
            size=shifted_leader_states.shape,
        )
        lagger_states = (
            (1.0 - cfg.noise_blend) * shifted_leader_states
            + cfg.noise_blend * lag_noise
        )
        lagger_states = self._zscore_2d(lagger_states)

        asset_a_tokens = self._rolling_windows(leader_states, cfg.sequence_length)
        asset_b_tokens = self._rolling_windows(lagger_states, cfg.sequence_length)

        anomaly_indices = self._select_anomaly_indices(cfg.num_samples)
        labels = np.ones(cfg.num_samples, dtype=np.float32)
        labels[anomaly_indices] = 0.0
        if anomaly_indices.size:
            asset_b_tokens[anomaly_indices] = self._independent_breakdown_windows(
                count=anomaly_indices.size,
                sequence_length=cfg.sequence_length,
                d_model=cfg.d_model,
            )

        asset_a_prices, asset_b_prices = self._generate_close_prices(
            factor_innovations=factor_innovations,
            anomaly_indices=anomaly_indices,
        )

        self.logger.info(
            "Generated synthetic Kronos states: samples=%d, seq_len=%d, d_model=%d, "
            "lag_steps=%d, anomalies=%d (%.2f%%).",
            cfg.num_samples,
            cfg.sequence_length,
            cfg.d_model,
            cfg.lag_steps,
            anomaly_indices.size,
            100.0 * anomaly_indices.size / cfg.num_samples,
        )

        return SyntheticKronosDataset(
            asset_a_tokens=asset_a_tokens.astype(np.float32),
            asset_b_tokens=asset_b_tokens.astype(np.float32),
            labels=labels,
            asset_a_prices=asset_a_prices.astype(np.float64),
            asset_b_prices=asset_b_prices.astype(np.float64),
            anomaly_indices=anomaly_indices,
        )

    def _factor_loadings(self, factor_count: int, d_model: int) -> np.ndarray:
        loadings = self.rng.normal(loc=0.0, scale=1.0, size=(factor_count, d_model))
        row_norms = np.linalg.norm(loadings, axis=1, keepdims=True)
        return loadings / np.maximum(row_norms, 1e-12)

    def _correlated_factors(self, steps: int, factor_count: int) -> np.ndarray:
        factor_index = np.arange(factor_count)
        covariance = 0.62 ** np.abs(factor_index[:, None] - factor_index[None, :])
        return self.rng.multivariate_normal(
            mean=np.zeros(factor_count),
            cov=covariance,
            size=steps,
        )

    @staticmethod
    def _zscore_2d(matrix: np.ndarray) -> np.ndarray:
        mean = matrix.mean(axis=0, keepdims=True)
        std = matrix.std(axis=0, keepdims=True)
        return (matrix - mean) / np.maximum(std, 1e-8)

    @staticmethod
    def _apply_temporal_shift_2d(matrix: np.ndarray, lag_steps: int) -> np.ndarray:
        """Apply an explicit lag matrix to a [time, d_model] state matrix."""
        time_steps = matrix.shape[0]
        lag_matrix = np.zeros((time_steps, time_steps), dtype=np.float64)
        source_indices = np.maximum(np.arange(time_steps) - lag_steps, 0)
        lag_matrix[np.arange(time_steps), source_indices] = 1.0
        return lag_matrix @ matrix

    @staticmethod
    def _apply_temporal_shift_1d(values: np.ndarray, lag_steps: int) -> np.ndarray:
        shifted = np.empty_like(values)
        if lag_steps == 0:
            shifted[:] = values
            return shifted
        shifted[:lag_steps] = values[0]
        shifted[lag_steps:] = values[:-lag_steps]
        return shifted

    @staticmethod
    def _rolling_windows(states: np.ndarray, sequence_length: int) -> np.ndarray:
        """Create [num_samples, seq_len, d_model] rolling token windows."""
        windows = np.lib.stride_tricks.sliding_window_view(
            states,
            window_shape=sequence_length,
            axis=0,
        )
        return np.moveaxis(windows, -1, 1).copy()

    def _select_anomaly_indices(self, num_samples: int) -> np.ndarray:
        anomaly_count = int(round(num_samples * self.config.deviation_fraction))
        if anomaly_count == 0:
            return np.array([], dtype=np.int64)
        indices = self.rng.choice(num_samples, size=anomaly_count, replace=False)
        return np.sort(indices.astype(np.int64))

    def _independent_breakdown_windows(
        self,
        count: int,
        sequence_length: int,
        d_model: int,
    ) -> np.ndarray:
        factor_count = min(8, d_model)
        loadings = self._factor_loadings(factor_count, d_model)
        factors = self.rng.normal(size=(count, sequence_length, factor_count))
        idiosyncratic = self.rng.normal(
            loc=0.0,
            scale=self.config.idiosyncratic_scale,
            size=(count, sequence_length, d_model),
        )
        innovations = self.config.innovation_scale * np.einsum(
            "bsf,fd->bsd",
            factors,
            loadings,
        ) + idiosyncratic
        states = np.cumsum(innovations, axis=1)
        mean = states.mean(axis=(1, 2), keepdims=True)
        std = states.std(axis=(1, 2), keepdims=True)
        return (states - mean) / np.maximum(std, 1e-8)

    def _generate_close_prices(
        self,
        factor_innovations: np.ndarray,
        anomaly_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        primary_factor = factor_innovations[:, 0]
        secondary_factor = factor_innovations[:, 1] if factor_innovations.shape[1] > 1 else 0.0
        leader_returns = cfg.price_return_scale * (
            0.80 * primary_factor + 0.20 * secondary_factor
        )
        leader_returns += self.rng.normal(
            loc=0.0,
            scale=cfg.price_noise_scale,
            size=leader_returns.shape[0],
        )

        lagged_returns = self._apply_temporal_shift_1d(leader_returns, cfg.lag_steps)
        lagged_returns += self.rng.normal(
            loc=0.0,
            scale=cfg.price_noise_scale * 0.85,
            size=lagged_returns.shape[0],
        )

        leader_prices = 100.0 * np.exp(np.cumsum(leader_returns))
        lagger_prices = 100.0 * np.exp(np.cumsum(lagged_returns))

        close_slice = slice(cfg.sequence_length - 1, None)
        asset_a_closes = leader_prices[close_slice].copy()
        asset_b_closes = lagger_prices[close_slice].copy()

        if anomaly_indices.size:
            breakdown_shocks = self.rng.normal(
                loc=0.0,
                scale=cfg.breakdown_price_shock_scale,
                size=anomaly_indices.size,
            )
            asset_b_closes[anomaly_indices] *= np.exp(breakdown_shocks)

        return asset_a_closes, asset_b_closes


__all__ = [
    "SyntheticKronosConfig",
    "SyntheticKronosDataEngine",
    "SyntheticKronosDataset",
]

