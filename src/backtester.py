"""Vectorized statistical arbitrage backtester driven by latent distances."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestConfig:
    """Execution and risk parameters for the distance-triggered stat-arb engine."""

    initial_capital: float = 1_000_000.0
    threshold_window: int = 40
    entry_sigma: float = 1.25
    spread_window: int = 40
    max_position_leverage: float = 1.0
    transaction_cost_bps: float = 1.0
    annualization_factor: int = 252

    def validate(self) -> None:
        if self.initial_capital <= 0.0:
            raise ValueError("initial_capital must be positive.")
        if self.threshold_window < 2:
            raise ValueError("threshold_window must be at least 2.")
        if self.spread_window < 2:
            raise ValueError("spread_window must be at least 2.")
        if self.entry_sigma <= 0.0:
            raise ValueError("entry_sigma must be positive.")
        if self.max_position_leverage <= 0.0:
            raise ValueError("max_position_leverage must be positive.")
        if self.transaction_cost_bps < 0.0:
            raise ValueError("transaction_cost_bps cannot be negative.")


@dataclass(frozen=True)
class BacktestResult:
    """Backtest outputs and diagnostics."""

    metrics: Dict[str, float]
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    trade_log: pd.DataFrame
    signal_frame: pd.DataFrame


class VectorizedStatArbBacktester:
    """Distance-gated market-neutral spread execution engine.

    Latent distances drive entry and exit timing. The direction of the spread is
    determined by the sign of the log-price spread relative to its trailing mean:
    positive spread deviations short the spread, while negative deviations long
    the spread.
    """

    def __init__(
        self,
        config: BacktestConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config or BacktestConfig()
        self.config.validate()
        self.logger = logger or LOGGER

    def run(
        self,
        distances: np.ndarray,
        asset_a_prices: np.ndarray,
        asset_b_prices: np.ndarray,
    ) -> BacktestResult:
        distance_series = self._as_series(distances, "distance")
        asset_a = self._as_series(asset_a_prices, "asset_a_price")
        asset_b = self._as_series(asset_b_prices, "asset_b_price")
        self._validate_inputs(distance_series, asset_a, asset_b)

        signal_frame = self._build_signal_frame(distance_series, asset_a, asset_b)
        decision_positions = self._resolve_positions(signal_frame)
        effective_positions = decision_positions.shift(1).fillna(0.0)

        asset_a_returns = asset_a.pct_change().fillna(0.0)
        asset_b_returns = asset_b.pct_change().fillna(0.0)
        spread_returns = 0.5 * (asset_a_returns - asset_b_returns)
        gross_returns = (
            self.config.max_position_leverage
            * effective_positions
            * spread_returns
        )

        position_turnover = effective_positions.diff().abs().fillna(
            effective_positions.abs()
        )
        trading_cost = position_turnover * self.config.transaction_cost_bps / 10_000.0
        net_returns = gross_returns - trading_cost
        equity_curve = self.config.initial_capital * (1.0 + net_returns).cumprod()

        drawdowns = equity_curve / equity_curve.cummax() - 1.0
        metrics = self._calculate_metrics(
            returns=net_returns,
            equity_curve=equity_curve,
            drawdowns=drawdowns,
            positions=decision_positions,
        )
        trade_log = self._build_trade_log(
            positions=decision_positions,
            equity_curve=equity_curve,
            distance_series=distance_series,
        )
        if not trade_log.empty:
            metrics["win_rate_pct"] = 100.0 * float((trade_log["trade_return"] > 0.0).mean())
        else:
            metrics["win_rate_pct"] = 0.0

        signal_frame = signal_frame.assign(
            position=decision_positions,
            effective_position=effective_positions,
            spread_return=spread_returns,
            net_return=net_returns,
            equity=equity_curve,
            drawdown=drawdowns,
        )

        self.logger.info(
            "Backtest complete: terminal_capital=%.2f, total_return=%.2f%%, "
            "max_drawdown=%.2f%%, sharpe=%.3f, trades=%d.",
            metrics["terminal_capital"],
            metrics["total_return_pct"],
            metrics["max_drawdown_pct"],
            metrics["annualized_sharpe"],
            int(metrics["num_trades"]),
        )

        return BacktestResult(
            metrics=metrics,
            equity_curve=equity_curve.rename("equity"),
            returns=net_returns.rename("net_return"),
            positions=decision_positions.rename("position"),
            trade_log=trade_log,
            signal_frame=signal_frame,
        )

    def _build_signal_frame(
        self,
        distance_series: pd.Series,
        asset_a: pd.Series,
        asset_b: pd.Series,
    ) -> pd.DataFrame:
        cfg = self.config
        min_distance_periods = max(5, cfg.threshold_window // 4)
        distance_mean = (
            distance_series.rolling(cfg.threshold_window, min_periods=min_distance_periods)
            .mean()
            .shift(1)
        )
        distance_std = (
            distance_series.rolling(cfg.threshold_window, min_periods=min_distance_periods)
            .std(ddof=0)
            .shift(1)
            .fillna(0.0)
        )
        entry_threshold = distance_mean + cfg.entry_sigma * distance_std
        exit_threshold = distance_mean

        log_spread = np.log(asset_a) - np.log(asset_b)
        spread_mean = (
            log_spread.rolling(cfg.spread_window, min_periods=max(5, cfg.spread_window // 4))
            .mean()
            .shift(1)
        )
        spread_deviation = log_spread - spread_mean
        direction = -np.sign(spread_deviation).replace(0.0, np.nan).fillna(0.0)

        entry_signal = (
            distance_series.gt(entry_threshold)
            & direction.ne(0.0)
            & entry_threshold.notna()
        )
        exit_signal = distance_series.lt(exit_threshold) & exit_threshold.notna()

        return pd.DataFrame(
            {
                "distance": distance_series,
                "asset_a_price": asset_a,
                "asset_b_price": asset_b,
                "log_spread": log_spread,
                "spread_deviation": spread_deviation,
                "direction": direction,
                "entry_threshold": entry_threshold,
                "exit_threshold": exit_threshold,
                "entry_signal": entry_signal,
                "exit_signal": exit_signal,
            }
        )

    @staticmethod
    def _resolve_positions(signal_frame: pd.DataFrame) -> pd.Series:
        positions = np.zeros(len(signal_frame), dtype=np.float64)
        current_position = 0.0
        entry_signal = signal_frame["entry_signal"].to_numpy(dtype=bool)
        exit_signal = signal_frame["exit_signal"].to_numpy(dtype=bool)
        directions = signal_frame["direction"].to_numpy(dtype=np.float64)

        for index in range(len(signal_frame)):
            if current_position != 0.0 and exit_signal[index]:
                current_position = 0.0
            elif current_position == 0.0 and entry_signal[index]:
                current_position = directions[index]
            positions[index] = current_position

        return pd.Series(positions, index=signal_frame.index, name="position")

    def _calculate_metrics(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
        drawdowns: pd.Series,
        positions: pd.Series,
    ) -> Dict[str, float]:
        terminal_capital = float(equity_curve.iloc[-1])
        total_return_pct = 100.0 * (terminal_capital / self.config.initial_capital - 1.0)
        max_drawdown_pct = 100.0 * float(drawdowns.min())
        return_std = float(returns.std(ddof=0))
        if return_std > 1e-12:
            annualized_sharpe = (
                float(returns.mean())
                / return_std
                * np.sqrt(self.config.annualization_factor)
            )
        else:
            annualized_sharpe = 0.0
        num_trades = int(((positions != 0.0) & (positions.shift(1).fillna(0.0) == 0.0)).sum())
        exposure_pct = 100.0 * float((positions != 0.0).mean())

        return {
            "terminal_capital": terminal_capital,
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "annualized_sharpe": float(annualized_sharpe),
            "num_trades": float(num_trades),
            "exposure_pct": exposure_pct,
        }

    @staticmethod
    def _build_trade_log(
        positions: pd.Series,
        equity_curve: pd.Series,
        distance_series: pd.Series,
    ) -> pd.DataFrame:
        previous_positions = positions.shift(1).fillna(0.0)
        entry_indices = positions[(positions != 0.0) & (previous_positions == 0.0)].index
        rows = []
        for entry_index in entry_indices:
            entry_position = float(positions.loc[entry_index])
            exit_candidates = positions.loc[entry_index:][positions.loc[entry_index:] == 0.0]
            if len(exit_candidates) > 0:
                exit_index = int(exit_candidates.index[0])
            else:
                exit_index = int(positions.index[-1])
            if exit_index <= entry_index:
                exit_index = min(int(entry_index) + 1, int(positions.index[-1]))
            entry_equity = float(equity_curve.loc[entry_index])
            exit_equity = float(equity_curve.loc[exit_index])
            rows.append(
                {
                    "entry_index": int(entry_index),
                    "exit_index": exit_index,
                    "side": "long_spread" if entry_position > 0.0 else "short_spread",
                    "entry_distance": float(distance_series.loc[entry_index]),
                    "exit_distance": float(distance_series.loc[exit_index]),
                    "trade_return": exit_equity / entry_equity - 1.0,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _as_series(values: np.ndarray, name: str) -> pd.Series:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        return pd.Series(array, name=name)

    @staticmethod
    def _validate_inputs(
        distances: pd.Series,
        asset_a_prices: pd.Series,
        asset_b_prices: pd.Series,
    ) -> None:
        lengths = {len(distances), len(asset_a_prices), len(asset_b_prices)}
        if len(lengths) != 1:
            raise ValueError("distances and price arrays must have matching lengths.")
        if len(distances) < 10:
            raise ValueError("At least 10 observations are required for backtesting.")
        if not np.isfinite(distances.to_numpy()).all():
            raise ValueError("distances contain NaN or infinite values.")
        if (asset_a_prices <= 0.0).any() or (asset_b_prices <= 0.0).any():
            raise ValueError("asset prices must be strictly positive.")


__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "VectorizedStatArbBacktester",
]

