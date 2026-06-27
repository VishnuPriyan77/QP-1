# Cross-Asset Latent Alignment: Modifying Kronos for Multi-Asset Statistical Arbitrage

This project is an end-to-end educational research prototype for cross-asset statistical arbitrage using deep latent representations. It simulates the kind of hidden token states that could come from a frozen foundation time-series model such as Kronos, learns a cross-asset alignment adapter with PyTorch, and sends the resulting latent distance profile into a vectorized stat-arb backtester.

The current implementation is fully runnable in this repository and has been verified with:

```bash
python -m compileall main.py src
python main.py
```

Important note: no software project can be guaranteed to run "without any errors whatsoever" on every machine. Dependency versions, Python versions, hardware acceleration backends, and local environment settings can differ. What this repository does provide is a complete, structured, tested-in-place baseline that runs end to end in the current workspace.

## What This Project Does

The project answers a practical research question:

> Can two assets be compared in a learned latent space so that abnormal divergence in their hidden states becomes a useful statistical arbitrage signal?

The workflow is:

1. Generate synthetic foundation-model token states for a leader asset.
2. Generate a lagging asset by applying a temporal delay plus noise.
3. Inject structural breakdown regimes into exactly 15 percent of the dataset.
4. Train a PyTorch cross-attention adapter to align the two assets.
5. Use the pairwise latent distance as a market-neutral spread signal.
6. Backtest entries, exits, equity, drawdown, Sharpe ratio, and trade outcomes.

This is for education and research only. It is not financial advice and should not be used for live trading without real data validation, transaction-cost modeling, risk controls, and compliance review.

## Repository Structure

```text
.
├── main.py
├── pyproject.toml
├── requirements.txt
├── README.md
├── docs
│   └── BEGINNER_GUIDE.md
└── src
    ├── __init__.py
    ├── backtester.py
    ├── data_engine.py
    └── model.py
```

## Core Modules

### `src/model.py`

Contains the neural network code:

- `CrossAttentionAdapter`
  - Accepts two tensors shaped `[batch, sequence_length, d_model]`.
  - Uses bidirectional `nn.MultiheadAttention`.
  - Asset A queries Asset B.
  - Asset B queries Asset A.
  - Applies residual connections and `nn.LayerNorm`.
  - Produces adapted latent states and a unified alignment vector.

- `ContrastiveAlignmentLoss`
  - Pulls co-moving pairs together when label is `1`.
  - Pushes broken-regime pairs apart when label is `0`.
  - Uses a configurable margin-based hinge penalty.

### `src/data_engine.py`

Contains the synthetic data generator:

- Generates correlated Gaussian random-walk latent states.
- Creates a leader asset and a lagged follower asset.
- Adds Gaussian white noise to mimic imperfect lead-lag behavior.
- Injects structural regime deviations into exactly 15 percent of observations.
- Returns token tensors, labels, anomaly indices, and synthetic close prices.

### `src/backtester.py`

Contains the vectorized stat-arb execution engine:

- Uses latent distances as divergence signals.
- Enters spread positions when distance exceeds a trailing threshold.
- Exits when distance contracts below a trailing historical mean.
- Calculates terminal capital, total return, drawdown, Sharpe ratio, exposure, trade count, and win rate.

### `main.py`

Runs the complete system:

- Configures logging.
- Builds synthetic train and test splits.
- Trains the adapter for 10 epochs.
- Evaluates out-of-sample latent distances.
- Runs the backtest.
- Prints final strategy metrics.

## Tech Stack

- Python 3.10 or newer
- PyTorch for neural network modeling and training
- NumPy for numerical simulation
- Pandas for vectorized signal and backtest calculations

The project intentionally avoids proprietary market data and proprietary Kronos internals. Instead, it builds a rigorous synthetic stand-in for latent token states.

## Quick Start

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

## Expected Console Output

You should see logs similar to:

```text
Using device: cpu
Generated synthetic Kronos states: samples=1200, seq_len=30, d_model=256, lag_steps=2, anomalies=180 (15.00%).
Starting adapter optimization for 10 epochs.
Epoch 01/10 | contrastive_alignment_loss=...
...
Out-of-sample alignment generated: distances_shape=(360,), alignment_vectors_shape=(360, 128).
Backtest complete: terminal_capital=..., total_return=...%, max_drawdown=...%, sharpe=..., trades=...
```

Exact results can vary slightly by hardware, PyTorch backend, and random-number behavior.

## How To Modify Experiments

Most research parameters are configured in `main.py`:

- `SyntheticKronosConfig`
  - `num_samples`
  - `sequence_length`
  - `d_model`
  - `lag_steps`
  - `deviation_fraction`

- `TrainingConfig`
  - `epochs`
  - `batch_size`
  - `learning_rate`
  - `train_fraction`

- `BacktestConfig`
  - `initial_capital`
  - `threshold_window`
  - `entry_sigma`
  - `spread_window`
  - `transaction_cost_bps`

## GitHub Readiness

This repository is structured in a GitHub-friendly way:

- Modular source code under `src/`
- Runnable entry point in `main.py`
- Dependency list in `requirements.txt`
- Project metadata in `pyproject.toml`
- Beginner documentation in `docs/BEGINNER_GUIDE.md`
- Ignore rules in `.gitignore`

Before publishing, consider adding:

- A license file, such as MIT or Apache-2.0, once you choose the licensing terms.
- Unit tests if this evolves from educational research into a maintained package.
- Real-data adapters if you want to connect it to production market data.

## Current Verification Snapshot

In the current workspace, the project ran successfully with:

- Python `3.13.11`
- PyTorch `2.11.0`
- NumPy `2.4.4`
- Pandas `2.2.2`

The latest run produced:

- Training loss improvement from about `11.59` to `0.19`
- Out-of-sample distance profile shape: `(360,)`
- Alignment vector matrix shape: `(360, 128)`
- Terminal capital around `$1.90M` from `$1.00M`
- Annualized Sharpe around `4.25`

These numbers are synthetic-research results, not live trading evidence.

