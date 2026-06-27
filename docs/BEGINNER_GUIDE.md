# Beginner Guide

This guide explains the project from first principles. It assumes you are familiar with basic Python but new to quantitative finance, PyTorch, or statistical arbitrage.

## 1. Big Picture

Financial assets often move together because they are exposed to similar forces. For example, two stocks in the same sector may react to the same macro news, earnings themes, interest-rate moves, or risk sentiment.

Statistical arbitrage tries to exploit temporary dislocations between related assets. If two assets usually move together and then suddenly diverge, a strategy may bet that the relationship will normalize.

This project studies that idea in latent space.

Instead of comparing only prices, the project compares hidden time-series representations. These hidden representations are synthetic stand-ins for what a foundation model like Kronos might produce from market candles or ticks.

## 2. What Is A Latent Vector?

A latent vector is a numerical representation learned or produced by a model. In this project, each asset has a sequence of latent vectors:

```text
[batch, sequence_length, d_model]
```

That means:

- `batch`: how many samples are processed at once.
- `sequence_length`: how many time steps are in each sample window.
- `d_model`: how many hidden features describe each time step.

The default shape is:

```text
[batch, 30, 256]
```

So each sample contains 30 time steps, and each time step has 256 hidden features.

## 3. Why Cross-Attention?

Attention lets one sequence look at another sequence and decide which time steps matter.

This project uses cross-attention in both directions:

```text
Asset A queries Asset B
Asset B queries Asset A
```

This is useful because one asset may lead the other. If Asset A tends to move first and Asset B reacts two steps later, cross-attention can learn that temporal relationship better than a simple same-time comparison.

## 4. The Neural Network

The neural core is `CrossAttentionAdapter` in `src/model.py`.

It does four main things:

1. Receives two latent tensors:

   ```text
   asset_a_hidden: [batch, sequence_length, d_model]
   asset_b_hidden: [batch, sequence_length, d_model]
   ```

2. Applies bidirectional cross-attention:

   ```text
   A attends to B
   B attends to A
   ```

3. Adds residual connections and layer normalization:

   ```text
   aligned_a = LayerNorm(asset_a + attention_output_a)
   aligned_b = LayerNorm(asset_b + attention_output_b)
   ```

4. Pools the temporal dimension and projects the pair into an alignment vector:

   ```text
   [batch, sequence_length, d_model] -> [batch, d_model] -> [batch, alignment_dim]
   ```

The default alignment dimension is 128.

## 5. The Loss Function

The loss is `ContrastiveAlignmentLoss` in `src/model.py`.

Each sample has a label:

```text
1 = the assets are structurally co-moving
0 = the relationship has broken down
```

For label `1`, the model is rewarded for making the two latent means close together.

For label `0`, the model is penalized if the two latent means are too close. This uses a margin. If the distance is already larger than the margin, the model does not need to push it farther.

Conceptually:

```text
co-moving pair: minimize distance
broken pair:    keep distance above margin
```

## 6. Synthetic Data Generation

The data engine is in `src/data_engine.py`.

Real foundation-model market tokens are proprietary, so this project creates a synthetic but controlled substitute.

The generator creates:

- A leader asset.
- A lagging asset.
- Labels showing whether the relationship is normal or broken.
- Synthetic prices for backtesting.

The leader is generated from correlated Gaussian random walks. This creates smooth, structured latent states rather than unrelated random noise.

The lagging asset is created by shifting the leader through time, then adding a small amount of white noise. With the default settings, the lag is 2 steps.

Then the engine injects regime deviations into exactly 15 percent of samples. These samples are assigned label `0`.

## 7. Training Flow

Training happens in `main.py`.

The main steps are:

1. Create synthetic data.
2. Split the data chronologically into train and test sets.
3. Create the PyTorch model.
4. Create the contrastive loss.
5. Optimize with Adam for 10 epochs.

During training, the model learns to produce smaller latent distances for co-moving pairs and larger distances for broken-regime pairs.

## 8. Backtesting Logic

The backtester is in `src/backtester.py`.

The model outputs a distance for each test sample. A high distance means the two assets look unusually far apart in latent space.

The strategy logic is:

1. Compute a trailing distance mean and standard deviation.
2. Enter a spread trade when distance exceeds:

   ```text
   trailing_mean + entry_sigma * trailing_std
   ```

3. Determine trade direction from the price spread:

   ```text
   log(asset_a_price) - log(asset_b_price)
   ```

4. Exit when distance contracts below its trailing mean.

The position is market-neutral in spirit:

- Long spread means long Asset A and short Asset B.
- Short spread means short Asset A and long Asset B.

## 9. Performance Metrics

The backtester reports:

- `terminal_capital`: final account value.
- `total_return_pct`: percentage return over the test period.
- `max_drawdown_pct`: worst peak-to-trough equity decline.
- `annualized_sharpe`: risk-adjusted return estimate.
- `num_trades`: number of spread entries.
- `exposure_pct`: percentage of time spent in a position.
- `win_rate_pct`: percentage of trades with positive return.

## 10. Important Caveats

This is a research scaffold, not a production trading system.

The current data is synthetic. Strong backtest numbers on synthetic data do not prove the method works on real markets.

A production version would need:

- Real market data ingestion.
- Proper train, validation, and test periods.
- Slippage and transaction-cost calibration.
- Borrow fees and shorting constraints.
- Risk limits.
- Position sizing.
- Outlier handling.
- Model checkpointing.
- Experiment tracking.
- Unit and integration tests.

## 11. How To Run The Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python main.py
```

You should see training logs followed by final backtest metrics.

## 12. Suggested Learning Path

If you are new to this topic, read the files in this order:

1. `README.md`
2. `src/data_engine.py`
3. `src/model.py`
4. `src/backtester.py`
5. `main.py`

This order starts with the data, then moves to the neural model, then to the trading simulation, then to the orchestration layer.

