"""End-to-end execution for Cross-Asset Latent Alignment stat-arb research."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.backtester import BacktestConfig, VectorizedStatArbBacktester
from src.data_engine import SyntheticKronosConfig, SyntheticKronosDataEngine, SyntheticKronosDataset
from src.model import ContrastiveAlignmentLoss, CrossAttentionAdapter


LOGGER = logging.getLogger("cross_asset_latent_alignment")


@dataclass(frozen=True)
class TrainingConfig:
    """Training parameters for the cross-attention adapter."""

    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    train_fraction: float = 0.70


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(
    dataset: SyntheticKronosDataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    asset_a = torch.from_numpy(dataset.asset_a_tokens).float()
    asset_b = torch.from_numpy(dataset.asset_b_tokens).float()
    labels = torch.from_numpy(dataset.labels).float()
    tensor_dataset = TensorDataset(asset_a, asset_b, labels)
    return DataLoader(
        tensor_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def train_adapter(
    model: CrossAttentionAdapter,
    criterion: ContrastiveAlignmentLoss,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
) -> None:
    model.train()
    for epoch in range(1, config.epochs + 1):
        epoch_losses: list[float] = []
        for asset_a, asset_b, labels in train_loader:
            # Input tensors: [batch, seq_len, d_model].
            asset_a = asset_a.to(device)
            asset_b = asset_b.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            output = model(asset_a, asset_b)
            loss = criterion(output.aligned_a, output.aligned_b, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        LOGGER.info(
            "Epoch %02d/%02d | contrastive_alignment_loss=%.6f",
            epoch,
            config.epochs,
            float(np.mean(epoch_losses)),
        )


@torch.no_grad()
def evaluate_alignment(
    model: CrossAttentionAdapter,
    dataset: SyntheticKronosDataset,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    distances: list[np.ndarray] = []
    alignment_vectors: list[np.ndarray] = []
    for asset_a, asset_b, _ in loader:
        asset_a = asset_a.to(device)
        asset_b = asset_b.to(device)
        output = model(asset_a, asset_b)
        distances.append(output.distances.detach().cpu().numpy())
        alignment_vectors.append(output.alignment_vector.detach().cpu().numpy())

    distance_profile = np.concatenate(distances, axis=0)
    alignment_matrix = np.concatenate(alignment_vectors, axis=0)
    return distance_profile, alignment_matrix


def log_dataset_summary(
    train_data: SyntheticKronosDataset,
    test_data: SyntheticKronosDataset,
) -> None:
    train_anomalies = int((train_data.labels == 0.0).sum())
    test_anomalies = int((test_data.labels == 0.0).sum())
    LOGGER.info(
        "Train split: samples=%d, anomalies=%d (%.2f%%).",
        train_data.num_samples,
        train_anomalies,
        100.0 * train_anomalies / train_data.num_samples,
    )
    LOGGER.info(
        "Test split: samples=%d, anomalies=%d (%.2f%%).",
        test_data.num_samples,
        test_anomalies,
        100.0 * test_anomalies / test_data.num_samples,
    )


def main() -> None:
    configure_logging()
    torch.manual_seed(42)
    np.random.seed(42)

    data_config = SyntheticKronosConfig(
        num_samples=1_200,
        sequence_length=30,
        d_model=256,
        lag_steps=2,
        deviation_fraction=0.15,
        seed=42,
    )
    training_config = TrainingConfig()
    device = choose_device()
    LOGGER.info("Using device: %s", device)

    data_engine = SyntheticKronosDataEngine(data_config, logger=LOGGER)
    full_dataset = data_engine.generate()
    train_data, test_data = full_dataset.train_test_split(training_config.train_fraction)
    log_dataset_summary(train_data, test_data)

    train_loader = make_loader(
        train_data,
        batch_size=training_config.batch_size,
        shuffle=True,
    )
    model = CrossAttentionAdapter(
        d_model=data_config.d_model,
        num_heads=8,
        alignment_dim=128,
        dropout=0.10,
    ).to(device)
    criterion = ContrastiveAlignmentLoss(margin=5.0, reduction="mean")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    LOGGER.info("Starting adapter optimization for %d epochs.", training_config.epochs)
    train_adapter(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        config=training_config,
        device=device,
    )

    distances, alignment_vectors = evaluate_alignment(
        model=model,
        dataset=test_data,
        batch_size=training_config.batch_size,
        device=device,
    )
    LOGGER.info(
        "Out-of-sample alignment generated: distances_shape=%s, alignment_vectors_shape=%s.",
        distances.shape,
        alignment_vectors.shape,
    )
    LOGGER.info(
        "Distance profile summary: mean=%.4f, std=%.4f, min=%.4f, max=%.4f.",
        float(distances.mean()),
        float(distances.std()),
        float(distances.min()),
        float(distances.max()),
    )

    backtester = VectorizedStatArbBacktester(
        BacktestConfig(
            initial_capital=1_000_000.0,
            threshold_window=40,
            entry_sigma=1.25,
            spread_window=40,
            max_position_leverage=1.0,
            transaction_cost_bps=1.0,
            annualization_factor=252,
        ),
        logger=LOGGER,
    )
    result = backtester.run(
        distances=distances,
        asset_a_prices=test_data.asset_a_prices,
        asset_b_prices=test_data.asset_b_prices,
    )

    LOGGER.info("Final strategy metrics:")
    for key, value in result.metrics.items():
        LOGGER.info("  %-24s %.6f", key + ":", value)

    if not result.trade_log.empty:
        LOGGER.info("Last five trades:\n%s", result.trade_log.tail().to_string(index=False))
    else:
        LOGGER.info("No trades were triggered by the configured distance thresholds.")


if __name__ == "__main__":
    main()

