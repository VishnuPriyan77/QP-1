"""Neural alignment modules for cross-asset latent state adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CrossAttentionAdapterOutput:
    """Structured output from the cross-asset attention adapter."""

    aligned_a: Tensor
    """Asset A adapted hidden states with shape [batch, seq_len, d_model]."""

    aligned_b: Tensor
    """Asset B adapted hidden states with shape [batch, seq_len, d_model]."""

    alignment_vector: Tensor
    """Unified alignment vector with shape [batch, alignment_dim]."""

    distances: Tensor
    """Euclidean distance between temporal mean adapted states, shape [batch]."""


class CrossAttentionAdapter(nn.Module):
    """Bidirectional cross-attention adapter for frozen foundation-model states.

    The module accepts two parallel latent state tensors, each shaped
    [batch, seq_len, d_model]. Asset A attends over Asset B, and Asset B attends
    over Asset A. The attention outputs are wrapped in residual connections and
    layer normalization before a pooled pair representation is projected into a
    unified alignment vector.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        alignment_dim: int = 128,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if alignment_dim <= 0:
            raise ValueError("alignment_dim must be positive.")
        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("num_heads must be positive and divide d_model.")

        self.d_model = d_model
        self.alignment_dim = alignment_dim

        self.asset_a_queries_b = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.asset_b_queries_a = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.attention_dropout = nn.Dropout(dropout)
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_b = nn.LayerNorm(d_model)

        # Pair features: mean(A), mean(B), absolute dislocation, signed spread.
        projection_input_dim = d_model * 4
        self.alignment_projection = nn.Sequential(
            nn.Linear(projection_input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, alignment_dim),
        )

    def forward(
        self,
        asset_a_hidden: Tensor,
        asset_b_hidden: Tensor,
    ) -> CrossAttentionAdapterOutput:
        """Adapt and align two latent token sequences.

        Args:
            asset_a_hidden: Asset A latent tokens, [batch, seq_len, d_model].
            asset_b_hidden: Asset B latent tokens, [batch, seq_len, d_model].

        Returns:
            CrossAttentionAdapterOutput containing adapted sequences, a unified
            pair vector, and pairwise latent distances.
        """
        self._validate_inputs(asset_a_hidden, asset_b_hidden)

        # A queries B: [B, S, D] attends over [B, S, D] -> [B, S, D].
        attended_a, _ = self.asset_a_queries_b(
            query=asset_a_hidden,
            key=asset_b_hidden,
            value=asset_b_hidden,
            need_weights=False,
        )
        aligned_a = self.norm_a(asset_a_hidden + self.attention_dropout(attended_a))

        # B queries A: [B, S, D] attends over [B, S, D] -> [B, S, D].
        attended_b, _ = self.asset_b_queries_a(
            query=asset_b_hidden,
            key=asset_a_hidden,
            value=asset_a_hidden,
            need_weights=False,
        )
        aligned_b = self.norm_b(asset_b_hidden + self.attention_dropout(attended_b))

        # Temporal means compress token sequences into asset-level latent states.
        pooled_a = aligned_a.mean(dim=1)  # [B, D]
        pooled_b = aligned_b.mean(dim=1)  # [B, D]
        dislocation = torch.abs(pooled_a - pooled_b)  # [B, D]
        signed_spread = pooled_a - pooled_b  # [B, D]

        pair_features = torch.cat(
            (pooled_a, pooled_b, dislocation, signed_spread),
            dim=-1,
        )  # [B, 4D]
        alignment_vector = self.alignment_projection(pair_features)  # [B, A]
        distances = torch.linalg.vector_norm(pooled_a - pooled_b, ord=2, dim=-1)

        return CrossAttentionAdapterOutput(
            aligned_a=aligned_a,
            aligned_b=aligned_b,
            alignment_vector=alignment_vector,
            distances=distances,
        )

    def _validate_inputs(self, asset_a_hidden: Tensor, asset_b_hidden: Tensor) -> None:
        if asset_a_hidden.ndim != 3 or asset_b_hidden.ndim != 3:
            raise ValueError("Both hidden state tensors must have rank 3: [B, S, D].")
        if asset_a_hidden.shape != asset_b_hidden.shape:
            raise ValueError(
                "Asset hidden state tensors must have identical shapes. "
                f"Received {tuple(asset_a_hidden.shape)} and {tuple(asset_b_hidden.shape)}."
            )
        if asset_a_hidden.size(-1) != self.d_model:
            raise ValueError(
                f"Expected d_model={self.d_model}, received {asset_a_hidden.size(-1)}."
            )


class ContrastiveAlignmentLoss(nn.Module):
    """Contrastive loss for structural co-movement and regime breakdowns.

    For label 1, the loss minimizes Euclidean distance between temporal means of
    the aligned latent sequences. For label 0, it applies a hinge penalty when
    the distance sits below the configured margin.
    """

    def __init__(
        self,
        margin: float = 5.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if margin <= 0.0:
            raise ValueError("margin must be positive.")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of: 'mean', 'sum', 'none'.")
        self.margin = margin
        self.reduction = reduction
        self.eps = eps

    def forward(self, latent_a: Tensor, latent_b: Tensor, labels: Tensor) -> Tensor:
        """Compute label-aware pairwise contrastive alignment loss.

        Args:
            latent_a: Asset A latent states, [batch, seq_len, d_model] or [batch, d_model].
            latent_b: Asset B latent states, [batch, seq_len, d_model] or [batch, d_model].
            labels: Binary structural labels, [batch], where 1 is co-moving and 0
                is regime breakdown.
        """
        distances = self.pairwise_distance(latent_a, latent_b)  # [B]
        clean_labels = labels.float().view(-1).clamp(min=0.0, max=1.0)
        if clean_labels.numel() != distances.numel():
            raise ValueError(
                f"labels must have {distances.numel()} entries, received {clean_labels.numel()}."
            )

        positive_loss = clean_labels * distances.pow(2)
        negative_margin = F.relu(self.margin - distances)
        negative_loss = (1.0 - clean_labels) * negative_margin.pow(2)
        loss = positive_loss + negative_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def pairwise_distance(self, latent_a: Tensor, latent_b: Tensor) -> Tensor:
        """Return Euclidean distances between temporal mean latent states."""
        mean_a = self._temporal_mean(latent_a)
        mean_b = self._temporal_mean(latent_b)
        if mean_a.shape != mean_b.shape:
            raise ValueError(
                f"Latent means must share shape, got {tuple(mean_a.shape)} and {tuple(mean_b.shape)}."
            )
        squared_distance = (mean_a - mean_b).pow(2).sum(dim=-1)
        return torch.sqrt(squared_distance + self.eps)

    @staticmethod
    def _temporal_mean(latent: Tensor) -> Tensor:
        if latent.ndim == 3:
            return latent.mean(dim=1)  # [B, S, D] -> [B, D]
        if latent.ndim == 2:
            return latent
        raise ValueError("Latent tensors must be rank 2 or rank 3.")


__all__ = [
    "ContrastiveAlignmentLoss",
    "CrossAttentionAdapter",
    "CrossAttentionAdapterOutput",
]

