"""
Compact model definitions for the reduced biologically informed workflow.

Only the three model families that remain central to the current work are kept:

- gene_only: SNP -> gene masked MLP
- gene_pathway: SNP -> gene -> pathway masked MLP
- attention_gene: impact-aware SNP attention followed by the gene MLP
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix


def _get_activation_module(activation: str) -> nn.Module:
    activation = activation.lower()
    if activation == "relu":
        return nn.ReLU()
    if activation == "sigmoid":
        return nn.Sigmoid()
    if activation == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {activation}")


def _init_dense(layer: nn.Linear, activation: str, final: bool = False) -> None:
    activation = activation.lower()
    if final:
        nn.init.xavier_normal_(layer.weight)
    elif activation in {"relu", "gelu"}:
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
    elif activation == "sigmoid":
        nn.init.xavier_normal_(layer.weight)
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    nn.init.zeros_(layer.bias)


class MaskedLinear(nn.Module):
    """Linear layer whose connectivity is constrained by a fixed binary mask."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask: torch.Tensor,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters(activation)

    def reset_parameters(self, activation: str) -> None:
        fan_in = torch.clamp(self.mask.sum(dim=1), min=1)
        if activation.lower() == "relu":
            std = torch.sqrt(2.0 / fan_in)
        else:
            std = torch.sqrt(1.0 / fan_in)

        with torch.no_grad():
            for row_index in range(self.weight.shape[0]):
                self.weight[row_index].normal_(0, std[row_index])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)


class MaskedBiologicalMLP(nn.Module):
    """Masked SNP -> gene -> [pathway] -> dense head architecture."""

    def __init__(
        self,
        input_dim: int,
        gene_dim: int,
        fc_layers: list[int],
        mask_snp_gene: torch.Tensor,
        pathway_dim: int = 0,
        mask_gene_pathway: torch.Tensor | None = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()

        self.activation_name = activation
        self.activation_fn = _get_activation_module(activation)
        self.has_pathway_layer = pathway_dim > 0 and mask_gene_pathway is not None

        self.masked1 = MaskedLinear(
            in_features=input_dim,
            out_features=gene_dim,
            mask=mask_snp_gene,
            activation=activation,
        )
        self.norm1 = nn.LayerNorm(gene_dim) if use_layernorm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if self.has_pathway_layer:
            self.masked2 = MaskedLinear(
                in_features=gene_dim,
                out_features=pathway_dim,
                mask=mask_gene_pathway,
                activation=activation,
            )
            self.norm2 = nn.LayerNorm(pathway_dim) if use_layernorm else nn.Identity()
            self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            final_input_dim = pathway_dim
        else:
            self.masked2 = None
            self.norm2 = None
            self.dropout2 = None
            final_input_dim = gene_dim

        dense_layers: list[nn.Module] = []
        previous_dim = final_input_dim
        for layer_dim in fc_layers:
            linear = nn.Linear(previous_dim, layer_dim)
            _init_dense(linear, activation)
            dense_layers.append(linear)
            dense_layers.append(
                nn.LayerNorm(layer_dim) if use_layernorm else nn.Identity()
            )
            dense_layers.append(_get_activation_module(activation))
            if dropout > 0:
                dense_layers.append(nn.Dropout(dropout))
            previous_dim = layer_dim

        final = nn.Linear(previous_dim, 1)
        _init_dense(final, activation, final=True)
        dense_layers.append(final)
        self.fc_stack = nn.Sequential(*dense_layers)

    def encode_genes(self, x: torch.Tensor) -> torch.Tensor:
        x = self.masked1(x)
        x = self.norm1(x)
        x = self.activation_fn(x)
        x = self.dropout1(x)
        return x

    def encode_pathways_from_genes(self, gene_features: torch.Tensor) -> torch.Tensor:
        if not self.has_pathway_layer:
            return gene_features
        x = self.masked2(gene_features)
        x = self.norm2(x)
        x = self.activation_fn(x)
        x = self.dropout2(x)
        return x

    def encode_pathways(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_pathways_from_genes(self.encode_genes(x))

    def forward_from_gene_features(self, gene_features: torch.Tensor) -> torch.Tensor:
        return self.fc_stack(self.encode_pathways_from_genes(gene_features))

    def forward_from_pathway_features(
        self,
        pathway_features: torch.Tensor,
    ) -> torch.Tensor:
        if not self.has_pathway_layer:
            raise ValueError("This model does not have a pathway layer.")
        return self.fc_stack(pathway_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_stack(self.encode_pathways(x))


class SNPImpactAttention(nn.Module):
    """Impact-conditioned multiplicative SNP gating."""

    def __init__(
        self,
        num_snps: int,
        num_impacts: int,
        impact_indices: torch.Tensor,
        embedding_dim: int = 16,
        activation: str = "gelu",
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if impact_indices.numel() != num_snps:
            raise ValueError(
                "impact_indices must contain exactly one label per SNP "
                f"(expected {num_snps}, got {impact_indices.numel()})."
            )

        self.register_buffer("impact_indices", impact_indices.long())
        self.activation_fn = _get_activation_module(activation)

        self.impact_embedding = nn.Embedding(num_impacts, embedding_dim)
        self.impact_projection = nn.Linear(embedding_dim, embedding_dim)
        self.impact_norm = nn.LayerNorm(embedding_dim)
        self.scale_head = nn.Linear(embedding_dim, 1)
        self.bias_head = nn.Linear(embedding_dim, 1)
        self.output_dropout = (
            nn.Dropout(attention_dropout) if attention_dropout > 0 else nn.Identity()
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.impact_embedding.weight, mean=0.0, std=0.02)
        nn.init.xavier_normal_(self.impact_projection.weight)
        nn.init.zeros_(self.impact_projection.bias)
        nn.init.normal_(self.scale_head.weight, mean=0.0, std=5e-3)
        nn.init.zeros_(self.scale_head.bias)
        nn.init.normal_(self.bias_head.weight, mean=0.0, std=5e-3)
        nn.init.zeros_(self.bias_head.bias)

    def _impact_hidden(self) -> torch.Tensor:
        hidden = self.impact_embedding(self.impact_indices)
        hidden = self.impact_projection(hidden)
        hidden = self.impact_norm(hidden)
        hidden = self.activation_fn(hidden)
        return hidden

    def get_attention_logits(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self._impact_hidden()
        scale = self.scale_head(hidden).squeeze(-1)
        bias = self.bias_head(hidden).squeeze(-1)
        return x * scale + bias

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * torch.sigmoid(self.get_attention_logits(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_dropout(x * self.get_attention_weights(x))


class AttentionGeneMLP(nn.Module):
    """Impact-attention SNP encoder followed by the gene masked MLP head."""

    def __init__(
        self,
        input_dim: int,
        gene_dim: int,
        fc_layers: list[int],
        mask_snp_gene: torch.Tensor,
        impact_indices: torch.Tensor,
        num_impacts: int,
        activation: str = "gelu",
        dropout: float = 0.0,
        use_layernorm: bool = True,
        impact_embedding_dim: int = 16,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.activation_name = activation
        self.activation_fn = _get_activation_module(activation)
        self.has_pathway_layer = False

        self.snp_attention = SNPImpactAttention(
            num_snps=input_dim,
            num_impacts=num_impacts,
            impact_indices=impact_indices,
            embedding_dim=impact_embedding_dim,
            activation=activation,
            attention_dropout=attention_dropout,
        )

        self.masked1 = MaskedLinear(
            in_features=input_dim,
            out_features=gene_dim,
            mask=mask_snp_gene,
            activation=activation,
        )
        self.norm1 = nn.LayerNorm(gene_dim) if use_layernorm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        dense_layers: list[nn.Module] = []
        previous_dim = gene_dim
        for layer_dim in fc_layers:
            linear = nn.Linear(previous_dim, layer_dim)
            _init_dense(linear, activation)
            dense_layers.append(linear)
            dense_layers.append(
                nn.LayerNorm(layer_dim) if use_layernorm else nn.Identity()
            )
            dense_layers.append(_get_activation_module(activation))
            if dropout > 0:
                dense_layers.append(nn.Dropout(dropout))
            previous_dim = layer_dim

        final = nn.Linear(previous_dim, 1)
        _init_dense(final, activation, final=True)
        dense_layers.append(final)
        self.fc_stack = nn.Sequential(*dense_layers)

    def encode_snps(self, x: torch.Tensor) -> torch.Tensor:
        return self.snp_attention(x)

    def get_attention_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.snp_attention.get_attention_logits(x)

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        return self.snp_attention.get_attention_weights(x)

    def encode_genes(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode_snps(x)
        x = self.masked1(x)
        x = self.norm1(x)
        x = self.activation_fn(x)
        x = self.dropout1(x)
        return x

    def forward_from_gene_features(self, gene_features: torch.Tensor) -> torch.Tensor:
        return self.fc_stack(gene_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_stack(self.encode_genes(x))


class GeneToOutputWrapper(nn.Module):
    """Small helper used for integrated gradients in gene space."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, gene_features: torch.Tensor) -> torch.Tensor:
        return self.model.forward_from_gene_features(gene_features)


class PathwayToOutputWrapper(nn.Module):
    """Small helper used for integrated gradients in pathway space."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pathway_features: torch.Tensor) -> torch.Tensor:
        return self.model.forward_from_pathway_features(pathway_features)


def build_model(
    model_type: str,
    input_dim: int,
    gene_dim: int,
    fc_layers: list[int],
    mask_snp_gene: torch.Tensor,
    pathway_dim: int = 0,
    mask_gene_pathway: torch.Tensor | None = None,
    impact_indices: torch.Tensor | None = None,
    num_impacts: int | None = None,
    activation: str = "gelu",
    dropout: float = 0.0,
    use_layernorm: bool = True,
    impact_embedding_dim: int = 16,
    attention_dropout: float = 0.0,
) -> nn.Module:
    """Build one of the three compact model families."""

    model_type = model_type.lower()
    if model_type == "gene_only":
        return MaskedBiologicalMLP(
            input_dim=input_dim,
            gene_dim=gene_dim,
            pathway_dim=0,
            fc_layers=fc_layers,
            mask_snp_gene=mask_snp_gene,
            mask_gene_pathway=None,
            activation=activation,
            dropout=dropout,
            use_layernorm=use_layernorm,
        )

    if model_type == "gene_pathway":
        if pathway_dim <= 0 or mask_gene_pathway is None:
            raise ValueError(
                "gene_pathway requires pathway_dim > 0 and mask_gene_pathway."
            )
        return MaskedBiologicalMLP(
            input_dim=input_dim,
            gene_dim=gene_dim,
            pathway_dim=pathway_dim,
            fc_layers=fc_layers,
            mask_snp_gene=mask_snp_gene,
            mask_gene_pathway=mask_gene_pathway,
            activation=activation,
            dropout=dropout,
            use_layernorm=use_layernorm,
        )

    if model_type == "attention_gene":
        if impact_indices is None or num_impacts is None:
            raise ValueError(
                "attention_gene requires impact_indices and num_impacts."
            )
        return AttentionGeneMLP(
            input_dim=input_dim,
            gene_dim=gene_dim,
            fc_layers=fc_layers,
            mask_snp_gene=mask_snp_gene,
            impact_indices=impact_indices,
            num_impacts=num_impacts,
            activation=activation,
            dropout=dropout,
            use_layernorm=use_layernorm,
            impact_embedding_dim=impact_embedding_dim,
            attention_dropout=attention_dropout,
        )

    raise ValueError(f"Unsupported model_type: {model_type}")


def compact_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Save weights without fixed masks/buffers that can be reconstructed later."""

    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if not name.endswith(".mask") and not name.endswith("impact_indices")
    }


def load_model_state_dict_compatibly(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None:
    """Load a compact state_dict while tolerating reconstructed mask buffers."""

    load_result = model.load_state_dict(state_dict, strict=False)
    unexpected = set(load_result.unexpected_keys)
    missing = set(load_result.missing_keys)
    allowed_missing = {
        name
        for name in missing
        if name.endswith(".mask") or name.endswith("impact_indices")
    }
    disallowed_missing = missing - allowed_missing
    if unexpected or disallowed_missing:
        raise RuntimeError(
            "Incompatible checkpoint state_dict. "
            f"Missing keys: {sorted(disallowed_missing)}. "
            f"Unexpected keys: {sorted(unexpected)}."
        )


def _rebuild_mask_from_annotation_edges(
    annotation_edges: pd.DataFrame,
    gene_map: pd.DataFrame,
    snp_map: pd.DataFrame,
) -> torch.Tensor:
    """Rebuild the SNP -> gene mask from compact checkpoint metadata."""

    gene_index = pd.Series(
        gene_map["gene_index"].to_numpy(),
        index=gene_map["ensembl_gene_id"],
    )
    snp_index = pd.Series(
        snp_map["snp_index"].to_numpy(),
        index=snp_map["snp_id"],
    )

    row_indices = annotation_edges["Gene"].map(gene_index).to_numpy()
    col_indices = annotation_edges["snp_id"].map(snp_index).to_numpy()
    valid = (~pd.isna(row_indices)) & (~pd.isna(col_indices))
    if not valid.all():
        missing_rows = int((~valid).sum())
        raise ValueError(
            "Could not rebuild the SNP-gene mask from checkpoint metadata. "
            f"Missing rows: {missing_rows}."
        )

    mask = coo_matrix(
        (
            np.ones(valid.sum(), dtype=np.float32),
            (row_indices[valid].astype(int), col_indices[valid].astype(int)),
        ),
        shape=(len(gene_map), len(snp_map)),
        dtype=np.float32,
    )
    mask.sum_duplicates()
    mask.data[:] = 1.0
    return torch.from_numpy(mask.toarray()).float()


def _rebuild_pathway_mask_from_checkpoint(
    checkpoint: dict[str, Any],
) -> torch.Tensor | None:
    """Rebuild the gene -> pathway mask when the checkpoint stores one."""

    row_indices = checkpoint.get("mask_gene_pathway_row_indices")
    col_indices = checkpoint.get("mask_gene_pathway_col_indices")
    shape = checkpoint.get("mask_gene_pathway_shape")
    if row_indices is None or col_indices is None or shape is None:
        return None

    mask = coo_matrix(
        (
            np.ones(len(row_indices), dtype=np.float32),
            (np.asarray(row_indices), np.asarray(col_indices)),
        ),
        shape=tuple(int(value) for value in shape),
        dtype=np.float32,
    )
    mask.sum_duplicates()
    mask.data[:] = 1.0
    return torch.from_numpy(mask.toarray()).float()


def load_model_from_checkpoint(
    checkpoint: dict[str, Any],
    device: torch.device | str | None = None,
) -> nn.Module:
    """Rebuild one of the compact models from a saved checkpoint."""

    architecture = checkpoint["architecture"]
    mask_snp_gene = _rebuild_mask_from_annotation_edges(
        annotation_edges=checkpoint["annotation_edges_dataframe"],
        gene_map=checkpoint["gene_map_dataframe"],
        snp_map=checkpoint["snp_map_dataframe"],
    )
    mask_gene_pathway = _rebuild_pathway_mask_from_checkpoint(checkpoint)

    impact_indices = checkpoint.get("impact_indices")
    if impact_indices is not None and not torch.is_tensor(impact_indices):
        impact_indices = torch.as_tensor(impact_indices, dtype=torch.long)

    model = build_model(
        model_type=architecture["model_type"],
        input_dim=int(architecture["input_dim"]),
        gene_dim=int(architecture["gene_dim"]),
        pathway_dim=int(architecture.get("pathway_dim", 0) or 0),
        fc_layers=list(architecture.get("fc_layers", [])),
        mask_snp_gene=mask_snp_gene,
        mask_gene_pathway=mask_gene_pathway,
        impact_indices=impact_indices,
        num_impacts=checkpoint.get("n_impacts"),
        activation=architecture.get("activation", "gelu"),
        dropout=float(architecture.get("dropout", 0.0)),
        use_layernorm=bool(architecture.get("use_layernorm", True)),
        impact_embedding_dim=int(architecture.get("impact_embedding_dim", 16)),
        attention_dropout=float(architecture.get("attention_dropout", 0.0)),
    )

    if device is not None:
        model = model.to(device)

    load_model_state_dict_compatibly(model, checkpoint["model_state_dict"])
    return model


def checkpoint_supports_pathways(checkpoint: dict[str, Any]) -> bool:
    """Return True when the checkpoint contains the pathway layer metadata."""

    architecture = checkpoint["architecture"]
    return architecture.get("model_type") == "gene_pathway"


def ensure_checkpoint_exists(checkpoint_path: str | Path) -> Path:
    """Validate a checkpoint path early and return it as a Path."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path
