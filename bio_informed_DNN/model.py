import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F


# ===============================
# Utilities
# ===============================


def _get_activation_module(activation):
    if activation == "relu":
        return nn.ReLU()
    if activation == "sigmoid":
        return nn.Sigmoid()
    if activation == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {activation}")


def _get_transformer_activation(activation):
    if activation == "relu":
        return F.relu
    if activation == "sigmoid":
        return torch.sigmoid
    if activation == "gelu":
        return F.gelu
    raise ValueError(f"Unsupported activation: {activation}")


def _init_dense(linear, activation, final=False):
    if final:
        nn.init.xavier_normal_(linear.weight)
    elif activation in ["relu", "gelu"]:
        nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
    elif activation == "sigmoid":
        nn.init.xavier_normal_(linear.weight)
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    nn.init.zeros_(linear.bias)


# ===============================
# Masked Linear Layer
# ===============================


class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask, activation="relu"):
        super().__init__()

        self.register_buffer("mask", mask)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.reset_parameters(activation)

    def reset_parameters(self, activation):
        fan_in = self.mask.sum(dim=1)
        fan_in = torch.clamp(fan_in, min=1)

        if activation.lower() == "relu":
            std = torch.sqrt(2.0 / fan_in)
        else:
            std = torch.sqrt(1.0 / fan_in)

        with torch.no_grad():
            for i in range(self.weight.shape[0]):
                self.weight[i].normal_(0, std[i])

    def forward(self, x):
        masked_weight = self.weight * self.mask
        return F.linear(x, masked_weight, self.bias)


# ===============================
# Flexible Partial Network
# ===============================


class PartialNet(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden1,
        hidden2,
        fc_layers,
        mask1,
        mask2,
        activation="relu",
        use_layernorm=True,
        dropout=0.0,
        encoder_dropout=None,
    ):
        super().__init__()

        effective_dropout = dropout if encoder_dropout is None else encoder_dropout

        self.activation_name = activation
        self.activation_fn = _get_activation_module(activation)
        self.has_pathway_layer = True

        self.masked1 = MaskedLinear(input_dim, hidden1, mask1, activation)
        self.masked2 = MaskedLinear(hidden1, hidden2, mask2, activation)

        self.norm1 = nn.LayerNorm(hidden1) if use_layernorm else nn.Identity()
        self.norm2 = nn.LayerNorm(hidden2) if use_layernorm else nn.Identity()

        self.dropout1 = (
            nn.Dropout(effective_dropout) if effective_dropout > 0 else nn.Identity()
        )
        self.dropout2 = (
            nn.Dropout(effective_dropout) if effective_dropout > 0 else nn.Identity()
        )

        layers = []
        prev_dim = hidden2

        for dim in fc_layers:
            linear = nn.Linear(prev_dim, dim)
            _init_dense(linear, activation)

            layers.append(linear)
            layers.append(nn.LayerNorm(dim) if use_layernorm else nn.Identity())
            layers.append(self.activation_fn)
            if effective_dropout > 0:
                layers.append(nn.Dropout(effective_dropout))

            prev_dim = dim

        final = nn.Linear(prev_dim, 1)
        _init_dense(final, activation, final=True)
        layers.append(final)

        self.fc_stack = nn.Sequential(*layers)

    def encode_genes(self, x):
        x = self.masked1(x)
        x = self.norm1(x)
        x = self.activation_fn(x)
        x = self.dropout1(x)
        return x

    def encode_pathways_from_genes(self, gene_features):
        x = self.masked2(gene_features)
        x = self.norm2(x)
        x = self.activation_fn(x)
        x = self.dropout2(x)
        return x

    def encode_pathways(self, x):
        gene_features = self.encode_genes(x)
        return self.encode_pathways_from_genes(gene_features)

    def forward(self, x):
        x = self.encode_pathways(x)
        x = self.fc_stack(x)
        return x


# ===============================
# SNP Impact Attention + Gene MLP
# ===============================


class SNPImpactAttention(nn.Module):
    def __init__(
        self,
        num_snps,
        num_impacts,
        impact_indices,
        embedding_dim=16,
        activation="relu",
        attention_dropout=0.0,
    ):
        super().__init__()

        if impact_indices.numel() != num_snps:
            raise ValueError(
                "impact_indices must contain exactly one impact label per SNP "
                f"(expected {num_snps}, got {impact_indices.numel()})"
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

    def reset_parameters(self):
        nn.init.normal_(self.impact_embedding.weight, mean=0.0, std=0.02)
        nn.init.xavier_normal_(self.impact_projection.weight)
        nn.init.zeros_(self.impact_projection.bias)

        # Keep the initial attention close to identity while still allowing
        # gradients to start shaping the impact-conditioned modulation.
        nn.init.normal_(self.scale_head.weight, mean=0.0, std=5e-3)
        nn.init.zeros_(self.scale_head.bias)
        nn.init.normal_(self.bias_head.weight, mean=0.0, std=5e-3)
        nn.init.zeros_(self.bias_head.bias)

    def _impact_hidden(self):
        hidden = self.impact_embedding(self.impact_indices)
        hidden = self.impact_projection(hidden)
        hidden = self.impact_norm(hidden)
        hidden = self.activation_fn(hidden)
        return hidden

    def get_attention_logits(self, x):
        impact_hidden = self._impact_hidden()
        scale = self.scale_head(impact_hidden).squeeze(-1)
        bias = self.bias_head(impact_hidden).squeeze(-1)
        return x * scale + bias

    def get_attention_weights(self, x):
        return 2.0 * torch.sigmoid(self.get_attention_logits(x))

    def forward(self, x):
        attention_weights = self.get_attention_weights(x)
        return self.output_dropout(x * attention_weights)


class ImpactAttentionNet(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden1,
        fc_layers,
        mask1,
        impact_indices,
        num_impacts,
        impact_embedding_dim=16,
        activation="relu",
        use_layernorm=True,
        dropout=0.0,
        attention_dropout=0.0,
    ):
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

        self.masked1 = MaskedLinear(input_dim, hidden1, mask1, activation)
        self.masked2 = None

        self.norm1 = nn.LayerNorm(hidden1) if use_layernorm else nn.Identity()
        self.norm2 = None

        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.dropout2 = None

        layers = []
        prev_dim = hidden1

        for dim in fc_layers:
            linear = nn.Linear(prev_dim, dim)
            _init_dense(linear, activation)

            layers.append(linear)
            layers.append(nn.LayerNorm(dim) if use_layernorm else nn.Identity())
            layers.append(self.activation_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = dim

        final = nn.Linear(prev_dim, 1)
        _init_dense(final, activation, final=True)
        layers.append(final)

        self.fc_stack = nn.Sequential(*layers)

    def encode_snps(self, x):
        return self.snp_attention(x)

    def get_attention_logits(self, x):
        return self.snp_attention.get_attention_logits(x)

    def get_attention_weights(self, x):
        return self.snp_attention.get_attention_weights(x)

    def encode_genes(self, x):
        x = self.encode_snps(x)
        x = self.masked1(x)
        x = self.norm1(x)
        x = self.activation_fn(x)
        x = self.dropout1(x)
        return x

    def forward(self, x):
        x = self.encode_genes(x)
        x = self.fc_stack(x)
        return x


# ===============================
# Transformer Head
# ===============================


class TransformerHead(nn.Module):
    def __init__(
        self,
        num_tokens,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        transformer_dropout=0.0,
        activation="gelu",
        max_tokens=None,
        pooling="mean",
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(
                f"transformer_d_model ({d_model}) must be divisible by "
                f"transformer_heads ({nhead})"
            )

        if pooling not in {"mean", "cls"}:
            raise ValueError(
                f"Unsupported transformer pooling '{pooling}'. Expected 'mean' or 'cls'."
            )

        self.pooling = pooling
        use_pooling = max_tokens is not None and max_tokens > 0 and max_tokens < num_tokens
        self.original_num_tokens = num_tokens
        self.sequence_length = max_tokens if use_pooling else num_tokens

        self.value_projection = nn.Linear(1, d_model)
        _init_dense(self.value_projection, activation)

        self.token_embedding = nn.Parameter(
            torch.zeros(1, self.sequence_length, d_model)
        )
        nn.init.normal_(self.token_embedding, mean=0.0, std=0.02)

        self.cls_token = None
        if self.pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
            activation=_get_transformer_activation(activation),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_norm = nn.LayerNorm(d_model)
        self.output_dropout = (
            nn.Dropout(transformer_dropout) if transformer_dropout > 0 else nn.Identity()
        )
        self.output = nn.Linear(d_model, 1)
        _init_dense(self.output, activation, final=True)

    def _pool_tokens(self, x):
        if x.size(1) == self.sequence_length:
            return x

        x = x.unsqueeze(1)
        x = F.adaptive_avg_pool1d(x, self.sequence_length)
        return x.squeeze(1)

    def forward(self, x):
        x = self._pool_tokens(x)
        tokens = self.value_projection(x.unsqueeze(-1))
        tokens = tokens + self.token_embedding

        if self.pooling == "cls":
            cls_token = self.cls_token.expand(tokens.size(0), -1, -1)
            tokens = torch.cat((cls_token, tokens), dim=1)

        tokens = self.encoder(tokens)

        if self.pooling == "cls":
            pooled = tokens[:, 0, :]
        else:
            pooled = tokens.mean(dim=1)

        pooled = self.output_norm(pooled)
        pooled = self.output_dropout(pooled)
        return self.output(pooled)


# ===============================
# Partial Network + Transformer Head
# ===============================


class PartialTransformerNet(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden1,
        hidden2,
        mask1,
        mask2,
        transformer_input="pathway",
        transformer_d_model=64,
        transformer_heads=4,
        transformer_layers=2,
        transformer_ff_dim=256,
        transformer_max_tokens=None,
        transformer_pooling="mean",
        activation="relu",
        use_layernorm=True,
        dropout=0.0,
        encoder_dropout=None,
        transformer_dropout=None,
    ):
        super().__init__()

        if transformer_input not in {"gene", "pathway"}:
            raise ValueError(
                "transformer_input must be either 'gene' or 'pathway'"
            )

        effective_encoder_dropout = dropout if encoder_dropout is None else encoder_dropout
        effective_transformer_dropout = (
            dropout if transformer_dropout is None else transformer_dropout
        )

        self.activation_name = activation
        self.activation_fn = _get_activation_module(activation)
        self.transformer_input = transformer_input
        self.has_pathway_layer = transformer_input == "pathway"

        self.masked1 = MaskedLinear(input_dim, hidden1, mask1, activation)
        self.norm1 = nn.LayerNorm(hidden1) if use_layernorm else nn.Identity()
        self.dropout1 = (
            nn.Dropout(effective_encoder_dropout)
            if effective_encoder_dropout > 0
            else nn.Identity()
        )

        if self.has_pathway_layer:
            self.masked2 = MaskedLinear(hidden1, hidden2, mask2, activation)
            self.norm2 = nn.LayerNorm(hidden2) if use_layernorm else nn.Identity()
            self.dropout2 = (
                nn.Dropout(effective_encoder_dropout)
                if effective_encoder_dropout > 0
                else nn.Identity()
            )
            transformer_tokens = hidden2
        else:
            self.masked2 = None
            self.norm2 = None
            self.dropout2 = None
            transformer_tokens = hidden1

        self.transformer_head = TransformerHead(
            num_tokens=transformer_tokens,
            d_model=transformer_d_model,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            dim_feedforward=transformer_ff_dim,
            transformer_dropout=effective_transformer_dropout,
            activation=activation,
            max_tokens=transformer_max_tokens,
            pooling=transformer_pooling,
        )

    def encode_genes(self, x):
        x = self.masked1(x)
        x = self.norm1(x)
        x = self.activation_fn(x)
        x = self.dropout1(x)
        return x

    def encode_pathways_from_genes(self, gene_features):
        if not self.has_pathway_layer:
            return None

        x = self.masked2(gene_features)
        x = self.norm2(x)
        x = self.activation_fn(x)
        x = self.dropout2(x)
        return x

    def encode_pathways(self, x):
        gene_features = self.encode_genes(x)
        return self.encode_pathways_from_genes(gene_features)

    def get_transformer_inputs(self, x):
        gene_features = self.encode_genes(x)
        if self.transformer_input == "gene":
            return gene_features
        return self.encode_pathways_from_genes(gene_features)

    def forward(self, x):
        x = self.get_transformer_inputs(x)
        return self.transformer_head(x)


# ===============================
# Factory
# ===============================


def build_biological_model(
    model_type,
    input_dim,
    hidden1,
    hidden2=0,
    fc_layers=None,
    mask1=None,
    mask2=None,
    activation="relu",
    use_layernorm=True,
    dropout=0.0,
    transformer_d_model=64,
    transformer_heads=4,
    transformer_layers=2,
    transformer_ff_dim=256,
    transformer_max_tokens=None,
    transformer_pooling="mean",
    encoder_dropout=None,
    transformer_dropout=None,
    impact_indices=None,
    num_impacts=None,
    impact_embedding_dim=16,
    attention_dropout=0.0,
):
    if model_type == "mlp":
        return PartialNet(
            input_dim=input_dim,
            hidden1=hidden1,
            hidden2=hidden2,
            fc_layers=fc_layers or [],
            mask1=mask1,
            mask2=mask2,
            activation=activation,
            use_layernorm=use_layernorm,
            dropout=dropout,
            encoder_dropout=encoder_dropout,
        )

    if model_type == "impact_attention_mlp":
        if impact_indices is None or num_impacts is None:
            raise ValueError(
                "impact_attention_mlp requires impact_indices and num_impacts"
            )

        return ImpactAttentionNet(
            input_dim=input_dim,
            hidden1=hidden1,
            fc_layers=fc_layers or [],
            mask1=mask1,
            impact_indices=impact_indices,
            num_impacts=num_impacts,
            impact_embedding_dim=impact_embedding_dim,
            activation=activation,
            use_layernorm=use_layernorm,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )

    if model_type == "pathway_transformer":
        return PartialTransformerNet(
            input_dim=input_dim,
            hidden1=hidden1,
            hidden2=hidden2,
            mask1=mask1,
            mask2=mask2,
            transformer_input="pathway",
            transformer_d_model=transformer_d_model,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            transformer_ff_dim=transformer_ff_dim,
            transformer_max_tokens=transformer_max_tokens,
            transformer_pooling=transformer_pooling,
            activation=activation,
            use_layernorm=use_layernorm,
            dropout=dropout,
            encoder_dropout=encoder_dropout,
            transformer_dropout=transformer_dropout,
        )

    if model_type == "gene_transformer":
        return PartialTransformerNet(
            input_dim=input_dim,
            hidden1=hidden1,
            hidden2=hidden2,
            mask1=mask1,
            mask2=mask2,
            transformer_input="gene",
            transformer_d_model=transformer_d_model,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            transformer_ff_dim=transformer_ff_dim,
            transformer_max_tokens=transformer_max_tokens,
            transformer_pooling=transformer_pooling,
            activation=activation,
            use_layernorm=use_layernorm,
            dropout=dropout,
            encoder_dropout=encoder_dropout,
            transformer_dropout=transformer_dropout,
        )

    raise ValueError(f"Unsupported model_type: {model_type}")


def _rebuild_mask_from_snp_annotation(snp_annotation_df, gene_map_df, snp_map_df):
    import numpy as np
    from scipy.sparse import coo_matrix

    if "Gene" not in snp_annotation_df.columns:
        raise KeyError("snp_annotation_dataframe must include a 'Gene' column")

    gene_index = pd.Series(
        gene_map_df["gene_index"].to_numpy(),
        index=gene_map_df["ensembl_gene_id"],
    )
    snp_index = pd.Series(
        snp_map_df["snp_index"].to_numpy(),
        index=snp_map_df["snp_id"],
    )

    row_indices = snp_annotation_df["Gene"].map(gene_index).to_numpy()
    col_indices = snp_annotation_df["snp_id"].map(snp_index).to_numpy()

    valid = (~pd.isna(row_indices)) & (~pd.isna(col_indices))
    if not valid.all():
        missing_rows = int((~valid).sum())
        raise ValueError(
            f"Could not rebuild SNP-gene mask from checkpoint metadata; "
            f"{missing_rows} SNP annotations could not be mapped."
        )

    mask_snp_gene = coo_matrix(
        (
            np.ones(valid.sum(), dtype=np.float32),
            (row_indices[valid].astype(int), col_indices[valid].astype(int)),
        ),
        shape=(len(gene_map_df), len(snp_map_df)),
        dtype=np.float32,
    )
    mask_snp_gene.sum_duplicates()
    mask_snp_gene.data[:] = 1.0
    return torch.from_numpy(mask_snp_gene.toarray()).float()


def _recover_impact_checkpoint_artifacts(checkpoint):
    snp_annotation_df = checkpoint.get("snp_annotation_dataframe")
    gene_map_df = checkpoint.get("gene_map_dataframe")
    snp_map_df = checkpoint.get("snp_map_dataframe")
    impact_map_df = checkpoint.get("impact_map_dataframe")

    if (
        snp_annotation_df is not None
        and gene_map_df is not None
        and snp_map_df is not None
    ):
        mask1 = _rebuild_mask_from_snp_annotation(
            snp_annotation_df=snp_annotation_df,
            gene_map_df=gene_map_df,
            snp_map_df=snp_map_df,
        )

        impact_indices = checkpoint.get("impact_indices")
        num_impacts = checkpoint.get("n_impacts")

        if impact_indices is None and impact_map_df is not None:
            impact_lookup = pd.Series(
                impact_map_df["impact_index"].to_numpy(),
                index=impact_map_df["new_impact"],
            )
            snp_impact_df = (
                snp_annotation_df.drop_duplicates(subset=["snp_id"], keep="first")
                [["snp_id", "new_impact"]]
                .set_index("snp_id")
                .reindex(snp_map_df["snp_id"])
                .reset_index()
            )
            impact_indices = torch.tensor(
                snp_impact_df["new_impact"].map(impact_lookup).to_numpy(),
                dtype=torch.long,
            )
            num_impacts = len(impact_map_df)

        if impact_indices is not None and not torch.is_tensor(impact_indices):
            impact_indices = torch.tensor(impact_indices, dtype=torch.long)

        return mask1, impact_indices, num_impacts

    training_args = checkpoint.get("training_args", {})
    annotation_path = training_args.get("annotation_path")
    if annotation_path is None:
        raise KeyError(
            "Checkpoint is missing mask_snp_gene and compact SNP annotation metadata."
        )

    try:
        from data_training_impact_attention import (
            build_annotation_resources,
            load_annotation_dataframe,
        )
    except ImportError:
        from bio_informed_DNN.data_training_impact_attention import (
            build_annotation_resources,
            load_annotation_dataframe,
        )

    annotation_df, _ = load_annotation_dataframe(annotation_path)
    resources = build_annotation_resources(annotation_df)
    return (
        resources["mask_tensor"],
        resources["impact_indices"],
        resources["n_impacts"],
    )


def load_model_state_dict_compatibly(model, state_dict):
    load_result = model.load_state_dict(state_dict, strict=False)

    unexpected = set(load_result.unexpected_keys)
    missing = set(load_result.missing_keys)
    allowed_missing = {
        name
        for name in missing
        if name.endswith('.mask') or name.endswith('impact_indices')
    }
    disallowed_missing = missing - allowed_missing

    if unexpected or disallowed_missing:
        raise RuntimeError(
            'Incompatible checkpoint state_dict. '
            f'Missing keys: {sorted(disallowed_missing)}. '
            f'Unexpected keys: {sorted(unexpected)}.'
        )

    return load_result


def build_model_from_checkpoint(checkpoint, map_location=None):
    architecture = checkpoint["architecture"]
    training_args = checkpoint.get("training_args", {})

    mask1 = checkpoint.get("mask_snp_gene")
    mask2 = checkpoint.get("mask_gene_pathway")
    impact_indices = checkpoint.get("impact_indices")
    num_impacts = checkpoint.get(
        "n_impacts",
        architecture.get("num_impacts", training_args.get("num_impacts")),
    )

    if mask1 is None or (
        architecture.get("model_type", training_args.get("model_type"))
        == "impact_attention_mlp"
        and (impact_indices is None or num_impacts is None)
    ):
        rebuilt_mask1, rebuilt_impact_indices, rebuilt_num_impacts = (
            _recover_impact_checkpoint_artifacts(checkpoint)
        )
        if mask1 is None:
            mask1 = rebuilt_mask1
        if impact_indices is None:
            impact_indices = rebuilt_impact_indices
        if num_impacts is None:
            num_impacts = rebuilt_num_impacts

    if mask1 is not None and not torch.is_tensor(mask1):
        mask1 = torch.as_tensor(mask1, dtype=torch.float32)
    if mask2 is not None and not torch.is_tensor(mask2):
        mask2 = torch.as_tensor(mask2, dtype=torch.float32)
    if impact_indices is not None and not torch.is_tensor(impact_indices):
        impact_indices = torch.as_tensor(impact_indices, dtype=torch.long)

    shared_dropout = architecture.get("dropout", training_args.get("dropout", 0.0))
    encoder_dropout = architecture.get(
        "encoder_dropout", training_args.get("encoder_dropout", shared_dropout)
    )
    transformer_dropout = architecture.get(
        "transformer_dropout",
        training_args.get("transformer_dropout", shared_dropout),
    )

    model = build_biological_model(
        model_type=architecture.get("model_type", training_args.get("model_type", "mlp")),
        input_dim=architecture["input_dim"],
        hidden1=architecture["hidden1"],
        hidden2=architecture.get("hidden2", 0),
        fc_layers=architecture.get("fc_layers", []),
        mask1=mask1,
        mask2=mask2,
        activation=architecture.get("activation", training_args.get("activation", "relu")),
        dropout=shared_dropout,
        encoder_dropout=encoder_dropout,
        transformer_dropout=transformer_dropout,
        transformer_d_model=architecture.get(
            "transformer_d_model",
            training_args.get("transformer_d_model", 64),
        ),
        transformer_heads=architecture.get(
            "transformer_heads",
            training_args.get("transformer_heads", 4),
        ),
        transformer_layers=architecture.get(
            "transformer_layers",
            training_args.get("transformer_layers", 2),
        ),
        transformer_ff_dim=architecture.get(
            "transformer_ff_dim",
            training_args.get("transformer_ff_dim", 256),
        ),
        transformer_max_tokens=architecture.get(
            "transformer_max_tokens",
            training_args.get("transformer_max_tokens"),
        ),
        transformer_pooling=architecture.get(
            "transformer_pooling",
            training_args.get("transformer_pooling", "mean"),
        ),
        impact_indices=impact_indices,
        num_impacts=num_impacts,
        impact_embedding_dim=architecture.get(
            "impact_embedding_dim",
            training_args.get("impact_embedding_dim", 16),
        ),
        attention_dropout=architecture.get(
            "attention_dropout",
            training_args.get("attention_dropout", 0.0),
        ),
    )

    if map_location is not None:
        model = model.to(map_location)

    load_model_state_dict_compatibly(model, checkpoint["model_state_dict"])
    return model


# ===============================
# Early Stopping epochs
# ===============================


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, max_delta=1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.max_delta = max_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, loss=None):
        if self.best_loss is None:
            self.best_loss = val_loss
            return

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif loss is None or (
            (val_loss > self.best_loss + self.max_delta)
            and (abs(val_loss - loss) > 3e-2)
        ):
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True
