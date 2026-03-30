"""
Integrated-gradients analysis for the compact biologically informed workflow.

This script is intentionally focused:

- validation/test reconstruction from compact checkpoints
- integrated gradients in SNP space for every model
- integrated gradients in gene space for every model
- integrated gradients in pathway space when the model includes a pathway layer
- IG-based plots and summary tables only
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

CACHE_DIR = Path(__file__).resolve().parent / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from bio_informed_DNN.resume_workflow.dataset import (
        inverse_scale,
        reconstruct_test_data_from_checkpoint,
        reconstruct_validation_data_from_checkpoint,
        sparse_mask_from_index_arrays,
    )
    from bio_informed_DNN.resume_workflow.models import (
        GeneToOutputWrapper,
        PathwayToOutputWrapper,
        checkpoint_supports_pathways,
        ensure_checkpoint_exists,
        load_model_from_checkpoint,
    )
except ImportError:
    from dataset import (  # type: ignore
        inverse_scale,
        reconstruct_test_data_from_checkpoint,
        reconstruct_validation_data_from_checkpoint,
        sparse_mask_from_index_arrays,
    )
    from models import (  # type: ignore
        GeneToOutputWrapper,
        PathwayToOutputWrapper,
        checkpoint_supports_pathways,
        ensure_checkpoint_exists,
        load_model_from_checkpoint,
    )


TRAIT_NAMES = {
    2: "shoulder",
    3: "top",
    4: "buttock_side",
    5: "buttock_rear",
    6: "size",
    7: "musculature",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the reduced integrated-gradients analysis on a compact model "
            "checkpoint produced by resume_workflow/train.py."
        )
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--full-phenotype-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--top-snps", type=int, default=30)
    parser.add_argument("--top-genes", type=int, default=20)
    parser.add_argument("--top-pathways", type=int, default=20)
    parser.add_argument(
        "--top-impact-sizes",
        type=int,
        nargs="+",
        default=[50, 100, 200],
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--ig-samples", type=int, default=128)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_checkpoint(checkpoint_path: str | Path) -> tuple[Path, dict[str, Any]]:
    checkpoint_path = ensure_checkpoint_exists(checkpoint_path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint_path, checkpoint


def infer_output_dir(checkpoint_path: Path, output_dir: str | None) -> Path:
    if output_dir is not None:
        out_dir = Path(output_dir)
    else:
        out_dir = checkpoint_path.parent / "integrated_gradients_analysis" / checkpoint_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def trait_label(trait: int) -> str:
    return TRAIT_NAMES.get(int(trait), f"trait_{trait}")


def trait_plot_title(trait: int, title: str) -> str:
    return f"Trait {int(trait)}: {title}"


def set_plot_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#f7f7f2",
            "axes.edgecolor": "#2e2e2e",
            "axes.labelcolor": "#222222",
            "axes.titleweight": "bold",
            "grid.color": "#d8d8cf",
            "grid.alpha": 0.55,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "savefig.bbox": "tight",
        }
    )


def safe_pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size == 0 or y.size == 0:
        return np.nan, np.nan
    if np.allclose(x.std(ddof=0), 0.0) or np.allclose(y.std(ddof=0), 0.0):
        return np.nan, np.nan
    return pearsonr(x, y)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mse = mean_squared_error(y_true, y_pred)
    pearson, _ = safe_pearson(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson": float(pearson),
    }


def batched_predict(
    model: torch.nn.Module,
    X_scaled: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, X_scaled.shape[0], batch_size):
            xb = torch.tensor(
                X_scaled[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            outputs.append(model(xb).detach().cpu().numpy())
    return np.vstack(outputs).astype(np.float32)


def compute_integrated_gradients(
    module: torch.nn.Module,
    feature_matrix: np.ndarray,
    device: torch.device,
    batch_size: int,
    steps: int = 32,
    max_samples: int = 128,
    random_seed: int = 42,
) -> dict[str, Any]:
    if max_samples is not None and max_samples < len(feature_matrix):
        rng = np.random.default_rng(random_seed)
        selected = np.sort(
            rng.choice(len(feature_matrix), size=max_samples, replace=False)
        )
    else:
        selected = np.arange(len(feature_matrix))

    inputs = feature_matrix[selected]
    total_abs = None
    total_signed = None
    total_samples = 0
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=device)[1:]

    module.eval()
    for start in range(0, inputs.shape[0], batch_size):
        xb = torch.tensor(
            inputs[start : start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        baseline = torch.zeros_like(xb)
        accumulated = torch.zeros_like(xb)

        for alpha in alphas:
            interpolated = (
                baseline + alpha * (xb - baseline)
            ).detach().requires_grad_(True)
            outputs = module(interpolated)
            gradients = torch.autograd.grad(outputs.sum(), interpolated)[0]
            accumulated += gradients.detach()

        attribution = (xb - baseline) * accumulated / max(int(steps), 1)
        attribution_abs = attribution.abs().sum(dim=0).detach().cpu().numpy()
        attribution_signed = attribution.sum(dim=0).detach().cpu().numpy()

        if total_abs is None:
            total_abs = attribution_abs
            total_signed = attribution_signed
        else:
            total_abs += attribution_abs
            total_signed += attribution_signed
        total_samples += xb.shape[0]

    return {
        "sample_indices": selected,
        "abs": total_abs / max(total_samples, 1),
        "signed": total_signed / max(total_samples, 1),
    }


def vectorized_correlation(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).reshape(-1)

    matrix_centered = matrix - matrix.mean(axis=0, keepdims=True)
    target_centered = target - target.mean()
    numerator = (matrix_centered * target_centered[:, None]).sum(axis=0)
    denominator = np.sqrt(
        (matrix_centered**2).sum(axis=0) * (target_centered**2).sum()
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros(matrix.shape[1], dtype=np.float64),
        where=denominator > 0,
    ).astype(np.float32)


def build_sparse_snp_gene_mask(
    annotation_edges: pd.DataFrame,
    gene_map: pd.DataFrame,
    snp_map: pd.DataFrame,
) -> coo_matrix:
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
    return coo_matrix(
        (
            np.ones(valid.sum(), dtype=np.float32),
            (row_indices[valid].astype(int), col_indices[valid].astype(int)),
        ),
        shape=(len(gene_map), len(snp_map)),
        dtype=np.float32,
    ).tocsr()


def encode_gene_features(
    model: torch.nn.Module,
    X_scaled: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, X_scaled.shape[0], batch_size):
            xb = torch.tensor(
                X_scaled[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            outputs.append(model.encode_genes(xb).detach().cpu().numpy())
    return np.vstack(outputs).astype(np.float32)


def encode_pathway_features(
    model: torch.nn.Module,
    X_scaled: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, X_scaled.shape[0], batch_size):
            xb = torch.tensor(
                X_scaled[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            outputs.append(model.encode_pathways(xb).detach().cpu().numpy())
    return np.vstack(outputs).astype(np.float32)


def build_snp_table(
    validation_data: dict[str, Any],
    checkpoint: dict[str, Any],
    ig_abs: np.ndarray,
    ig_signed: np.ndarray,
) -> pd.DataFrame:
    annotation_edges = checkpoint["annotation_edges_dataframe"]
    snp_map = checkpoint["snp_map_dataframe"].copy()
    bim = validation_data["bim"]

    keep_columns = [
        column
        for column in ["snp_id", "new_impact", "Consequence", "IMPACT", "cm", "bp"]
        if column in annotation_edges.columns
    ]
    snp_metadata = annotation_edges.drop_duplicates("snp_id")[keep_columns].copy()

    gene_summary = (
        annotation_edges.groupby("snp_id")
        .agg(
            n_genes=("Gene", "nunique"),
            mapped_genes=("Gene", lambda values: ";".join(sorted(pd.unique(values))[:5])),
            mapped_symbols=(
                "SYMBOL",
                lambda values: ";".join(
                    sorted(
                        symbol
                        for symbol in pd.unique(values)
                        if isinstance(symbol, str) and symbol
                    )
                )[:120],
            )
            if "SYMBOL" in annotation_edges.columns
            else ("Gene", lambda values: ""),
        )
        .reset_index()
    )

    bim_meta = bim.loc[snp_map["snp_id"]].reset_index()
    rename_map = {
        "snp": "snp_id",
        "chrom": "chromosome",
        "pos": "position",
        "cm": "plink_cm",
        "a0": "plink_allele0",
        "a1": "plink_allele1",
    }
    bim_meta = bim_meta.rename(columns=rename_map)

    snp_table = snp_map.merge(snp_metadata, on="snp_id", how="left")
    snp_table = snp_table.merge(gene_summary, on="snp_id", how="left")
    snp_table = snp_table.merge(bim_meta, on="snp_id", how="left")
    snp_table["mapped_symbols"] = snp_table["mapped_symbols"].fillna("")
    snp_table["mapped_genes"] = snp_table["mapped_genes"].fillna("")
    snp_table["display_gene_label"] = np.where(
        snp_table["mapped_symbols"].astype(str).str.len() > 0,
        snp_table["mapped_symbols"],
        snp_table["mapped_genes"],
    )
    snp_table["integrated_gradients_abs"] = snp_table["snp_index"].map(
        lambda value: float(ig_abs[int(value)])
    )
    snp_table["integrated_gradients_signed"] = snp_table["snp_index"].map(
        lambda value: float(ig_signed[int(value)])
    )

    snp_corr = vectorized_correlation(validation_data["X_raw"], validation_data["y_raw"])
    snp_table["genotype_trait_corr"] = snp_corr
    snp_table["abs_genotype_trait_corr"] = np.abs(snp_corr)
    return snp_table.sort_values(
        "integrated_gradients_abs",
        ascending=False,
    ).reset_index(drop=True)


def build_gene_table(
    validation_data: dict[str, Any],
    checkpoint: dict[str, Any],
    gene_feature_matrix: np.ndarray,
    ig_abs: np.ndarray,
    ig_signed: np.ndarray,
) -> pd.DataFrame:
    gene_map = checkpoint["gene_map_dataframe"].copy()
    annotation_edges = checkpoint["annotation_edges_dataframe"]
    snp_map = checkpoint["snp_map_dataframe"]

    gene_map["gene_label"] = np.where(
        gene_map["gene_name"].astype(str).str.len() > 0,
        gene_map["gene_name"],
        gene_map["ensembl_gene_id"],
    )

    sparse_mask = build_sparse_snp_gene_mask(annotation_edges, gene_map, snp_map)
    snps_per_gene = np.asarray(sparse_mask.sum(axis=1)).reshape(-1).astype(np.float32)
    snps_per_gene = np.where(snps_per_gene == 0, 1.0, snps_per_gene)
    gene_burden = np.asarray(validation_data["X_raw"] @ sparse_mask.T, dtype=np.float32)
    gene_burden = gene_burden / snps_per_gene[None, :]

    gene_feature_corr = vectorized_correlation(gene_feature_matrix, validation_data["y_raw"])
    gene_burden_corr = vectorized_correlation(gene_burden, validation_data["y_raw"])

    gene_map["n_snps"] = snps_per_gene.astype(int)
    gene_map["integrated_gradients_abs"] = gene_map["gene_index"].map(
        lambda value: float(ig_abs[int(value)])
    )
    gene_map["integrated_gradients_signed"] = gene_map["gene_index"].map(
        lambda value: float(ig_signed[int(value)])
    )
    gene_map["gene_feature_trait_corr"] = gene_feature_corr
    gene_map["abs_gene_feature_trait_corr"] = np.abs(gene_feature_corr)
    gene_map["gene_burden_trait_corr"] = gene_burden_corr
    gene_map["abs_gene_burden_trait_corr"] = np.abs(gene_burden_corr)
    return gene_map.sort_values(
        "integrated_gradients_abs",
        ascending=False,
    ).reset_index(drop=True)


def build_pathway_table(
    validation_data: dict[str, Any],
    checkpoint: dict[str, Any],
    pathway_feature_matrix: np.ndarray,
    ig_abs: np.ndarray,
    ig_signed: np.ndarray,
) -> pd.DataFrame:
    pathway_map = checkpoint["pathway_map_dataframe"].copy()
    if "pathway_id" not in pathway_map.columns:
        pathway_map["pathway_id"] = pathway_map["pathway_index"].map(
            lambda value: f"pathway_{int(value):05d}"
        )
    pathway_map["pathway_label"] = pathway_map["pathway_id"].astype(str)
    pathway_corr = vectorized_correlation(pathway_feature_matrix, validation_data["y_raw"])
    pathway_map["integrated_gradients_abs"] = pathway_map["pathway_index"].map(
        lambda value: float(ig_abs[int(value)])
    )
    pathway_map["integrated_gradients_signed"] = pathway_map["pathway_index"].map(
        lambda value: float(ig_signed[int(value)])
    )
    pathway_map["pathway_trait_corr"] = pathway_corr
    pathway_map["abs_pathway_trait_corr"] = np.abs(pathway_corr)
    return pathway_map.sort_values(
        "integrated_gradients_abs",
        ascending=False,
    ).reset_index(drop=True)


def build_impact_tables(
    snp_table: pd.DataFrame,
    top_sizes: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    top_sizes = sorted({int(size) for size in top_sizes if int(size) > 0})
    overall_counts = (
        snp_table["new_impact"].fillna("unknown").value_counts().sort_values(ascending=False)
    )
    categories = overall_counts.index.tolist()
    overall_total = int(overall_counts.sum())

    composition_rows = []
    for impact_class, count in overall_counts.items():
        composition_rows.append(
            {
                "subset": "Overall",
                "top_n": overall_total,
                "new_impact": impact_class,
                "count": int(count),
                "percentage": float(100.0 * count / overall_total),
                "ranking_label": "integrated gradients",
            }
        )

    enrichment_rows = []
    for top_n in top_sizes:
        top_subset = snp_table.head(min(top_n, len(snp_table)))
        top_counts = (
            top_subset["new_impact"]
            .fillna("unknown")
            .value_counts()
            .reindex(categories, fill_value=0)
        )
        top_total = int(top_counts.sum())
        for impact_class in categories:
            overall_pct = overall_counts[impact_class] / max(overall_total, 1)
            top_pct = top_counts[impact_class] / max(top_total, 1)
            composition_rows.append(
                {
                    "subset": f"Top {top_n}",
                    "top_n": int(top_n),
                    "new_impact": impact_class,
                    "count": int(top_counts[impact_class]),
                    "percentage": float(100.0 * top_pct),
                    "ranking_label": "integrated gradients",
                }
            )
            enrichment_rows.append(
                {
                    "top_n": int(top_n),
                    "new_impact": impact_class,
                    "overall_pct": float(overall_pct),
                    "top_pct": float(top_pct),
                    "log2_enrichment": float(
                        np.log2((top_pct + 1e-8) / (overall_pct + 1e-8))
                    ),
                    "ranking_label": "integrated gradients",
                }
            )

    return pd.DataFrame(composition_rows), pd.DataFrame(enrichment_rows)


def build_palette(categories: list[str]) -> dict[str, Any]:
    cmap = plt.get_cmap("tab20", max(len(categories), 3))
    return {category: cmap(index) for index, category in enumerate(categories)}


def format_snp_label(row: Any) -> str:
    gene_label = (
        row.display_gene_label
        if isinstance(getattr(row, "display_gene_label", ""), str)
        and getattr(row, "display_gene_label", "")
        else "no_gene_label"
    )
    return f"{row.snp_id} | {gene_label}"


def plot_ranked_snp_metric(
    snp_table: pd.DataFrame,
    output_path: Path,
    top_n: int,
    trait: int,
    dpi: int,
) -> None:
    top = snp_table.head(min(top_n, len(snp_table))).copy()
    top = top.sort_values("integrated_gradients_abs", ascending=True)
    categories = top["new_impact"].fillna("unknown").tolist()
    palette = build_palette(sorted(pd.unique(categories)))
    colors = [palette.get(category if isinstance(category, str) else "unknown") for category in categories]
    labels = [format_snp_label(row) for row in top.itertuples(index=False)]

    fig_height = max(7, 0.35 * len(top) + 2)
    plt.figure(figsize=(13, fig_height))
    plt.barh(labels, top["integrated_gradients_abs"], color=colors, edgecolor="none")
    plt.xlabel("Mean |integrated gradients|")
    plt.ylabel("SNP")
    plt.title(
        trait_plot_title(
            trait,
            f"Top {min(top_n, len(snp_table))} SNPs by integrated gradients",
        )
    )

    handles = []
    used = set()
    for category in categories:
        key = category if isinstance(category, str) else "unknown"
        if key in used:
            continue
        used.add(key)
        handles.append(plt.Rectangle((0, 0), 1, 1, color=palette[key], label=key))
    if handles:
        plt.legend(handles=handles, title="new_impact", loc="lower right")

    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_ranked_feature_metric(
    table: pd.DataFrame,
    output_path: Path,
    top_n: int,
    label_column: str,
    title: str,
    x_label: str,
    trait: int,
    dpi: int,
) -> None:
    if table.empty:
        return

    top = table.head(min(top_n, len(table))).copy()
    top = top.sort_values("integrated_gradients_abs", ascending=True)

    fig_height = max(6, 0.35 * len(top) + 2)
    plt.figure(figsize=(11, fig_height))
    plt.barh(top[label_column], top["integrated_gradients_abs"], color="#bc6c25", edgecolor="none")
    plt.xlabel(x_label)
    plt.ylabel(label_column.replace("_", " ").title())
    plt.title(trait_plot_title(trait, title))
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_metric_vs_correlation(
    table: pd.DataFrame,
    output_path: Path,
    corr_column: str,
    corr_xlabel: str,
    title: str,
    trait: int,
    dpi: int,
    label_column: str | None = None,
) -> None:
    working = table.dropna(subset=[corr_column, "integrated_gradients_abs"]).copy()
    if working.empty:
        return

    plt.figure(figsize=(8, 6))
    plt.scatter(
        np.abs(working[corr_column]),
        working["integrated_gradients_abs"],
        alpha=0.2,
        s=16,
        color="#7a7a7a",
        edgecolor="none",
    )
    top = working.head(min(20, len(working))).copy()
    plt.scatter(
        np.abs(top[corr_column]),
        top["integrated_gradients_abs"],
        alpha=0.9,
        s=28,
        color="#d95f02",
        edgecolor="none",
    )
    if label_column is not None and label_column in top.columns:
        for row in top.head(12).itertuples(index=False):
            plt.annotate(
                getattr(row, label_column),
                (abs(getattr(row, corr_column)), row.integrated_gradients_abs),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                alpha=0.92,
            )
    plt.xlabel(corr_xlabel)
    plt.ylabel("Integrated gradients")
    plt.title(trait_plot_title(trait, title))
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def chrom_sort_key(value: Any) -> tuple[int, Any]:
    string_value = str(value).replace("chr", "").replace("CHR", "")
    try:
        return 0, int(string_value)
    except ValueError:
        return 1, string_value


def plot_snp_manhattan(
    snp_table: pd.DataFrame,
    output_path: Path,
    trait: int,
    dpi: int,
) -> None:
    required_columns = {"chromosome", "position", "integrated_gradients_abs"}
    if not required_columns.issubset(snp_table.columns):
        return

    manhattan = snp_table.dropna(subset=["chromosome", "position"]).copy()
    if manhattan.empty:
        return

    manhattan["chromosome"] = manhattan["chromosome"].astype(str)
    manhattan["position"] = pd.to_numeric(manhattan["position"], errors="coerce")
    manhattan = manhattan.dropna(subset=["position"])
    manhattan = manhattan.sort_values(
        ["chromosome", "position"],
        key=lambda series: (
            series.map(chrom_sort_key) if series.name == "chromosome" else series
        ),
    )

    chrom_order = sorted(manhattan["chromosome"].unique(), key=chrom_sort_key)
    colors = ["#5f0f40", "#0f4c5c"]
    tick_positions = []
    tick_labels = []
    offset = 0.0
    x_values = np.zeros(len(manhattan), dtype=np.float64)

    for index, chromosome in enumerate(chrom_order):
        chrom_mask = manhattan["chromosome"] == chromosome
        chrom_positions = manhattan.loc[chrom_mask, "position"].to_numpy(dtype=np.float64)
        x_values[chrom_mask.to_numpy()] = chrom_positions + offset
        tick_positions.append(offset + chrom_positions.max() / 2.0)
        tick_labels.append(chromosome)
        offset += chrom_positions.max() + 1e6

    manhattan["plot_x"] = x_values

    plt.figure(figsize=(14, 5.5))
    for index, chromosome in enumerate(chrom_order):
        chrom_df = manhattan.loc[manhattan["chromosome"] == chromosome]
        plt.scatter(
            chrom_df["plot_x"],
            chrom_df["integrated_gradients_abs"],
            s=10,
            alpha=0.7,
            color=colors[index % len(colors)],
            edgecolor="none",
        )

    highlight = manhattan.head(min(25, len(manhattan))).copy()
    highlight["plot_label"] = [format_snp_label(row) for row in highlight.itertuples(index=False)]
    plt.scatter(
        highlight["plot_x"],
        highlight["integrated_gradients_abs"],
        s=20,
        alpha=0.95,
        color="#f4a261",
        edgecolor="none",
    )
    for row in highlight.head(12).itertuples(index=False):
        plt.annotate(
            row.plot_label,
            (row.plot_x, row.integrated_gradients_abs),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            alpha=0.92,
        )

    plt.xticks(tick_positions, tick_labels)
    plt.xlabel("Chromosome")
    plt.ylabel("Integrated gradients")
    plt.title(
        trait_plot_title(trait, "Genome-wide integrated-gradients landscape")
    )
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_impact_composition(
    composition_table: pd.DataFrame,
    output_path: Path,
    trait: int,
    dpi: int,
) -> None:
    if composition_table.empty:
        return

    pivot = (
        composition_table.pivot(index="new_impact", columns="subset", values="percentage")
        .fillna(0.0)
    )
    ordered_classes = composition_table.loc[
        composition_table["subset"] == "Overall"
    ].sort_values("percentage", ascending=False)["new_impact"]
    pivot = pivot.reindex(ordered_classes)

    subsets = list(pivot.columns)
    x = np.arange(len(pivot.index))
    width = 0.8 / max(len(subsets), 1)
    colors = plt.get_cmap("PuOr")(np.linspace(0.15, 0.85, len(subsets)))

    fig_width = max(12, 0.6 * len(pivot.index) + 4)
    plt.figure(figsize=(fig_width, 6))
    for idx, subset in enumerate(subsets):
        plt.bar(
            x + (idx - (len(subsets) - 1) / 2) * width,
            pivot[subset].to_numpy(),
            width=width,
            label=subset,
            color=colors[idx],
            edgecolor="none",
        )

    plt.xticks(x, pivot.index, rotation=45, ha="right")
    plt.ylabel("Percentage of SNPs")
    plt.title(
        trait_plot_title(
            trait,
            "impact-class composition in the full model vs top-ranked SNPs by integrated gradients",
        )
    )
    plt.legend()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_impact_enrichment(
    enrichment_table: pd.DataFrame,
    output_path: Path,
    trait: int,
    dpi: int,
) -> None:
    if enrichment_table.empty:
        return

    preferred_n = 100 if 100 in set(enrichment_table["top_n"]) else int(enrichment_table["top_n"].max())
    subset = enrichment_table.loc[enrichment_table["top_n"] == preferred_n].copy()
    subset = subset.sort_values("log2_enrichment", ascending=False)
    subset = pd.concat([subset.head(8), subset.tail(8)]).drop_duplicates("new_impact")
    subset = subset.sort_values("log2_enrichment", ascending=True)

    plt.figure(figsize=(10, max(6, 0.35 * len(subset) + 2)))
    colors = ["#4c956c" if value >= 0 else "#bc4749" for value in subset["log2_enrichment"]]
    plt.barh(subset["new_impact"], subset["log2_enrichment"], color=colors, edgecolor="none")
    plt.axvline(0.0, color="#2e2e2e", linestyle="--")
    plt.xlabel("log2 enrichment relative to all SNPs")
    plt.ylabel("new_impact")
    plt.title(
        trait_plot_title(
            trait,
            f"impact-class enrichment among top {preferred_n} SNPs by integrated gradients",
        )
    )
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def write_summary(
    output_dir: Path,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float],
    snp_table: pd.DataFrame,
    gene_table: pd.DataFrame,
    pathway_table: pd.DataFrame | None = None,
) -> None:
    trait = int(checkpoint["training_args"]["trait"])
    lines = [
        "# Integrated Gradients Summary",
        "",
        f"- Checkpoint: {checkpoint_path}",
        f"- Model type: {checkpoint['architecture']['model_type']}",
        f"- Trait: {trait_label(trait)} ({trait})",
        (
            "- Validation metrics: "
            f"MAE={validation_metrics['mae']:.6f}, "
            f"MSE={validation_metrics['mse']:.6f}, "
            f"RMSE={validation_metrics['rmse']:.6f}, "
            f"R2={validation_metrics['r2']:.6f}, "
            f"Pearson={validation_metrics['pearson']:.6f}"
        ),
        (
            "- Test metrics: "
            f"MAE={test_metrics['mae']:.6f}, "
            f"MSE={test_metrics['mse']:.6f}, "
            f"RMSE={test_metrics['rmse']:.6f}, "
            f"R2={test_metrics['r2']:.6f}, "
            f"Pearson={test_metrics['pearson']:.6f}"
        ),
        "",
        "## Top SNPs",
    ]

    for row in snp_table.head(10).itertuples(index=False):
        gene_label = (
            row.display_gene_label
            if isinstance(row.display_gene_label, str) and row.display_gene_label
            else "NA"
        )
        lines.append(
            f"- {row.snp_id} | gene={gene_label} | impact={row.new_impact} | "
            f"IG={row.integrated_gradients_abs:.6f} | "
            f"|corr|={row.abs_genotype_trait_corr:.6f}"
        )

    lines.extend(["", "## Top Genes"])
    for row in gene_table.head(10).itertuples(index=False):
        lines.append(
            f"- {row.gene_label} ({row.ensembl_gene_id}) | n_snps={row.n_snps} | "
            f"IG={row.integrated_gradients_abs:.6f} | "
            f"|gene_feature_corr|={row.abs_gene_feature_trait_corr:.6f}"
        )

    if pathway_table is not None and not pathway_table.empty:
        lines.extend(["", "## Top Pathways"])
        for row in pathway_table.head(10).itertuples(index=False):
            lines.append(
                f"- {row.pathway_label} | "
                f"IG={row.integrated_gradients_abs:.6f} | "
                f"|pathway_corr|={row.abs_pathway_trait_corr:.6f}"
            )

    (output_dir / "summary.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    set_plot_style()

    checkpoint_path, checkpoint = load_checkpoint(args.checkpoint)
    output_dir = infer_output_dir(checkpoint_path, args.output_dir)
    device = torch.device(args.device)

    validation_data = reconstruct_validation_data_from_checkpoint(
        checkpoint,
        max_samples=args.max_samples,
    )
    test_data = reconstruct_test_data_from_checkpoint(
        checkpoint,
        full_phenotype_path=args.full_phenotype_path,
    )
    trait = int(validation_data["trait"])

    model = load_model_from_checkpoint(checkpoint, device=device)
    model.eval()

    validation_pred_scaled = batched_predict(
        model,
        validation_data["X_scaled"],
        device,
        args.batch_size,
    )
    validation_pred = inverse_scale(
        validation_pred_scaled,
        validation_data["scaler_y_mean"],
        validation_data["scaler_y_scale"],
    ).reshape(-1)
    validation_metrics = compute_regression_metrics(
        validation_data["y_raw"],
        validation_pred,
    )

    test_pred_scaled = batched_predict(
        model,
        test_data["X_scaled"],
        device,
        args.batch_size,
    )
    test_pred = inverse_scale(
        test_pred_scaled,
        test_data["scaler_y_mean"],
        test_data["scaler_y_scale"],
    ).reshape(-1)
    test_metrics = compute_regression_metrics(
        test_data["y_raw"],
        test_pred,
    )

    pd.DataFrame(
        {
            "sample_id": validation_data["sample_ids_eval"],
            "y_true": validation_data["y_raw"],
            "y_pred": validation_pred,
            "residual": validation_data["y_raw"] - validation_pred,
        }
    ).to_csv(output_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(
        {
            "sample_id": test_data["sample_ids_eval"],
            "y_true": test_data["y_raw"],
            "y_pred": test_pred,
            "residual": test_data["y_raw"] - test_pred,
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)

    snp_ig_result = compute_integrated_gradients(
        model,
        validation_data["X_scaled"],
        device,
        args.batch_size,
        steps=args.ig_steps,
        max_samples=args.ig_samples,
    )
    snp_table = build_snp_table(
        validation_data=validation_data,
        checkpoint=checkpoint,
        ig_abs=np.asarray(snp_ig_result["abs"], dtype=np.float32),
        ig_signed=np.asarray(snp_ig_result["signed"], dtype=np.float32),
    )

    gene_feature_matrix = encode_gene_features(
        model,
        validation_data["X_scaled"],
        device,
        args.batch_size,
    )
    gene_ig_result = compute_integrated_gradients(
        GeneToOutputWrapper(model),
        gene_feature_matrix,
        device,
        args.batch_size,
        steps=args.ig_steps,
        max_samples=args.ig_samples,
    )
    gene_table = build_gene_table(
        validation_data=validation_data,
        checkpoint=checkpoint,
        gene_feature_matrix=gene_feature_matrix,
        ig_abs=np.asarray(gene_ig_result["abs"], dtype=np.float32),
        ig_signed=np.asarray(gene_ig_result["signed"], dtype=np.float32),
    )

    pathway_table = None
    if checkpoint_supports_pathways(checkpoint):
        pathway_feature_matrix = encode_pathway_features(
            model,
            validation_data["X_scaled"],
            device,
            args.batch_size,
        )
        pathway_ig_result = compute_integrated_gradients(
            PathwayToOutputWrapper(model),
            pathway_feature_matrix,
            device,
            args.batch_size,
            steps=args.ig_steps,
            max_samples=args.ig_samples,
        )
        pathway_table = build_pathway_table(
            validation_data=validation_data,
            checkpoint=checkpoint,
            pathway_feature_matrix=pathway_feature_matrix,
            ig_abs=np.asarray(pathway_ig_result["abs"], dtype=np.float32),
            ig_signed=np.asarray(pathway_ig_result["signed"], dtype=np.float32),
        )
        pathway_table.to_csv(output_dir / "integrated_gradients_pathway.csv", index=False)

    composition_table, enrichment_table = build_impact_tables(
        snp_table,
        args.top_impact_sizes,
    )

    snp_table.to_csv(output_dir / "integrated_gradients_snp.csv", index=False)
    gene_table.to_csv(output_dir / "integrated_gradients_gene.csv", index=False)
    composition_table.to_csv(
        output_dir / "impact_class_composition_integrated_gradients.csv",
        index=False,
    )
    enrichment_table.to_csv(
        output_dir / "impact_class_enrichment_integrated_gradients.csv",
        index=False,
    )

    metadata = {
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "model_type": checkpoint["architecture"]["model_type"],
        "trait": trait,
        "trait_label": trait_label(trait),
        "n_validation_samples": int(len(validation_data["sample_ids_eval"])),
        "n_test_samples": int(len(test_data["sample_ids_eval"])),
        "ig_steps": int(args.ig_steps),
        "ig_samples_requested": int(args.ig_samples),
        "ig_samples_used_snp": int(len(snp_ig_result["sample_indices"])),
        "ig_samples_used_gene": int(len(gene_ig_result["sample_indices"])),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    (output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2))

    plot_ranked_snp_metric(
        snp_table,
        output_dir / "top_snps_integrated_gradients.png",
        args.top_snps,
        trait,
        args.dpi,
    )
    plot_metric_vs_correlation(
        snp_table,
        output_dir / "snp_integrated_gradients_vs_correlation.png",
        corr_column="genotype_trait_corr",
        corr_xlabel="|Pearson correlation| between SNP genotype and trait",
        title="Integrated gradients vs univariate SNP-trait signal",
        trait=trait,
        dpi=args.dpi,
        label_column="display_gene_label",
    )
    plot_snp_manhattan(
        snp_table,
        output_dir / "snp_integrated_gradients_manhattan.png",
        trait,
        args.dpi,
    )
    plot_ranked_feature_metric(
        gene_table,
        output_dir / "top_genes_integrated_gradients.png",
        args.top_genes,
        label_column="gene_label",
        title=f"Top {min(args.top_genes, len(gene_table))} genes by integrated gradients",
        x_label="Mean |integrated gradients|",
        trait=trait,
        dpi=args.dpi,
    )
    plot_metric_vs_correlation(
        gene_table,
        output_dir / "gene_integrated_gradients_vs_correlation.png",
        corr_column="gene_feature_trait_corr",
        corr_xlabel="|Pearson correlation| between learned gene feature and trait",
        title="Integrated gradients vs learned gene-feature correlation",
        trait=trait,
        dpi=args.dpi,
        label_column="gene_label",
    )
    plot_impact_composition(
        composition_table,
        output_dir / "impact_class_composition_integrated_gradients.png",
        trait,
        args.dpi,
    )
    plot_impact_enrichment(
        enrichment_table,
        output_dir / "impact_class_enrichment_integrated_gradients.png",
        trait,
        args.dpi,
    )

    if pathway_table is not None:
        plot_ranked_feature_metric(
            pathway_table,
            output_dir / "top_pathways_integrated_gradients.png",
            args.top_pathways,
            label_column="pathway_label",
            title=(
                f"Top {min(args.top_pathways, len(pathway_table))} pathways by integrated gradients"
            ),
            x_label="Mean |integrated gradients|",
            trait=trait,
            dpi=args.dpi,
        )
        plot_metric_vs_correlation(
            pathway_table,
            output_dir / "pathway_integrated_gradients_vs_correlation.png",
            corr_column="pathway_trait_corr",
            corr_xlabel="|Pearson correlation| between pathway feature and trait",
            title="Integrated gradients vs learned pathway-feature correlation",
            trait=trait,
            dpi=args.dpi,
            label_column="pathway_label",
        )

    write_summary(
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        snp_table=snp_table,
        gene_table=gene_table,
        pathway_table=pathway_table,
    )


if __name__ == "__main__":
    main()
