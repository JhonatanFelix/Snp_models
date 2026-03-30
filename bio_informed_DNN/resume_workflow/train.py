"""
Compact training entrypoint for the reduced biologically informed workflow.

The goal of this script to provide one clean and documented trainer that 
covers the three model families used in the current work:

- gene_only
- gene_pathway
- attention_gene
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
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
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    from bio_informed_DNN.resume_workflow.dataset import (
        DEFAULT_ANNOTATION_PATH,
        DEFAULT_FULL_PHENOTYPE_PATH,
        DEFAULT_GENOTYPE_PREFIX,
        DEFAULT_PATHWAY_GENE_MAPPING_PATH,
        DEFAULT_PATHWAY_MAPPING_PATH,
        DEFAULT_PATHWAY_MASK_PATH,
        DEFAULT_PEDIGREE_PATH,
        DEFAULT_PHENOTYPE_PATH,
        attach_pathway_resources,
        build_annotation_resources,
        load_annotation_dataframe,
        load_trait_dataset,
        load_trait_test_dataset,
        sparse_mask_to_index_arrays,
        split_last_n_samples,
    )
    from bio_informed_DNN.resume_workflow.models import (
        build_model,
        compact_state_dict,
    )
except ImportError:
    from dataset import (  # type: ignore
        DEFAULT_ANNOTATION_PATH,
        DEFAULT_FULL_PHENOTYPE_PATH,
        DEFAULT_GENOTYPE_PREFIX,
        DEFAULT_PATHWAY_GENE_MAPPING_PATH,
        DEFAULT_PATHWAY_MAPPING_PATH,
        DEFAULT_PATHWAY_MASK_PATH,
        DEFAULT_PEDIGREE_PATH,
        DEFAULT_PHENOTYPE_PATH,
        attach_pathway_resources,
        build_annotation_resources,
        load_annotation_dataframe,
        load_trait_dataset,
        load_trait_test_dataset,
        sparse_mask_to_index_arrays,
        split_last_n_samples,
    )
    from models import build_model, compact_state_dict  # type: ignore


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "outputs"

TRAIT_NAMES = {
    2: "shoulder",
    3: "top",
    4: "buttock_side",
    5: "buttock_rear",
    6: "size",
    7: "musculature",
}


class EarlyStopping:
    """Validation-loss early stopping used by the current project."""

    def __init__(self, patience: int = 9, min_delta: float = 1e-5) -> None:
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_loss: float | None = None
        self.counter = 0
        self.early_stop = False

    def step(self, val_loss: float) -> None:
        if self.best_loss is None:
            self.best_loss = val_loss
            return

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return

        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one of the compact biologically informed models with the "
            "deterministic birth-date split used in the project."
        )
    )

    parser.add_argument(
        "--model-type",
        choices=["gene_only", "gene_pathway", "attention_gene"],
        required=True,
        help="Model family to train.",
    )
    parser.add_argument(
        "--trait",
        type=int,
        default=2,
        help="Trait index. Project convention: 2..7.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[50],
        help="Dense hidden layers after the biological encoder.",
    )

    parser.add_argument(
        "--annotation-path",
        type=str,
        default=str(DEFAULT_ANNOTATION_PATH),
    )
    parser.add_argument(
        "--genotype-prefix",
        type=str,
        default=str(DEFAULT_GENOTYPE_PREFIX),
    )
    parser.add_argument(
        "--phenotype-path",
        type=str,
        default=str(DEFAULT_PHENOTYPE_PATH),
    )
    parser.add_argument(
        "--full-phenotype-path",
        type=str,
        default=str(DEFAULT_FULL_PHENOTYPE_PATH),
    )
    parser.add_argument(
        "--pedigree-path",
        type=str,
        default=str(DEFAULT_PEDIGREE_PATH),
    )
    parser.add_argument(
        "--pathway-mask-path",
        type=str,
        default=str(DEFAULT_PATHWAY_MASK_PATH),
        help="Required only for model_type=gene_pathway.",
    )
    parser.add_argument(
        "--pathway-mapping-path",
        type=str,
        default=str(DEFAULT_PATHWAY_MAPPING_PATH),
        help="Optional pathway index mapping for model_type=gene_pathway.",
    )
    parser.add_argument(
        "--pathway-gene-mapping-path",
        type=str,
        default=str(DEFAULT_PATHWAY_GENE_MAPPING_PATH),
        help="Optional gene mapping used to realign the saved pathway mask.",
    )

    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--criterion", type=str, default="HuberLoss")
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--attention-dropout", type=float, default=0.0)
    parser.add_argument("--impact-embedding-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stop", type=str, default="true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-layernorm", type=str, default="true")

    parser.add_argument(
        "--save-model",
        type=str,
        default="true",
        help="If true, save a compact checkpoint for later analysis.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(OUTPUT_ROOT),
        help="Root directory where run folders are created.",
    )

    return parser.parse_args()


def flag_is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def trait_label(trait: int) -> str:
    return TRAIT_NAMES.get(int(trait), f"trait_{trait}")


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_criterion(name: str) -> nn.Module:
    name = str(name)
    if name == "MSE":
        return nn.MSELoss()
    if name == "MAE":
        return nn.L1Loss()
    if name == "HuberLoss":
        return nn.HuberLoss()
    raise ValueError(f"Unsupported criterion: {name}")


def build_optimizer(
    name: str,
    parameters: Any,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    name = str(name).lower()
    if name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def compute_real_scale_metrics(
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    scaler_y: StandardScaler,
) -> dict[str, float]:
    y_true_real = scaler_y.inverse_transform(y_true_scaled).reshape(-1)
    y_pred_real = scaler_y.inverse_transform(y_pred_scaled).reshape(-1)
    pearson_corr, _ = pearsonr(y_true_real, y_pred_real)
    return {
        "MAE": float(mean_absolute_error(y_true_real, y_pred_real)),
        "MSE": float(mean_squared_error(y_true_real, y_pred_real)),
        "R2": float(r2_score(y_true_real, y_pred_real)),
        "Pearson": float(pearson_corr),
    }


def safe_torch_save(checkpoint: dict[str, Any], checkpoint_path: Path) -> None:
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    try:
        torch.save(checkpoint, tmp_path, _use_new_zipfile_serialization=False)
        os.replace(tmp_path, checkpoint_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def build_run_name(args: argparse.Namespace) -> str:
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    layers = "-".join(str(layer) for layer in args.layers)
    return (
        f"{args.model_type}_trait{args.trait}_{trait_label(args.trait)}"
        f"_layers{layers}_lr{args.learning_rate}"
        f"_crit{args.criterion}_act{args.activation}"
        f"_seed{args.seed}_{timestamp}"
    )


def create_run_directory(args: argparse.Namespace) -> Path:
    run_name = build_run_name(args)
    run_dir = Path(args.output_root) / args.model_type / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def configure_logging(log_path: Path) -> None:
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )


def plot_training_curve(
    train_losses: list[float],
    val_losses: list[float],
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.savefig(output_path, dpi=220)
    plt.close()


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    scaler_y: StandardScaler,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray]:
    predictions = []
    y_batches = []
    running_loss = 0.0

    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            outputs = model(xb)
            running_loss += criterion(outputs, yb).item()
            predictions.append(outputs.detach().cpu().numpy())
            y_batches.append(yb.detach().cpu().numpy())

    y_pred_scaled = np.vstack(predictions).astype(np.float32)
    y_true_scaled = np.vstack(y_batches).astype(np.float32)
    metrics = compute_real_scale_metrics(y_true_scaled, y_pred_scaled, scaler_y)
    metrics["scaled_loss"] = float(running_loss / max(len(loader), 1))
    return metrics, y_pred_scaled


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    run_dir = create_run_directory(args)
    configure_logging(run_dir / "train.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Starting compact training workflow.")
    logging.info("Run directory: %s", run_dir)
    logging.info("Device: %s", device)
    logging.info("Configuration: %s", json.dumps(vars(args), indent=2))

    annotation_df, dropped_rows = load_annotation_dataframe(args.annotation_path)
    resources = build_annotation_resources(annotation_df)
    duplicate_snp_count = int(annotation_df["snp_id"].duplicated().sum())
    conflicting_impacts = int(
        (annotation_df.groupby("snp_id")["new_impact"].nunique() > 1).sum()
    )

    if args.model_type == "gene_pathway":
        resources = attach_pathway_resources(
            resources,
            mask_gene_pathway_path=args.pathway_mask_path,
            pathway_mapping_path=args.pathway_mapping_path,
            gene_mapping_path=args.pathway_gene_mapping_path,
        )
        logging.info(
            "Loaded pathway resources with %d pathways.",
            resources.n_pathways,
        )

    logging.info(
        "Annotation rows kept: %d (dropped %d rows with missing SNP/gene/new_impact).",
        len(annotation_df),
        dropped_rows,
    )
    logging.info("Unique SNPs: %d", resources.n_snps)
    logging.info("Unique genes: %d", resources.n_genes)
    logging.info("Unique impact classes: %d", resources.n_impacts)
    logging.info("Duplicate SNP annotations: %d", duplicate_snp_count)
    logging.info("SNPs with conflicting impact labels: %d", conflicting_impacts)

    dataset = load_trait_dataset(
        genotype_prefix=args.genotype_prefix,
        phenotype_path=args.phenotype_path,
        trait=args.trait,
        snp_ids=resources.snp_map["snp_id"].to_numpy(),
        pedigree_path=args.pedigree_path,
    )
    test_dataset = load_trait_test_dataset(
        genotype_prefix=args.genotype_prefix,
        masked_phenotype_path=args.phenotype_path,
        full_phenotype_path=args.full_phenotype_path,
        trait=args.trait,
        snp_ids=resources.snp_map["snp_id"].to_numpy(),
        pedigree_path=args.pedigree_path,
    )

    split = split_last_n_samples(
        dataset["X"],
        dataset["y"],
        sample_ids=dataset["sample_ids_valid"],
        val_size=args.val_size,
    )

    logging.info(
        "Using deterministic validation holdout with the last %d valid samples after %s ordering.",
        split["val_size"],
        dataset["sample_ordering"],
    )
    logging.info(
        "Validation sample IDs span %s to %s.",
        split["sample_ids_val"][0],
        split["sample_ids_val"][-1],
    )
    logging.info(
        "Held-out masked test set size: %d samples.",
        len(test_dataset["sample_ids_test"]),
    )

    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(split["X_train"]).astype(np.float32)
    X_val = scaler_X.transform(split["X_val"]).astype(np.float32)
    X_test = scaler_X.transform(test_dataset["X"]).astype(np.float32)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(split["y_train"].reshape(-1, 1)).astype(np.float32)
    y_val = scaler_y.transform(split["y_val"].reshape(-1, 1)).astype(np.float32)
    y_test = scaler_y.transform(test_dataset["y"].reshape(-1, 1)).astype(np.float32)

    model = build_model(
        model_type=args.model_type,
        input_dim=resources.n_snps,
        gene_dim=resources.n_genes,
        pathway_dim=resources.n_pathways,
        fc_layers=args.layers,
        mask_snp_gene=resources.mask_snp_gene_tensor,
        mask_gene_pathway=resources.mask_gene_pathway_tensor,
        impact_indices=resources.impact_indices,
        num_impacts=resources.n_impacts,
        activation=args.activation,
        dropout=args.dropout,
        use_layernorm=flag_is_true(args.use_layernorm),
        impact_embedding_dim=args.impact_embedding_dim,
        attention_dropout=args.attention_dropout,
    ).to(device)

    criterion = build_criterion(args.criterion)
    optimizer = build_optimizer(
        args.optimizer,
        model.parameters(),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=4,
        factor=0.5,
        threshold=min(1e-6, args.learning_rate / 1000.0),
    )
    early_stopper = EarlyStopping(patience=9, min_delta=1e-5)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_train_loss: float | None = None
    best_state_dict: dict[str, torch.Tensor] | None = None

    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_grad_norm = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=args.grad_clip,
                )

            batch_grad_norm = 0.0
            for parameter in model.parameters():
                if parameter.grad is not None:
                    batch_grad_norm += parameter.grad.data.norm(2).item()
            epoch_grad_norm += batch_grad_norm

            optimizer.step()
            epoch_train_loss += loss.item()

        epoch_train_loss /= max(len(train_loader), 1)
        epoch_grad_norm /= max(len(train_loader), 1)
        train_losses.append(epoch_train_loss)
        logging.info(
            "Epoch %d | Train Loss: %.6f | Avg Grad Norm: %.6f",
            epoch,
            epoch_train_loss,
            epoch_grad_norm,
        )

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                epoch_val_loss += criterion(model(xb), yb).item()
        epoch_val_loss /= max(len(val_loader), 1)
        val_losses.append(epoch_val_loss)
        logging.info(
            "Epoch %d | Val Loss: %.6f | LR: %.6e",
            epoch,
            epoch_val_loss,
            optimizer.param_groups[0]["lr"],
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            best_train_loss = epoch_train_loss
            best_state_dict = compact_state_dict(model)

        scheduler.step(epoch_val_loss)

        if flag_is_true(args.early_stop):
            early_stopper.step(epoch_val_loss)
            if early_stopper.early_stop:
                logging.info("Early stopping triggered.")
                break

    if best_state_dict is not None:
        missing = model.load_state_dict(best_state_dict, strict=False)
        if missing.unexpected_keys:
            raise RuntimeError(
                f"Unexpected keys when restoring best model: {missing.unexpected_keys}"
            )
        logging.info(
            "Restored best model from epoch %d with train loss %.6f and val loss %.6f.",
            best_epoch,
            best_train_loss if best_train_loss is not None else float("nan"),
            best_val_loss,
        )

    validation_metrics, _ = evaluate_model(
        model,
        val_loader,
        criterion,
        scaler_y,
        device,
    )
    test_metrics, _ = evaluate_model(
        model,
        test_loader,
        criterion,
        scaler_y,
        device,
    )
    validation_metrics["best_epoch"] = int(best_epoch)
    validation_metrics["best_val_loss"] = float(best_val_loss)
    validation_metrics["best_train_loss"] = (
        None if best_train_loss is None else float(best_train_loss)
    )

    logging.info("Final validation metrics: %s", validation_metrics)
    logging.info("Final test metrics: %s", test_metrics)

    plot_training_curve(
        train_losses,
        val_losses,
        run_dir / "training_curve.png",
    )

    metrics_payload = {
        "model_type": args.model_type,
        "trait": int(args.trait),
        "trait_label": trait_label(args.trait),
        "validation": validation_metrics,
        "test": test_metrics,
        "run_directory": str(run_dir),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))

    if flag_is_true(args.save_model):
        checkpoint_path = run_dir / "checkpoint.pt"
        pathway_mask_payload = sparse_mask_to_index_arrays(resources.mask_gene_pathway)
        checkpoint = {
            "checkpoint_format": "resume_workflow_v1",
            "model_state_dict": compact_state_dict(model),
            "architecture": {
                "model_type": args.model_type,
                "input_dim": int(resources.n_snps),
                "gene_dim": int(resources.n_genes),
                "pathway_dim": int(resources.n_pathways),
                "fc_layers": list(args.layers),
                "activation": args.activation,
                "dropout": float(args.dropout),
                "use_layernorm": bool(flag_is_true(args.use_layernorm)),
                "impact_embedding_dim": int(args.impact_embedding_dim),
                "attention_dropout": float(args.attention_dropout),
            },
            "training_args": {
                **vars(args),
                "annotation_path": str(args.annotation_path),
                "genotype_prefix": str(args.genotype_prefix),
                "phenotype_path": str(args.phenotype_path),
                "full_phenotype_path": str(args.full_phenotype_path),
                "pedigree_path": str(args.pedigree_path),
                "val_size": int(args.val_size),
                "sample_ordering": dataset["sample_ordering"],
                "split_strategy": split["split_strategy"],
            },
            "scaler_X_mean": np.asarray(scaler_X.mean_, dtype=np.float32),
            "scaler_X_scale": np.asarray(scaler_X.scale_, dtype=np.float32),
            "scaler_y_mean": np.asarray(scaler_y.mean_, dtype=np.float32),
            "scaler_y_scale": np.asarray(scaler_y.scale_, dtype=np.float32),
            "annotation_edges_dataframe": resources.annotation_edges,
            "snp_map_dataframe": resources.snp_map,
            "gene_map_dataframe": resources.gene_map,
            "impact_map_dataframe": resources.impact_map,
            "snp_annotation_dataframe": resources.snp_annotation,
            "pathway_map_dataframe": resources.pathway_map,
            "impact_indices": resources.impact_indices.cpu().numpy(),
            "n_impacts": int(resources.n_impacts),
            "sample_ids_valid": np.asarray(dataset["sample_ids_valid"], dtype=str),
            "sample_ids_val": np.asarray(split["sample_ids_val"], dtype=str),
            "mask_gene_pathway_row_indices": pathway_mask_payload["row_indices"],
            "mask_gene_pathway_col_indices": pathway_mask_payload["col_indices"],
            "mask_gene_pathway_shape": pathway_mask_payload["shape"],
            "final_metrics": metrics_payload,
        }
        safe_torch_save(checkpoint, checkpoint_path)
        logging.info("Saved compact checkpoint at: %s", checkpoint_path)
    else:
        logging.info("Checkpoint saving skipped.")


if __name__ == "__main__":
    main()
