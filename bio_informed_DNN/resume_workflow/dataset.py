"""
Compact dataset utilities for the biologically informed SNP models.

This module keeps only the data logic that is central to the current work:

- VEP-driven SNP -> gene resources
- optional gene -> pathway resources
- the deterministic validation split used in the project
  (the last 1000 birth-date ordered valid animals)
- reconstruction of the masked test cohort from the full phenotype file

The implementation is intentionally self-contained so it can serve as a
readable reference version of the larger experimental pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from pandas_plink import read_plink
from scipy.sparse import coo_matrix, load_npz


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_ANNOTATION_PATH = PROJECT_ROOT / "data/ML/vep_ingenes_newimpact.csv"
DEFAULT_GENOTYPE_PREFIX = PROJECT_ROOT / "data/ML/BBB2023_MD"
DEFAULT_PHENOTYPE_PATH = PROJECT_ROOT / "data/ML/pheno_2023bbb_0twins_6traits_mask"
DEFAULT_FULL_PHENOTYPE_PATH = PROJECT_ROOT / "data/ML/pheno_20000bbb_6traits"
DEFAULT_PEDIGREE_PATH = PROJECT_ROOT / "data/ML/pedi_full_list.txt"
DEFAULT_PATHWAY_MASK_PATH = (
    PROJECT_ROOT / "data/preprocessed/Bos_taurus_new/mask_gene_pathway.npz"
)
DEFAULT_PATHWAY_MAPPING_PATH = (
    PROJECT_ROOT / "data/preprocessed/Bos_taurus_new/pathway_index_mapping.csv"
)
DEFAULT_PATHWAY_GENE_MAPPING_PATH = (
    PROJECT_ROOT / "data/preprocessed/Bos_taurus_new/gene_index_mapping.csv"
)


@dataclass
class BiologicalResources:
    """Biological lookup tables and masks aligned to the training genotype order."""

    annotation_edges: pd.DataFrame
    snp_map: pd.DataFrame
    gene_map: pd.DataFrame
    impact_map: pd.DataFrame
    snp_annotation: pd.DataFrame
    mask_snp_gene: coo_matrix
    mask_snp_gene_tensor: torch.Tensor
    impact_indices: torch.Tensor
    n_snps: int
    n_genes: int
    n_impacts: int
    pathway_map: pd.DataFrame | None = None
    mask_gene_pathway: coo_matrix | None = None
    mask_gene_pathway_tensor: torch.Tensor | None = None
    n_pathways: int = 0


def safe_scale(array: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Scale with sklearn-like mean/scale arrays while protecting zero variance."""

    safe_scale_values = np.where(np.asarray(scale) == 0, 1.0, scale)
    return (array - mean) / safe_scale_values


def inverse_scale(
    array: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Undo sklearn StandardScaler-style normalization."""

    return array * scale + mean


def load_annotation_dataframe(annotation_path: str | Path) -> tuple[pd.DataFrame, int]:
    """Load and sanitize the VEP-derived annotation table."""

    annotation_df = pd.read_csv(annotation_path)

    required_columns = {"snp_id", "Gene", "new_impact"}
    missing_columns = sorted(required_columns - set(annotation_df.columns))
    if missing_columns:
        raise ValueError(
            "Annotation file is missing required columns: "
            + ", ".join(missing_columns)
        )

    annotation_df = annotation_df.copy()
    annotation_df["snp_id"] = (
        annotation_df["snp_id"].fillna("").astype(str).str.strip()
    )
    annotation_df["Gene"] = annotation_df["Gene"].fillna("").astype(str).str.strip()
    annotation_df["new_impact"] = (
        annotation_df["new_impact"].fillna("").astype(str).str.strip()
    )

    if "SYMBOL" in annotation_df.columns:
        annotation_df["SYMBOL"] = (
            annotation_df["SYMBOL"].fillna("").astype(str).str.strip()
        )
    else:
        annotation_df["SYMBOL"] = ""

    initial_rows = len(annotation_df)
    annotation_df = annotation_df.loc[
        (annotation_df["snp_id"] != "")
        & (annotation_df["Gene"] != "")
        & (annotation_df["new_impact"] != "")
    ].copy()

    return annotation_df, initial_rows - len(annotation_df)


def build_annotation_resources(annotation_df: pd.DataFrame) -> BiologicalResources:
    """Build ordered SNP/gene/impact resources from the annotation table."""

    snp_order = pd.Index(pd.unique(annotation_df["snp_id"]), name="snp_id")
    gene_order = pd.Index(pd.unique(annotation_df["Gene"]), name="Gene")
    impact_order = pd.Index(
        sorted(annotation_df["new_impact"].unique()),
        name="new_impact",
    )

    snp_index = pd.Series(np.arange(len(snp_order)), index=snp_order)
    gene_index = pd.Series(np.arange(len(gene_order)), index=gene_order)
    impact_index = pd.Series(np.arange(len(impact_order)), index=impact_order)

    row_indices = annotation_df["Gene"].map(gene_index).to_numpy()
    col_indices = annotation_df["snp_id"].map(snp_index).to_numpy()

    mask_snp_gene = coo_matrix(
        (
            np.ones(len(annotation_df), dtype=np.float32),
            (row_indices, col_indices),
        ),
        shape=(len(gene_order), len(snp_order)),
        dtype=np.float32,
    )
    mask_snp_gene.sum_duplicates()
    mask_snp_gene.data[:] = 1.0

    snp_map = pd.DataFrame(
        {
            "snp_index": np.arange(len(snp_order)),
            "snp_id": snp_order,
        }
    )

    gene_symbol_lookup = (
        annotation_df.loc[annotation_df["SYMBOL"] != "", ["Gene", "SYMBOL"]]
        .drop_duplicates(subset=["Gene"])
        .set_index("Gene")["SYMBOL"]
    )

    gene_map = pd.DataFrame(
        {
            "gene_index": np.arange(len(gene_order)),
            "ensembl_gene_id": gene_order,
        }
    )
    gene_map["gene_name"] = (
        gene_map["ensembl_gene_id"].map(gene_symbol_lookup).fillna("")
    )

    annotation_edge_columns = [
        column
        for column in [
            "snp_id",
            "Gene",
            "SYMBOL",
            "new_impact",
            "Consequence",
            "IMPACT",
            "bp",
            "cm",
            "allele1",
            "allele2",
        ]
        if column in annotation_df.columns
    ]
    annotation_edges = annotation_df[annotation_edge_columns].drop_duplicates().copy()

    snp_annotation = (
        annotation_df.drop_duplicates(subset=["snp_id"], keep="first")
        [[column for column in annotation_edge_columns if column != "Gene"]]
        .set_index("snp_id")
        .reindex(snp_order)
        .reset_index()
    )

    impact_indices = torch.tensor(
        snp_annotation["new_impact"].map(impact_index).to_numpy(),
        dtype=torch.long,
    )

    impact_counts = (
        snp_annotation["new_impact"]
        .value_counts()
        .reindex(impact_order, fill_value=0)
    )
    impact_map = pd.DataFrame(
        {
            "impact_index": np.arange(len(impact_order)),
            "new_impact": impact_order,
            "n_snps": impact_counts.to_numpy(),
        }
    )

    return BiologicalResources(
        annotation_edges=annotation_edges,
        snp_map=snp_map,
        gene_map=gene_map,
        impact_map=impact_map,
        snp_annotation=snp_annotation,
        mask_snp_gene=mask_snp_gene,
        mask_snp_gene_tensor=torch.from_numpy(mask_snp_gene.toarray()).float(),
        impact_indices=impact_indices,
        n_snps=len(snp_order),
        n_genes=len(gene_order),
        n_impacts=len(impact_order),
    )


def _load_plink_selection(
    genotype_prefix: str | Path,
    snp_ids: np.ndarray | None,
) -> tuple[pd.DataFrame, Any, pd.Series, np.ndarray, np.ndarray | None]:
    """Load PLINK data and align SNP order if a subset is requested."""

    bim, fam, bed = read_plink(str(genotype_prefix))
    bim = bim.set_index("snp")
    fam = fam.copy()
    fam["iid"] = fam["iid"].astype(str)

    if not fam["iid"].is_unique:
        duplicate_ids = fam.loc[fam["iid"].duplicated(), "iid"].iloc[:10].tolist()
        raise ValueError(
            "Duplicate sample IDs found in the PLINK FAM data. "
            f"First duplicates: {duplicate_ids}"
        )

    sample_to_index = fam.set_index("iid")["i"].astype(np.int64)

    if snp_ids is not None:
        snp_ids = pd.Index(np.asarray(snp_ids, dtype=str), name="snp_id")
        missing_snps = snp_ids[~snp_ids.isin(bim.index)]
        if not missing_snps.empty:
            raise ValueError(
                "Some requested SNPs are missing from the genotype BIM file. "
                f"First missing SNPs: {missing_snps[:10].tolist()}"
            )
        snp_to_index = bim.loc[snp_ids, "i"].to_numpy(dtype=np.int64)
    else:
        snp_ids = bim.index.astype(str)
        snp_to_index = None

    return bim, bed, sample_to_index, np.asarray(snp_ids, dtype=str), snp_to_index


def _read_phenotype_dataframe(phenotype_path: str | Path) -> pd.DataFrame:
    """Read the tab-delimited phenotype matrix used by the project."""

    pheno_df = pd.read_csv(
        phenotype_path,
        delimiter="\t",
        header=None,
    )
    pheno_df = pheno_df.drop(columns=[0]).set_index(1)
    pheno_df.index = pheno_df.index.astype(str)

    if not pheno_df.index.is_unique:
        duplicate_ids = (
            pheno_df.index[pheno_df.index.duplicated()].unique()[:10].tolist()
        )
        raise ValueError(
            "Duplicate sample IDs found in the phenotype file. "
            f"First duplicates: {duplicate_ids}"
        )

    return pheno_df


def _apply_birth_date_order(
    pheno_df: pd.DataFrame,
    pedigree_path: str | Path | None,
) -> tuple[pd.DataFrame, str]:
    """Order the phenotype table exactly as in the current project workflow."""

    sample_ordering = "phenotype_file"
    if pedigree_path is None:
        return pheno_df, sample_ordering

    pedigree_df = pd.read_csv(
        pedigree_path,
        sep="\t",
        names=["sample_id", "sex", "birth_date"],
        header=None,
        dtype={"sample_id": str, "sex": str, "birth_date": str},
    ).set_index("sample_id")
    pedigree_df.index = pedigree_df.index.astype(str)

    missing_ids = pheno_df.index.difference(pedigree_df.index)
    if not missing_ids.empty:
        raise ValueError(
            "Some phenotype sample IDs are missing from the pedigree file. "
            f"First missing IDs: {missing_ids[:10].tolist()}"
        )

    pedigree_df["birth_date"] = pd.to_datetime(
        pedigree_df["birth_date"],
        format="%Y%m%d",
    )

    observed_ids = pheno_df.dropna(how="all").index
    masked_ids = pheno_df.index.difference(observed_ids)
    if len(masked_ids) > 0:
        latest_observed_birth = pedigree_df.loc[observed_ids, "birth_date"].max()
        earliest_masked_birth = pedigree_df.loc[masked_ids, "birth_date"].min()
        if latest_observed_birth >= earliest_masked_birth:
            raise ValueError(
                "The masked cohort must be younger than the observed animals for "
                "the current dataset ordering convention."
            )

    pheno_df = pheno_df.reindex(pedigree_df.sort_values("birth_date").index)
    return pheno_df, "birth_date"


def _extract_aligned_genotypes(
    bed: Any,
    sample_to_index: pd.Series,
    sample_ids: np.ndarray,
    snp_to_index: np.ndarray | None,
    dtype: np.dtype,
) -> np.ndarray:
    """Extract a dense genotype matrix aligned to the requested samples/SNPs."""

    sample_ids = np.asarray(sample_ids, dtype=str)
    missing_ids = pd.Index(sample_ids).difference(sample_to_index.index)
    if not missing_ids.empty:
        raise ValueError(
            "Some phenotype sample IDs are missing from the PLINK FAM data. "
            f"First missing IDs: {missing_ids[:10].tolist()}"
        )

    sample_positions = sample_to_index.loc[sample_ids].to_numpy(dtype=np.int64)
    if snp_to_index is not None:
        X = bed[snp_to_index].T[sample_positions].compute()
    else:
        X = bed.T[sample_positions].compute()
    return np.asarray(X, dtype=dtype)


def load_trait_dataset(
    genotype_prefix: str | Path,
    phenotype_path: str | Path,
    trait: int,
    snp_ids: np.ndarray | None = None,
    dtype: np.dtype = np.float32,
    pedigree_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the observed cohort for one trait in birth-date order."""

    bim, bed, sample_to_index, snp_ids, snp_to_index = _load_plink_selection(
        genotype_prefix,
        snp_ids,
    )
    pheno_df = _read_phenotype_dataframe(phenotype_path)
    pheno_df, sample_ordering = _apply_birth_date_order(pheno_df, pedigree_path)

    trait = int(trait)
    if trait not in pheno_df.columns:
        raise ValueError(
            f"Trait column {trait} not found in phenotype file {phenotype_path}."
        )

    valid_series = pheno_df[trait].dropna()
    sample_ids_valid = valid_series.index.astype(str).to_numpy()
    y = valid_series.to_numpy(dtype=dtype)
    X = _extract_aligned_genotypes(
        bed,
        sample_to_index,
        sample_ids_valid,
        snp_to_index,
        dtype,
    )

    return {
        "bim": bim,
        "X": X,
        "y": np.asarray(y, dtype=dtype),
        "sample_ids_valid": sample_ids_valid,
        "sample_ordering": sample_ordering,
        "trait": trait,
        "snp_ids": np.asarray(snp_ids, dtype=str),
    }


def load_trait_test_dataset(
    genotype_prefix: str | Path,
    masked_phenotype_path: str | Path,
    full_phenotype_path: str | Path,
    trait: int,
    snp_ids: np.ndarray | None = None,
    dtype: np.dtype = np.float32,
    pedigree_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Reconstruct the masked test cohort.

    The masked phenotype file defines the observed train/validation animals.
    Every remaining non-missing individual in the full phenotype file is treated
    as the held-out test set.
    """

    bim, bed, sample_to_index, snp_ids, snp_to_index = _load_plink_selection(
        genotype_prefix,
        snp_ids,
    )
    masked_pheno_df = _read_phenotype_dataframe(masked_phenotype_path)
    masked_pheno_df, _ = _apply_birth_date_order(masked_pheno_df, pedigree_path)
    full_pheno_df = _read_phenotype_dataframe(full_phenotype_path)

    trait = int(trait)
    if trait not in full_pheno_df.columns:
        raise ValueError(
            f"Trait column {trait} not found in phenotype file {full_phenotype_path}."
        )

    train_val_ids = pd.Index(
        masked_pheno_df.dropna(how="all").index.astype(str),
        name="sample_id",
    )
    test_series = full_pheno_df.loc[
        ~full_pheno_df.index.astype(str).isin(train_val_ids),
        trait,
    ].dropna()

    sample_ids_test = test_series.index.astype(str).to_numpy()
    y = test_series.to_numpy(dtype=dtype)
    X = _extract_aligned_genotypes(
        bed,
        sample_to_index,
        sample_ids_test,
        snp_to_index,
        dtype,
    )

    return {
        "bim": bim,
        "X": X,
        "y": np.asarray(y, dtype=dtype),
        "sample_ids_test": sample_ids_test,
        "sample_ordering": "full_phenotype_file",
        "trait": trait,
        "snp_ids": np.asarray(snp_ids, dtype=str),
    }


def split_last_n_samples(
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: np.ndarray | None = None,
    val_size: int = 1000,
) -> dict[str, Any]:
    """Use the last `val_size` birth-date ordered valid animals as validation."""

    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have the same number of samples. Got {X.shape[0]} and {y.shape[0]}."
        )

    val_size = int(val_size)
    if val_size <= 0:
        raise ValueError(f"val_size must be > 0, got {val_size}.")
    if X.shape[0] <= val_size:
        raise ValueError(
            f"Need more than {val_size} samples to create a holdout. Got {X.shape[0]}."
        )

    split_index = X.shape[0] - val_size
    result = {
        "X_train": X[:split_index],
        "X_val": X[split_index:],
        "y_train": y[:split_index],
        "y_val": y[split_index:],
        "split_index": split_index,
        "val_size": val_size,
        "split_strategy": "last_n",
    }

    if sample_ids is not None:
        sample_ids = np.asarray(sample_ids, dtype=str)
        if sample_ids.shape[0] != X.shape[0]:
            raise ValueError(
                "sample_ids must have the same length as X/y. "
                f"Got {sample_ids.shape[0]} and {X.shape[0]}."
            )
        result["sample_ids_train"] = sample_ids[:split_index]
        result["sample_ids_val"] = sample_ids[split_index:]

    return result


def _reorder_pathway_mask_to_gene_map(
    mask_gene_pathway: coo_matrix,
    source_gene_mapping: pd.DataFrame,
    target_gene_map: pd.DataFrame,
) -> coo_matrix:
    """Align a gene -> pathway mask to the gene order used by the SNP annotation."""

    if "ensembl_gene_id" not in source_gene_mapping.columns:
        raise ValueError(
            "gene_mapping_path must include an 'ensembl_gene_id' column."
        )

    source_order = pd.Index(
        source_gene_mapping["ensembl_gene_id"].astype(str),
        name="ensembl_gene_id",
    )
    target_order = pd.Index(
        target_gene_map["ensembl_gene_id"].astype(str),
        name="ensembl_gene_id",
    )

    reorder = source_order.get_indexer(target_order)
    if (reorder < 0).any():
        missing = target_order[reorder < 0][:10].tolist()
        raise ValueError(
            "The pathway gene mapping does not cover the annotation gene set. "
            f"First missing genes: {missing}"
        )

    reordered = mask_gene_pathway.tocsr()[:, reorder]
    return reordered.tocoo()


def attach_pathway_resources(
    resources: BiologicalResources,
    mask_gene_pathway_path: str | Path,
    pathway_mapping_path: str | Path | None = None,
    gene_mapping_path: str | Path | None = None,
) -> BiologicalResources:
    """
    Attach an optional gene -> pathway mask to the annotation-derived resources.

    The current project stores pathway masks externally. This helper aligns them
    to the gene order used by the SNP annotation resources.
    """

    mask_gene_pathway = load_npz(str(mask_gene_pathway_path)).astype(np.float32).tocoo()

    if gene_mapping_path is not None and Path(gene_mapping_path).exists():
        gene_mapping = pd.read_csv(gene_mapping_path)
        mask_gene_pathway = _reorder_pathway_mask_to_gene_map(
            mask_gene_pathway=mask_gene_pathway,
            source_gene_mapping=gene_mapping,
            target_gene_map=resources.gene_map,
        )
    elif mask_gene_pathway.shape[1] != resources.n_genes:
        raise ValueError(
            "The pathway mask does not match the annotation gene dimension. "
            f"Expected {resources.n_genes} columns, got {mask_gene_pathway.shape[1]}. "
            "Provide gene_mapping_path to enable reordering."
        )

    if pathway_mapping_path is not None and Path(pathway_mapping_path).exists():
        pathway_map = pd.read_csv(pathway_mapping_path).copy()
        if "pathway_index" not in pathway_map.columns:
            pathway_map["pathway_index"] = np.arange(len(pathway_map))
        if "pathway_id" not in pathway_map.columns:
            raise ValueError(
                "pathway_mapping_path must include a 'pathway_id' column."
            )
        pathway_map = pathway_map.sort_values("pathway_index").reset_index(drop=True)
    else:
        pathway_map = pd.DataFrame(
            {
                "pathway_index": np.arange(mask_gene_pathway.shape[0]),
                "pathway_id": [
                    f"pathway_{index:05d}"
                    for index in range(mask_gene_pathway.shape[0])
                ],
            }
        )

    if len(pathway_map) != mask_gene_pathway.shape[0]:
        raise ValueError(
            "The pathway mapping and pathway mask disagree on the number of pathways."
        )

    resources.pathway_map = pathway_map
    resources.mask_gene_pathway = mask_gene_pathway
    resources.mask_gene_pathway_tensor = torch.from_numpy(
        mask_gene_pathway.toarray()
    ).float()
    resources.n_pathways = int(mask_gene_pathway.shape[0])
    return resources


def sparse_mask_to_index_arrays(mask: coo_matrix | None) -> dict[str, Any]:
    """Serialize a sparse mask into simple numpy arrays for compact checkpoints."""

    if mask is None:
        return {
            "row_indices": None,
            "col_indices": None,
            "shape": None,
        }

    mask = mask.tocoo()
    return {
        "row_indices": np.asarray(mask.row, dtype=np.int32),
        "col_indices": np.asarray(mask.col, dtype=np.int32),
        "shape": tuple(int(value) for value in mask.shape),
    }


def sparse_mask_from_index_arrays(
    row_indices: np.ndarray | None,
    col_indices: np.ndarray | None,
    shape: tuple[int, int] | list[int] | None,
) -> coo_matrix | None:
    """Rebuild a sparse binary mask serialized with `sparse_mask_to_index_arrays`."""

    if row_indices is None or col_indices is None or shape is None:
        return None

    row_indices = np.asarray(row_indices, dtype=np.int32)
    col_indices = np.asarray(col_indices, dtype=np.int32)
    return coo_matrix(
        (
            np.ones(len(row_indices), dtype=np.float32),
            (row_indices, col_indices),
        ),
        shape=tuple(int(value) for value in shape),
        dtype=np.float32,
    )


def reconstruct_validation_data_from_checkpoint(
    checkpoint: dict[str, Any],
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Rebuild the saved validation cohort exactly from a compact checkpoint."""

    training_args = checkpoint["training_args"]
    snp_map = checkpoint["snp_map_dataframe"].copy()

    dataset = load_trait_dataset(
        genotype_prefix=training_args["genotype_prefix"],
        phenotype_path=training_args["phenotype_path"],
        trait=int(training_args["trait"]),
        snp_ids=snp_map["snp_id"].to_numpy(),
        pedigree_path=training_args.get("pedigree_path"),
    )
    split = split_last_n_samples(
        dataset["X"],
        dataset["y"],
        sample_ids=dataset["sample_ids_valid"],
        val_size=int(training_args.get("val_size", 1000)),
    )

    X_val = split["X_val"]
    y_val = split["y_val"]
    sample_ids_val = np.asarray(split["sample_ids_val"], dtype=str)

    if max_samples is not None and max_samples < len(sample_ids_val):
        rng = np.random.default_rng(42)
        keep = np.sort(
            rng.choice(len(sample_ids_val), size=max_samples, replace=False)
        )
        X_val = X_val[keep]
        y_val = y_val[keep]
        sample_ids_val = sample_ids_val[keep]

    scaler_X_mean = np.asarray(checkpoint["scaler_X_mean"], dtype=np.float32)
    scaler_X_scale = np.asarray(checkpoint["scaler_X_scale"], dtype=np.float32)
    scaler_y_mean = np.asarray(checkpoint["scaler_y_mean"], dtype=np.float32)
    scaler_y_scale = np.asarray(checkpoint["scaler_y_scale"], dtype=np.float32)

    return {
        "X_raw": np.asarray(X_val, dtype=np.float32),
        "y_raw": np.asarray(y_val, dtype=np.float32),
        "X_scaled": safe_scale(
            np.asarray(X_val, dtype=np.float32),
            scaler_X_mean,
            scaler_X_scale,
        ).astype(np.float32),
        "y_scaled": safe_scale(
            np.asarray(y_val, dtype=np.float32).reshape(-1, 1),
            scaler_y_mean,
            scaler_y_scale,
        ).astype(np.float32),
        "sample_ids_eval": sample_ids_val,
        "sample_ids_valid": np.asarray(dataset["sample_ids_valid"], dtype=str),
        "sample_ordering": dataset["sample_ordering"],
        "bim": dataset["bim"],
        "trait": int(training_args["trait"]),
        "snp_map": checkpoint["snp_map_dataframe"].copy(),
        "gene_map": checkpoint["gene_map_dataframe"].copy(),
        "impact_map": checkpoint["impact_map_dataframe"].copy(),
        "annotation_edges": checkpoint["annotation_edges_dataframe"].copy(),
        "pathway_map": checkpoint.get("pathway_map_dataframe"),
        "scaler_y_mean": scaler_y_mean,
        "scaler_y_scale": scaler_y_scale,
        "evaluation_source": "checkpoint_validation_holdout",
        "evaluation_source_note": (
            "validation holdout reconstructed from the saved sample ordering "
            "and the deterministic last-1000 split"
        ),
    }


def reconstruct_test_data_from_checkpoint(
    checkpoint: dict[str, Any],
    full_phenotype_path: str | Path | None = None,
) -> dict[str, Any]:
    """Rebuild the masked test cohort from a compact checkpoint."""

    training_args = checkpoint["training_args"]
    snp_map = checkpoint["snp_map_dataframe"].copy()

    full_phenotype_path = (
        training_args.get("full_phenotype_path")
        if full_phenotype_path is None
        else str(full_phenotype_path)
    )
    if full_phenotype_path is None:
        raise ValueError(
            "full_phenotype_path must be provided when it was not stored in the checkpoint."
        )

    test_dataset = load_trait_test_dataset(
        genotype_prefix=training_args["genotype_prefix"],
        masked_phenotype_path=training_args["phenotype_path"],
        full_phenotype_path=full_phenotype_path,
        trait=int(training_args["trait"]),
        snp_ids=snp_map["snp_id"].to_numpy(),
        pedigree_path=training_args.get("pedigree_path"),
    )

    scaler_X_mean = np.asarray(checkpoint["scaler_X_mean"], dtype=np.float32)
    scaler_X_scale = np.asarray(checkpoint["scaler_X_scale"], dtype=np.float32)
    scaler_y_mean = np.asarray(checkpoint["scaler_y_mean"], dtype=np.float32)
    scaler_y_scale = np.asarray(checkpoint["scaler_y_scale"], dtype=np.float32)

    return {
        "X_raw": np.asarray(test_dataset["X"], dtype=np.float32),
        "y_raw": np.asarray(test_dataset["y"], dtype=np.float32),
        "X_scaled": safe_scale(
            np.asarray(test_dataset["X"], dtype=np.float32),
            scaler_X_mean,
            scaler_X_scale,
        ).astype(np.float32),
        "y_scaled": safe_scale(
            np.asarray(test_dataset["y"], dtype=np.float32).reshape(-1, 1),
            scaler_y_mean,
            scaler_y_scale,
        ).astype(np.float32),
        "sample_ids_eval": np.asarray(test_dataset["sample_ids_test"], dtype=str),
        "bim": test_dataset["bim"],
        "trait": int(training_args["trait"]),
        "snp_map": checkpoint["snp_map_dataframe"].copy(),
        "gene_map": checkpoint["gene_map_dataframe"].copy(),
        "impact_map": checkpoint["impact_map_dataframe"].copy(),
        "annotation_edges": checkpoint["annotation_edges_dataframe"].copy(),
        "pathway_map": checkpoint.get("pathway_map_dataframe"),
        "scaler_y_mean": scaler_y_mean,
        "scaler_y_scale": scaler_y_scale,
        "evaluation_source": "masked_test_set",
        "evaluation_source_note": (
            "test cohort reconstructed as animals present in the full phenotype "
            "file but absent from the masked training phenotype file"
        ),
    }
