import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pandas_plink import read_plink
from scipy.sparse import coo_matrix
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from model import build_model_from_checkpoint
    from data_training_impact_attention import (
        build_annotation_resources,
        load_annotation_dataframe,
    )
except ImportError:
    from bio_informed_DNN.model import build_model_from_checkpoint
    from bio_informed_DNN.data_training_impact_attention import (
        build_annotation_resources,
        load_annotation_dataframe,
    )


TRAIT_NAMES = {
    2: 'shoulder',
    3: 'top',
    4: 'buttock_side',
    5: 'buttock_rear',
    6: 'size',
    7: 'musculature',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Analyze a saved impact-attention checkpoint to identify important '
            'SNPs, genes, and impact classes with publication-ready plots.'
        )
    )
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--top-snps', type=int, default=30)
    parser.add_argument('--top-genes', type=int, default=20)
    parser.add_argument(
        '--top-impact-sizes',
        type=int,
        nargs='+',
        default=[50, 100, 200],
        help='Top-N SNP cutoffs used for impact-class composition/enrichment.',
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Optional cap on validation samples used for attribution.',
    )
    parser.add_argument(
        '--gene-global-top-k',
        type=int,
        default=0,
        help='Number of genes to score with held-out perturbation/permutation. Use 0 to score all genes.',
    )
    parser.add_argument(
        '--correlation-block-top-k',
        dest='correlation_block_top_k',
        type=int,
        default=0,
        help='Number of correlation blocks to score with held-out perturbation/permutation. Use 0 to score all blocks.',
    )
    parser.add_argument(
        '--ld-block-top-k',
        dest='correlation_block_top_k',
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--snp-global-top-k',
        type=int,
        default=0,
        help='Number of correlation-block representative SNPs to score globally. Use 0 to score all representatives.',
    )
    parser.add_argument(
        '--permutation-repeats',
        type=int,
        default=3,
        help='Number of held-out permutations per group for perturbation importance.',
    )
    parser.add_argument(
        '--correlation-threshold',
        dest='correlation_threshold',
        type=float,
        default=0.60,
        help='Absolute adjacent genotype-correlation threshold used to build approximate correlation blocks.',
    )
    parser.add_argument(
        '--ld-corr-threshold',
        dest='correlation_threshold',
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--correlation-max-gap-kb',
        dest='correlation_max_gap_kb',
        type=int,
        default=500,
        help='Maximum genomic gap in kb allowed inside an approximate correlation block.',
    )
    parser.add_argument(
        '--ld-max-gap-kb',
        dest='correlation_max_gap_kb',
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--ig-steps',
        type=int,
        default=32,
        help='Number of steps for Integrated Gradients.',
    )
    parser.add_argument(
        '--ig-samples',
        type=int,
        default=128,
        help='Maximum held-out samples used for Integrated Gradients.',
    )
    parser.add_argument(
        '--run-shap',
        action='store_true',
        help='If set, run a careful small-sample SHAP approximation when the shap package is available.',
    )
    parser.add_argument(
        '--shap-background-size',
        type=int,
        default=64,
        help='Background sample count used for optional SHAP.',
    )
    parser.add_argument(
        '--shap-eval-size',
        type=int,
        default=64,
        help='Evaluation sample count used for optional SHAP.',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
    )
    parser.add_argument(
        '--sample-id-file',
        type=str,
        default=None,
        help='Optional newline-delimited or table-based list of sample IDs to analyze instead of the checkpoint validation holdout.',
    )
    parser.add_argument(
        '--sample-id-column',
        type=str,
        default=None,
        help='Optional column name or zero-based column index used when reading --sample-id-file.',
    )
    parser.add_argument('--dpi', type=int, default=220)
    return parser.parse_args()


def load_checkpoint(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

    return checkpoint_path, checkpoint


def infer_output_dir(checkpoint_path, output_dir=None):
    if output_dir is not None:
        out_dir = Path(output_dir)
    else:
        out_dir = (
            checkpoint_path.parent.parent.parent
            / 'interpretability'
            / checkpoint_path.stem
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def trait_label(trait_idx):
    return TRAIT_NAMES.get(int(trait_idx), f'trait_{trait_idx}')


def set_plot_style():
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        plt.style.use('default')
    plt.rcParams.update(
        {
            'figure.facecolor': 'white',
            'axes.facecolor': '#f7f7f2',
            'axes.edgecolor': '#2e2e2e',
            'axes.labelcolor': '#222222',
            'axes.titleweight': 'bold',
            'grid.color': '#d8d8cf',
            'grid.alpha': 0.55,
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'savefig.bbox': 'tight',
        }
    )


def safe_scale(array, mean, scale):
    safe_scale_values = np.where(scale == 0, 1.0, scale)
    return (array - mean) / safe_scale_values


def inverse_scale(array, mean, scale):
    return array * scale + mean


def compute_regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mse = mean_squared_error(y_true, y_pred)
    pearson, _ = safe_pearson(y_true, y_pred)
    return {
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'mse': float(mse),
        'rmse': float(np.sqrt(mse)),
        'r2': float(r2_score(y_true, y_pred)),
        'pearson': float(pearson),
    }


def resolve_top_k(top_k, total_count):
    if top_k is None:
        return int(total_count)
    top_k = int(top_k)
    if top_k <= 0:
        return int(total_count)
    return min(top_k, int(total_count))


def benjamini_hochberg(pvalues):
    pvalues = np.asarray(pvalues, dtype=np.float64)
    qvalues = np.full(pvalues.shape, np.nan, dtype=np.float64)

    valid = np.isfinite(pvalues)
    if not valid.any():
        return qvalues

    valid_pvalues = pvalues[valid]
    order = np.argsort(valid_pvalues)
    ranked = valid_pvalues[order]
    n_tests = ranked.size

    adjusted = ranked * n_tests / np.arange(1, n_tests + 1, dtype=np.float64)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    reordered = np.empty_like(adjusted)
    reordered[order] = adjusted
    qvalues[valid] = reordered
    return qvalues


def load_requested_sample_ids(sample_id_file, sample_id_column=None):
    sample_id_path = Path(sample_id_file)
    if not sample_id_path.exists():
        raise FileNotFoundError(f'Sample ID file not found: {sample_id_path}')

    raw_lines = [
        line.strip()
        for line in sample_id_path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith('#')
    ]
    if raw_lines and sample_id_column is None:
        if all('\t' not in line and ',' not in line for line in raw_lines):
            requested_ids = np.asarray(raw_lines, dtype=str)
            if len(pd.Index(requested_ids)) != pd.Index(requested_ids).nunique():
                raise ValueError('Sample ID file contains duplicate sample IDs.')
            return requested_ids

    frame = pd.read_csv(sample_id_path, sep=None, engine='python')
    if frame.empty:
        raise ValueError(f'Sample ID file is empty: {sample_id_path}')

    if sample_id_column is None:
        series = frame.iloc[:, 0]
    elif sample_id_column in frame.columns:
        series = frame[sample_id_column]
    else:
        try:
            column_index = int(sample_id_column)
        except ValueError as exc:
            raise ValueError(
                f'Could not find sample ID column "{sample_id_column}" in {sample_id_path}.'
            ) from exc
        series = frame.iloc[:, column_index]

    requested_ids = series.dropna().astype(str).str.strip()
    requested_ids = requested_ids[requested_ids != ''].to_numpy()
    if requested_ids.size == 0:
        raise ValueError(f'No sample IDs were found in {sample_id_path}.')
    if len(pd.Index(requested_ids)) != pd.Index(requested_ids).nunique():
        raise ValueError('Sample ID file contains duplicate sample IDs.')
    return requested_ids


def summarize_evaluation_source(sample_id_file, checkpoint_has_sample_ids):
    if sample_id_file is not None:
        return 'custom_sample_ids'
    if checkpoint_has_sample_ids:
        return 'checkpoint_validation_holdout'
    return 'reconstructed_validation_holdout'


def evaluation_warning_for_source(evaluation_source):
    if evaluation_source == 'custom_sample_ids':
        return ''
    if evaluation_source == 'checkpoint_validation_holdout':
        return (
            'Interpretability is being computed on the checkpoint validation holdout, '
            'which was also used for model selection during training.'
        )
    return (
        'Interpretability is being computed on a reconstructed validation holdout '
        'because the checkpoint did not store explicit evaluation sample IDs.'
    )


def evaluate_model_on_scaled_inputs(
    model,
    X_scaled,
    y_true,
    scaler_y_mean,
    scaler_y_scale,
    device,
    batch_size,
):
    prediction_scaled = batched_predict(model, X_scaled, device, batch_size)
    prediction_real = inverse_scale(
        prediction_scaled,
        scaler_y_mean,
        scaler_y_scale,
    ).reshape(-1)
    metrics = compute_regression_metrics(y_true, prediction_real)
    return prediction_real, metrics


def batched_forward(module, feature_matrix, device, batch_size):
    outputs = []
    module.eval()
    with torch.no_grad():
        for start in range(0, feature_matrix.shape[0], batch_size):
            xb = torch.tensor(
                feature_matrix[start:start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            outputs.append(module(xb).detach().cpu().numpy())
    return np.vstack(outputs).astype(np.float32)


def evaluate_module_on_feature_matrix(
    module,
    feature_matrix,
    y_true,
    scaler_y_mean,
    scaler_y_scale,
    device,
    batch_size,
):
    prediction_scaled = batched_forward(module, feature_matrix, device, batch_size)
    prediction_real = inverse_scale(
        prediction_scaled,
        scaler_y_mean,
        scaler_y_scale,
    ).reshape(-1)
    metrics = compute_regression_metrics(y_true, prediction_real)
    return prediction_real, metrics


def safe_pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1:
        x = x.reshape(-1)
    if y.ndim != 1:
        y = y.reshape(-1)
    if x.size == 0 or y.size == 0:
        return np.nan, np.nan
    if np.allclose(x.std(ddof=0), 0.0) or np.allclose(y.std(ddof=0), 0.0):
        return np.nan, np.nan
    return pearsonr(x, y)


def vectorized_correlation(matrix, target):
    matrix = np.asarray(matrix, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).reshape(-1)

    if matrix.ndim != 2:
        raise ValueError('matrix must be 2D')
    if matrix.shape[0] != target.shape[0]:
        raise ValueError('matrix rows must match target length')

    matrix_centered = matrix - matrix.mean(axis=0, keepdims=True)
    target_centered = target - target.mean()

    numerator = (matrix_centered * target_centered[:, None]).sum(axis=0)
    denominator = np.sqrt(
        (matrix_centered**2).sum(axis=0) * (target_centered**2).sum()
    )
    corr = np.divide(
        numerator,
        denominator,
        out=np.zeros(matrix.shape[1], dtype=np.float64),
        where=denominator > 0,
    )
    return corr.astype(np.float32)


def build_sparse_snp_gene_mask(annotation_edges, gene_map, snp_map):
    gene_index = pd.Series(
        gene_map['gene_index'].to_numpy(),
        index=gene_map['ensembl_gene_id'],
    )
    snp_index = pd.Series(
        snp_map['snp_index'].to_numpy(),
        index=snp_map['snp_id'],
    )

    row_indices = annotation_edges['Gene'].map(gene_index).to_numpy()
    col_indices = annotation_edges['snp_id'].map(snp_index).to_numpy()

    valid = (~pd.isna(row_indices)) & (~pd.isna(col_indices))
    if not valid.all():
        missing_rows = int((~valid).sum())
        raise ValueError(
            'Could not map all SNP-gene edges from checkpoint metadata. '
            f'Missing rows: {missing_rows}'
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
    return mask.tocsr()


def aggregate_snp_scores_to_gene_scores(snp_scores, sparse_mask):
    snp_scores = np.asarray(snp_scores, dtype=np.float32).reshape(-1)
    snp_membership = np.asarray(sparse_mask.sum(axis=0)).reshape(-1).astype(np.float32)
    snp_membership = np.where(snp_membership == 0, 1.0, snp_membership)
    weighted_mask = sparse_mask.multiply(1.0 / snp_membership)
    gene_scores = weighted_mask @ snp_scores
    return np.asarray(gene_scores, dtype=np.float32).reshape(-1)


def build_gene_group_table(gene_table, sparse_mask, snp_table):
    snp_lookup = snp_table.set_index('snp_index')['snp_id']
    rows = []
    for row in gene_table.itertuples(index=False):
        member_snp_indices = sparse_mask.getrow(int(row.gene_index)).indices.astype(int)
        if member_snp_indices.size == 0:
            continue
        member_snp_ids = snp_lookup.reindex(member_snp_indices).fillna('').tolist()
        rows.append(
            {
                'group_id': row.ensembl_gene_id,
                'group_label': row.gene_label,
                'candidate_score': float(row.local_differential_sensitivity_abs),
                'member_indices': [int(row.gene_index)],
                'member_snp_ids': ';'.join(str(value) for value in member_snp_ids if value),
                'n_snps': int(member_snp_indices.size),
                'gene_name': row.gene_name,
            }
        )
    return pd.DataFrame(rows)


def build_correlation_blocks(snp_table, X_raw, corr_threshold=0.60, max_gap_kb=500):
    work = snp_table.copy()
    if 'chromosome' in work.columns and 'position' in work.columns:
        work['chromosome_label'] = work['chromosome'].astype(str)
        work['position_numeric'] = pd.to_numeric(work['position'], errors='coerce')
    else:
        work['chromosome_label'] = 'NA'
        work['position_numeric'] = np.arange(len(work), dtype=np.float64)

    if work['position_numeric'].isna().all():
        work['position_numeric'] = np.arange(len(work), dtype=np.float64)

    chrom_keys = work['chromosome_label'].map(chrom_sort_key)
    work = work.assign(
        chrom_order_major=[value[0] for value in chrom_keys],
        chrom_order_minor=[str(value[1]) for value in chrom_keys],
    )
    work = work.sort_values(
        ['chrom_order_major', 'chrom_order_minor', 'position_numeric', 'snp_index']
    ).reset_index(drop=True)

    ordered_indices = work['snp_index'].to_numpy(dtype=int)
    if len(ordered_indices) <= 1:
        single_block = pd.DataFrame(
            [
                {
                    'correlation_block_id': 'corr_block_0001',
                    'chromosome': work.iloc[0]['chromosome_label'] if len(work) else 'NA',
                    'start_position': float(work.iloc[0]['position_numeric']) if len(work) else np.nan,
                    'end_position': float(work.iloc[0]['position_numeric']) if len(work) else np.nan,
                    'n_snps': int(len(work)),
                    'member_indices': ordered_indices.tolist(),
                    'member_snp_ids': ';'.join(work['snp_id'].astype(str).tolist()),
                    'candidate_score': float(work['local_differential_sensitivity_abs'].sum()) if len(work) else 0.0,
                    'representative_snp_id': work.iloc[0]['snp_id'] if len(work) else '',
                    'representative_snp_index': int(work.iloc[0]['snp_index']) if len(work) else -1,
                }
            ]
        )
        snp_to_block = {int(idx): 'corr_block_0001' for idx in ordered_indices}
        return single_block, snp_to_block

    left = X_raw[:, ordered_indices[:-1]].astype(np.float64)
    right = X_raw[:, ordered_indices[1:]].astype(np.float64)
    left_centered = left - left.mean(axis=0, keepdims=True)
    right_centered = right - right.mean(axis=0, keepdims=True)
    numerator = (left_centered * right_centered).sum(axis=0)
    denominator = np.sqrt(
        (left_centered**2).sum(axis=0) * (right_centered**2).sum(axis=0)
    )
    adjacent_corr = np.divide(
        numerator,
        denominator,
        out=np.zeros(left.shape[1], dtype=np.float64),
        where=denominator > 0,
    )

    max_gap_bp = float(max_gap_kb) * 1000.0
    rows = []
    snp_to_block = {}
    block_start = 0

    def flush_block(start_idx, end_idx, block_number):
        block = work.iloc[start_idx:end_idx].copy()
        member_indices = block['snp_index'].astype(int).tolist()
        representative = block.nlargest(1, 'local_differential_sensitivity_abs').iloc[0]
        block_id = f'corr_block_{block_number:04d}'
        for snp_index in member_indices:
            snp_to_block[int(snp_index)] = block_id
        rows.append(
            {
                'correlation_block_id': block_id,
                'chromosome': block.iloc[0]['chromosome_label'],
                'start_position': float(block['position_numeric'].min()),
                'end_position': float(block['position_numeric'].max()),
                'n_snps': int(len(block)),
                'member_indices': member_indices,
                'member_snp_ids': ';'.join(block['snp_id'].astype(str).tolist()),
                'candidate_score': float(block['local_differential_sensitivity_abs'].sum()),
                'representative_snp_id': representative['snp_id'],
                'representative_snp_index': int(representative['snp_index']),
            }
        )

    block_number = 1
    for idx in range(1, len(work)):
        same_chromosome = (
            work.iloc[idx]['chromosome_label'] == work.iloc[idx - 1]['chromosome_label']
        )
        gap_bp = abs(
            float(work.iloc[idx]['position_numeric'])
            - float(work.iloc[idx - 1]['position_numeric'])
        )
        high_ld = abs(adjacent_corr[idx - 1]) >= float(corr_threshold)
        if not (same_chromosome and gap_bp <= max_gap_bp and high_ld):
            flush_block(block_start, idx, block_number)
            block_start = idx
            block_number += 1
    flush_block(block_start, len(work), block_number)
    return pd.DataFrame(rows), snp_to_block


def build_block_representative_snp_table(snp_table, correlation_block_table):
    if correlation_block_table.empty:
        return pd.DataFrame(columns=['group_id', 'group_label', 'candidate_score'])

    representatives = correlation_block_table[
        [
            'correlation_block_id',
            'representative_snp_id',
            'representative_snp_index',
            'candidate_score',
            'n_snps',
        ]
    ].copy()
    representatives = representatives.rename(
        columns={
            'correlation_block_id': 'correlation_block_id',
            'representative_snp_id': 'snp_id',
            'representative_snp_index': 'snp_index',
        }
    )
    representatives = representatives.merge(
        snp_table,
        on=['snp_id', 'snp_index', 'correlation_block_id'],
        how='left',
        suffixes=('', '_snp'),
    )
    representatives['group_id'] = representatives['snp_id']
    representatives['group_label'] = representatives['snp_id']
    representatives['member_indices'] = representatives['snp_index'].apply(lambda value: [int(value)])
    representatives['member_snp_ids'] = representatives['snp_id']
    return representatives


def summarize_global_importance_row(
    group_row,
    baseline_metrics,
    mean_replacement_metrics,
    permutation_metrics,
):
    row = {
        'group_id': group_row.group_id,
        'group_label': group_row.group_label,
        'candidate_score': float(group_row.candidate_score),
        'member_snp_ids': getattr(group_row, 'member_snp_ids', ''),
        'n_snps': int(getattr(group_row, 'n_snps', len(group_row.member_indices))),
        'mean_replacement_mae': float(mean_replacement_metrics['mae']),
        'mean_replacement_mse': float(mean_replacement_metrics['mse']),
        'mean_replacement_rmse': float(mean_replacement_metrics['rmse']),
        'mean_replacement_r2': float(mean_replacement_metrics['r2']),
        'mean_replacement_pearson': float(mean_replacement_metrics['pearson']),
        'mean_replacement_delta_mae': float(
            mean_replacement_metrics['mae'] - baseline_metrics['mae']
        ),
        'mean_replacement_delta_mse': float(
            mean_replacement_metrics['mse'] - baseline_metrics['mse']
        ),
        'mean_replacement_delta_rmse': float(
            mean_replacement_metrics['rmse'] - baseline_metrics['rmse']
        ),
        'mean_replacement_delta_r2': float(
            baseline_metrics['r2'] - mean_replacement_metrics['r2']
        ),
        'mean_replacement_delta_pearson': float(
            baseline_metrics['pearson'] - mean_replacement_metrics['pearson']
        ),
    }

    for metric_name in ['mae', 'mse', 'rmse', 'r2', 'pearson']:
        values = np.asarray([metrics[metric_name] for metrics in permutation_metrics], dtype=np.float64)
        row[f'permutation_{metric_name}_mean'] = float(values.mean())
        row[f'permutation_{metric_name}_sd'] = float(values.std(ddof=0))
    row['permutation_delta_mae_mean'] = float(row['permutation_mae_mean'] - baseline_metrics['mae'])
    row['permutation_delta_mse_mean'] = float(row['permutation_mse_mean'] - baseline_metrics['mse'])
    row['permutation_delta_rmse_mean'] = float(row['permutation_rmse_mean'] - baseline_metrics['rmse'])
    row['permutation_delta_r2_mean'] = float(baseline_metrics['r2'] - row['permutation_r2_mean'])
    row['permutation_delta_pearson_mean'] = float(baseline_metrics['pearson'] - row['permutation_pearson_mean'])

    # Legacy aliases retained for downstream compatibility.
    for metric_name in ['mae', 'mse', 'rmse', 'r2', 'pearson']:
        row[f'ablation_{metric_name}'] = row[f'mean_replacement_{metric_name}']
        row[f'ablation_delta_{metric_name}'] = row[
            f'mean_replacement_delta_{metric_name}'
        ]
    return row


def evaluate_group_global_importance(
    module,
    feature_matrix,
    y_true,
    scaler_y_mean,
    scaler_y_scale,
    group_table,
    top_k,
    permutation_repeats,
    device,
    batch_size,
    random_seed=42,
):
    _, baseline_metrics = evaluate_module_on_feature_matrix(
        module,
        feature_matrix,
        y_true,
        scaler_y_mean,
        scaler_y_scale,
        device,
        batch_size,
    )
    if group_table.empty:
        return pd.DataFrame(), baseline_metrics

    top_groups = group_table.sort_values(
        'candidate_score',
        ascending=False,
    ).head(resolve_top_k(top_k, len(group_table)))
    rng = np.random.default_rng(random_seed)
    rows = []

    for group_row in top_groups.itertuples(index=False):
        member_indices = np.asarray(group_row.member_indices, dtype=int)
        if member_indices.size == 0:
            continue

        mean_replaced = feature_matrix.copy()
        mean_replaced[:, member_indices] = 0.0
        _, mean_replacement_metrics = evaluate_module_on_feature_matrix(
            module,
            mean_replaced,
            y_true,
            scaler_y_mean,
            scaler_y_scale,
            device,
            batch_size,
        )

        permutation_metrics = []
        for _ in range(max(1, int(permutation_repeats))):
            permuted = feature_matrix.copy()
            permutation = rng.permutation(feature_matrix.shape[0])
            permuted[:, member_indices] = permuted[permutation][:, member_indices]
            _, metrics = evaluate_module_on_feature_matrix(
                module,
                permuted,
                y_true,
                scaler_y_mean,
                scaler_y_scale,
                device,
                batch_size,
            )
            permutation_metrics.append(metrics)

        rows.append(
            summarize_global_importance_row(
                group_row,
                baseline_metrics,
                mean_replacement_metrics,
                permutation_metrics,
            )
        )

    result = pd.DataFrame(rows).sort_values(
        'mean_replacement_delta_mse',
        ascending=False,
    )
    return result, baseline_metrics


def encode_gene_features(model, X_scaled, device, batch_size):
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, X_scaled.shape[0], batch_size):
            xb = torch.tensor(
                X_scaled[start:start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            outputs.append(model.encode_genes(xb).detach().cpu().numpy())
    return np.vstack(outputs).astype(np.float32)


def compute_integrated_gradients(
    module,
    feature_matrix,
    device,
    batch_size,
    steps=32,
    max_samples=128,
    random_seed=42,
):
    if max_samples is not None and max_samples < len(feature_matrix):
        rng = np.random.default_rng(random_seed)
        selected = np.sort(rng.choice(len(feature_matrix), size=max_samples, replace=False))
    else:
        selected = np.arange(len(feature_matrix))

    inputs = feature_matrix[selected]
    total_abs = None
    total_signed = None
    total_samples = 0
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=device)[1:]

    module.eval()
    for start in range(0, inputs.shape[0], batch_size):
        xb = torch.tensor(inputs[start:start + batch_size], dtype=torch.float32, device=device)
        baseline = torch.zeros_like(xb)
        accumulated = torch.zeros_like(xb)

        for alpha in alphas:
            interpolated = (baseline + alpha * (xb - baseline)).detach().requires_grad_(True)
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
        'sample_indices': selected,
        'abs': total_abs / max(total_samples, 1),
        'signed': total_signed / max(total_samples, 1),
    }


def compute_shap_attributions(
    module,
    feature_matrix,
    device,
    background_size=64,
    eval_size=64,
    random_seed=42,
):
    try:
        import shap
    except ImportError:
        return None, 'shap package not installed'

    sample_count = len(feature_matrix)
    if sample_count == 0:
        return None, 'no samples available for SHAP'

    rng = np.random.default_rng(random_seed)
    background_idx = np.sort(
        rng.choice(sample_count, size=min(background_size, sample_count), replace=False)
    )
    eval_idx = np.sort(
        rng.choice(sample_count, size=min(eval_size, sample_count), replace=False)
    )

    background = torch.tensor(feature_matrix[background_idx], dtype=torch.float32, device=device)
    evaluation = torch.tensor(feature_matrix[eval_idx], dtype=torch.float32, device=device)

    try:
        explainer = shap.GradientExplainer(module, background)
        shap_values = explainer.shap_values(evaluation)
    except Exception as exc:
        return None, f'shap execution failed: {exc}'

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values, dtype=np.float32)
    if shap_values.ndim == 3:
        shap_values = shap_values[..., 0]

    return {
        'sample_indices': eval_idx,
        'abs': np.abs(shap_values).mean(axis=0),
        'signed': shap_values.mean(axis=0),
    }, None


def chrom_sort_key(value):
    string_value = str(value).replace('chr', '').replace('CHR', '')
    try:
        return (0, int(string_value))
    except ValueError:
        return (1, string_value)


def build_snp_metadata(annotation_edges, checkpoint, bim):
    snp_map = checkpoint['snp_map_dataframe'].copy()

    if checkpoint.get('snp_metadata_dataframe') is not None:
        snp_metadata = checkpoint['snp_metadata_dataframe'].copy()
    else:
        keep_columns = [
            column
            for column in ['snp_id', 'new_impact', 'Consequence', 'IMPACT', 'cm', 'bp']
            if column in annotation_edges.columns
        ]
        snp_metadata = annotation_edges.drop_duplicates('snp_id')[keep_columns].copy()

    gene_summary = (
        annotation_edges.groupby('snp_id')
        .agg(
            n_genes=('Gene', 'nunique'),
            mapped_genes=('Gene', lambda series: ';'.join(sorted(pd.unique(series))[:5])),
            mapped_symbols=(
                'SYMBOL',
                lambda series: ';'.join(
                    sorted(symbol for symbol in pd.unique(series) if isinstance(symbol, str) and symbol)
                )[:120],
            )
            if 'SYMBOL' in annotation_edges.columns
            else ('Gene', lambda series: ''),
        )
        .reset_index()
    )

    bim_meta = bim.loc[snp_map['snp_id']].reset_index()
    rename_map = {
        'snp': 'snp_id',
        'chrom': 'chromosome',
        'pos': 'position',
        'cm': 'plink_cm',
        'a0': 'plink_allele0',
        'a1': 'plink_allele1',
    }
    bim_meta = bim_meta.rename(columns=rename_map)

    snp_table = snp_map.merge(snp_metadata, on='snp_id', how='left')
    snp_table = snp_table.merge(gene_summary, on='snp_id', how='left')
    snp_table = snp_table.merge(bim_meta, on='snp_id', how='left')

    snp_table['mapped_symbols'] = snp_table['mapped_symbols'].fillna('')
    snp_table['mapped_genes'] = snp_table['mapped_genes'].fillna('')
    snp_table['display_gene_label'] = np.where(
        snp_table['mapped_symbols'].astype(str).str.len() > 0,
        snp_table['mapped_symbols'],
        snp_table['mapped_genes'],
    )
    return snp_table


def resolve_annotation_metadata(checkpoint):
    annotation_edges = checkpoint.get('snp_annotation_dataframe')
    gene_map = checkpoint.get('gene_map_dataframe')
    snp_map = checkpoint.get('snp_map_dataframe')
    impact_map = checkpoint.get('impact_map_dataframe')

    if (
        annotation_edges is not None
        and gene_map is not None
        and snp_map is not None
        and impact_map is not None
    ):
        return annotation_edges.copy(), gene_map.copy(), snp_map.copy(), impact_map.copy()

    training_args = checkpoint['training_args']
    annotation_df, _ = load_annotation_dataframe(training_args['annotation_path'])
    resources = build_annotation_resources(annotation_df)
    return (
        resources['annotation_edges'].copy(),
        resources['gene_map'].copy(),
        resources['snp_map'].copy(),
        resources['impact_map'].copy(),
    )


def reconstruct_validation_data(
    checkpoint,
    max_samples=None,
    sample_id_file=None,
    sample_id_column=None,
):
    training_args = checkpoint['training_args']
    annotation_edges, gene_map, snp_map, impact_map = resolve_annotation_metadata(checkpoint)

    bim, _, bed = read_plink(training_args['genotype_prefix'])
    bim = bim.set_index('snp')

    missing_snps = snp_map.loc[~snp_map['snp_id'].isin(bim.index), 'snp_id']
    if not missing_snps.empty:
        raise ValueError(
            'Some checkpoint SNPs are missing from the genotype BIM file. '
            f'First missing SNPs: {missing_snps.iloc[:10].tolist()}'
        )

    pheno_df = pd.read_csv(
        training_args['phenotype_path'],
        delimiter='	',
        header=None,
    )
    pheno_df = pheno_df.drop(columns=[0]).set_index(1)

    trait = int(training_args['trait'])
    valid_mask = ~pheno_df[trait].isna()
    valid_mask_array = valid_mask.to_numpy()
    valid_positions = np.flatnonzero(valid_mask_array)
    valid_sample_ids = pheno_df.index[valid_mask].astype(str).to_numpy()
    y_valid = pheno_df.loc[valid_mask, trait].to_numpy(dtype=np.float32)

    checkpoint_sample_ids = checkpoint.get('sample_ids_val')
    evaluation_source = summarize_evaluation_source(
        sample_id_file=sample_id_file,
        checkpoint_has_sample_ids=checkpoint_sample_ids is not None,
    )
    evaluation_source_note = ''

    if sample_id_file is not None:
        sample_ids_val = load_requested_sample_ids(sample_id_file, sample_id_column)
        evaluation_source_note = f'custom evaluation cohort from {sample_id_file}'
        valid_index = pd.Index(valid_sample_ids)
        if valid_index.has_duplicates:
            raise ValueError(
                'Evaluation sample IDs are duplicated in the phenotype file, '
                'so exact cohort reconstruction is ambiguous.'
            )
        selected_valid_indices = valid_index.get_indexer(sample_ids_val)
        if (selected_valid_indices < 0).any():
            missing = sample_ids_val[selected_valid_indices < 0][:10].tolist()
            raise ValueError(
                'Some requested evaluation sample IDs were not found in the phenotype '
                f'file: {missing}'
            )
    elif checkpoint_sample_ids is not None:
        sample_ids_val = np.asarray(checkpoint_sample_ids).astype(str)
        evaluation_source_note = 'checkpoint validation holdout reused for interpretation'
        valid_index = pd.Index(valid_sample_ids)
        if valid_index.has_duplicates:
            raise ValueError(
                'Validation sample IDs are duplicated in the phenotype file, '
                'so exact checkpoint reconstruction is ambiguous.'
            )
        selected_valid_indices = valid_index.get_indexer(sample_ids_val)
        if (selected_valid_indices < 0).any():
            missing = sample_ids_val[selected_valid_indices < 0][:10].tolist()
            raise ValueError(
                'Some validation sample IDs stored in the checkpoint were not found '
                f'in the phenotype file: {missing}'
            )
    else:
        evaluation_source_note = (
            'validation holdout reconstructed from training_args '
            '(sample_ids_val missing from checkpoint)'
        )
        full_indices = np.arange(len(valid_sample_ids))
        _, selected_valid_indices = train_test_split(
            full_indices,
            test_size=int(training_args.get('val_size', 1000)),
            random_state=int(training_args.get('split_random_state', 42)),
        )
        sample_ids_val = valid_sample_ids[selected_valid_indices]

    if max_samples is not None and max_samples < len(selected_valid_indices):
        rng = np.random.default_rng(42)
        keep = np.sort(rng.choice(len(selected_valid_indices), size=max_samples, replace=False))
        selected_valid_indices = selected_valid_indices[keep]
        sample_ids_val = sample_ids_val[keep]

    snp_to_index = bim.loc[snp_map['snp_id'], 'i'].to_numpy()
    bed_positions = valid_positions[selected_valid_indices]
    X_raw = bed[snp_to_index].T[bed_positions].compute()
    X_raw = np.asarray(X_raw, dtype=np.float32)
    y_raw = y_valid[selected_valid_indices].astype(np.float32)

    scaler_X_mean = np.asarray(checkpoint['scaler_X_mean'], dtype=np.float32)
    scaler_X_scale = np.asarray(checkpoint['scaler_X_scale'], dtype=np.float32)
    scaler_y_mean = np.asarray(checkpoint['scaler_y_mean'], dtype=np.float32)
    scaler_y_scale = np.asarray(checkpoint['scaler_y_scale'], dtype=np.float32)

    X_scaled = safe_scale(X_raw, scaler_X_mean, scaler_X_scale).astype(np.float32)
    y_scaled = safe_scale(y_raw.reshape(-1, 1), scaler_y_mean, scaler_y_scale).astype(np.float32)

    return {
        'annotation_edges': annotation_edges,
        'gene_map': gene_map,
        'snp_map': snp_map,
        'impact_map': impact_map,
        'bim': bim,
        'X_raw': X_raw,
        'X_scaled': X_scaled,
        'y_raw': y_raw,
        'y_scaled': y_scaled,
        'sample_ids_val': sample_ids_val,
        'sample_ids_eval': sample_ids_val,
        'evaluation_source': evaluation_source,
        'evaluation_source_note': evaluation_source_note,
        'evaluation_warning': evaluation_warning_for_source(evaluation_source),
        'scaler_y_mean': scaler_y_mean,
        'scaler_y_scale': scaler_y_scale,
        'trait': trait,
    }


def batched_predict(model, X_scaled, device, batch_size):
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, X_scaled.shape[0], batch_size):
            xb = torch.tensor(X_scaled[start:start + batch_size], dtype=torch.float32, device=device)
            outputs.append(model(xb).detach().cpu().numpy())
    return np.vstack(outputs).astype(np.float32)


def compute_snp_importance(model, X_scaled, device, batch_size):
    n_snps = X_scaled.shape[1]
    importance_abs = torch.zeros(n_snps, dtype=torch.float32, device=device)
    importance_signed = torch.zeros(n_snps, dtype=torch.float32, device=device)
    gate_mean = torch.zeros(n_snps, dtype=torch.float32, device=device)
    total_samples = 0

    model.eval()
    for start in range(0, X_scaled.shape[0], batch_size):
        xb = torch.tensor(X_scaled[start:start + batch_size], dtype=torch.float32, device=device)
        xb.requires_grad_(True)

        model.zero_grad(set_to_none=True)
        outputs = model(xb)
        outputs.sum().backward()

        contribution = xb.grad * xb
        importance_abs += contribution.abs().sum(dim=0)
        importance_signed += contribution.sum(dim=0)
        gate_mean += model.get_attention_weights(xb).detach().sum(dim=0)
        total_samples += xb.shape[0]

    return {
        'importance_abs': (importance_abs / total_samples).detach().cpu().numpy(),
        'importance_signed': (importance_signed / total_samples).detach().cpu().numpy(),
        'gate_mean': (gate_mean / total_samples).detach().cpu().numpy(),
        'attention_mean': (gate_mean / total_samples).detach().cpu().numpy(),
    }


def compute_gene_importance(model, X_scaled, device, batch_size):
    with torch.no_grad():
        first_batch = torch.tensor(X_scaled[:1], dtype=torch.float32, device=device)
        n_genes = model.encode_genes(first_batch).shape[1]

    importance_abs = torch.zeros(n_genes, dtype=torch.float32, device=device)
    importance_signed = torch.zeros(n_genes, dtype=torch.float32, device=device)
    activation_abs = torch.zeros(n_genes, dtype=torch.float32, device=device)
    total_samples = 0

    model.eval()
    for start in range(0, X_scaled.shape[0], batch_size):
        xb = torch.tensor(X_scaled[start:start + batch_size], dtype=torch.float32, device=device)

        model.zero_grad(set_to_none=True)
        gene_features = model.encode_genes(xb)
        gene_features.retain_grad()
        outputs = model.fc_stack(gene_features)
        outputs.sum().backward()

        contribution = gene_features.grad * gene_features
        importance_abs += contribution.abs().sum(dim=0)
        importance_signed += contribution.sum(dim=0)
        activation_abs += gene_features.detach().abs().sum(dim=0)
        total_samples += xb.shape[0]

    return {
        'importance_abs': (importance_abs / total_samples).detach().cpu().numpy(),
        'importance_signed': (importance_signed / total_samples).detach().cpu().numpy(),
        'mean_activation_abs': (activation_abs / total_samples).detach().cpu().numpy(),
    }


def attach_top_feature_pvalues(table, feature_matrix, target, top_n, corr_column, prefix):
    table = table.copy()
    table[f'{prefix}_pearson_pvalue'] = np.nan
    table[f'{prefix}_pearson_qvalue'] = np.nan
    ranking_column = 'importance_abs' if 'importance_abs' in table.columns else 'local_differential_sensitivity_abs'
    top_indices = table.nlargest(top_n, ranking_column)['feature_index'].to_numpy()
    for feature_index in top_indices:
        corr, pvalue = safe_pearson(feature_matrix[:, feature_index], target)
        table.loc[
            table['feature_index'] == feature_index,
            [corr_column, f'{prefix}_pearson_pvalue'],
        ] = [corr, pvalue]
    computed = table[f'{prefix}_pearson_pvalue'].notna()
    if computed.any():
        table.loc[computed, f'{prefix}_pearson_qvalue'] = benjamini_hochberg(
            table.loc[computed, f'{prefix}_pearson_pvalue'].to_numpy()
        )
    return table


def prepare_snp_table(validation_data, checkpoint, snp_scores):
    annotation_edges = validation_data['annotation_edges']
    bim = validation_data['bim']
    X_raw = validation_data['X_raw']
    y_raw = validation_data['y_raw']

    snp_table = build_snp_metadata(annotation_edges, checkpoint, bim)
    snp_table['feature_index'] = snp_table['snp_index']
    snp_table['importance_abs'] = snp_scores['importance_abs']
    snp_table['importance_signed'] = snp_scores['importance_signed']
    snp_table['mean_gate_weight'] = snp_scores['gate_mean']
    snp_table['local_differential_sensitivity_abs'] = snp_table['importance_abs']
    snp_table['local_differential_sensitivity_signed'] = snp_table['importance_signed']
    snp_table['attention'] = snp_scores['gate_mean']
    snp_table['mean_attention'] = snp_scores['gate_mean']
    snp_table['gate_weight'] = snp_scores['gate_mean']
    snp_table['importance'] = snp_table['local_differential_sensitivity_abs']

    snp_corr = vectorized_correlation(X_raw, y_raw)
    snp_table['genotype_trait_corr'] = snp_corr
    snp_table['abs_genotype_trait_corr'] = np.abs(snp_corr)

    snp_table = snp_table.sort_values(
        'local_differential_sensitivity_abs', ascending=False
    ).reset_index(drop=True)
    snp_table['importance_rank'] = np.arange(1, len(snp_table) + 1)
    snp_table = attach_top_feature_pvalues(
        snp_table,
        X_raw,
        y_raw,
        top_n=min(200, len(snp_table)),
        corr_column='genotype_trait_corr',
        prefix='genotype',
    )
    return snp_table


def prepare_gene_table(validation_data, gene_scores, gene_feature_matrix):
    annotation_edges = validation_data['annotation_edges']
    gene_map = validation_data['gene_map'].copy()
    snp_map = validation_data['snp_map']
    X_raw = validation_data['X_raw']
    y_raw = validation_data['y_raw']

    gene_map['feature_index'] = gene_map['gene_index']
    gene_map['importance_abs'] = gene_scores['importance_abs']
    gene_map['importance_signed'] = gene_scores['importance_signed']
    gene_map['mean_activation_abs'] = gene_scores['mean_activation_abs']
    gene_map['local_differential_sensitivity_abs'] = gene_map['importance_abs']
    gene_map['local_differential_sensitivity_signed'] = gene_map['importance_signed']
    gene_map['importance'] = gene_map['local_differential_sensitivity_abs']
    gene_map['gene_label'] = np.where(
        gene_map['gene_name'].astype(str).str.len() > 0,
        gene_map['gene_name'],
        gene_map['ensembl_gene_id'],
    )
    gene_map['gene'] = gene_map['gene_label']

    sparse_mask = build_sparse_snp_gene_mask(annotation_edges, gene_map, snp_map)
    snps_per_gene = np.asarray(sparse_mask.sum(axis=1)).reshape(-1).astype(np.float32)
    snps_per_gene = np.where(snps_per_gene == 0, 1.0, snps_per_gene)
    gene_burden = np.asarray(X_raw @ sparse_mask.T, dtype=np.float32)
    gene_burden = gene_burden / snps_per_gene[None, :]

    gene_feature_corr = vectorized_correlation(gene_feature_matrix, y_raw)
    gene_burden_corr = vectorized_correlation(gene_burden, y_raw)
    gene_map['gene_feature_trait_corr'] = gene_feature_corr
    gene_map['abs_gene_feature_trait_corr'] = np.abs(gene_feature_corr)
    gene_map['gene_burden_trait_corr'] = gene_burden_corr
    gene_map['abs_gene_burden_trait_corr'] = np.abs(gene_burden_corr)
    gene_map['gene_trait_corr'] = gene_feature_corr
    gene_map['abs_gene_trait_corr'] = np.abs(gene_feature_corr)
    gene_map['n_snps'] = snps_per_gene.astype(int)

    gene_map = gene_map.sort_values(
        'local_differential_sensitivity_abs', ascending=False
    ).reset_index(drop=True)
    gene_map['importance_rank'] = np.arange(1, len(gene_map) + 1)
    gene_map = attach_top_feature_pvalues(
        gene_map,
        gene_feature_matrix,
        y_raw,
        top_n=min(200, len(gene_map)),
        corr_column='gene_trait_corr',
        prefix='gene',
    )
    return gene_map, gene_burden, sparse_mask


def build_impact_tables(snp_table, top_sizes):
    top_sizes = sorted({int(size) for size in top_sizes if size > 0})
    overall_counts = snp_table['new_impact'].fillna('unknown').value_counts().sort_values(ascending=False)
    categories = overall_counts.index.tolist()

    composition_rows = []
    overall_total = overall_counts.sum()
    for impact_class, count in overall_counts.items():
        composition_rows.append(
            {
                'subset': 'Overall',
                'top_n': overall_total,
                'new_impact': impact_class,
                'count': int(count),
                'percentage': float(100.0 * count / overall_total),
            }
        )

    enrichment_rows = []
    for top_n in top_sizes:
        top_subset = snp_table.nlargest(min(top_n, len(snp_table)), 'importance_abs')
        top_counts = top_subset['new_impact'].fillna('unknown').value_counts().reindex(categories, fill_value=0)
        top_total = int(top_counts.sum())
        for impact_class in categories:
            overall_pct = overall_counts[impact_class] / overall_total
            top_pct = top_counts[impact_class] / max(top_total, 1)
            enrichment_rows.append(
                {
                    'top_n': top_n,
                    'new_impact': impact_class,
                    'overall_pct': float(overall_pct),
                    'top_pct': float(top_pct),
                    'log2_enrichment': float(np.log2((top_pct + 1e-8) / (overall_pct + 1e-8))),
                }
            )
            composition_rows.append(
                {
                    'subset': f'Top {top_n}',
                    'top_n': top_n,
                    'new_impact': impact_class,
                    'count': int(top_counts[impact_class]),
                    'percentage': float(100.0 * top_pct),
                }
            )

    impact_summary = (
        snp_table.assign(new_impact=snp_table['new_impact'].fillna('unknown'))
        .groupby('new_impact', as_index=False)
        .agg(
            n_snps=('snp_id', 'size'),
            mean_importance_abs=('importance_abs', 'mean'),
            median_importance_abs=('importance_abs', 'median'),
            mean_gate_weight=('mean_gate_weight', 'mean'),
            mean_abs_corr=('abs_genotype_trait_corr', 'mean'),
        )
        .sort_values(['mean_importance_abs', 'n_snps'], ascending=[False, False])
    )
    impact_summary['attention'] = impact_summary['mean_gate_weight']
    impact_summary['mean_attention'] = impact_summary['mean_gate_weight']

    return (
        pd.DataFrame(composition_rows),
        pd.DataFrame(enrichment_rows),
        impact_summary,
    )


def build_palette(categories):
    cmap = plt.get_cmap('tab20', max(len(categories), 3))
    return {category: cmap(index) for index, category in enumerate(categories)}


def plot_ranked_snp_metric(
    snp_table,
    output_path,
    top_n,
    value_column,
    xlabel,
    title,
    dpi,
):
    if snp_table.empty or value_column not in snp_table.columns:
        return

    top = snp_table.nlargest(min(top_n, len(snp_table)), value_column).copy()
    top = top.sort_values(value_column, ascending=True)
    categories = top['new_impact'].fillna('unknown').tolist()
    palette = build_palette(sorted(pd.unique(categories)))
    colors = [palette.get(category if isinstance(category, str) else 'unknown') for category in categories]

    labels = []
    for row in top.itertuples(index=False):
        gene_label = row.display_gene_label if isinstance(row.display_gene_label, str) and row.display_gene_label else 'no_gene_label'
        labels.append(f'{row.snp_id} | {gene_label}')

    fig_height = max(7, 0.35 * len(top) + 2)
    plt.figure(figsize=(13, fig_height))
    plt.barh(labels, top[value_column], color=colors, edgecolor='none')
    plt.xlabel(xlabel)
    plt.ylabel('SNP')
    plt.title(title)

    handles = []
    used = set()
    for category in categories:
        key = category if isinstance(category, str) else 'unknown'
        if key in used:
            continue
        used.add(key)
        handles.append(plt.Rectangle((0, 0), 1, 1, color=palette[key], label=key))
    if handles:
        plt.legend(handles=handles, title='new_impact', loc='lower right')

    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_ranked_gene_metric(
    gene_table,
    output_path,
    top_n,
    value_column,
    xlabel,
    title,
    dpi,
):
    if gene_table.empty or value_column not in gene_table.columns:
        return

    top = gene_table.nlargest(min(top_n, len(gene_table)), value_column).copy()
    top = top.sort_values(value_column, ascending=True)
    labels = top['gene_label'].tolist()

    fig_height = max(6, 0.35 * len(top) + 2)
    plt.figure(figsize=(11, fig_height))
    plt.barh(labels, top[value_column], color='#bc6c25', edgecolor='none')
    plt.xlabel(xlabel)
    plt.ylabel('Gene')
    plt.title(title)
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_global_importance_bars(
    table,
    output_path,
    label_column,
    value_column,
    top_n,
    xlabel,
    title,
    dpi,
    color='#6c8ead',
):
    if table.empty or value_column not in table.columns:
        return

    top = table.nlargest(min(top_n, len(table)), value_column).copy()
    top = top.sort_values(value_column, ascending=True)

    fig_height = max(6, 0.35 * len(top) + 2)
    plt.figure(figsize=(11, fig_height))
    plt.barh(top[label_column], top[value_column], color=color, edgecolor='none')
    plt.xlabel(xlabel)
    plt.ylabel(label_column.replace('_', ' ').title())
    plt.title(title)
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_prediction_scatter(y_true, y_pred, output_path, dpi):
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, color='#4c956c', edgecolor='none')
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='#2e2e2e')
    plt.xlabel('Observed trait value')
    plt.ylabel('Predicted trait value')
    plt.title('Evaluation predictions vs observed values')
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_top_snp_bars(snp_table, output_path, top_n, dpi):
    plot_ranked_snp_metric(
        snp_table,
        output_path,
        top_n,
        'local_differential_sensitivity_abs',
        'Mean |local differential sensitivity|',
        f'Top {min(top_n, len(snp_table))} SNPs by local differential sensitivity',
        dpi,
    )


def plot_top_gene_bars(gene_table, output_path, top_n, dpi):
    plot_ranked_gene_metric(
        gene_table,
        output_path,
        top_n,
        'local_differential_sensitivity_abs',
        'Mean |local differential sensitivity|',
        f'Top {min(top_n, len(gene_table))} genes by local differential sensitivity',
        dpi,
    )


def plot_impact_composition(composition_table, output_path, dpi):
    pivot = composition_table.pivot(index='new_impact', columns='subset', values='percentage').fillna(0.0)
    ordered_classes = composition_table.loc[
        composition_table['subset'] == 'Overall'
    ].sort_values('percentage', ascending=False)['new_impact']
    pivot = pivot.reindex(ordered_classes)

    subsets = list(pivot.columns)
    x = np.arange(len(pivot.index))
    width = 0.8 / max(len(subsets), 1)
    colors = plt.get_cmap('PuOr')(np.linspace(0.15, 0.85, len(subsets)))

    fig_width = max(12, 0.6 * len(pivot.index) + 4)
    plt.figure(figsize=(fig_width, 6))
    for idx, subset in enumerate(subsets):
        plt.bar(
            x + (idx - (len(subsets) - 1) / 2) * width,
            pivot[subset].to_numpy(),
            width=width,
            label=subset,
            color=colors[idx],
            edgecolor='none',
        )

    plt.xticks(x, pivot.index, rotation=45, ha='right')
    plt.ylabel('Percentage of SNPs')
    plt.title('Impact-class composition in the full model vs top-ranked SNPs')
    plt.legend()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_impact_enrichment(enrichment_table, output_path, dpi):
    if enrichment_table.empty:
        return

    preferred_n = 100 if 100 in set(enrichment_table['top_n']) else int(enrichment_table['top_n'].max())
    subset = enrichment_table.loc[enrichment_table['top_n'] == preferred_n].copy()
    subset = subset.sort_values('log2_enrichment', ascending=False)
    subset = pd.concat([subset.head(8), subset.tail(8)]).drop_duplicates('new_impact')
    subset = subset.sort_values('log2_enrichment', ascending=True)

    plt.figure(figsize=(10, max(6, 0.35 * len(subset) + 2)))
    colors = ['#4c956c' if value >= 0 else '#bc4749' for value in subset['log2_enrichment']]
    plt.barh(subset['new_impact'], subset['log2_enrichment'], color=colors, edgecolor='none')
    plt.axvline(0.0, color='#2e2e2e', linestyle='--')
    plt.xlabel('log2 enrichment relative to all SNPs')
    plt.ylabel('new_impact')
    plt.title(f'Impact-class enrichment among top {preferred_n} SNPs')
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_attention_by_impact(impact_summary, output_path, dpi):
    ordered = impact_summary.sort_values('mean_gate_weight', ascending=False)
    fig_width = max(11, 0.55 * len(ordered) + 3)
    plt.figure(figsize=(fig_width, 6))
    plt.bar(
        ordered['new_impact'],
        ordered['mean_gate_weight'],
        color='#577590',
        edgecolor='none',
    )
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Mean gate weight')
    plt.title('Average learned impact-conditioned gate weight by impact class')
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_snp_importance_vs_correlation(snp_table, output_path, dpi):
    plt.figure(figsize=(8, 6))
    plt.scatter(
        snp_table['abs_genotype_trait_corr'],
        snp_table['local_differential_sensitivity_abs'],
        alpha=0.18,
        s=14,
        color='#7a7a7a',
        edgecolor='none',
    )
    top = snp_table.nlargest(min(20, len(snp_table)), 'local_differential_sensitivity_abs')
    plt.scatter(
        top['abs_genotype_trait_corr'],
        top['local_differential_sensitivity_abs'],
        alpha=0.9,
        s=26,
        color='#d95f02',
        edgecolor='none',
    )
    plt.xlabel('|Pearson correlation| between SNP genotype and trait')
    plt.ylabel('Local differential sensitivity')
    plt.title('Local differential sensitivity vs univariate SNP-trait signal')
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_gene_importance_vs_correlation(gene_table, output_path, dpi):
    plt.figure(figsize=(8, 6))
    plt.scatter(
        gene_table['abs_gene_feature_trait_corr'],
        gene_table['local_differential_sensitivity_abs'],
        alpha=0.22,
        s=16,
        color='#7a7a7a',
        edgecolor='none',
    )
    top = gene_table.nlargest(min(20, len(gene_table)), 'local_differential_sensitivity_abs')
    plt.scatter(
        top['abs_gene_feature_trait_corr'],
        top['local_differential_sensitivity_abs'],
        alpha=0.9,
        s=28,
        color='#2a9d8f',
        edgecolor='none',
    )
    plt.xlabel('|Pearson correlation| between learned gene feature and trait')
    plt.ylabel('Local differential sensitivity')
    plt.title('Local differential sensitivity vs learned gene-feature correlation')
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_snp_manhattan(snp_table, output_path, dpi):
    required_columns = {'chromosome', 'position', 'local_differential_sensitivity_abs'}
    if not required_columns.issubset(snp_table.columns):
        return

    manhattan = snp_table.dropna(subset=['chromosome', 'position']).copy()
    if manhattan.empty:
        return

    manhattan['chromosome'] = manhattan['chromosome'].astype(str)
    manhattan['position'] = pd.to_numeric(manhattan['position'], errors='coerce')
    manhattan = manhattan.dropna(subset=['position'])
    manhattan = manhattan.sort_values(['chromosome', 'position'], key=lambda series: series.map(chrom_sort_key) if series.name == 'chromosome' else series)

    chrom_order = sorted(manhattan['chromosome'].unique(), key=chrom_sort_key)
    colors = ['#5f0f40', '#0f4c5c']
    tick_positions = []
    tick_labels = []
    offset = 0.0
    x_values = np.zeros(len(manhattan), dtype=np.float64)

    for idx, chromosome in enumerate(chrom_order):
        chrom_mask = manhattan['chromosome'] == chromosome
        chrom_positions = manhattan.loc[chrom_mask, 'position'].to_numpy(dtype=np.float64)
        x_values[chrom_mask.to_numpy()] = chrom_positions + offset
        tick_positions.append(offset + chrom_positions.max() / 2.0)
        tick_labels.append(chromosome)
        offset += chrom_positions.max() + 1e6

    manhattan['plot_x'] = x_values

    plt.figure(figsize=(14, 5.5))
    for idx, chromosome in enumerate(chrom_order):
        chrom_df = manhattan.loc[manhattan['chromosome'] == chromosome]
        plt.scatter(
            chrom_df['plot_x'],
            chrom_df['local_differential_sensitivity_abs'],
            s=10,
            alpha=0.7,
            color=colors[idx % len(colors)],
            edgecolor='none',
        )

    highlight = manhattan.nlargest(min(25, len(manhattan)), 'local_differential_sensitivity_abs')
    plt.scatter(
        highlight['plot_x'],
        highlight['local_differential_sensitivity_abs'],
        s=20,
        alpha=0.95,
        color='#f4a261',
        edgecolor='none',
    )
    plt.xticks(tick_positions, tick_labels)
    plt.xlabel('Chromosome')
    plt.ylabel('Local differential sensitivity')
    plt.title('Genome-wide local differential sensitivity landscape')
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def write_summary(
    output_dir,
    checkpoint_path,
    checkpoint,
    validation_data,
    snp_table,
    gene_table,
    impact_summary,
    enrichment_table,
    baseline_metrics,
    gene_global_table=None,
    correlation_block_global_table=None,
    snp_global_table=None,
    shap_note=None,
):
    top_snp_lines = []
    for row in snp_table.head(10).itertuples(index=False):
        gene_label = row.display_gene_label if isinstance(row.display_gene_label, str) and row.display_gene_label else 'NA'
        impact_label = row.new_impact if isinstance(row.new_impact, str) and row.new_impact else 'unknown'
        ig_value = getattr(row, 'integrated_gradients_abs', np.nan)
        top_snp_lines.append(
            f'- {row.snp_id} | gene={gene_label} | impact={impact_label} | '
            f'local_differential_sensitivity={row.local_differential_sensitivity_abs:.6f} | '
            f'integrated_gradients={ig_value:.6f} | '
            f'gate_weight={row.mean_gate_weight:.6f} | abs_corr={row.abs_genotype_trait_corr:.6f}'
        )
    if not top_snp_lines:
        top_snp_lines = ['- No SNP-level results available.']

    top_gene_lines = []
    for row in gene_table.head(10).itertuples(index=False):
        ig_value = getattr(row, 'integrated_gradients_abs', np.nan)
        top_gene_lines.append(
            f'- {row.gene_label} ({row.ensembl_gene_id}) | n_snps={row.n_snps} | '
            f'local_differential_sensitivity={row.local_differential_sensitivity_abs:.6f} | '
            f'integrated_gradients={ig_value:.6f} | abs_gene_feature_corr={row.abs_gene_feature_trait_corr:.6f}'
        )
    if not top_gene_lines:
        top_gene_lines = ['- No gene-level results available.']

    enrichment_lines = []
    if enrichment_table is not None and not enrichment_table.empty:
        preferred_n = 100 if 100 in set(enrichment_table['top_n']) else int(enrichment_table['top_n'].max())
        top_enrichment = enrichment_table.loc[
            enrichment_table['top_n'] == preferred_n
        ].sort_values('log2_enrichment', ascending=False).head(8)
        enrichment_lines = [
            f'- {row.new_impact}: top_pct={100.0 * row.top_pct:.2f}% | '
            f'overall_pct={100.0 * row.overall_pct:.2f}% | log2_enrichment={row.log2_enrichment:.3f}'
            for row in top_enrichment.itertuples(index=False)
        ]
    preferred_n = 100 if enrichment_table is not None and not enrichment_table.empty and 100 in set(enrichment_table['top_n']) else (
        int(enrichment_table['top_n'].max()) if enrichment_table is not None and not enrichment_table.empty else 0
    )
    if not enrichment_lines:
        enrichment_lines = ['- No impact-class enrichment table available.']

    global_gene_lines = []
    if gene_global_table is not None and not gene_global_table.empty:
        for row in gene_global_table.head(8).itertuples(index=False):
            global_gene_lines.append(
                f'- {row.group_label}: mean_replacement_delta_mse={row.mean_replacement_delta_mse:.6f} | '
                f'permutation_delta_mse_mean={row.permutation_delta_mse_mean:.6f} | n_snps={row.n_snps}'
            )

    global_block_lines = []
    if correlation_block_global_table is not None and not correlation_block_global_table.empty:
        for row in correlation_block_global_table.head(6).itertuples(index=False):
            global_block_lines.append(
                f'- {row.group_label}: mean_replacement_delta_mse={row.mean_replacement_delta_mse:.6f} | '
                f'permutation_delta_mse_mean={row.permutation_delta_mse_mean:.6f} | n_snps={row.n_snps}'
            )

    global_snp_lines = []
    if snp_global_table is not None and not snp_global_table.empty:
        for row in snp_global_table.head(8).itertuples(index=False):
            global_snp_lines.append(
                f'- {row.group_label}: mean_replacement_delta_mse={row.mean_replacement_delta_mse:.6f} | '
                f'permutation_delta_mse_mean={row.permutation_delta_mse_mean:.6f} | correlation_block_snps={row.n_snps}'
            )

    impact_gate_lines = []
    if impact_summary is not None and not impact_summary.empty:
        impact_gate_lines = [
            f"- {row.new_impact}: gate_weight={row.mean_gate_weight:.6f} | n_snps={row.n_snps}"
            for row in impact_summary.head(10).itertuples(index=False)
        ]
    if not impact_gate_lines:
        impact_gate_lines = ['- No impact-class gate-weight summary available.']

    trait_name = trait_label(validation_data['trait'])
    n_evaluation_samples = len(validation_data['sample_ids_eval'])
    summary_lines = [
        '# Impact-Attention Model Interpretation Summary',
        '',
        f'- Checkpoint: {checkpoint_path}',
        f'- Trait: {trait_name}',
        f"- Evaluation cohort source: {validation_data['evaluation_source']}",
        f"- Evaluation cohort note: {validation_data['evaluation_source_note']}",
        f'- Evaluation samples analyzed: {n_evaluation_samples}',
        f"- Held-out MAE: {baseline_metrics['mae']:.6f}",
        f"- Held-out MSE: {baseline_metrics['mse']:.6f}",
        f"- Held-out RMSE: {baseline_metrics['rmse']:.6f}",
        f"- Held-out R2: {baseline_metrics['r2']:.6f}",
        f"- Held-out Pearson: {baseline_metrics['pearson']:.6f}",
    ]
    if validation_data.get('evaluation_warning'):
        summary_lines.extend(['', f"- Evaluation warning: {validation_data['evaluation_warning']}"])
    if shap_note:
        summary_lines.extend(['', f'- SHAP status: {shap_note}'])

    summary_lines.extend(['', '## Top SNPs by Local Differential Sensitivity', *top_snp_lines])
    summary_lines.extend(['', '## Top Genes by Local Differential Sensitivity and Integrated Gradients', *top_gene_lines])
    if global_gene_lines:
        summary_lines.extend(['', '## Top Genes by Held-out Perturbation Importance', *global_gene_lines])
    if global_block_lines:
        summary_lines.extend(['', '## Top Correlation Blocks by Held-out Perturbation Importance', *global_block_lines])
    if global_snp_lines:
        summary_lines.extend(['', '## Correlation-Block Representative SNPs by Held-out Perturbation Importance', *global_snp_lines])
    if preferred_n > 0:
        summary_lines.extend(['', f'## Most Enriched Impact Classes in Top {preferred_n} SNPs', *enrichment_lines])
    else:
        summary_lines.extend(['', '## Most Enriched Impact Classes', *enrichment_lines])
    summary_lines.extend(['', '## Impact-Class Mean Gate Weights', *impact_gate_lines])
    (output_dir / 'interpretation_summary.md').write_text('\n'.join(summary_lines))


def main():
    args = parse_args()
    set_plot_style()

    checkpoint_path, checkpoint = load_checkpoint(args.checkpoint)
    if checkpoint['architecture'].get('model_type', checkpoint.get('training_args', {}).get('model_type')) != 'impact_attention_mlp':
        raise ValueError('This analysis script currently expects an impact_attention_mlp checkpoint.')

    output_dir = infer_output_dir(checkpoint_path, args.output_dir)
    device = torch.device(args.device)

    validation_data = reconstruct_validation_data(
        checkpoint,
        max_samples=args.max_samples,
        sample_id_file=args.sample_id_file,
        sample_id_column=args.sample_id_column,
    )
    model = build_model_from_checkpoint(checkpoint, map_location=device)
    model.eval()

    checkpoint['snp_map_dataframe'] = validation_data['snp_map']
    checkpoint['gene_map_dataframe'] = validation_data['gene_map']
    checkpoint['impact_map_dataframe'] = validation_data['impact_map']
    checkpoint['snp_annotation_dataframe'] = validation_data['annotation_edges']

    prediction_scaled = batched_predict(model, validation_data['X_scaled'], device, args.batch_size)
    prediction_real = inverse_scale(
        prediction_scaled,
        validation_data['scaler_y_mean'],
        validation_data['scaler_y_scale'],
    ).reshape(-1)
    baseline_metrics = compute_regression_metrics(validation_data['y_raw'], prediction_real)

    predictions = pd.DataFrame(
        {
            'sample_id': validation_data['sample_ids_eval'],
            'y_true': validation_data['y_raw'],
            'y_pred': prediction_real,
            'residual': validation_data['y_raw'] - prediction_real,
        }
    )
    predictions.to_csv(output_dir / 'evaluation_predictions.csv', index=False)
    predictions.to_csv(output_dir / 'validation_predictions.csv', index=False)

    snp_scores = compute_snp_importance(
        model,
        validation_data['X_scaled'],
        device,
        args.batch_size,
    )
    gene_scores = compute_gene_importance(
        model,
        validation_data['X_scaled'],
        device,
        args.batch_size,
    )
    gene_feature_matrix = encode_gene_features(
        model,
        validation_data['X_scaled'],
        device,
        args.batch_size,
    )

    snp_table = prepare_snp_table(validation_data, checkpoint, snp_scores)
    gene_table, gene_burden, sparse_mask = prepare_gene_table(
        validation_data,
        gene_scores,
        gene_feature_matrix,
    )

    correlation_block_table, snp_to_block = build_correlation_blocks(
        snp_table,
        validation_data['X_raw'],
        corr_threshold=args.correlation_threshold,
        max_gap_kb=args.correlation_max_gap_kb,
    )
    snp_table['correlation_block_id'] = snp_table['snp_index'].map(
        lambda value: snp_to_block.get(int(value), 'unassigned')
    )
    snp_table['is_correlation_block_representative'] = False

    if not correlation_block_table.empty:
        correlation_block_table = correlation_block_table.copy()

        def _format_block_label(row):
            chrom = row.chromosome
            start = row.start_position
            end = row.end_position
            if pd.notna(start) and pd.notna(end):
                return f'chr{chrom}:{int(start):,}-{int(end):,} ({int(row.n_snps)} SNPs)'
            return f'chr{chrom} {row.correlation_block_id} ({int(row.n_snps)} SNPs)'

        correlation_block_table['group_id'] = correlation_block_table['correlation_block_id']
        correlation_block_table['group_label'] = correlation_block_table.apply(
            _format_block_label,
            axis=1,
        )
        snp_table = snp_table.merge(
            correlation_block_table[
                [
                    'correlation_block_id',
                    'n_snps',
                    'representative_snp_id',
                    'representative_snp_index',
                ]
            ].rename(
                columns={
                    'n_snps': 'correlation_block_n_snps',
                    'representative_snp_id': 'correlation_block_representative_snp_id',
                    'representative_snp_index': 'correlation_block_representative_snp_index',
                }
            ),
            on='correlation_block_id',
            how='left',
        )
        snp_table['is_correlation_block_representative'] = (
            snp_table['snp_id'] == snp_table['correlation_block_representative_snp_id']
        )
    else:
        snp_table['correlation_block_n_snps'] = 1
        snp_table['correlation_block_representative_snp_id'] = snp_table['snp_id']
        snp_table['correlation_block_representative_snp_index'] = snp_table['snp_index']

    snp_table['ld_block_id'] = snp_table['correlation_block_id']
    snp_table['ld_block_n_snps'] = snp_table['correlation_block_n_snps']
    snp_table['ld_block_representative_snp_id'] = snp_table['correlation_block_representative_snp_id']
    snp_table['ld_block_representative_snp_index'] = snp_table['correlation_block_representative_snp_index']
    snp_table['is_ld_block_representative'] = snp_table['is_correlation_block_representative']

    if not correlation_block_table.empty:
        correlation_block_table['ld_block_id'] = correlation_block_table['correlation_block_id']

    gene_group_table = build_gene_group_table(gene_table, sparse_mask, snp_table)
    representative_snp_table = build_block_representative_snp_table(
        snp_table,
        correlation_block_table,
    )

    gene_global_table, gene_global_baseline = evaluate_group_global_importance(
        model.fc_stack,
        gene_feature_matrix,
        validation_data['y_raw'],
        validation_data['scaler_y_mean'],
        validation_data['scaler_y_scale'],
        gene_group_table,
        args.gene_global_top_k,
        args.permutation_repeats,
        device,
        args.batch_size,
    )
    correlation_block_global_table, _ = evaluate_group_global_importance(
        model,
        validation_data['X_scaled'],
        validation_data['y_raw'],
        validation_data['scaler_y_mean'],
        validation_data['scaler_y_scale'],
        correlation_block_table,
        args.correlation_block_top_k,
        args.permutation_repeats,
        device,
        args.batch_size,
    )
    snp_global_table, _ = evaluate_group_global_importance(
        model,
        validation_data['X_scaled'],
        validation_data['y_raw'],
        validation_data['scaler_y_mean'],
        validation_data['scaler_y_scale'],
        representative_snp_table,
        args.snp_global_top_k,
        args.permutation_repeats,
        device,
        args.batch_size,
    )

    if not gene_global_table.empty:
        gene_global_table = gene_global_table.merge(
            gene_table[
                [
                    'ensembl_gene_id',
                    'gene_label',
                    'gene_name',
                    'local_differential_sensitivity_abs',
                    'abs_gene_feature_trait_corr',
                    'abs_gene_burden_trait_corr',
                    'n_snps',
                ]
            ],
            left_on='group_id',
            right_on='ensembl_gene_id',
            how='left',
            suffixes=('', '_gene'),
        )
        gene_table = gene_table.merge(
            gene_global_table[
                [
                    'group_id',
                    'mean_replacement_delta_mae',
                    'mean_replacement_delta_mse',
                    'mean_replacement_delta_rmse',
                    'mean_replacement_delta_r2',
                    'mean_replacement_delta_pearson',
                    'permutation_delta_mae_mean',
                    'permutation_delta_mse_mean',
                    'permutation_delta_rmse_mean',
                    'permutation_delta_r2_mean',
                    'permutation_delta_pearson_mean',
                ]
            ].rename(columns={'group_id': 'ensembl_gene_id'}),
            on='ensembl_gene_id',
            how='left',
        )

    if not snp_global_table.empty:
        snp_global_table = snp_global_table.rename(
            columns={'n_snps': 'correlation_block_n_snps'}
        )
        snp_global_table['ld_block_n_snps'] = snp_global_table['correlation_block_n_snps']

    snp_ig_result = compute_integrated_gradients(
        model,
        validation_data['X_scaled'],
        device,
        args.batch_size,
        steps=args.ig_steps,
        max_samples=args.ig_samples,
    )
    ig_abs = np.asarray(snp_ig_result['abs'], dtype=np.float32)
    ig_signed = np.asarray(snp_ig_result['signed'], dtype=np.float32)
    snp_table['integrated_gradients_abs'] = snp_table['snp_index'].map(lambda value: float(ig_abs[int(value)]))
    snp_table['integrated_gradients_signed'] = snp_table['snp_index'].map(lambda value: float(ig_signed[int(value)]))

    gene_ig_result = compute_integrated_gradients(
        model.fc_stack,
        gene_feature_matrix,
        device,
        args.batch_size,
        steps=args.ig_steps,
        max_samples=args.ig_samples,
    )
    gene_ig_abs = np.asarray(gene_ig_result['abs'], dtype=np.float32)
    gene_ig_signed = np.asarray(gene_ig_result['signed'], dtype=np.float32)
    gene_table['integrated_gradients_abs'] = gene_table['gene_index'].map(
        lambda value: float(gene_ig_abs[int(value)])
    )
    gene_table['integrated_gradients_signed'] = gene_table['gene_index'].map(
        lambda value: float(gene_ig_signed[int(value)])
    )

    shap_note = 'not run (use --run-shap to enable)'
    if args.run_shap:
        shap_notes = []
        snp_shap_result, snp_shap_note = compute_shap_attributions(
            model,
            validation_data['X_scaled'],
            device,
            background_size=args.shap_background_size,
            eval_size=args.shap_eval_size,
        )
        if snp_shap_result is not None:
            shap_abs = np.asarray(snp_shap_result['abs'], dtype=np.float32)
            shap_signed = np.asarray(snp_shap_result['signed'], dtype=np.float32)
            snp_table['shap_abs'] = snp_table['snp_index'].map(lambda value: float(shap_abs[int(value)]))
            snp_table['shap_signed'] = snp_table['snp_index'].map(lambda value: float(shap_signed[int(value)]))
            shap_notes.append(
                f"SNP-space SHAP completed on {len(snp_shap_result['sample_indices'])} evaluation samples with "
                f'{min(args.shap_background_size, len(validation_data["X_scaled"]))} background samples'
            )
        else:
            shap_notes.append(f'SNP-space SHAP not available: {snp_shap_note}')

        gene_shap_result, gene_shap_note = compute_shap_attributions(
            model.fc_stack,
            gene_feature_matrix,
            device,
            background_size=args.shap_background_size,
            eval_size=args.shap_eval_size,
        )
        if gene_shap_result is not None:
            gene_shap_abs = np.asarray(gene_shap_result['abs'], dtype=np.float32)
            gene_shap_signed = np.asarray(gene_shap_result['signed'], dtype=np.float32)
            gene_table['shap_abs'] = gene_table['gene_index'].map(
                lambda value: float(gene_shap_abs[int(value)])
            )
            gene_table['shap_signed'] = gene_table['gene_index'].map(
                lambda value: float(gene_shap_signed[int(value)])
            )
            shap_notes.append(
                f"gene-feature SHAP completed on {len(gene_shap_result['sample_indices'])} evaluation samples with "
                f'{min(args.shap_background_size, len(gene_feature_matrix))} background samples'
            )
        else:
            shap_notes.append(f'gene-feature SHAP not available: {gene_shap_note}')
        shap_note = '; '.join(shap_notes)

    snp_table = snp_table.sort_values(
        ['local_differential_sensitivity_abs', 'integrated_gradients_abs'],
        ascending=[False, False],
    ).reset_index(drop=True)
    snp_table['importance_rank'] = np.arange(1, len(snp_table) + 1)

    gene_table = gene_table.sort_values(
        ['local_differential_sensitivity_abs', 'integrated_gradients_abs'],
        ascending=[False, False],
    ).reset_index(drop=True)
    gene_table['importance_rank'] = np.arange(1, len(gene_table) + 1)

    composition_table, enrichment_table, impact_summary = build_impact_tables(
        snp_table,
        args.top_impact_sizes,
    )

    ig_snp_table = snp_table[
        [
            'snp_id',
            'snp_index',
            'new_impact',
            'chromosome',
            'position',
            'display_gene_label',
            'integrated_gradients_abs',
            'integrated_gradients_signed',
            'local_differential_sensitivity_abs',
            'mean_gate_weight',
            'correlation_block_id',
            'correlation_block_n_snps',
        ]
    ].sort_values('integrated_gradients_abs', ascending=False)
    ig_gene_table = gene_table[
        [
            'ensembl_gene_id',
            'gene_label',
            'gene_name',
            'n_snps',
            'integrated_gradients_abs',
            'integrated_gradients_signed',
            'local_differential_sensitivity_abs',
            'abs_gene_feature_trait_corr',
            'abs_gene_burden_trait_corr',
        ]
    ].sort_values('integrated_gradients_abs', ascending=False)

    if not gene_global_table.empty:
        gene_global_table = gene_global_table.sort_values(
            ['mean_replacement_delta_mse', 'permutation_delta_mse_mean'],
            ascending=[False, False],
        ).reset_index(drop=True)
    if not correlation_block_global_table.empty:
        correlation_block_global_table = correlation_block_global_table.sort_values(
            ['mean_replacement_delta_mse', 'permutation_delta_mse_mean'],
            ascending=[False, False],
        ).reset_index(drop=True)
    if not snp_global_table.empty:
        snp_global_table = snp_global_table.sort_values(
            ['mean_replacement_delta_mse', 'permutation_delta_mse_mean'],
            ascending=[False, False],
        ).reset_index(drop=True)

    snp_table.to_csv(output_dir / 'snp_importance.csv', index=False)
    gene_table.to_csv(output_dir / 'gene_importance.csv', index=False)
    composition_table.to_csv(output_dir / 'impact_class_composition.csv', index=False)
    enrichment_table.to_csv(output_dir / 'impact_class_enrichment.csv', index=False)
    impact_summary.to_csv(output_dir / 'impact_class_summary.csv', index=False)
    correlation_block_table.to_csv(output_dir / 'correlation_block_catalog.csv', index=False)
    representative_snp_table.to_csv(
        output_dir / 'correlation_block_representative_snps.csv',
        index=False,
    )
    correlation_block_table.to_csv(output_dir / 'ld_block_catalog.csv', index=False)
    representative_snp_table.to_csv(output_dir / 'ld_block_representative_snps.csv', index=False)
    gene_global_table.to_csv(output_dir / 'gene_global_importance.csv', index=False)
    correlation_block_global_table.to_csv(
        output_dir / 'correlation_block_global_importance.csv',
        index=False,
    )
    correlation_block_global_table.to_csv(
        output_dir / 'ld_block_global_importance.csv',
        index=False,
    )
    snp_global_table.to_csv(output_dir / 'snp_global_importance.csv', index=False)
    ig_snp_table.to_csv(output_dir / 'integrated_gradients_snp.csv', index=False)
    ig_gene_table.to_csv(output_dir / 'integrated_gradients_gene.csv', index=False)

    if 'shap_abs' in snp_table.columns:
        snp_table[
            [
                'snp_id',
                'snp_index',
                'new_impact',
                'chromosome',
                'position',
                'display_gene_label',
                'shap_abs',
                'shap_signed',
                'correlation_block_id',
                'correlation_block_n_snps',
            ]
        ].sort_values('shap_abs', ascending=False).to_csv(output_dir / 'shap_snp.csv', index=False)
    if 'shap_abs' in gene_table.columns:
        gene_table[
            [
                'ensembl_gene_id',
                'gene_label',
                'gene_name',
                'n_snps',
                'shap_abs',
                'shap_signed',
            ]
        ].sort_values('shap_abs', ascending=False).to_csv(output_dir / 'shap_gene.csv', index=False)

    top_snps = snp_table.head(min(200, len(snp_table))).copy()
    top_snps.to_csv(output_dir / 'top_snp_candidates.csv', index=False)
    top_genes = gene_table.head(min(200, len(gene_table))).copy()
    top_genes.to_csv(output_dir / 'top_gene_candidates.csv', index=False)

    metadata = {
        'checkpoint': str(checkpoint_path),
        'output_dir': str(output_dir),
        'trait': int(validation_data['trait']),
        'trait_label': trait_label(validation_data['trait']),
        'evaluation_source': validation_data['evaluation_source'],
        'evaluation_source_note': validation_data['evaluation_source_note'],
        'evaluation_warning': validation_data['evaluation_warning'],
        'n_validation_samples': int(len(validation_data['sample_ids_eval'])),
        'n_evaluation_samples': int(len(validation_data['sample_ids_eval'])),
        'n_snps': int(len(snp_table)),
        'n_genes': int(len(gene_table)),
        'n_correlation_blocks': int(len(correlation_block_table)),
        'final_metrics': checkpoint.get('final_metrics', {}),
        'baseline_metrics': baseline_metrics,
        'global_importance_baseline_metrics': gene_global_baseline,
        'gene_global_top_k': int(args.gene_global_top_k),
        'correlation_block_top_k': int(args.correlation_block_top_k),
        'snp_global_top_k': int(args.snp_global_top_k),
        'permutation_repeats': int(args.permutation_repeats),
        'correlation_threshold': float(args.correlation_threshold),
        'correlation_max_gap_kb': int(args.correlation_max_gap_kb),
        'ig_steps': int(args.ig_steps),
        'ig_samples': int(min(args.ig_samples, len(validation_data['X_scaled']))),
        'run_shap': bool(args.run_shap),
        'shap_note': shap_note,
        'gene_integrated_gradients_space': 'learned_gene_feature',
        'gene_shap_space': 'learned_gene_feature' if args.run_shap else 'not_run',
    }
    (output_dir / 'analysis_metadata.json').write_text(json.dumps(metadata, indent=2))

    plot_prediction_scatter(
        predictions['y_true'].to_numpy(),
        predictions['y_pred'].to_numpy(),
        output_dir / 'predicted_vs_observed.png',
        args.dpi,
    )
    plot_top_snp_bars(snp_table, output_dir / 'top_snps_importance.png', args.top_snps, args.dpi)
    plot_top_gene_bars(gene_table, output_dir / 'top_genes_importance.png', args.top_genes, args.dpi)
    plot_ranked_snp_metric(
        snp_table,
        output_dir / 'top_snps_integrated_gradients.png',
        args.top_snps,
        'integrated_gradients_abs',
        'Mean |integrated gradients|',
        f'Top {min(args.top_snps, len(snp_table))} SNPs by integrated gradients',
        args.dpi,
    )
    plot_ranked_gene_metric(
        gene_table,
        output_dir / 'top_genes_integrated_gradients.png',
        args.top_genes,
        'integrated_gradients_abs',
        'Mean |integrated gradients|',
        f'Top {min(args.top_genes, len(gene_table))} genes by integrated gradients',
        args.dpi,
    )
    if 'shap_abs' in snp_table.columns:
        plot_ranked_snp_metric(
            snp_table,
            output_dir / 'top_snps_shap.png',
            args.top_snps,
            'shap_abs',
            'Mean |SHAP value|',
            f'Top {min(args.top_snps, len(snp_table))} SNPs by SHAP',
            args.dpi,
        )
    if 'shap_abs' in gene_table.columns:
        plot_ranked_gene_metric(
            gene_table,
            output_dir / 'top_genes_shap.png',
            args.top_genes,
            'shap_abs',
            'Mean |SHAP value|',
            f'Top {min(args.top_genes, len(gene_table))} genes by SHAP',
            args.dpi,
        )
    plot_impact_composition(composition_table, output_dir / 'impact_class_composition.png', args.dpi)
    plot_impact_enrichment(enrichment_table, output_dir / 'impact_class_enrichment.png', args.dpi)
    plot_attention_by_impact(impact_summary, output_dir / 'impact_class_gate_weight.png', args.dpi)
    plot_snp_importance_vs_correlation(snp_table, output_dir / 'snp_importance_vs_correlation.png', args.dpi)
    plot_gene_importance_vs_correlation(gene_table, output_dir / 'gene_importance_vs_correlation.png', args.dpi)
    plot_snp_manhattan(snp_table, output_dir / 'snp_importance_manhattan.png', args.dpi)
    plot_global_importance_bars(
        gene_global_table,
        output_dir / 'gene_global_mean_replacement_importance.png',
        'group_label',
        'mean_replacement_delta_mse',
        min(args.top_genes, len(gene_global_table)),
        'Held-out delta MSE after gene mean-replacement perturbation',
        'Top genes by held-out perturbation importance (mean replacement)',
        args.dpi,
        color='#8c564b',
    )
    plot_global_importance_bars(
        gene_global_table,
        output_dir / 'gene_global_permutation_importance.png',
        'group_label',
        'permutation_delta_mse_mean',
        min(args.top_genes, len(gene_global_table)),
        'Held-out delta MSE after gene permutation',
        'Top genes by held-out perturbation importance (permutation)',
        args.dpi,
        color='#9c6644',
    )
    plot_global_importance_bars(
        correlation_block_global_table,
        output_dir / 'correlation_block_global_mean_replacement_importance.png',
        'group_label',
        'mean_replacement_delta_mse',
        min(args.top_snps, len(correlation_block_global_table)),
        'Held-out delta MSE after correlation-block mean replacement',
        'Top correlation blocks by held-out perturbation importance (mean replacement)',
        args.dpi,
        color='#3d5a80',
    )
    plot_global_importance_bars(
        correlation_block_global_table,
        output_dir / 'correlation_block_global_permutation_importance.png',
        'group_label',
        'permutation_delta_mse_mean',
        min(args.top_snps, len(correlation_block_global_table)),
        'Held-out delta MSE after correlation-block permutation',
        'Top correlation blocks by held-out perturbation importance (permutation)',
        args.dpi,
        color='#457b9d',
    )
    plot_global_importance_bars(
        snp_global_table,
        output_dir / 'snp_global_mean_replacement_importance.png',
        'group_label',
        'mean_replacement_delta_mse',
        min(args.top_snps, len(snp_global_table)),
        'Held-out delta MSE after representative-SNP mean replacement',
        'Top correlation-block representative SNPs by held-out perturbation importance (mean replacement)',
        args.dpi,
        color='#6a994e',
    )
    plot_global_importance_bars(
        snp_global_table,
        output_dir / 'snp_global_permutation_importance.png',
        'group_label',
        'permutation_delta_mse_mean',
        min(args.top_snps, len(snp_global_table)),
        'Held-out delta MSE after representative-SNP permutation',
        'Top correlation-block representative SNPs by held-out perturbation importance (permutation)',
        args.dpi,
        color='#588157',
    )

    write_summary(
        output_dir,
        checkpoint_path,
        checkpoint,
        validation_data,
        snp_table,
        gene_table,
        impact_summary,
        enrichment_table,
        baseline_metrics,
        gene_global_table=gene_global_table,
        correlation_block_global_table=correlation_block_global_table,
        snp_global_table=snp_global_table,
        shap_note=shap_note,
    )

    print(f'Analysis written to: {output_dir}')
    print(f'Top SNP table: {output_dir / "snp_importance.csv"}')
    print(f'Top gene table: {output_dir / "gene_importance.csv"}')
    print(f'Gene global importance table: {output_dir / "gene_global_importance.csv"}')
    print(f'Correlation-block global importance table: {output_dir / "correlation_block_global_importance.csv"}')
    print(f'SNP global importance table: {output_dir / "snp_global_importance.csv"}')


if __name__ == '__main__':
    main()
