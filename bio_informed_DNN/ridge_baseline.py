import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from data_utils import (
        build_annotation_resources,
        load_annotation_dataframe,
        load_trait_dataset,
        load_trait_test_dataset,
        split_last_n_samples,
    )
    from normalization import (
        GENOTYPE_NORMALIZATION_CHOICES,
        build_genotype_scaler,
        genotype_normalization_description,
        genotype_normalization_formula,
        genotype_normalization_tag,
    )
except ImportError:
    from bio_informed_DNN.data_utils import (
        build_annotation_resources,
        load_annotation_dataframe,
        load_trait_dataset,
        load_trait_test_dataset,
        split_last_n_samples,
    )
    from bio_informed_DNN.normalization import (
        GENOTYPE_NORMALIZATION_CHOICES,
        build_genotype_scaler,
        genotype_normalization_description,
        genotype_normalization_formula,
        genotype_normalization_tag,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_ANNOTATION_PATH = str(PROJECT_ROOT / 'data/ML/vep_ingenes_newimpact.csv')
DEFAULT_GENOTYPE_PREFIX = str(PROJECT_ROOT / 'data/ML/BBB2023_MD')
DEFAULT_PHENOTYPE_PATH = str(PROJECT_ROOT / 'data/ML/pheno_2023bbb_0twins_6traits_mask')
DEFAULT_FULL_PHENOTYPE_PATH = str(PROJECT_ROOT / 'data/ML/pheno_20000bbb_6traits')
DEFAULT_PEDIGREE_PATH = str(PROJECT_ROOT / 'data/ML/pedi_full_list.txt')


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Compare two GBLUP/kernel-ridge baselines using the last valid samples '
            'as validation: one model with all SNPs and one restricted to the '
            'annotation-backed SNP subset.'
        )
    )
    parser.add_argument('-t', '--trait', type=int, default=2)
    parser.add_argument('--annotation-path', type=str, default=DEFAULT_ANNOTATION_PATH)
    parser.add_argument('--genotype-prefix', type=str, default=DEFAULT_GENOTYPE_PREFIX)
    parser.add_argument('--phenotype-path', type=str, default=DEFAULT_PHENOTYPE_PATH)
    parser.add_argument(
        '--genotype-normalization',
        type=str,
        choices=GENOTYPE_NORMALIZATION_CHOICES,
        default='standard',
        help=(
            'Genotype feature normalization: standard uses StandardScaler; '
            'allele_frequency uses z_ij = (x_ij - 2 p_j) / sqrt(2 p_j (1 - p_j)).'
        ),
    )
    parser.add_argument(
        '--full-phenotype-path',
        type=str,
        default=DEFAULT_FULL_PHENOTYPE_PATH,
        help='Full phenotype file used to recover the held-out test targets.',
    )
    parser.add_argument(
        '--pedigree-path',
        type=str,
        default=DEFAULT_PEDIGREE_PATH,
        help='Pedigree file used to sort individuals by birth date before taking the final validation holdout.',
    )
    parser.add_argument(
        '--val-size',
        type=int,
        default=1000,
        help='Number of final valid samples reserved for validation.',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results/gblup_baseline',
    )
    return parser.parse_args()


def safe_pearson(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size == 0 or y_pred.size == 0:
        return np.nan
    if np.allclose(y_true.std(ddof=0), 0.0) or np.allclose(y_pred.std(ddof=0), 0.0):
        return np.nan
    return float(pearsonr(y_true, y_pred)[0])


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mse = mean_squared_error(y_true, y_pred)
    return {
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'mse': float(mse),
        'rmse': float(np.sqrt(mse)),
        'r2': float(r2_score(y_true, y_pred)),
        'pearson': safe_pearson(y_true, y_pred),
    }


def project_out_intercept(y):
    """
    Remove the intercept from y: M y, with M = I - 11'/n.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    return y - y.mean()


def estimate_gblup_lambda_reml(kernel_matrix, y_train_scaled, lambda_bounds=(1e-6, 1e6)):
    """
    Estimate lambda = sigma_e^2 / sigma_g^2 by REML for:

        y = 1*mu + g + e
        g ~ N(0, sigma_g^2 K)
        e ~ N(0, sigma_e^2 I)

    After projecting out the intercept, REML can be optimized over lambda.
    """
    K = np.asarray(kernel_matrix, dtype=np.float64)
    y = np.asarray(y_train_scaled, dtype=np.float64).reshape(-1)

    if K.shape[0] != K.shape[1]:
        raise ValueError('kernel_matrix must be square.')
    if K.shape[0] != y.shape[0]:
        raise ValueError('kernel_matrix and y_train_scaled dimensions do not match.')

    n = y.shape[0]
    if n <= 1:
        raise ValueError('Need at least 2 samples.')

    # Project out intercept from y
    y_centered = project_out_intercept(y)

    # Eigen decomposition of K
    evals, evecs = eigh(K)
    evals = np.maximum(evals, 0.0)

    # One eigenvector corresponds roughly to the mean direction; since we centered y,
    # REML can be approximated well by using centered y with the full eigensystem.
    uy = evecs.T @ y_centered

    def neg_reml_loglik(log_lambda):
        lam = np.exp(log_lambda)
        d = evals + lam

        # sigma_g^2_hat under REML-like profile likelihood
        quad = np.sum((uy ** 2) / d)
        sigma_g2_hat = quad / (n - 1)
        sigma_g2_hat = max(sigma_g2_hat, 1e-12)

        # Profile REML criterion up to constants
        value = (
            (n - 1) * np.log(sigma_g2_hat)
            + np.sum(np.log(d))
        )
        return float(value)

    result = minimize_scalar(
        neg_reml_loglik,
        bounds=(np.log(lambda_bounds[0]), np.log(lambda_bounds[1])),
        method='bounded',
        options={'xatol': 1e-6},
    )

    if not result.success:
        raise RuntimeError(f'REML optimization failed: {result.message}')

    lambda_gblup = float(np.exp(result.x))
    d_opt = evals + lambda_gblup
    quad_opt = np.sum((uy ** 2) / d_opt)
    sigma_g2_hat = float(max(quad_opt / (n - 1), 1e-12))
    sigma_e2_hat = float(max(lambda_gblup * sigma_g2_hat, 1e-12))

    return {
        'sigma_g': sigma_g2_hat,
        'sigma_e': sigma_e2_hat,
        'lambda_gblup': lambda_gblup,
        'optimizer_success': bool(result.success),
        'optimizer_fun': float(result.fun),
        'optimizer_log_lambda': float(result.x),
        'n_samples': int(n),
    }


def fit_gblup_kernel(kernel_train, y_train_scaled, lambda_value):
    n_train = kernel_train.shape[0]
    regularized = kernel_train + float(lambda_value) * np.eye(n_train, dtype=np.float64)
    factor = cho_factor(regularized, lower=True, check_finite=False)
    dual_coef = cho_solve(factor, y_train_scaled.reshape(-1), check_finite=False)
    return dual_coef


def predict_gblup_kernel(Z_reference, Z_query, dual_coef):
    kernel_query = Z_query @ Z_reference.T
    return kernel_query @ dual_coef


def prepare_feature_set_result(
    name,
    X,
    y,
    sample_ids,
    X_test,
    y_test,
    sample_ids_test,
    sample_ordering,
    val_size,
    genotype_normalization,
):
    split_data = split_last_n_samples(
        X,
        y,
        sample_ids=sample_ids,
        val_size=val_size,
    )

    scaler_X = build_genotype_scaler(genotype_normalization)
    X_train = scaler_X.fit_transform(split_data['X_train']).astype(np.float64)
    X_val = scaler_X.transform(split_data['X_val']).astype(np.float64)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(
        split_data['y_train'].reshape(-1, 1)
    ).reshape(-1).astype(np.float64)

    n_features = X_train.shape[1]
    scale_factor = np.sqrt(float(n_features))
    Z_train = X_train / scale_factor
    Z_val = X_val / scale_factor

    kernel_train = Z_train @ Z_train.T

    gblup_summary = estimate_gblup_lambda_reml(
        kernel_matrix=kernel_train,
        y_train_scaled=y_train_scaled,
    )

    lambda_value = float(gblup_summary['lambda_gblup'])

    dual_coef = fit_gblup_kernel(
        kernel_train=kernel_train,
        y_train_scaled=y_train_scaled,
        lambda_value=lambda_value,
    )

    primal_coef = Z_train.T @ dual_coef

    y_val_pred_scaled = predict_gblup_kernel(Z_train, Z_val, dual_coef)
    y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).reshape(-1)
    y_val = split_data['y_val'].reshape(-1)
    validation_metrics = compute_metrics(y_val, y_val_pred)

    X_test_scaled = scaler_X.transform(np.asarray(X_test, dtype=np.float64))
    Z_test = X_test_scaled / scale_factor
    y_test_pred_scaled = predict_gblup_kernel(Z_train, Z_test, dual_coef)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).reshape(-1)
    y_test = np.asarray(y_test, dtype=np.float64).reshape(-1)
    test_metrics = compute_metrics(y_test, y_test_pred)

    return {
        'name': name,
        'n_snps': int(n_features),
        'valid_sample_count': int(X.shape[0]),
        'train_sample_count': int(split_data['X_train'].shape[0]),
        'test_sample_count': int(y_test.shape[0]),
        'split_strategy': split_data['split_strategy'],
        'sample_ordering': sample_ordering,
        'val_size': int(split_data['val_size']),
        'val_sample_id_first': split_data['sample_ids_val'][0],
        'val_sample_id_last': split_data['sample_ids_val'][-1],
        'sample_ids_val': split_data['sample_ids_val'].tolist(),
        'test_sample_id_first': str(sample_ids_test[0]) if len(sample_ids_test) else None,
        'test_sample_id_last': str(sample_ids_test[-1]) if len(sample_ids_test) else None,
        'sample_ids_test': np.asarray(sample_ids_test, dtype=str).tolist(),
        'genotype_normalization': genotype_normalization,
        'metrics': validation_metrics,
        'test_metrics': test_metrics,
        'gblup': gblup_summary,
        'lambda_summary': {
            'lambda_estimated': lambda_value,
            'alpha_used': lambda_value,
            'selection_method': 'reml_gblup',
            'fallback_triggered': False,
        },
        'ridge': {
            'lambda_estimated': lambda_value,
            'alpha_used': lambda_value,
            'dual_coef_norm': float(np.linalg.norm(dual_coef)),
            'coef_norm': float(np.linalg.norm(primal_coef)),
        },
        'predictions': {
            'validation': {
                'y_true': y_val.tolist(),
                'y_pred': y_val_pred.tolist(),
            },
            'test': {
                'y_true': y_test.tolist(),
                'y_pred': y_test_pred.tolist(),
            },
        },
    }


def plot_metric_comparison(results_by_name, output_path):
    def annotate_bars(axis, bars, values):
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                label = f'{value:.4f}'
            else:
                label = 'nan'

            if value >= 0:
                offset = 3
                va = 'bottom'
            else:
                offset = -3
                va = 'top'

            axis.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, offset),
                textcoords='offset points',
                ha='center',
                va=va,
                fontsize=9,
            )

    labels = [
        results_by_name['annotated_22k']['name'],
        results_by_name['all_36k']['name'],
    ]
    mae_values = [
        results_by_name['annotated_22k']['metrics']['mae'],
        results_by_name['all_36k']['metrics']['mae'],
    ]
    corr_values = [
        results_by_name['annotated_22k']['metrics']['pearson'],
        results_by_name['all_36k']['metrics']['pearson'],
    ]
    test_mae_values = [
        results_by_name['annotated_22k']['test_metrics']['mae'],
        results_by_name['all_36k']['test_metrics']['mae'],
    ]
    test_corr_values = [
        results_by_name['annotated_22k']['test_metrics']['pearson'],
        results_by_name['all_36k']['test_metrics']['pearson'],
    ]

    colors = ['#bc6c25', '#457b9d']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    val_mae_bars = axes[0, 0].bar(labels, mae_values, color=colors, edgecolor='none')
    axes[0, 0].set_title('Validation MAE')
    axes[0, 0].set_ylabel('MAE')
    annotate_bars(axes[0, 0], val_mae_bars, mae_values)

    val_corr_bars = axes[0, 1].bar(labels, corr_values, color=colors, edgecolor='none')
    axes[0, 1].set_title('Validation Pearson Correlation')
    axes[0, 1].set_ylabel('Pearson r')
    axes[0, 1].set_ylim(
        min(0.0, np.nanmin(corr_values) - 0.05),
        max(1.0, np.nanmax(corr_values) + 0.05),
    )
    annotate_bars(axes[0, 1], val_corr_bars, corr_values)

    test_mae_bars = axes[1, 0].bar(labels, test_mae_values, color=colors, edgecolor='none')
    axes[1, 0].set_title('Test MAE')
    axes[1, 0].set_ylabel('MAE')
    annotate_bars(axes[1, 0], test_mae_bars, test_mae_values)

    test_corr_bars = axes[1, 1].bar(labels, test_corr_values, color=colors, edgecolor='none')
    axes[1, 1].set_title('Test Pearson Correlation')
    axes[1, 1].set_ylabel('Pearson r')
    axes[1, 1].set_ylim(
        min(0.0, np.nanmin(test_corr_values) - 0.05),
        max(1.0, np.nanmax(test_corr_values) + 0.05),
    )
    annotate_bars(axes[1, 1], test_corr_bars, test_corr_values)

    for axis in axes.flat:
        axis.tick_params(axis='x', rotation=10)

    fig.suptitle('GBLUP comparison with REML-estimated lambda on validation and test')
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Genotype normalization: {args.genotype_normalization}')
    print(genotype_normalization_description(args.genotype_normalization))
    print(
        'Genotype normalization formula: '
        f'{genotype_normalization_formula(args.genotype_normalization)}'
    )

    annotation_df, _ = load_annotation_dataframe(args.annotation_path)
    resources = build_annotation_resources(annotation_df)
    annotated_snp_ids = resources['snp_map']['snp_id'].to_numpy()

    full_dataset = load_trait_dataset(
        genotype_prefix=args.genotype_prefix,
        phenotype_path=args.phenotype_path,
        trait=args.trait,
        snp_ids=None,
        pedigree_path=args.pedigree_path,
    )
    full_test_dataset = load_trait_test_dataset(
        genotype_prefix=args.genotype_prefix,
        masked_phenotype_path=args.phenotype_path,
        full_phenotype_path=args.full_phenotype_path,
        trait=args.trait,
        snp_ids=None,
        pedigree_path=args.pedigree_path,
    )
    annotated_dataset = load_trait_dataset(
        genotype_prefix=args.genotype_prefix,
        phenotype_path=args.phenotype_path,
        trait=args.trait,
        snp_ids=annotated_snp_ids,
        pedigree_path=args.pedigree_path,
    )
    annotated_test_dataset = load_trait_test_dataset(
        genotype_prefix=args.genotype_prefix,
        masked_phenotype_path=args.phenotype_path,
        full_phenotype_path=args.full_phenotype_path,
        trait=args.trait,
        snp_ids=annotated_snp_ids,
        pedigree_path=args.pedigree_path,
    )

    if not np.array_equal(
        full_dataset['sample_ids_valid'],
        annotated_dataset['sample_ids_valid'],
    ):
        raise ValueError(
            'The full-SNP and annotated-SNP datasets do not share the same valid '
            'sample order, so the deterministic split would not match.'
        )

    if not np.array_equal(
        full_test_dataset['sample_ids_test'],
        annotated_test_dataset['sample_ids_test'],
    ):
        raise ValueError(
            'The full-SNP and annotated-SNP test datasets do not share the same '
            'test sample order, so the baseline comparison would not match.'
        )

    results_22k = prepare_feature_set_result(
        name='Annotated 22k SNPs',
        X=annotated_dataset['X'],
        y=annotated_dataset['y'],
        sample_ids=annotated_dataset['sample_ids_valid'],
        X_test=annotated_test_dataset['X'],
        y_test=annotated_test_dataset['y'],
        sample_ids_test=annotated_test_dataset['sample_ids_test'],
        sample_ordering=annotated_dataset['sample_ordering'],
        val_size=args.val_size,
        genotype_normalization=args.genotype_normalization,
    )

    results_36k = prepare_feature_set_result(
        name='All 36k SNPs',
        X=full_dataset['X'],
        y=full_dataset['y'],
        sample_ids=full_dataset['sample_ids_valid'],
        X_test=full_test_dataset['X'],
        y_test=full_test_dataset['y'],
        sample_ids_test=full_test_dataset['sample_ids_test'],
        sample_ordering=full_dataset['sample_ordering'],
        val_size=args.val_size,
        genotype_normalization=args.genotype_normalization,
    )

    if results_22k['sample_ids_val'] != results_36k['sample_ids_val']:
        raise ValueError(
            'Validation sample IDs differ between the 22k and 36k runs.'
        )

    comparison = {
        'trait': int(args.trait),
        'annotation_path': args.annotation_path,
        'genotype_prefix': args.genotype_prefix,
        'phenotype_path': args.phenotype_path,
        'full_phenotype_path': args.full_phenotype_path,
        'pedigree_path': args.pedigree_path,
        'genotype_normalization': args.genotype_normalization,
        'genotype_normalization_formula': genotype_normalization_formula(
            args.genotype_normalization
        ),
        'split_strategy': 'last_n',
        'sample_ordering': full_dataset['sample_ordering'],
        'val_size': int(args.val_size),
        'test_split_source': 'full_phenotype_excluding_masked_train_validation_ids',
        'lambda_estimation_method': 'reml_gblup',
        'results': {
            'annotated_22k': results_22k,
            'all_36k': results_36k,
        },
    }

    normalization_tag = genotype_normalization_tag(args.genotype_normalization)
    json_path = os.path.join(
        args.output_dir,
        f'gblup_comparison_trait{args.trait}_{normalization_tag}.json',
    )
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(comparison, handle, indent=2)

    plot_path = os.path.join(
        args.output_dir,
        f'gblup_comparison_trait{args.trait}_{normalization_tag}.png',
    )
    plot_metric_comparison(comparison['results'], plot_path)

    print('===== GBLUP COMPARISON RESULTS =====')
    for key in ['annotated_22k', 'all_36k']:
        result = comparison['results'][key]
        val_metrics = result['metrics']
        test_metrics = result['test_metrics']
        print(
            f"{result['name']} | lambda={result['lambda_summary']['lambda_estimated']:.6f}"
        )
        print(
            '  Validation: '
            f"MAE={val_metrics['mae']:.6f} "
            f"MSE={val_metrics['mse']:.6f} "
            f"RMSE={val_metrics['rmse']:.6f} "
            f"R2={val_metrics['r2']:.6f} "
            f"Pearson={val_metrics['pearson']:.6f}"
        )
        print(
            '  Test: '
            f"MAE={test_metrics['mae']:.6f} "
            f"MSE={test_metrics['mse']:.6f} "
            f"RMSE={test_metrics['rmse']:.6f} "
            f"R2={test_metrics['r2']:.6f} "
            f"Pearson={test_metrics['pearson']:.6f}"
        )
    print(f'JSON saved to: {json_path}')
    print(f'Plot saved to: {plot_path}')


if __name__ == '__main__':
    main()
