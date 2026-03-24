import argparse
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pandas_plink import read_plink
from scipy.sparse import coo_matrix
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    from model import (
        EarlyStopping,
        build_biological_model,
        load_model_state_dict_compatibly,
    )
except ImportError:
    from bio_informed_DNN.model import (
        EarlyStopping,
        build_biological_model,
        load_model_state_dict_compatibly,
    )


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


DEFAULT_ANNOTATION_PATH = './data/preprocessed/vep_ingenes_newimpact.csv'
DEFAULT_GENOTYPE_PREFIX = '../data/ML/BBB2023_MD'
DEFAULT_PHENOTYPE_PATH = '../data/ML/pheno_2023bbb_0twins_6traits_mask'
MODEL_TYPE = 'impact_attention_mlp'
MODEL_VARIANT = 'impact_attention'


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Train an impact-aware biologically informed model with a '
            'SNP->gene mask built directly from a VEP annotation table.'
        )
    )

    parser.add_argument(
        '-t',
        '--trait',
        type=int,
        default=2,
        help='Trait index to train on: 2.shoulder 3.top 4.buttock(side) '
        '5.buttock(rear) 6.size 7.musculature',
    )
    parser.add_argument(
        '-l',
        '--layers',
        type=int,
        nargs='+',
        default=[50],
        help='Hidden layer sizes after the gene layer (default: 50).',
    )
    parser.add_argument(
        '--annotation-path',
        type=str,
        default=DEFAULT_ANNOTATION_PATH,
        help='CSV containing snp_id, Gene, SYMBOL and new_impact columns.',
    )
    parser.add_argument(
        '--genotype-prefix',
        type=str,
        default=DEFAULT_GENOTYPE_PREFIX,
        help='PLINK prefix used by pandas_plink (default: ../data/ML/BBB2023_MD).',
    )
    parser.add_argument(
        '--phenotype-path',
        type=str,
        default=DEFAULT_PHENOTYPE_PATH,
        help='Phenotype file used by the current training scripts.',
    )
    parser.add_argument(
        '--impact-embedding-dim',
        type=int,
        default=16,
        help='Embedding width used for new_impact-conditioned attention.',
    )
    parser.add_argument(
        '--attention-dropout',
        type=float,
        default=0.05,
        help='Dropout applied after the SNP impact attention reweighting.',
    )
    parser.add_argument(
        '-lr',
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Learning rate to use on Adam optimizer (default: 1e-4).',
    )
    parser.add_argument(
        '-e',
        '--epochs',
        type=int,
        default=150,
        help='Number of training epochs (default: 150).',
    )
    parser.add_argument(
        '-c',
        '--criterion',
        type=str,
        default='HuberLoss',
        help='Choose the loss for training: MSE, MAE, HuberLoss.',
    )
    parser.add_argument(
        '-a',
        '--activation',
        type=str,
        default='gelu',
        help='Choose the activation function: relu, sigmoid, gelu.',
    )
    parser.add_argument(
        '--save-model',
        type=str,
        default='true',
        help='If true, saves a compact model checkpoint with metadata.',
    )
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--early-stop', type=str, default='true')
    parser.add_argument(
        '--grad-clip',
        type=float,
        default=1.0,
        help='Max gradient norm. Set to 0 to disable clipping (default: 1.0).',
    )
    parser.add_argument('--num-workers', type=int, default=4)

    return parser.parse_args()


def flag_is_true(value):
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y'}


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_criterion(name):
    if name == 'MSE':
        return nn.MSELoss()
    if name == 'MAE':
        return nn.L1Loss()
    if name == 'HuberLoss':
        return nn.HuberLoss()
    raise ValueError(f'Unsupported criterion: {name}')


def build_args_string(args, annotation_tag):
    layers_str = '-'.join(str(layer) for layer in args.layers)
    return (
        f'{annotation_tag}_{MODEL_VARIANT}'
        f'_layers{layers_str}'
        f'_emb{args.impact_embedding_dim}'
        f'_attndrop{args.attention_dropout}'
        f'_lr{args.learning_rate}'
        f'_trait{args.trait}'
        f'_epoch{args.epochs}'
        f'_crit{args.criterion}'
        f'_act{args.activation}'
        f'_batch{args.batch_size}'
        f'_wdecay{args.weight_decay}'
        f'_dropout{args.dropout}'
        f'_gclip{args.grad_clip}'
        f'_seed{args.seed}'
        f'_ea{flag_is_true(args.early_stop)}'
    )


def compact_model_state_dict(model):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if not name.endswith('.mask') and not name.endswith('impact_indices')
    }


def safe_torch_save(checkpoint, model_path):
    tmp_path = f'{model_path}.tmp'
    try:
        torch.save(checkpoint, tmp_path, _use_new_zipfile_serialization=False)
        os.replace(tmp_path, model_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_annotation_dataframe(annotation_path):
    annotation_df = pd.read_csv(annotation_path)

    required_columns = {'snp_id', 'Gene', 'new_impact'}
    missing_columns = sorted(required_columns - set(annotation_df.columns))
    if missing_columns:
        raise ValueError(
            'Annotation file is missing required columns: '
            + ', '.join(missing_columns)
        )

    annotation_df = annotation_df.copy()
    annotation_df['snp_id'] = annotation_df['snp_id'].fillna('').astype(str).str.strip()
    annotation_df['Gene'] = annotation_df['Gene'].fillna('').astype(str).str.strip()
    annotation_df['new_impact'] = (
        annotation_df['new_impact'].fillna('').astype(str).str.strip()
    )

    if 'SYMBOL' in annotation_df.columns:
        annotation_df['SYMBOL'] = (
            annotation_df['SYMBOL'].fillna('').astype(str).str.strip()
        )
    else:
        annotation_df['SYMBOL'] = ''

    initial_rows = len(annotation_df)
    annotation_df = annotation_df.loc[
        (annotation_df['snp_id'] != '')
        & (annotation_df['Gene'] != '')
        & (annotation_df['new_impact'] != '')
    ].copy()

    dropped_rows = initial_rows - len(annotation_df)
    return annotation_df, dropped_rows


def build_annotation_resources(annotation_df):
    snp_order = pd.Index(pd.unique(annotation_df['snp_id']), name='snp_id')
    gene_order = pd.Index(pd.unique(annotation_df['Gene']), name='Gene')
    impact_order = pd.Index(
        sorted(annotation_df['new_impact'].unique()),
        name='new_impact',
    )

    snp_index = pd.Series(np.arange(len(snp_order)), index=snp_order)
    gene_index = pd.Series(np.arange(len(gene_order)), index=gene_order)
    impact_index = pd.Series(np.arange(len(impact_order)), index=impact_order)

    row_indices = annotation_df['Gene'].map(gene_index).to_numpy()
    col_indices = annotation_df['snp_id'].map(snp_index).to_numpy()

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
            'snp_index': np.arange(len(snp_order)),
            'snp_id': snp_order,
        }
    )

    gene_symbol_lookup = (
        annotation_df.loc[annotation_df['SYMBOL'] != '', ['Gene', 'SYMBOL']]
        .drop_duplicates(subset=['Gene'])
        .set_index('Gene')['SYMBOL']
    )

    gene_map = pd.DataFrame(
        {
            'gene_index': np.arange(len(gene_order)),
            'ensembl_gene_id': gene_order,
        }
    )
    gene_map['gene_name'] = (
        gene_map['ensembl_gene_id'].map(gene_symbol_lookup).fillna('')
    )

    annotation_edge_columns = [
        column
        for column in [
            'snp_id',
            'Gene',
            'SYMBOL',
            'new_impact',
            'Consequence',
            'IMPACT',
            'bp',
            'cm',
            'allele1',
            'allele2',
        ]
        if column in annotation_df.columns
    ]
    annotation_edges = annotation_df[annotation_edge_columns].drop_duplicates().copy()

    snp_annotation = (
        annotation_df.drop_duplicates(subset=['snp_id'], keep='first')
        [[column for column in annotation_edge_columns if column != 'Gene']]
        .set_index('snp_id')
        .reindex(snp_order)
        .reset_index()
    )

    impact_indices = torch.tensor(
        snp_annotation['new_impact'].map(impact_index).to_numpy(),
        dtype=torch.long,
    )

    impact_counts = (
        snp_annotation['new_impact']
        .value_counts()
        .reindex(impact_order, fill_value=0)
    )
    impact_map = pd.DataFrame(
        {
            'impact_index': np.arange(len(impact_order)),
            'new_impact': impact_order,
            'n_snps': impact_counts.to_numpy(),
        }
    )

    return {
        'mask_snp_gene': mask_snp_gene,
        'mask_tensor': torch.from_numpy(mask_snp_gene.toarray()).float(),
        'annotation_edges': annotation_edges,
        'snp_map': snp_map,
        'gene_map': gene_map,
        'impact_map': impact_map,
        'snp_annotation': snp_annotation,
        'impact_indices': impact_indices,
        'n_snps': len(snp_order),
        'n_genes': len(gene_order),
        'n_impacts': len(impact_order),
    }


def main():
    args = parse_args()
    set_random_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    annotation_tag = os.path.splitext(os.path.basename(args.annotation_path))[0]
    args_str = build_args_string(args, annotation_tag)

    path_crit = f'crit{args.criterion}_{args.activation}'
    path_results = './results/test_hyp'
    path_to_save = os.path.join(annotation_tag, path_crit, MODEL_VARIANT)
    path_models = os.path.join('./saved_models', path_to_save)
    path_logs = os.path.join('./logs_models', path_to_save)
    path_figs = os.path.join(path_results, path_to_save)

    os.makedirs(path_models, exist_ok=True)
    os.makedirs(path_logs, exist_ok=True)
    os.makedirs(path_figs, exist_ok=True)

    log_name = os.path.join(path_logs, f'training_{args_str}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[logging.FileHandler(log_name), logging.StreamHandler()],
    )

    logging.info(f'Using device: {device}')
    logging.info('\n' * 3 + '======== STARTED TRAINING ========' + '\n' * 3)
    logging.info(
        'Parameters:\n'
        f' annotation_path: {args.annotation_path}\n'
        f' genotype_prefix: {args.genotype_prefix}\n'
        f' phenotype_path: {args.phenotype_path}\n'
        f' impact_embedding_dim: {args.impact_embedding_dim}\n'
        f' attention_dropout: {args.attention_dropout}\n'
        f' learning_rate: {args.learning_rate}\n'
        f' trait: {args.trait}\n'
        f' epochs: {args.epochs}\n'
        f' criterion: {args.criterion}\n'
        f' activation: {args.activation}\n'
        f' layers: {args.layers}\n'
        f' batch: {args.batch_size}\n'
        f' weight_decay: {args.weight_decay}\n'
        f' dropout: {args.dropout}\n'
        f' grad_clip: {args.grad_clip}\n'
        f' seed: {args.seed}\n'
        f' early_stop: {flag_is_true(args.early_stop)}'
    )

    annotation_df, dropped_rows = load_annotation_dataframe(args.annotation_path)
    duplicate_snp_count = int(annotation_df['snp_id'].duplicated().sum())
    conflicting_impacts = int(
        (annotation_df.groupby('snp_id')['new_impact'].nunique() > 1).sum()
    )

    resources = build_annotation_resources(annotation_df)
    mask_snp_gene = resources['mask_snp_gene']
    mask1 = resources['mask_tensor']
    annotation_edges = resources['annotation_edges']
    snp_map = resources['snp_map']
    gene_map = resources['gene_map']
    impact_map = resources['impact_map']
    snp_annotation = resources['snp_annotation']
    impact_indices = resources['impact_indices']
    n_snps = resources['n_snps']
    n_genes = resources['n_genes']
    n_impacts = resources['n_impacts']

    logging.info(
        f'Annotation rows kept: {len(annotation_df)} '
        f'(dropped {dropped_rows} rows with missing SNP/gene/new_impact).'
    )
    logging.info(f'Unique SNPs from annotation: {n_snps}')
    logging.info(f'Unique genes from annotation: {n_genes}')
    logging.info(f'Unique impact classes: {n_impacts}')
    logging.info(f'Duplicate SNP annotations: {duplicate_snp_count}')
    logging.info(f'SNPs with conflicting impact labels: {conflicting_impacts}')
    logging.info(mask_snp_gene.shape)
    logging.info(mask_snp_gene.nnz)
    logging.info(
        'Mask density: '
        f'{mask_snp_gene.nnz / (mask_snp_gene.shape[0] * mask_snp_gene.shape[1]):.8f}'
    )

    for impact_row in impact_map.itertuples(index=False):
        logging.info(
            f'Impact class {impact_row.new_impact}: {impact_row.n_snps} SNPs'
        )

    bim, _, bed = read_plink(args.genotype_prefix)
    bim = bim.set_index('snp')

    missing_snps = snp_map.loc[~snp_map['snp_id'].isin(bim.index), 'snp_id']
    if not missing_snps.empty:
        raise ValueError(
            'Some annotation SNPs are missing from the PLINK BIM file. '
            f'First missing SNPs: {missing_snps.iloc[:10].tolist()}'
        )

    snp_to_index = bim.loc[snp_map['snp_id'], 'i']
    X = bed[snp_to_index]

    pheno_df = pd.read_csv(
        args.phenotype_path,
        delimiter='\t',
        header=None,
    )
    pheno_df = pheno_df.drop(columns=[0]).set_index(1)

    valid_mask = ~pheno_df[args.trait].isna()
    valid_mask_array = valid_mask.to_numpy()
    sample_ids_valid = pheno_df.index[valid_mask].astype(str).to_numpy()
    y = pheno_df.loc[valid_mask, args.trait].to_numpy()
    X = X.T[valid_mask_array].compute()

    split_random_state = 42
    val_size = 1000
    X_train, X_val, y_train, y_val, _, sample_ids_val = train_test_split(
        X,
        y,
        sample_ids_valid,
        test_size=val_size,
        random_state=split_random_state,
    )

    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_val = scaler_y.transform(y_val.reshape(-1, 1))

    logging.info(
        f'Using fixed validation holdout of {val_size} samples with random_state='
        f'{split_random_state}'
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    model = build_biological_model(
        model_type=MODEL_TYPE,
        input_dim=n_snps,
        hidden1=n_genes,
        hidden2=0,
        fc_layers=args.layers,
        mask1=mask1,
        mask2=None,
        activation=args.activation,
        dropout=args.dropout,
        impact_indices=impact_indices,
        num_impacts=n_impacts,
        impact_embedding_dim=args.impact_embedding_dim,
        attention_dropout=args.attention_dropout,
    ).to(device)

    criterion = build_criterion(args.criterion)
    logging.info(f'Setup of Loss confirmed as {args.criterion}')

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=4,
        factor=0.5,
        threshold=min(1e-6, (args.learning_rate / 1000)),
    )

    use_early_stop = flag_is_true(args.early_stop)
    early_stopper = (
        EarlyStopping(patience=9, min_delta=1e-5) if use_early_stop else None
    )

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_train_loss = None
    best_epoch = -1
    best_model_state = None

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    for epoch in range(args.epochs):
        model.train()

        epoch_train_loss = 0.0
        total_norm = 0.0

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

            batch_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    batch_norm += param.grad.data.norm(2).item()
            total_norm += batch_norm

            optimizer.step()
            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)
        total_norm /= len(train_loader)
        train_losses.append(epoch_train_loss)

        logging.info(
            f'Epoch {epoch} | '
            f'Train Loss: {epoch_train_loss:.6f} | '
            f'Avg Grad Norm: {total_norm:.6f}'
        )

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                outputs = model(xb)
                val_loss += criterion(outputs, yb).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        logging.info(
            f'Epoch {epoch} | '
            f'Val Loss: {val_loss:.6f} | '
            f'LR: {optimizer.param_groups[0]["lr"]:.6e}'
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = epoch_train_loss
            best_epoch = epoch
            best_model_state = compact_model_state_dict(model)

        scheduler.step(val_loss)

        if use_early_stop:
            early_stopper(val_loss, epoch_train_loss)
            if early_stopper.early_stop:
                logging.info('Early stopping triggered')
                break

    if best_model_state is not None:
        load_model_state_dict_compatibly(model, best_model_state)
        logging.info(
            'Restored best model from epoch '
            f'{best_epoch} with train loss {best_train_loss:.6f} '
            f'and val loss {best_val_loss:.6f}'
        )

    logging.info('\n' * 3 + '======== FINISHED TRAINING ========')

    plt.figure()
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.xlabel('Epoch')
    if args.criterion == 'MSE':
        plt.ylabel('MSE Loss')
        plt.ylim([0, 2])
    elif args.criterion == 'MAE':
        plt.ylabel('MAE Loss')
    elif args.criterion == 'HuberLoss':
        plt.ylabel('Huber Loss')
    plt.title('Training and val Loss')
    plt.savefig(os.path.join(path_figs, f'val_architecture_{args_str}.png'))
    plt.close()

    logging.info('\n======== PLOTTED IMAGE ========')

    model.eval()
    probe_x = X_val[: min(32, X_val.shape[0])].to(device)

    prediction_batches = []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device, non_blocking=True)
            prediction_batches.append(model(xb).cpu())

    y_pred_scaled = torch.cat(prediction_batches, dim=0).numpy()
    y_val_scaled = y_val.cpu().numpy()
    y_pred_real = scaler_y.inverse_transform(y_pred_scaled)
    y_val_real = scaler_y.inverse_transform(y_val_scaled)

    mae = mean_absolute_error(y_val_real, y_pred_real)
    mse = mean_squared_error(y_val_real, y_pred_real)
    r2 = r2_score(y_val_real, y_pred_real)
    pearson_corr, _ = pearsonr(y_val_real.flatten(), y_pred_real.flatten())

    logging.info('\n======== FINAL METRICS (REAL SCALE) ========')
    logging.info(f'MAE: {mae:.6f}')
    logging.info(f'MSE: {mse:.6f}')
    logging.info(f'R2: {r2:.6f}')
    logging.info(f'Pearson Correlation: {pearson_corr:.6f}')

    with torch.no_grad():
        attended_x = model.encode_snps(probe_x)
        x1 = model.masked1(attended_x)
        x1_norm = model.norm1(x1)

    mean_ln = x1_norm.mean().item()
    std_ln = x1_norm.std().item()

    logging.info('\n======== LAYERNORM CHECK ========')
    logging.info(
        'Outputs should stay roughly centered; std can drift after LayerNorm '
        'learns per-feature scale and bias.'
    )
    logging.info(f'LayerNorm output mean: {mean_ln:.6f}')
    logging.info(f'LayerNorm output std: {std_ln:.6f}')

    with torch.no_grad():
        attention_weights = model.get_attention_weights(probe_x)

    logging.info('\n======== IMPACT ATTENTION CHECK ========')
    logging.info(f'Attention weight mean: {attention_weights.mean().item():.6f}')
    logging.info(f'Attention weight std: {attention_weights.std().item():.6f}')
    logging.info(f'Attention weight min: {attention_weights.min().item():.6f}')
    logging.info(f'Attention weight max: {attention_weights.max().item():.6f}')

    impact_indices_device = model.snp_attention.impact_indices
    for impact_row in impact_map.itertuples(index=False):
        impact_mask = impact_indices_device == impact_row.impact_index
        if impact_mask.any().item():
            impact_mean = attention_weights[:, impact_mask].mean().item()
            impact_std = attention_weights[:, impact_mask].std().item()
            logging.info(
                f'{impact_row.new_impact}: '
                f'mean_attention={impact_mean:.6f} '
                f'std_attention={impact_std:.6f} '
                f'n_snps={impact_row.n_snps}'
            )

    with torch.no_grad():
        a1 = model.masked1(attended_x)
        gene_features = model.activation_fn(model.norm1(a1))

    logging.info('\n======== ACTIVATION VARIANCE CHECK ========')
    logging.info(
        'If variance drops drastically then initialization may be too weak; '
        'if it explodes, propagation may be unstable.'
    )
    logging.info(f'Attended SNP variance: {attended_x.var().item():.6f}')
    logging.info(f'Masked1 variance: {a1.var().item():.6f}')
    logging.info(f'Gene feature variance: {gene_features.var().item():.6f}')

    with torch.no_grad():
        effective_weights = model.masked1.weight * model.masked1.mask
        violations = (effective_weights[model.masked1.mask == 0] != 0).sum().item()

    logging.info('\n======== MASK CHECK ========')
    logging.info(f'masked1 zero-mask weight violations: {violations}')

    logging.info('\n======== FORWARD SANITY CHECK ========')

    with torch.no_grad():
        test_attention = model.get_attention_weights(probe_x[:1])
        test_out = model.masked1(model.encode_snps(probe_x[:1]))

        if torch.isnan(test_out).any():
            logging.info('NaNs detected in masked1 output!')
        else:
            logging.info('No NaNs in masked1 output.')

        logging.info(f'attention output shape: {test_attention.shape}')
        logging.info(f'attention output mean: {test_attention.mean().item():.6f}')
        logging.info(f'attention output std: {test_attention.std().item():.6f}')
        logging.info(f'masked1 output shape: {test_out.shape}')
        logging.info(f'masked1 output mean: {test_out.mean().item():.6f}')
        logging.info(f'masked1 output std: {test_out.std().item():.6f}')

    model.zero_grad(set_to_none=True)
    diagnostic_x, diagnostic_y = next(iter(val_loader))
    diagnostic_x = diagnostic_x.to(device, non_blocking=True)
    diagnostic_y = diagnostic_y.to(device, non_blocking=True)
    diagnostic_loss = criterion(model(diagnostic_x), diagnostic_y)
    diagnostic_loss.backward()

    logging.info('\n======== FINAL GRADIENT CHECK ========')
    logging.info(
        f'Diagnostic loss on restored best model: {diagnostic_loss.item():.6f}'
    )
    for name, param in model.named_parameters():
        if param.grad is not None:
            logging.info(f'{name} grad norm: {param.grad.norm().item():.6f}')

    plt.figure()
    plt.scatter(y_val_real, y_pred_real, alpha=0.5)
    plt.xlabel('True Phenotype')
    plt.ylabel('Predicted Phenotype')
    plt.title('Prediction vs True')
    plt.savefig(os.path.join(path_figs, f'pred_vs_true_{args_str}.png'))
    plt.close()

    all_grads = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        grad = param.grad.detach()

        if 'masked1.weight' in name:
            grad = grad[model.masked1.mask == 1]
        else:
            grad = grad.view(-1)

        grad = grad[grad != 0]
        if grad.numel() > 0:
            all_grads.append(grad.view(-1))

    if all_grads:
        all_grads = torch.cat(all_grads).cpu().numpy()
        lower = np.percentile(all_grads, 1)
        upper = np.percentile(all_grads, 99)

        plt.figure()
        plt.hist(all_grads, bins=100, range=(lower, upper))
        plt.xlim(lower, upper)
        plt.title('Gradient Distribution (Non-masked weights)')
        plt.xlabel('Gradient value')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(path_figs, f'grad_distribution_{args_str}.png'))
        plt.close()

    model_path = None
    if flag_is_true(args.save_model):
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        model_path = os.path.join(path_models, f'model_{args_str}_{timestamp}.pt')

        checkpoint = {
            'checkpoint_format': 'impact_attention_compact_v1',
            'model_state_dict': compact_model_state_dict(model),
            'architecture': {
                'model_type': MODEL_TYPE,
                'input_dim': n_snps,
                'hidden1': n_genes,
                'hidden2': 0,
                'fc_layers': args.layers,
                'activation': args.activation,
                'dropout': args.dropout,
                'impact_embedding_dim': args.impact_embedding_dim,
                'attention_dropout': args.attention_dropout,
            },
            'training_args': {
                **vars(args),
                'annotation_tag': annotation_tag,
                'model_type': MODEL_TYPE,
                'split_random_state': split_random_state,
                'val_size': val_size,
            },
            'scaler_X_mean': np.asarray(scaler_X.mean_, dtype=np.float32),
            'scaler_X_scale': np.asarray(scaler_X.scale_, dtype=np.float32),
            'scaler_y_mean': np.asarray(scaler_y.mean_, dtype=np.float32),
            'scaler_y_scale': np.asarray(scaler_y.scale_, dtype=np.float32),
            'snp_annotation_dataframe': annotation_edges,
            'snp_metadata_dataframe': snp_annotation,
            'gene_map_dataframe': gene_map,
            'snp_map_dataframe': snp_map,
            'impact_map_dataframe': impact_map,
            'snp_names': snp_map['snp_id'].to_numpy(),
            'gene_names': gene_map['ensembl_gene_id'].to_numpy(),
            'impact_names': impact_map['new_impact'].to_numpy(),
            'mask_snp_gene': None,
            'mask_snp_gene_shape': tuple(int(dim) for dim in mask1.shape),
            'mask_snp_gene_nnz': int(mask_snp_gene.nnz),
            'mask_gene_pathway': None,
            'impact_indices': impact_indices.cpu().numpy().astype(np.int64),
            'sample_ids_valid': sample_ids_valid,
            'sample_ids_val': sample_ids_val,
            'n_snps': n_snps,
            'n_genes': n_genes,
            'n_impacts': n_impacts,
            'final_metrics': {
                'MAE': float(mae),
                'MSE': float(mse),
                'R2': float(r2),
                'Pearson': float(pearson_corr),
                'best_epoch': int(best_epoch),
                'best_train_loss': None if best_train_loss is None else float(best_train_loss),
                'best_val_loss': float(best_val_loss),
            },
        }

        logging.info(
            '\nSaving compact checkpoint without raw train/val tensors or dense masks.'
        )
        safe_torch_save(checkpoint, model_path)
        logging.info(f'Saved compact model checkpoint at: {model_path}')
    else:
        logging.info('\nModel checkpoint saving skipped.')


if __name__ == '__main__':
    main()
