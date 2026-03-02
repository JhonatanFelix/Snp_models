import argparse
import logging

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pandas_plink import read_plink
from scipy.sparse import load_npz
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ===============================
# Arguments
# ===============================

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a biologically informed model with configurable parameters"
    )

    parser.add_argument(
        "-t",
        "--trait",
        type=int,
        required=False,
        default = 2,
        help="Trait index to train on: 2.shoulder  3.top 4.buttock(side) \
            5.buttock(rear) 6.size 7.musculature",
    )
    parser.add_argument(
        "-l",
        "--layers",
        type=int,
        nargs="+",  # this means one or more integers
        required=False,
        default=[512, 256],
        help="Hidden layer sizes (e.g. 1024 500 200 100) (default:512 256)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        required=False,
        default=1e-3,
        help="Learning rate to use on Adam algorithm (default: 1e-3)"
    )
    parser.add_argument(
        "-e", 
        "--epochs",
        type=int,
        required=False,
        default= 150,
        help="Number of epochs to the training (default: 150)"
    )
    parser.add_argument(
        '-c',
        '--criterion',
        type=str,
        required=False,
        default='MSE',
        help="Choose the losses for your training: MSE, MAE, HuberLoss"
        )
    parser.add_argument(
        '-a',
        '--activation',
        type=str,
        required=False,
        default = "gelu",
        help='Choose the activation function for training: relu, sigmoid, gelu \
            (default: gelu)'
    )

    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
        
    return parser.parse_args()

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
    def __init__(self, input_dim, hidden1, hidden2,
                 fc_layers, mask1, mask2,
                 activation="relu", use_layernorm=True, dropout=0.0):

        super().__init__()

        if activation == "relu":
            self.activation_fn = nn.ReLU()
        elif activation == "sigmoid":
            self.activation_fn = nn.Sigmoid()
        elif activation == "gelu":
            self.activation_fn = nn.GELU()
        else:
            raise ValueError("Unsupported activation")

        self.masked1 = MaskedLinear(input_dim, hidden1, mask1, activation)
        self.masked2 = MaskedLinear(hidden1, hidden2, mask2, activation)

        self.norm1 = nn.LayerNorm(hidden1) if use_layernorm else nn.Identity()
        self.norm2 = nn.LayerNorm(hidden2) if use_layernorm else nn.Identity()

        layers = []
        prev_dim = hidden2

        for dim in fc_layers:
            linear = nn.Linear(prev_dim, dim)
            if activation == "relu":
                nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            elif activation in ['gelu','sigmoid']:
                nn.init.kaiming_normal_(linear.weight, nonlinearity="linear")
            
            nn.init.zeros_(linear.bias)
            
            layers.append(linear)
            layers.append(nn.LayerNorm(dim) if use_layernorm else nn.Identity())
            layers.append(self.activation_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = dim

        final = nn.Linear(prev_dim, 1)
        nn.init.xavier_normal_(final.weight)

        layers.append(final)

        self.fc_stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.masked1(x)
        x = self.norm1(x)
        x = self.activation_fn(x)

        x = self.masked2(x)
        x = self.norm2(x)
        x = self.activation_fn(x)

        x = self.fc_stack(x)
        return x

# ===============================
# Early Stopping epochs
# ===============================

class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True

# ===============================
# Logger
# ===============================

# def setup_logger(log_filename):
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s | %(levelname)s | %(message)s",
#         handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
#     )


# ===============================
# Main
# ===============================

def main():

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    print(logging.getLogger().handlers)

    args_str = (
        f"{args.layers}_lr{args.learning_rate}_trait{args.trait}"
        f"_epoch{args.epochs}_crit{args.criterion}_act{args.activation}"
        f"_batch{args.batch_size}_wdecay{args.weight_decay}_dropout{args.dropout}"
        f"_seed{args.seed}"
    )
    log_name = f'./logs_models/training_{args_str}.log'

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_name),
            logging.StreamHandler()
        ]   
    )

    logging.info('\n'*3+"======== STARTED TRAINING ========"+'\n'*3)
    logging.info(f"Parameters: \n \
                 Layers:{args.layers} \n Learning rate = {args.learning_rate}\
                \n trait:{args.trait} \n epochs: {args.epochs} \n \
                criterion: {args.criterion} \n activation: {args.activation} \
                \n batch: {args.batch_size} \n wdecay{args.weight_decay} \
                \n dropout{args.dropout} \n seed{args.seed}'")

    # Load mappings
    gene_map = pd.read_csv('./data/preprocessed/gene_index_mapping.csv')
    pathway_map = pd.read_csv('./data/preprocessed/pathway_index_mapping.csv')
    snp_map = pd.read_csv('./data/preprocessed/snp_index_mapping.csv')

    n_genes = gene_map.shape[0]
    n_pathway = pathway_map.shape[0]
    n_snps = snp_map.shape[0]

    # Load sparse masks properly
    mask_snp_gene = load_npz("./data/preprocessed/mask_snp_gene.npz")
    mask_gene_pathway = load_npz("./data/preprocessed/mask_gene_pathway.npz")

    # Convert sparse → dense tensor
    mask1 = torch.from_numpy(mask_snp_gene.toarray()).float()
    mask2 = torch.from_numpy(mask_gene_pathway.toarray()).float()

    # Load genotype
    gen_path = '../data/ML/BBB2023_MD'
    bim, _, bed = read_plink(gen_path)

    bim = bim.set_index('snp')
    snp_used = snp_map.set_index('snp_id')
    snp_to_index = bim.loc[snp_used.index]['i']

    X = bed[snp_to_index]

    # Load phenotype
    pheno_df = pd.read_csv(
        "../data/ML/pheno_2023bbb_0twins_6traits_mask",
        delimiter="\t", header=None
    )

    pheno_df = pheno_df.drop(columns=[0]).set_index(1)

    y = pheno_df[args.trait][~pheno_df[args.trait].isna()].to_numpy()
    X = X.T[~pheno_df[args.trait].isna()].compute()

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=1000, random_state=42
    )

    # Scale
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_val = scaler_y.transform(y_val.reshape(-1, 1))

    # Torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    # Model
    fc_architecture = args.layers 

    model = PartialNet(
        input_dim=n_snps,
        hidden1=n_genes,
        hidden2=n_pathway,
        fc_layers=fc_architecture,
        mask1=mask1,
        mask2=mask2,
        dropout=args.dropout
    )
    if args.criterion == "MSE":
        criterion = nn.MSELoss()
        logging.info("Setup of Loss confirmed as MSE")
    elif args.criterion == "MAE":
        criterion = nn.L1Loss()
        logging.info("Setup of Loss confirmed as MAE")
    elif args.criterion == "HuberLoss":
        criterion = nn.HuberLoss()
        logging.info("Setup of Loss confirmed as HuberLoss")
    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

    num_epochs = args.epochs
    train_losses = []
    val_losses = []
    #early_stopper = EarlyStopping(patience=20, min_delta=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    X_train = X_train.to(device)
    X_val = X_val.to(device)
    y_train = y_train.to(device)
    y_val = y_val.to(device)
    model = model.to(device)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
)
    for epoch in range(num_epochs):

        model.train()
        optimizer.zero_grad()

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

        # Gradient monitoring
        total_norm = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                total_norm += grad_norm

        logging.info(f"Gradient L2 norm: {total_norm:.6f}")

        optimizer.step()

        train_losses.append(loss.item())

        # Evaluation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                val_loss += criterion(outputs, yb).item()

        val_loss /= len(val_loader)

        logging.info(f"Epoch {epoch} | "
                     f"Train Loss: {loss.item():.6f} | "
                     f"val Loss: {val_loss:.6f}")
        
        #early_stopper(val_loss)

        # if early_stopper.early_stop:
        #     print("Early stopping triggered")
        #     break
        scheduler.step(val_loss)

    logging.info('\n'*3+"======== FINISHED TRAINING ========")

    # ===============================
    # Plot Loss Curves
    # ===============================

    plt.figure()
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.xlabel("Epoch")
    if args.criterion == "MSE":
        plt.ylabel("MSE Loss")
        logging.info("Setup of Loss confirmed as MSE on the plot")
    elif args.criterion == "MAE":
        plt.ylabel("Huber Loss")
        logging.info("Setup of Loss confirmed as MAE on the plot")
    elif args.criterion == "HuberLoss":
        plt.ylabel("Huber Loss")
        logging.info("Setup of Loss confirmed as HuberLoss on the plot")
    plt.title("Training and val Loss")
    plt.savefig(f"./results/test_hyp/val_architecture_{args_str}.png")
    plt.show()

    logging.info('\n'+"======== PLOTTED IMAGE ========")

    # ===============================
    # Final Evaluation (on real Scale)
    # ===============================

    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_val)

    # Convert to numpy
    y_pred_scaled = y_pred_scaled.cpu().numpy()
    y_val_scaled = y_val.cpu().numpy()

    # Inverse scaling
    y_pred_real = scaler_y.inverse_transform(y_pred_scaled)
    y_val_real = scaler_y.inverse_transform(y_val_scaled)

    # Compute them
    mae = mean_absolute_error(y_val_real, y_pred_real)
    mse = mean_squared_error(y_val_real, y_pred_real)
    r2 = r2_score(y_val_real, y_pred_real)
    pearson_corr, _ = pearsonr(y_val_real.flatten(), y_pred_real.flatten())

    logging.info("\n======== FINAL METRICS (REAL SCALE) ========")
    logging.info(f"MAE: {mae:.6f}")
    logging.info(f"MSE: {mse:.6f}")
    logging.info(f"R2: {r2:.6f}")
    logging.info(f"Pearson Correlation: {pearson_corr:.6f}")

    # ===============================
    # LayerNorm Sanity Check
    # ===============================

    with torch.no_grad():
        x1 = model.masked1(X_val)
        x1_norm = model.norm1(x1)

    mean_ln = x1_norm.mean().item()
    std_ln = x1_norm.std().item()

    logging.info("\n======== LAYERNORM CHECK ========")
    logging.info("Mean should be 0 and std 1.")
    logging.info(f"LayerNorm output mean: {mean_ln:.6f}")
    logging.info(f"LayerNorm output std: {std_ln:.6f}")

    # ===============================
    # Activation Variance Check
    # ===============================

    with torch.no_grad():
        a1 = model.masked1(X_val)
        a2 = model.masked2(model.activation_fn(model.norm1(a1)))

    logging.info("\n======== ACTIVATION VARIANCE CHECK ========")
    logging.info("If variance drops drastically then bad initialization, \
                 if it explodes, means unstable propagation.")
    logging.info(f"Masked1 variance: {a1.var().item():.6f}")
    logging.info(f"Masked2 variance: {a2.var().item():.6f}")

    # ===============================
    # Mask Integrity Check
    # ===============================

    with torch.no_grad():
        effective_weights = model.masked1.weight * model.masked1.mask
        violations = (effective_weights[model.masked1.mask == 0] != 0).sum().item()

    logging.info("\n======== MASK CHECK ========")
    logging.info("There are accidental connections?")
    logging.info(f"Zero-mask weight violations: {violations}")

    # ===============================
    # Forward Pass Sanity Check (Masked Layer)
    # ===============================

    logging.info("\n======== FORWARD SANITY CHECK ========")

    with torch.no_grad():
        test_out = model.masked1(X_val[:1])

        if torch.isnan(test_out).any():
            logging.info("NaNs detected in masked1 output!")
        else:
            logging.info("No NaNs in masked1 output.")

        logging.info(f"masked1 output shape: {test_out.shape}")
        logging.info(f"masked1 output mean: {test_out.mean().item():.6f}")
        logging.info(f"masked1 output std: {test_out.std().item():.6f}")
        
    # ===============================
    # Final Gradient Check
    # ===============================

    logging.info("\n======== FINAL GRADIENT CHECK ========")
    for name, param in model.named_parameters():
        if param.grad is not None:
            logging.info(f"{name} grad norm: {param.grad.norm().item():.6f}")

    plt.figure()
    plt.scatter(y_val_real, y_pred_real, alpha=0.5)
    plt.xlabel("True Phenotype")
    plt.ylabel("Predicted Phenotype")
    plt.title("Prediction vs True ")
    plt.savefig(f"./results/test_hyp/pred_vs_true_{args_str}.png")
    plt.close()

    # ===============================
    # Gradient Distribution Plot
    # ===============================

    all_grads = []

    for param in model.parameters():
        if param.grad is not None:
            all_grads.append(param.grad.view(-1))

    all_grads = torch.cat(all_grads).cpu().numpy()

    plt.figure()
    plt.hist(all_grads, bins=100)
    plt.title("Gradient Distribution")
    plt.savefig(f"./results/test_hyp/grad_distribution_{args_str}.png")
    plt.close()

if __name__ == '__main__':
    main()