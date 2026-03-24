import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from captum.attr import GradientShap
from model import PartialNet
from scipy.stats import zscore

# ===============================
# Load checkpoint
# ===============================

print("================== \n \n \n")
print("Started SHAP")

path_model = "./saved_models/ARS-UCD2.0.115/critHuberLoss_gelu/model_[50]_lr0.0001_trait2_epoch150_critHuberLoss_actgelu_batch256_wdecay0.0001_dropout0.5_seed43_eaTrue.pt"

checkpoint = torch.load(path_model, map_location="cpu", weights_only=False)

arch = checkpoint["architecture"]

mask1 = checkpoint["mask_snp_gene"]
mask2 = checkpoint["mask_gene_pathway"]

model = PartialNet(
    input_dim=arch["input_dim"],
    hidden1=arch["hidden1"],
    hidden2=arch["hidden2"],
    fc_layers=arch["fc_layers"],
    mask1=mask1,
    mask2=mask2,
    activation=arch["activation"],
    dropout=arch["dropout"],
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# ===============================
# Load data
# ===============================

X_val = checkpoint["X_val"]

snp_names = checkpoint["snp_names"]
gene_names = checkpoint["gene_names"]
pathway_names = checkpoint["pathway_names"]

mask_snp_gene = checkpoint["mask_snp_gene"].numpy()
mask_gene_pathway = checkpoint["mask_gene_pathway"].numpy()

print(gene_names.shape)
print(gene_names)
# ===============================
# Background distribution
# ===============================

# SHAP requires background samples
background = X_val[np.random.choice(X_val.shape[0], 100, replace=False)]


# ===============================
# Gradient SHAP
# ===============================

gs = GradientShap(model)

attributions = gs.attribute(X_val, baselines=background, n_samples=100, stdevs=0.05)

attributions = attributions.detach().numpy()


# ===============================
# SNP importance
# ===============================

snp_importance = np.mean(np.abs(attributions), axis=0)

snp_importance = zscore(snp_importance)

snp_df = pd.DataFrame({"snp": snp_names, "importance": snp_importance})

snp_df = snp_df.sort_values("importance", ascending=False)

print(f"The snp_df shape{snp_df.shape}")
# ===============================
# Gene importance
# ===============================
print(mask_snp_gene.shape)
print(len(snp_importance))
print(f"The mask_snp_gene has the shape {mask_snp_gene.shape}")

gene_importance = np.zeros(mask_snp_gene.shape[0])

for g in range(mask_snp_gene.shape[0]):
    snps = np.where(mask_snp_gene.T[:, g] == 1)[0]
    gene_importance[g] = snp_importance[snps].sum()

gene_importance = zscore(gene_importance)

gene_df = pd.DataFrame({"gene": gene_names, "importance": gene_importance})

gene_df = gene_df.sort_values("importance", ascending=False)

# ===============================
# Gene importance
# ===============================

# n_genes = mask_snp_gene.shape[1]

# print(f"The number of genes on mask snp genes{n_genes}")
# print(f"The number of genes on {len(gene_names)}")
# gene_importance = np.zeros(n_genes)

# for g in range(n_genes):
#     snps = np.where(mask_snp_gene.T[:, g] == 1)[0]
#     gene_importance[g] = snp_importance[snps].sum()

# gene_importance = zscore(gene_importance)

# # Ensure same length
# gene_names = np.array(gene_names)[:n_genes]

# gene_df = pd.DataFrame({
#     "gene": gene_names,
#     "importance": gene_importance
# })

# gene_df = gene_df.sort_values("importance", ascending=False)

# ===============================
# Pathway importance
# ===============================

pathway_importance = np.zeros(mask_gene_pathway.shape[0])

for p in range(mask_gene_pathway.shape[0]):
    genes = np.where(mask_gene_pathway.T[:, p] == 1)[0]
    pathway_importance[p] = gene_importance[genes].sum()

pathway_importance = zscore(pathway_importance)

pathway_df = pd.DataFrame({"pathway": pathway_names, "importance": pathway_importance})

pathway_df = pathway_df.sort_values("importance", ascending=False)


# ===============================
# Save tables
# ===============================

snp_df.to_csv("gradientshap_snp_importance.csv", index=False)
gene_df.to_csv("gradientshap_gene_importance.csv", index=False)
pathway_df.to_csv("gradientshap_pathway_importance.csv", index=False)


# ===============================
# Plot top features
# ===============================


def plot_top(df, label, n=20):

    top = df.head(n)

    plt.figure(figsize=(8, 6))
    plt.barh(top.iloc[::-1, 0], top.iloc[::-1, 1])
    plt.xlabel("Importance")
    plt.title(f"Top {n} {label} (Gradient SHAP)")
    plt.tight_layout()

    plt.savefig(f"gradientshap_top_{label}.png")
    plt.close()


plot_top(snp_df, "snps")
plot_top(gene_df, "genes")
plot_top(pathway_df, "pathways")


# ===============================
# Display results
# ===============================

print("\nTop SNPs")
print(snp_df.head(10))

print("\nTop Genes")
print(gene_df.head(10))

print("\nTop Pathways")
print(pathway_df.head(10))

print("Finished SHAP")
print("================== \n \n \n")