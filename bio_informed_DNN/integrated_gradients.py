import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from model import PartialNet
from scipy.stats import zscore

# ===============================
# Load checkpoint
# ===============================
print("Started IG")

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


# ===============================
# Integrated Gradients
# ===============================

ig = IntegratedGradients(model)

baseline = torch.zeros_like(X_val[:1])

attributions = ig.attribute(X_val, baselines=baseline, n_steps=50)

attributions = attributions.detach().numpy()


# ===============================
# SNP importance
# ===============================

snp_importance = np.mean(np.abs(attributions), axis=0)

snp_importance = zscore(snp_importance)

snp_df = pd.DataFrame({"snp": snp_names, "importance": snp_importance})

snp_df = snp_df.sort_values("importance", ascending=False)


# ===============================
# Gene importance
# ===============================
print(mask_snp_gene.shape)
print(len(snp_importance))
print(f"The mask_snp_gene has the shape {mask_snp_gene.shape}")

gene_importance = mask_snp_gene @ snp_importance

gene_importance = zscore(gene_importance)

gene_df = pd.DataFrame({"gene": gene_names, "importance": gene_importance}).sort_values(
    "importance", ascending=False
)


# ===============================
# Pathway importance
# ===============================

pathway_importance = mask_gene_pathway @ gene_importance

pathway_importance = pathway_importance / mask_gene_pathway.sum(axis=1)

pathway_importance = zscore(pathway_importance)

pathway_df = pd.DataFrame(
    {"pathway": pathway_names, "importance": pathway_importance}
).sort_values("importance", ascending=False)

# ===============================
# Save tables
# ===============================

snp_df.to_csv("snp_importance.csv", index=False)
gene_df.to_csv("gene_importance.csv", index=False)
pathway_df.to_csv("pathway_importance.csv", index=False)


# ===============================
# Plot top features
# ===============================


def plot_top(df, name, n=20):

    top = df.head(n)

    plt.figure(figsize=(8, 6))
    plt.barh(top.iloc[::-1, 0], top.iloc[::-1, 1])
    plt.xlabel("Importance")
    plt.title(f"Top {n} {name}")
    plt.tight_layout()

    plt.savefig(f"top_{name}_ig.png")
    plt.close()


plot_top(snp_df, "snps_ig")
plot_top(gene_df, "genes_ig")
plot_top(pathway_df, "pathways_ig")
print("Finished IG")
print("================== \n \n \n")
