# Snp_models

Repository for phenotype prediction from SNP data, with a focus on biologically informed deep neural networks.

The intended workflow in this project is:

1. Run [bio_informed_DNN/data_preprocessing.py](/home/jhonatan/Documents/Doutorado/projects/cleaning_repo/Snp_models/bio_informed_DNN/data_preprocessing.py)
2. Run [bio_informed_DNN/data_training.py](/home/jhonatan/Documents/Doutorado/projects/cleaning_repo/Snp_models/bio_informed_DNN/data_training.py)
3. Run [bio_informed_DNN/analyze_logs.py](/home/jhonatan/Documents/Doutorado/projects/cleaning_repo/Snp_models/bio_informed_DNN/analyze_logs.py)
4. Run one interpretation script:
   [bio_informed_DNN/gradientshap.py](/home/jhonatan/Documents/Doutorado/projects/cleaning_repo/Snp_models/bio_informed_DNN/gradientshap.py)
   or [bio_informed_DNN/integrated_gradients.py](/home/jhonatan/Documents/Doutorado/projects/cleaning_repo/Snp_models/bio_informed_DNN/integrated_gradients.py)

## Project Idea

This repository is designed to train models that predict quantitative phenotypes from SNP genotype data while preserving biological structure in the network.

Instead of using a fully dense neural network from input SNPs to output phenotype, the `bio_informed_DNN` pipeline builds biologically constrained connections:

- SNPs are linked to genes using genomic overlap.
- Genes are linked to pathways using enrichment results.
- The neural network uses these sparse masks to enforce biologically meaningful connectivity.

This makes the model more interpretable and helps organize the prediction problem around known biology.

## Main Pipeline

### 1. Data preprocessing

Script: [bio_informed_DNN/data_preprocessing.py](/home/jhonatan/Documents/Doutorado/projects/cleaning_repo/Snp_models/bio_informed_DNN/data_preprocessing.py)

This is the first step of the workflow and should be executed before training.

What it does:

- Loads genotype metadata from a PLINK dataset.
- Reads the genome annotation GTF file.
- Maps SNPs to genes by genomic overlap.
- Uses `gprofiler` to map genes to biological pathways from sources such as `KEGG`, `GO:BP`, and `REAC`.
- Builds sparse masks for:
  `SNP -> Gene`
  `Gene -> Pathway`
- Saves index mapping tables for SNPs, genes, and pathways.

Main outputs:

- `./data/preprocessed/Bos_taurus_new/mask_snp_gene.npz`
- `./data/preprocessed/Bos_taurus_new/mask_gene_pathway.npz`
- `./data/preprocessed/Bos_taurus_new/snp_index_mapping.csv`
- `./data/preprocessed/Bos_taurus_new/gene_index_mapping.csv`
- `./data/preprocessed/Bos_taurus_new/pathway_index_mapping.csv`

Typical use case:

- You received a new cattle SNP dataset and genome annotation and want to create the biological masks needed by the neural network.
- You want to rebuild pathway-aware connectivity after changing annotation or enrichment sources.

Important note:

- This script uses project-specific file locations such as `../data/ML/BBB2023_MD` and `../data/Bos_taurus.ARS-UCD2.0.115.gtf`. Adjust these paths to match your local data layout if needed.

### 2. Model training

Script: [bio_informed_DNN/data_training.py](/home/jhonatan/Documents/Doutorado/projects/cleaning_repo/Snp_models/bio_informed_DNN/data_training.py)

This script trains the biologically informed deep neural network using the preprocessed masks.

What it does:

- Loads the sparse biological masks generated during preprocessing.
- Loads genotype and phenotype data.
- Filters samples according to the selected trait.
- Splits the data into training and validation sets.
- Standardizes genotype and phenotype values.
- Builds a masked neural network where:
  input layer = SNPs
  hidden layer 1 = genes
  hidden layer 2 = pathways
  final layers = configurable dense layers
- Trains the model with configurable loss, activation, dropout, weight decay, batch size, and early stopping.
- Logs metrics and saves plots.
- Optionally saves a checkpoint with model weights, masks, validation data, and metadata for later interpretation.

Example arguments:

```bash
python bio_informed_DNN/data_training.py \
  --trait 2 \
  --layers 512 256 \
  --learning-rate 1e-3 \
  --epochs 150 \
  --criterion HuberLoss \
  --activation gelu \
  --batch-size 512 \
  --dropout 0.0 \
  --weight-decay 0.0 \
  --seed 43 \
  --early-stop true \
  --save-model true
```

Important training options:

- `--trait`: phenotype column to predict
- `--layers`: fully connected layers after the pathway layer
- `--criterion`: `MSE`, `MAE`, or `HuberLoss`
- `--activation`: `relu`, `sigmoid`, or `gelu`
- `--batch-size`: mini-batch size
- `--dropout`: dropout used in masked and dense sections
- `--weight-decay`: L2 regularization through Adam
- `--save-model true`: useful when you want to run attribution methods later

Typical outputs:

- Training logs in `./logs_models/...`
- Loss plots in `./results/test_hyp/...`
- Model checkpoints in `./saved_models/...` when saving is enabled

Typical use case:

- You want to compare multiple hyperparameter settings for a phenotype such as musculature, size, or shoulder score.
- You want to train a model whose architecture follows biological prior knowledge rather than a fully unconstrained DNN.

### 3. Training log analysis

Script: [bio_informed_DNN/analyze_logs.py](/home/jhonatan/Documents/Doutorado/projects/cleaning_repo/Snp_models/bio_informed_DNN/analyze_logs.py)

After training several models, this script summarizes the experimental logs.

What it does:

- Scans a log directory for training logs.
- Extracts hyperparameters from log filenames.
- Extracts final evaluation metrics such as MAE and Pearson correlation.
- Aggregates experiment statistics by hyperparameter.
- Finds the best runs by:
  lowest `MAE`
  highest `Correlation`
- Saves summary tables and bar plots.

Typical outputs:

- CSV summary files in `./log_analysis_results/...`
- Bar plots for mean MAE and mean correlation by hyperparameter

Typical use case:

- You launched many training jobs on a cluster and need a quick comparison of layer size, learning rate, dropout, or batch size.
- You want to identify the best checkpoint before running biological interpretation.

Important note:

- This script currently uses hardcoded `LOG_DIR` and `OUTPUT_DIR`. Update them before execution so they point to the experiment set you want to analyze.

### 4. Model interpretation

After selecting a trained model, the repository provides two post hoc attribution approaches.

#### Option A: Gradient SHAP

Script: [bio_informed_DNN/gradientshap.py](/home/jhonatan/Documents/Doutorado/projects/cleaning_repo/Snp_models/bio_informed_DNN/gradientshap.py)

What it does:

- Loads a saved checkpoint.
- Reconstructs the trained `PartialNet`.
- Uses Captum `GradientShap` to compute feature attributions on validation samples.
- Aggregates importance from SNP level to gene level and pathway level using the saved masks.
- Saves ranked tables and plots for the most important SNPs, genes, and pathways.

Typical outputs:

- `gradientshap_snp_importance.csv`
- `gradientshap_gene_importance.csv`
- `gradientshap_pathway_importance.csv`
- `gradientshap_top_snps.png`
- `gradientshap_top_genes.png`
- `gradientshap_top_pathways.png`

Typical use case:

- You want a stochastic gradient-based explanation that uses a background distribution from validation data.
- You want to identify which SNP groups, genes, and pathways most influenced phenotype prediction in a trained model.

#### Option B: Integrated Gradients

Script: [bio_informed_DNN/integrated_gradients.py](/home/jhonatan/Documents/Doutorado/projects/cleaning_repo/Snp_models/bio_informed_DNN/integrated_gradients.py)

What it does:

- Loads a saved checkpoint.
- Rebuilds the trained model.
- Uses Captum `IntegratedGradients` with a zero baseline.
- Computes attribution scores for validation samples.
- Aggregates importance scores from SNPs to genes and pathways.
- Saves ranked importance tables and top-feature plots.

Typical outputs:

- `snp_importance.csv`
- `gene_importance.csv`
- `pathway_importance.csv`
- `top_snps_ig.png`
- `top_genes_ig.png`
- `top_pathways_ig.png`

Typical use case:

- You want a deterministic path-based attribution method for the trained network.
- You want to compare explanations from Integrated Gradients against Gradient SHAP for robustness.

Important note for both interpretation scripts:

- Both scripts currently point to a specific saved checkpoint through a hardcoded `path_model`. Change this variable before running the script on another model.
- These scripts expect that the training checkpoint contains masks, validation data, and feature names, so interpretation works best when the model was saved with complete metadata.

## Recommended End-to-End Usage

For a standard project run, the recommended order is:

```bash
python bio_informed_DNN/data_preprocessing.py
python bio_informed_DNN/data_training.py --save-model true
python bio_informed_DNN/analyze_logs.py
python bio_informed_DNN/gradientshap.py
```

Or, if you prefer Integrated Gradients:

```bash
python bio_informed_DNN/data_preprocessing.py
python bio_informed_DNN/data_training.py --save-model true
python bio_informed_DNN/analyze_logs.py
python bio_informed_DNN/integrated_gradients.py
```


## Repository Structure

```text
Snp_models/
├── README.md
├── bio_informed_DNN/
│   ├── data_preprocessing.py
│   ├── data_training.py
│   ├── analyze_logs.py
│   ├── gradientshap.py
│   ├── integrated_gradients.py
│   ├── model.py
│   └── requirements.txt
└── LassoNet/
    └── train.py
```

## Environment

The repository was originally developed with:

- Python `3.9.21`
- Rocky Linux `9.5`

Package definitions are available in:

- [bio_informed_DNN/requirements.txt](/home/jhonatan/Documents/Doutorado/projects/cleaning_repo/Snp_models/bio_informed_DNN/requirements.txt)
- [pyproject.toml](/home/jhonatan/Documents/Doutorado/projects/cleaning_repo/Snp_models/pyproject.toml)
