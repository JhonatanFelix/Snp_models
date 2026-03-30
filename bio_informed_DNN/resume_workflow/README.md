# Resume Workflow

This folder is a compact, documented reference version of the biologically informed SNP modeling work in this repository.

It keeps only the core pieces that are needed to understand and rerun the current approach:

- `dataset.py`
  Loads the VEP-based SNP annotation, applies the birth-date ordering from the pedigree file, uses the last 1000 valid animals as validation, and reconstructs the masked animals as the test set.
- `models.py`
  Defines the three model families used in the current work: `gene_only`, `gene_pathway`, and `attention_gene`.
- `train.py`
  Unified training entrypoint for the three models, with standardization, early stopping, LR scheduling, compact checkpoint saving, and masked-test evaluation.
- `analyze_integrated_gradients.py`
  Compact integrated-gradients analysis script that adapts to all three checkpoint types and produces IG-based tables and figures only.

## Design Principles

- Keep the exact cohort logic used in the main project.
- Remove old experiment-specific branches and large pipeline-only utilities.
- Save compact checkpoints that still contain enough metadata to reconstruct the biological masks and evaluation cohorts later.
- Prefer readability and documentation over exhaustive experimental flexibility.

## Typical Usage

Train a gene-only model:

```bash
python3 bio_informed_DNN/resume_workflow/train.py \
  --model-type gene_only \
  --trait 4 \
  --layers 50 \
  --activation gelu \
  --criterion HuberLoss
```

Train an attention model:

```bash
python3 bio_informed_DNN/resume_workflow/train.py \
  --model-type attention_gene \
  --trait 4 \
  --layers 100 \
  --activation gelu \
  --criterion HuberLoss \
  --attention-dropout 0.0 \
  --impact-embedding-dim 16
```

Train a gene+pathway model:

```bash
python3 bio_informed_DNN/resume_workflow/train.py \
  --model-type gene_pathway \
  --trait 4 \
  --layers 50 \
  --pathway-mask-path /path/to/mask_gene_pathway.npz \
  --pathway-mapping-path /path/to/pathway_index_mapping.csv \
  --pathway-gene-mapping-path /path/to/gene_index_mapping.csv
```

Analyze a saved checkpoint with integrated gradients:

```bash
python3 bio_informed_DNN/resume_workflow/analyze_integrated_gradients.py \
  --checkpoint /path/to/checkpoint.pt
```

## Notes

- The default paths in the scripts are aligned with the current repository layout.
- `gene_pathway` expects external pathway-mask artifacts. The reduced workflow keeps support for them, but does not regenerate them.
- Checkpoints from `train.py` are designed to be self-sufficient for later validation/test reconstruction and IG analysis.
