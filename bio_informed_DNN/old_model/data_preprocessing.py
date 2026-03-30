import logging
import numpy as np
import pandas as pd
import pyranges as pr
from gprofiler import GProfiler
from pandas_plink import read_plink
from scipy.sparse import coo_matrix, save_npz
import os


# ===============================
# Logger
# ===============================

def setup_logger(log_filename):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
    )


# ===============================
# Main
# ===============================

def main():

    log_filename = "./logs/btaurus_new_data_preprocessing.log"
    setup_logger(log_filename)
    skip = "\n"*3
    logging.info(skip+"========== STARTING PREPROCESSING =========="+skip)

    # ==========================================================
    # 1 Load PLINK data
    # ==========================================================

    gen_path = '../data/ML/BBB2023_MD'
    bim, fam, bed = read_plink(gen_path)

    logging.info(f"PLINK loaded | bim: {bim.shape} fam: {fam.shape} bed: {bed.shape}")

    # Normalize chromosome format (string, no 'chr')
    bim['chrom'] = bim['chrom'].astype(str).str.replace("chr", "", regex=False)

    # ==========================================================
    # 2 SNP → Gene (genomic overlap)
    # ==========================================================

    snps_pos = bim[['snp', 'chrom', 'pos']].copy()
    snps_pos.rename(columns={
        'snp': 'snp_id',
        'chrom': 'Chromosome',
        'pos': 'End'
    }, inplace=True)

    snps_pos['Start'] = snps_pos['End'] - 1

    snps_pos['Chromosome'] = snps_pos['Chromosome'].astype(str)

    snps_gr = pr.PyRanges(snps_pos)

    genes = pr.read_gtf("../data/Bos_taurus.ARS-UCD2.0.115.gtf")
    genes = genes[genes.Feature == "gene"]

    # Normalize chromosome format in GTF
    genes.Chromosome = genes.Chromosome.astype(str).str.replace("chr", "", regex=False)

    overlap = snps_gr.join(genes)

    overlap_df = overlap.df[['snp_id', 'gene_id', 'gene_name']].dropna()

    logging.info(f"SNP-Gene overlaps found: {overlap_df.shape[0]}")
    logging.info(f"Unique SNPs on SNP-Gene overlaps:{len(overlap_df['snp_id'].unique())}")

    # ==========================================================
    # 3 Gene → Pathway (via gProfiler)
    # ==========================================================

    gp = GProfiler(return_dataframe=True)

    gene_symbols = overlap_df['gene_name'].unique().tolist()

    gprofiler_results = gp.profile(
        organism='btaurus',
        query=gene_symbols,
        sources=['KEGG', 'GO:BP', 'REAC'],
        no_evidences=False
    )

    logging.info(f"gProfiler results shape: {gprofiler_results.shape}")

    # ==========================================================
    # 4 Build SNP → Gene Mask (Sparse)
    # ==========================================================

    unique_snps = overlap_df['snp_id'].unique()
    unique_genes = overlap_df['gene_id'].unique()

    snp_idx = {snp: i for i, snp in enumerate(unique_snps)}
    gene_idx = {gene: i for i, gene in enumerate(unique_genes)}

    row_indices = []
    col_indices = []

    for row in overlap_df.itertuples(index=False):
        gene_i = gene_idx[row.gene_id]
        snp_j = snp_idx[row.snp_id]

        row_indices.append(gene_i)
        col_indices.append(snp_j)

    mask_snp_gene = coo_matrix(
        (np.ones(len(row_indices), dtype=np.float32),
         (row_indices, col_indices)),
        shape=(len(unique_genes), len(unique_snps))
    )

    logging.info(f"SNP-Gene mask shape: {mask_snp_gene.shape}")
    logging.info(f"Connections: {mask_snp_gene.nnz}")

    # ==========================================================
    # 5 Build Gene → Pathway Mask (Sparse)
    # ==========================================================

    pathway_list = gprofiler_results['native'].unique()
    pathway_idx = {p: i for i, p in enumerate(pathway_list)}

    gene_name_to_id = overlap_df[['gene_name', 'gene_id']].drop_duplicates()
    gene_name_to_id = dict(zip(gene_name_to_id['gene_name'],
                               gene_name_to_id['gene_id']))

    row_indices = []
    col_indices = []

    for row in gprofiler_results.itertuples(index=False):

        pathway_i = pathway_idx[row.native]

        for gene_symbol in row.intersections:

            gene_id = gene_name_to_id.get(gene_symbol)
            if gene_id is None:
                continue

            gene_j = gene_idx.get(gene_id)
            if gene_j is None:
                continue

            row_indices.append(pathway_i)
            col_indices.append(gene_j)

    mask_gene_pathway = coo_matrix(
        (np.ones(len(row_indices), dtype=np.float32),
         (row_indices, col_indices)),
        shape=(len(pathway_list), len(unique_genes))
    )

    logging.info(f"Gene-Pathway mask shape: {mask_gene_pathway.shape}")
    logging.info(f"Connections: {mask_gene_pathway.nnz}")

    # ==========================================================
    # 6 Save Masks to Disk
    # ==========================================================

    os.makedirs('./data/preprocessed/Bos_taurus_new/') #! Make the correct pathways in everyplace
    save_npz("./data/preprocessed/Bos_taurus_new/mask_snp_gene.npz", mask_snp_gene)
    save_npz("./data/preprocessed/Bos_taurus_new/mask_gene_pathway.npz", mask_gene_pathway)

    logging.info("Masks saved to disk.")

    # ==========================================================
    # 7 Save Index Mappings
    # ==========================================================

    # SNP mapping
    snp_mapping_df = pd.DataFrame({
        "snp_index": range(len(unique_snps)),
        "snp_id": unique_snps
    })
    snp_mapping_df.to_csv("./data/preprocessed/Bos_taurus_new/snp_index_mapping.csv", index=False)

    # Gene mapping
    gene_metadata = overlap_df[['gene_id', 'gene_name']].drop_duplicates()

    gene_mapping_df = pd.DataFrame({
        "gene_index": range(len(unique_genes)),
        "ensembl_gene_id": unique_genes
    })

    gene_mapping_df = gene_mapping_df.merge(
        gene_metadata,
        left_on="ensembl_gene_id",
        right_on="gene_id",
        how="left"
    ).drop(columns="gene_id")

    gene_mapping_df.to_csv("./data/preprocessed/Bos_taurus_new/gene_index_mapping.csv", index=False)

    # Pathway mapping
    pathway_mapping_df = pd.DataFrame({
        "pathway_index": range(len(pathway_list)),
        "pathway_id": pathway_list
    })
    pathway_mapping_df.to_csv("./data/preprocessed/Bos_taurus_new/pathway_index_mapping.csv", index=False)

    logging.info("Index mappings saved to disk.")
    
    logging.info(skip+"========== PREPROCESSING FINISHED =========="+skip)


if __name__ == "__main__":
    main()