

"""
Robust scATAQ‑seq Analysis Pipeline using Muon

This script builds an analysis pipeline for single‑cell ATAC‑seq data 
from scratch using Muon. It uses four user-provided inputs:
    1. Filtered count matrix (filtered_peak_bc_matrix.h5) – in 10× H5 format.
    2. Barcodes file (barcodes.tsv) – a TSV file listing cell barcodes.
    3. Features file (peaks.bed) – a BED file with peak coordinates.
    4. Cell metadata (singlecell.csv) – per‐cell metadata.

The pipeline will:
  • Load the filtered count matrix using Muon.
  • Override barcodes and feature names with those provided.
  • Merge cell metadata.
  • Run quality control and normalization.
  • Perform PCA, compute neighbors, generate a UMAP embedding, and apply Leiden clustering.
  • Produce visualizations (UMAP, PCA, etc.) for exploratory analysis.
    • Generate violin plots and a heatmap of top variable peaks across clusters.
    • Save the processed data to an H5AD file.


Ensure you run this with Python ≥ 3.6 (for f-string support) and that the file paths you provide are correct.
"""

import os
import sys
import muon as mu
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

# ------------------ User Input ------------------
print("Please provide the full paths for the following four files.")
raw_peak_file = input("1. Filtered count matrix (filtered_peak_bc_matrix.h5): ").strip()
barcodes_file = input("2. Barcodes file (barcodes.tsv): ").strip()
peaks_file    = input("3. Features file (peaks.bed): ").strip()
metadata_file = input("4. Cell metadata file (singlecell.csv): ").strip()

# Validate that files exist
for file_path in [raw_peak_file, barcodes_file, peaks_file, metadata_file]:
    if not os.path.exists(file_path):
        sys.exit(f"Error: The file {file_path} does not exist.")

# ------------------ Data Loading ------------------
print("\nLoading raw ATAC-seq count matrix using Muon...")
try:
    # Read the raw_peak_bc_matrix.h5 file (assumed to be in 10× H5 format)
    mdata = mu.read_10x_h5(raw_peak_file)
    print("Raw count matrix loaded successfully.")
except Exception as e:
    sys.exit(f"Error loading raw count matrix: {e}")

# For simplicity, work with the first modality (should be ATAC)
modality = list(mdata.mod.keys())[0]
adata = mdata.mod[modality]
print("Using modality:", modality)


# ------------------ Override Barcodes and Features ------------------
# Read barcodes.tsv (assumed to have one barcode per line; no header)
try:
    barcodes_df = pd.read_csv(barcodes_file, sep="\t", header=None)
    barcodes_list = barcodes_df.iloc[:, 0].tolist()  # Extract first column as list
    adata.obs_names = barcodes_list
    print(f"Overridden cell barcodes with {len(barcodes_list)} entries from {barcodes_file}.")
except Exception as e:
    sys.exit(f"Error reading barcodes file: {e}")

# Read peaks.bed (assumed to have at least three columns: chrom, start, end)
try:
    # Use engine='python' and skip comment lines (if any)
    peaks = pd.read_csv(peaks_file, sep="\t", header=None, comment='#', engine='python')
    # Create a unique peak ID by combining the first three columns (chrom, start, end)
    peaks['peak_id'] = peaks[0].astype(str) + "_" + peaks[1].astype(str) + "_" + peaks[2].astype(str)
    adata.var_names = peaks['peak_id'].tolist()
    print(f"Overridden feature names with {len(adata.var_names)} peak IDs from {peaks_file}.")
except Exception as e:
    sys.exit(f"Error reading peaks file: {e}")

# ------------------ Merge Cell Metadata ------------------
if os.path.exists(metadata_file):
    print("\nLoading cell metadata from:", metadata_file)
    try:
        cell_meta = pd.read_csv(metadata_file, index_col=0)
        # Merge: ensure that cell_meta indices match adata.obs_names
        adata.obs = adata.obs.join(cell_meta, how='left')
        print("Cell metadata merged successfully.")
    except Exception as e:
        sys.exit(f"Error merging cell metadata: {e}")
else:
    print("Cell metadata file not found. Proceeding without additional metadata.")

# ------------------ Preprocessing ------------------
print("\nStarting preprocessing...")

# Filter out cells with very low counts; threshold may need adjustment for ATAC data
sc.pp.filter_cells(adata, min_counts=1000)
# Filter peaks that are detected in very few cells
sc.pp.filter_genes(adata, min_cells=3)

# Normalize counts per cell and log-transform (common practice for single-cell data)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Identify highly variable peaks (features); adjust n_top_genes as needed
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)

# ------------------ Dimensionality Reduction & Clustering ------------------
print("\nRunning PCA...")
sc.tl.pca(adata, svd_solver='arpack', n_comps=50)

print("Computing the neighborhood graph...")
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)

print("Calculating UMAP embedding...")
sc.tl.umap(adata)

print("Performing Leiden clustering (resolution=0.5)...")
sc.tl.leiden(adata, resolution=0.5)

# Get top variable peaks
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat')
top_var_peaks = adata.var.sort_values('dispersions_norm', ascending=False)[:50].index.tolist()

# ------------------ Generate Violin Plots ------------------
print("\nGenerating violin plots...")

# Plot total counts per cluster
sc.pl.violin(adata, 
             'n_counts',
             groupby='leiden',
             rotation=45,
             save='_counts_violin.pdf')

# Plot distribution of top peaks
for peak in top_var_peaks[:5]:
    sc.pl.violin(adata,
                peak,
                groupby='leiden',
                rotation=45,
                save=f'_peak_{peak}_violin.pdf')

# Quality metrics violin plot
sc.pl.violin(adata,
             ['total', 'n_counts'],
             groupby='leiden',
             rotation=45,
             multi_panel=True,
             save='_qc_metrics_violin.pdf')

# ------------------ Generate Heatmap ------------------
print("\nGenerating heatmap of top variable peaks across clusters...")

# Get top variable peaks
sc.pp.highly_variable_genes(adata, n_top_genes=50, flavor='seurat')
top_var_peaks = adata.var.sort_values('dispersions_norm', ascending=False)[:50].index.tolist()

# Calculate differential peaks between clusters
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')

# Generate comprehensive heatmap
sc.pl.heatmap(adata, 
              var_names=top_var_peaks,
              groupby='leiden',
              show_gene_labels=True,
              dendrogram=True,
              cmap='viridis',
              figsize=(12, 8),
              swap_axes=True,
              save='_top_peaks_heatmap.pdf')

# Optional: Create a more detailed visualization
sc.pl.rank_genes_groups_heatmap(adata,
                               n_genes=25,
                               show_gene_labels=True,
                               dendrogram=True,
                               save='_differential_peaks_heatmap.pdf')

# ------------------ Visualization ------------------
print("\nGenerating UMAP plot with Leiden clusters...")
sc.pl.umap(adata, color=["leiden"], title="scATAQ‑seq Clusters", save="_umap_clusters.png", show=True)

print("Generating PCA plot...")
sc.pl.pca(adata, color=["leiden"], title="PCA Plot", save="_pca.png", show=True)

# ------------------ Save Processed Data ------------------
output_file = os.path.join(os.path.dirname(raw_peak_file), "processed_scataq.h5ad")
adata.write(output_file)
print(f"\nProcessed data saved to: {output_file}")