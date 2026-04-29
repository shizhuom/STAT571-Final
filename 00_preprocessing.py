# 00_preprocessing.py
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

sc.settings.verbosity = 1
sc.settings.set_figure_params(dpi=100, facecolor='white')

# ── Load the pre-processed PBMC 3k dataset ──
adata = sc.datasets.pbmc3k_processed()
print(adata)

print(f"Dataset shape: {adata.shape}")
print(f"Cell types: {adata.obs['louvain'].value_counts().to_dict()}")

# ── If starting from RAW data instead, uncomment this block ──
# adata_raw = sc.datasets.pbmc3k()
# adata_raw.var_names_make_unique()
# sc.pp.filter_cells(adata_raw, min_genes=200)
# sc.pp.filter_genes(adata_raw, min_cells=3)
# adata_raw.var['mt'] = adata_raw.var_names.str.startswith('MT-')
# sc.pp.calculate_qc_metrics(adata_raw, qc_vars=['mt'], inplace=True)
# adata_raw = adata_raw[adata_raw.obs.pct_counts_mt < 5].copy()
# sc.pp.normalize_total(adata_raw, target_sum=1e4)
# sc.pp.log1p(adata_raw)
# sc.pp.highly_variable_genes(adata_raw, n_top_genes=2000)
# sc.pp.scale(adata_raw, max_value=10)
# sc.pp.pca(adata_raw, n_comps=50)
# sc.pp.neighbors(adata_raw, n_neighbors=10, n_pcs=40)
# sc.tl.umap(adata_raw)

# ── Ensure PCA is computed ──
if 'X_pca' not in adata.obsm:
    sc.pp.pca(adata, n_comps=50, svd_solver='arpack')

# ── Ensure neighbor graph and UMAP exist ──
if 'X_umap' not in adata.obsm:
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.umap(adata)

# ── Visualization: UMAP colored by cell type ──
sc.pl.umap(adata, color='louvain', title='PBMC 3k - Cell Types',
           frameon=False, save='_cell_types.png')

# ── PCA variance explained ──
sc.pl.pca_variance_ratio(adata, n_pcs=50, log=True, save='_variance.png')

# ── Save processed data for downstream scripts ──
import anndata
anndata.settings.allow_write_nullable_strings = True
adata.write('data/pbmc3k_processed.h5ad')
print("Preprocessing complete. Saved to data/pbmc3k_processed.h5ad")

