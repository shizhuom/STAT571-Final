# 03_clustering.py
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score, confusion_matrix)

# ── Load data ──
adata = sc.read('data/pbmc3k_processed.h5ad')
ground_truth = adata.obs['louvain'].values
n_true_types = len(adata.obs['louvain'].unique())
X_pca = adata.obsm['X_pca']

print(f"Number of true cell types: {n_true_types}")
print(f"Cell type distribution:\n{adata.obs['louvain'].value_counts()}\n")

# ── Ensure neighbor graph exists (needed for Leiden) ──
if 'neighbors' not in adata.uns:
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

# ══════════════════════════════════════════════
# K-MEANS CLUSTERING
# ══════════════════════════════════════════════

# Elbow method and silhouette analysis to find optimal k
K_range = range(2, 16)
inertias, silhouettes, ari_scores = [], [], []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_pca)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_pca, labels, sample_size=2000,
                                         random_state=42))
    ari_scores.append(adjusted_rand_score(ground_truth, labels))

# Plot elbow and silhouette
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].plot(K_range, inertias, 'bo-')
axes[0].set_xlabel('k'); axes[0].set_ylabel('Inertia'); axes[0].set_title('Elbow Method')
axes[1].plot(K_range, silhouettes, 'ro-')
axes[1].set_xlabel('k'); axes[1].set_ylabel('Silhouette'); axes[1].set_title('Silhouette Score')
axes[2].plot(K_range, ari_scores, 'go-')
axes[2].set_xlabel('k'); axes[2].set_ylabel('ARI'); axes[2].set_title('ARI vs Ground Truth')
axes[2].axvline(x=n_true_types, color='gray', linestyle='--', label=f'True k={n_true_types}')
axes[2].legend()
plt.tight_layout()
plt.savefig('figures/kmeans_k_selection.png', dpi=150, bbox_inches='tight')
plt.show()

# Run K-means with the true number of cell types
kmeans_best = KMeans(n_clusters=n_true_types, random_state=42, n_init=20)
adata.obs['kmeans'] = kmeans_best.fit_predict(X_pca).astype(str)
adata.obs['kmeans'] = adata.obs['kmeans'].astype('category')

# ══════════════════════════════════════════════
# LEIDEN CLUSTERING (multiple resolutions)
# ══════════════════════════════════════════════

resolutions = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
leiden_results = []

for res in resolutions:
    key = f'leiden_{res}'
    sc.tl.leiden(adata, resolution=res, key_added=key)
    n_clust = adata.obs[key].nunique()
    ari = adjusted_rand_score(ground_truth, adata.obs[key])
    nmi = normalized_mutual_info_score(ground_truth, adata.obs[key])
    leiden_results.append({
        'resolution': res, 'n_clusters': n_clust, 'ARI': ari, 'NMI': nmi
    })

leiden_df = pd.DataFrame(leiden_results)
print("Leiden clustering results across resolutions:")
print(leiden_df.to_string(index=False))

# Select best Leiden resolution (highest ARI)
best_res = leiden_df.loc[leiden_df['ARI'].idxmax(), 'resolution']
adata.obs['leiden_best'] = adata.obs[f'leiden_{best_res}']
print(f"\nBest Leiden resolution: {best_res}")

# ══════════════════════════════════════════════
# EVALUATION: Compare all clustering methods
# ══════════════════════════════════════════════

methods = {
    'K-means': adata.obs['kmeans'],
    f'Leiden (res={best_res})': adata.obs['leiden_best'],
    'Leiden (res=1.0)': adata.obs['leiden_1.0'],
}

print(f"\n{'Method':<25} {'Clusters':>8} {'ARI':>8} {'NMI':>8}")
print('-' * 52)
for name, labels in methods.items():
    ari = adjusted_rand_score(ground_truth, labels)
    nmi = normalized_mutual_info_score(ground_truth, labels)
    n_cl = labels.nunique()
    print(f"{name:<25} {n_cl:>8} {ari:>8.4f} {nmi:>8.4f}")

# ══════════════════════════════════════════════
# CONFUSION MATRICES: Cluster vs Cell Type
# ══════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for ax, (name, labels) in zip(axes, [('K-means', adata.obs['kmeans']),
                                       (f'Leiden (res={best_res})',
                                        adata.obs['leiden_best'])]):
    ct = pd.crosstab(ground_truth, labels, normalize='index')
    sns.heatmap(ct, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                cbar_kws={'shrink': 0.8})
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Cell Type')
    ax.set_title(f'{name} vs Ground Truth')

plt.tight_layout()
plt.savefig('figures/clustering_confusion_matrices.png', dpi=150,
            bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════
# UMAP VISUALIZATION: Side by side
# ══════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
sc.pl.umap(adata, color='louvain', ax=axes[0], show=False,
           title='Ground Truth', frameon=False)
sc.pl.umap(adata, color='kmeans', ax=axes[1], show=False,
           title='K-means', frameon=False)
sc.pl.umap(adata, color='leiden_best', ax=axes[2], show=False,
           title=f'Leiden (res={best_res})', frameon=False)
plt.tight_layout()
plt.savefig('figures/clustering_umap_comparison.png', dpi=150,
            bbox_inches='tight')
plt.show()