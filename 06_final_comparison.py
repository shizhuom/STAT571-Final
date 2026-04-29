"""
=============================================================================
  FINAL COMPARISON: Predicting Blood Cell Types from Gene Expression
  Data Mining Final Project — All Methods + Evaluation in One Script
=============================================================================

This script reruns all baseline methods (Logistic Regression, Clustering,
MLP, Random Forest) on the PBMC 3k dataset and produces a unified
comparison of results. If 05_transformer_based.py has already been run,
its outputs in figures/transformer_results.csv are merged into the final
benchmark plot.

Run order: 01 → 02 → 03 → 04 → 05 → 06.

Usage:
  cd D:/260414_final_571
  python 06_final_comparison.py
=============================================================================
"""

import scanpy as sc
import anndata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

# ── Create output directories ─────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 0: LOAD AND PREPARE DATA
# ══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("SECTION 0: Loading data")
print("=" * 60)

# Load the processed dataset
# If you used sc.datasets.pbmc3k_processed(), the label column is 'louvain'
# If you preprocessed from raw data, the label column is 'cell_type'
adata = sc.read("data/pbmc3k_processed.h5ad")

# Auto-detect the cell type label column
if "louvain" in adata.obs.columns:
    label_col = "louvain"
elif "cell_type" in adata.obs.columns:
    label_col = "cell_type"
else:
    raise ValueError(
        f"Cannot find cell type labels. Available columns: {list(adata.obs.columns)}"
    )
print(f"Using label column: '{label_col}'")

# Ensure PCA exists
if "X_pca" not in adata.obsm:
    sc.pp.pca(adata, n_comps=50, svd_solver="arpack")

# Ensure neighbor graph and UMAP exist
if "X_umap" not in adata.obsm:
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.umap(adata)

# Prepare feature matrices
X_expr = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
X_pca = adata.obsm["X_pca"]

# Encode labels
le = LabelEncoder()
y = le.fit_transform(adata.obs[label_col].values)
cell_type_names = le.classes_
num_classes = len(cell_type_names)
ground_truth = adata.obs[label_col].values

print(f"Dataset shape: {adata.shape}")
print(f"Cell types ({num_classes}): {list(cell_type_names)}")
print(f"Class distribution:\n{adata.obs[label_col].value_counts().to_string()}\n")

# Stratified train/test split (shared across all supervised methods)
X_train_expr, X_test_expr, y_train, y_test = train_test_split(
    X_expr, y, test_size=0.2, random_state=42, stratify=y
)
X_train_pca, X_test_pca, _, _ = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {X_train_expr.shape[0]}, Test size: {X_test_expr.shape[0]}")

# Collector for all results
supervised_results = []
all_predictions = {}  # store predictions for combined confusion matrix plot


# ══════════════════════════════════════════════════════════════════════════
# SECTION 1: LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 1: Logistic Regression")
print("=" * 60)

# --- 1a: L2 on full HVG expression ---
print("\n--- 1a: L2 Logistic Regression (HVG expression) ---")
t0 = time.time()
lr_expr = LogisticRegression(
    C=1.0, l1_ratio=0, solver="lbfgs", max_iter=2000, random_state=42
)
lr_expr.fit(X_train_expr, y_train)
y_pred_lr_expr = lr_expr.predict(X_test_expr)
t_lr_expr = time.time() - t0

acc = accuracy_score(y_test, y_pred_lr_expr)
report = classification_report(
    y_test, y_pred_lr_expr, target_names=cell_type_names, output_dict=True
)
print(classification_report(y_test, y_pred_lr_expr, target_names=cell_type_names))
supervised_results.append(
    {
        "method": "LR L2 (HVG)",
        "accuracy": acc,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "train_time_sec": round(t_lr_expr, 2),
    }
)
all_predictions["LR L2 (HVG)"] = y_pred_lr_expr

# --- 1b: L2 on PCA features ---
print("--- 1b: L2 Logistic Regression (PCA 50 components) ---")
t0 = time.time()
lr_pca = LogisticRegression(
    C=1.0, l1_ratio=0, solver="lbfgs", max_iter=2000, random_state=42
)
lr_pca.fit(X_train_pca, y_train)
y_pred_lr_pca = lr_pca.predict(X_test_pca)
t_lr_pca = time.time() - t0

acc = accuracy_score(y_test, y_pred_lr_pca)
report = classification_report(
    y_test, y_pred_lr_pca, target_names=cell_type_names, output_dict=True
)
print(classification_report(y_test, y_pred_lr_pca, target_names=cell_type_names))
supervised_results.append(
    {
        "method": "LR L2 (PCA)",
        "accuracy": acc,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "train_time_sec": round(t_lr_pca, 2),
    }
)
all_predictions["LR L2 (PCA)"] = y_pred_lr_pca

# --- 1c: L1 Logistic Regression (sparse feature selection) ---
print("--- 1c: L1 Logistic Regression (HVG expression) ---")
t0 = time.time()
lr_l1 = LogisticRegression(
    C=0.1, l1_ratio=1, solver="saga", max_iter=500, random_state=42, tol=1e-3
)
lr_l1.fit(X_train_expr, y_train)
y_pred_lr_l1 = lr_l1.predict(X_test_expr)
t_lr_l1 = time.time() - t0

n_nonzero = np.sum(np.any(lr_l1.coef_ != 0, axis=0))
acc = accuracy_score(y_test, y_pred_lr_l1)
report = classification_report(
    y_test, y_pred_lr_l1, target_names=cell_type_names, output_dict=True
)
print(classification_report(y_test, y_pred_lr_l1, target_names=cell_type_names))
print(f"Non-zero genes selected by L1: {n_nonzero} / {X_train_expr.shape[1]}")
supervised_results.append(
    {
        "method": "LR L1 (HVG)",
        "accuracy": acc,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "train_time_sec": round(t_lr_l1, 2),
    }
)
all_predictions["LR L1 (HVG)"] = y_pred_lr_l1

# 5-fold cross-validation for the best LR model
print("\n--- Cross-validation (LR L2 HVG) ---")
cv_scores = cross_val_score(
    LogisticRegression(C=1.0, l1_ratio=0, solver="lbfgs", max_iter=2000, random_state=42),
    X_expr,
    y,
    cv=5,
    scoring="accuracy",
)
print(f"5-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# --- Top marker genes from LR L2 coefficients ---
gene_names = adata.var_names
top_n = 10
print(f"\n--- Top {top_n} marker genes per cell type (LR L2 coefficients) ---")
for i, ct in enumerate(cell_type_names):
    top_idx = np.argsort(lr_expr.coef_[i])[::-1][:top_n]
    top_genes = gene_names[top_idx]
    print(f"  {ct}: {', '.join(top_genes)}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2: RANDOM FOREST
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 2: Random Forest")
print("=" * 60)

# --- 2a: RF on full HVG expression ---
print("\n--- 2a: Random Forest (HVG expression) ---")
t0 = time.time()
rf_expr = RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf_expr.fit(X_train_expr, y_train)
y_pred_rf_expr = rf_expr.predict(X_test_expr)
t_rf_expr = time.time() - t0

acc = accuracy_score(y_test, y_pred_rf_expr)
report = classification_report(
    y_test, y_pred_rf_expr, target_names=cell_type_names, output_dict=True
)
print(classification_report(y_test, y_pred_rf_expr, target_names=cell_type_names))
supervised_results.append(
    {
        "method": "RF (HVG)",
        "accuracy": acc,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "train_time_sec": round(t_rf_expr, 2),
    }
)
all_predictions["RF (HVG)"] = y_pred_rf_expr

# --- 2b: RF on PCA features ---
print("\n--- 2b: Random Forest (PCA features) ---")
t0 = time.time()
rf_pca = RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf_pca.fit(X_train_pca, y_train)
y_pred_rf_pca = rf_pca.predict(X_test_pca)
t_rf_pca = time.time() - t0

acc = accuracy_score(y_test, y_pred_rf_pca)
report = classification_report(
    y_test, y_pred_rf_pca, target_names=cell_type_names, output_dict=True
)
print(classification_report(y_test, y_pred_rf_pca, target_names=cell_type_names))
supervised_results.append(
    {
        "method": "RF (PCA)",
        "accuracy": acc,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "train_time_sec": round(t_rf_pca, 2),
    }
)
all_predictions["RF (PCA)"] = y_pred_rf_pca

# Top genes by RF feature importance
top_n = 10
top_idx = np.argsort(rf_expr.feature_importances_)[::-1][:top_n]
print(f"\n--- Top {top_n} genes by RF feature importance (HVG) ---")
for rank, idx in enumerate(top_idx, 1):
    print(f"  {rank:2d}. {gene_names[idx]:<12} "
          f"importance={rf_expr.feature_importances_[idx]:.4f}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 3: DEEP LEARNING (PyTorch MLP)
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 3: Deep Learning (MLP)")
print("=" * 60)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_expr)
X_test_scaled = scaler.transform(X_test_expr)

# Further split train into train/val (75/25 of train = 60/20 of total)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.25, random_state=42, stratify=y_train
)


class GeneExpressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Weighted sampler for class imbalance
counts = Counter(y_tr)
class_weights = {c: len(y_tr) / (num_classes * n) for c, n in counts.items()}
sample_weights = [class_weights[label] for label in y_tr]
sampler = WeightedRandomSampler(
    sample_weights, num_samples=len(sample_weights), replacement=True
)

train_loader = DataLoader(
    GeneExpressionDataset(X_tr, y_tr), batch_size=128, sampler=sampler
)
val_loader = DataLoader(
    GeneExpressionDataset(X_val, y_val), batch_size=256, shuffle=False
)
test_loader = DataLoader(
    GeneExpressionDataset(X_test_scaled, y_test), batch_size=256, shuffle=False
)


class CellTypeClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.network(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = CellTypeClassifier(X_tr.shape[1], num_classes).to(device)
weight_tensor = torch.FloatTensor(
    [class_weights.get(i, 1.0) for i in range(num_classes)]
).to(device)
criterion = nn.CrossEntropyLoss(weight=weight_tensor)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=10, factor=0.5
)

# Training loop
n_epochs = 100
best_val_acc = 0
patience_counter = 0
patience = 20
train_losses, val_accs = [], []

t0 = time.time()
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch).argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

    val_acc = correct / total
    val_accs.append(val_acc)
    scheduler.step(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "data/best_model.pt")
        patience_counter = 0
    else:
        patience_counter += 1

    if epoch % 10 == 0:
        print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

    if patience_counter >= patience:
        print(f"  Early stopping at epoch {epoch}")
        break

t_dl = time.time() - t0

# Load best model and evaluate on test set
model.load_state_dict(torch.load("data/best_model.pt", weights_only=True))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        preds = model(X_batch).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

y_pred_dl = np.array(all_preds)
y_test_dl = np.array(all_labels)

acc = accuracy_score(y_test_dl, y_pred_dl)
report = classification_report(
    y_test_dl, y_pred_dl, target_names=cell_type_names, output_dict=True
)
print(f"\n{classification_report(y_test_dl, y_pred_dl, target_names=cell_type_names)}")
supervised_results.append(
    {
        "method": "Deep Learning (MLP)",
        "accuracy": acc,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "train_time_sec": round(t_dl, 2),
    }
)
all_predictions["Deep Learning (MLP)"] = y_pred_dl

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(train_losses, color="steelblue")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training Loss")
axes[1].plot(val_accs, color="darkorange")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Validation Accuracy")
plt.tight_layout()
plt.savefig("figures/dl_training_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/dl_training_curves.png")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 4: CLUSTERING (K-means and Leiden)
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 4: Clustering")
print("=" * 60)

# Ensure neighbor graph exists (needed for Leiden)
if "neighbors" not in adata.uns:
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

clustering_results = []

# --- 3a: K-means with elbow / silhouette analysis ---
print("\n--- 3a: K-means Clustering ---")
K_range = range(2, 16)
inertias, silhouettes, kmeans_ari_scores = [], [], []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_pca)
    inertias.append(km.inertia_)
    silhouettes.append(
        silhouette_score(X_pca, labels, sample_size=min(2000, len(labels)), random_state=42)
    )
    kmeans_ari_scores.append(adjusted_rand_score(ground_truth, labels))

# Plot K selection
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].plot(K_range, inertias, "bo-")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow Method")
axes[1].plot(K_range, silhouettes, "ro-")
axes[1].set_xlabel("k")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Analysis")
axes[2].plot(K_range, kmeans_ari_scores, "go-")
axes[2].set_xlabel("k")
axes[2].set_ylabel("ARI")
axes[2].set_title("ARI vs Ground Truth")
axes[2].axvline(
    x=num_classes, color="gray", linestyle="--", label=f"True k={num_classes}"
)
axes[2].legend()
plt.tight_layout()
plt.savefig("figures/kmeans_k_selection.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/kmeans_k_selection.png")

# Best K-means (k = number of true cell types)
kmeans_final = KMeans(n_clusters=num_classes, random_state=42, n_init=20)
kmeans_labels = kmeans_final.fit_predict(X_pca)
adata.obs["kmeans"] = pd.Categorical(kmeans_labels.astype(str))

ari_km = adjusted_rand_score(ground_truth, kmeans_labels)
nmi_km = normalized_mutual_info_score(ground_truth, kmeans_labels)
sil_km = silhouette_score(X_pca, kmeans_labels, sample_size=min(2000, len(kmeans_labels)), random_state=42)
print(f"K-means (k={num_classes}): ARI={ari_km:.4f}, NMI={nmi_km:.4f}, Silhouette={sil_km:.4f}")
clustering_results.append(
    {
        "method": f"K-means (k={num_classes})",
        "n_clusters": num_classes,
        "ARI": ari_km,
        "NMI": nmi_km,
        "silhouette": sil_km,
    }
)

# --- 3b: Leiden clustering across resolutions ---
print("\n--- 3b: Leiden Clustering ---")
resolutions = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]

for res in resolutions:
    key = f"leiden_{res}"
    sc.tl.leiden(adata, resolution=res, key_added=key)
    n_cl = adata.obs[key].nunique()
    ari = adjusted_rand_score(ground_truth, adata.obs[key])
    nmi = normalized_mutual_info_score(ground_truth, adata.obs[key])
    print(f"  Leiden res={res}: {n_cl} clusters, ARI={ari:.4f}, NMI={nmi:.4f}")
    clustering_results.append(
        {
            "method": f"Leiden (res={res})",
            "n_clusters": n_cl,
            "ARI": ari,
            "NMI": nmi,
            "silhouette": np.nan,
        }
    )

# Find best Leiden resolution by ARI
clust_df = pd.DataFrame(clustering_results)
leiden_df = clust_df[clust_df["method"].str.startswith("Leiden")]
best_row = leiden_df.loc[leiden_df["ARI"].idxmax()]
best_res = float(best_row["method"].split("=")[1].rstrip(")"))
best_leiden_key = f"leiden_{best_res}"
adata.obs["leiden_best"] = adata.obs[best_leiden_key]

print(f"\nBest Leiden resolution: {best_res}")
print(f"  ARI={best_row['ARI']:.4f}, NMI={best_row['NMI']:.4f}, "
      f"Clusters={int(best_row['n_clusters'])}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 5: VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 5: Generating Figures")
print("=" * 60)

# --- Figure 1: Confusion matrices for all supervised methods ---
n_sup = len(all_predictions)
fig, axes = plt.subplots(1, n_sup, figsize=(7 * n_sup, 6))
if n_sup == 1:
    axes = [axes]

for ax, (name, y_pred) in zip(axes, all_predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=cell_type_names)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    a = accuracy_score(y_test, y_pred)
    ax.set_title(f"{name}\nAccuracy: {a:.4f}")

plt.tight_layout()
plt.savefig("figures/all_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/all_confusion_matrices.png")

# --- Figure 2: Clustering confusion heatmaps ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, (name, labels) in zip(
    axes,
    [
        ("K-means", adata.obs["kmeans"]),
        (f"Leiden (res={best_res})", adata.obs["leiden_best"]),
    ],
):
    ct = pd.crosstab(ground_truth, labels, normalize="index")
    sns.heatmap(ct, annot=True, fmt=".2f", cmap="Blues", ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Cell Type")
    ax.set_title(f"{name} vs Ground Truth")

plt.tight_layout()
plt.savefig("figures/clustering_heatmaps.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/clustering_heatmaps.png")

# --- Figure 3: UMAP comparison ---
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
sc.pl.umap(adata, color=label_col, ax=axes[0], show=False, title="Ground Truth", frameon=False)
sc.pl.umap(adata, color="kmeans", ax=axes[1], show=False, title="K-means", frameon=False)
sc.pl.umap(adata, color="leiden_best", ax=axes[2], show=False, title=f"Leiden (res={best_res})", frameon=False)
plt.tight_layout()
plt.savefig("figures/umap_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/umap_comparison.png")

# --- Figure 4: Supervised method comparison bar chart ---
sup_df = pd.DataFrame(supervised_results)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(sup_df))
width = 0.25

bars1 = ax.bar(x - width, sup_df["accuracy"], width, label="Accuracy", color="steelblue")
bars2 = ax.bar(x, sup_df["macro_f1"], width, label="Macro F1", color="darkorange")
bars3 = ax.bar(x + width, sup_df["weighted_f1"], width, label="Weighted F1", color="seagreen")

ax.set_ylabel("Score")
ax.set_title("Supervised Methods Comparison")
ax.set_xticks(x)
ax.set_xticklabels(sup_df["method"], rotation=20, ha="right")
ax.set_ylim(0, 1.08)
ax.legend()

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig("figures/supervised_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/supervised_comparison.png")

# --- Figure 5: Clustering comparison bar chart ---
# Take only K-means and best Leiden for clean comparison
clust_compare = clust_df[
    clust_df["method"].isin([f"K-means (k={num_classes})", f"Leiden (res={best_res})"])
].copy()

fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(clust_compare))
width = 0.3
bars1 = ax.bar(x - width / 2, clust_compare["ARI"], width, label="ARI", color="steelblue")
bars2 = ax.bar(x + width / 2, clust_compare["NMI"], width, label="NMI", color="darkorange")

ax.set_ylabel("Score")
ax.set_title("Clustering Methods Comparison")
ax.set_xticks(x)
ax.set_xticklabels(clust_compare["method"].values, rotation=10, ha="right")
ax.set_ylim(0, 1.08)
ax.legend()

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("figures/clustering_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/clustering_comparison.png")

# --- Figure 6: Leiden resolution scan ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(resolutions, leiden_df["ARI"].values, "o-", color="steelblue", label="ARI")
ax.plot(resolutions, leiden_df["NMI"].values, "s-", color="darkorange", label="NMI")
ax.axvline(x=best_res, color="gray", linestyle="--", alpha=0.7, label=f"Best res={best_res}")
ax.set_xlabel("Resolution")
ax.set_ylabel("Score")
ax.set_title("Leiden Clustering: ARI/NMI vs Resolution")
ax.legend()
plt.tight_layout()
plt.savefig("figures/leiden_resolution_scan.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/leiden_resolution_scan.png")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 6: FINAL SUMMARY TABLE (+ optional transformer merge)
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 6: FINAL SUMMARY")
print("=" * 60)

# Supervised results
print("\n╔══════════════════════════════════════════════════════════════════════════╗")
print("║                     SUPERVISED METHOD RESULTS                          ║")
print("╠══════════════════════════════════════════════════════════════════════════╣")
sup_display = sup_df[["method", "accuracy", "macro_f1", "weighted_f1", "train_time_sec"]].copy()
sup_display.columns = ["Method", "Accuracy", "Macro F1", "Weighted F1", "Time (s)"]
for col in ["Accuracy", "Macro F1", "Weighted F1"]:
    sup_display[col] = sup_display[col].map(lambda x: f"{x:.4f}")
print(sup_display.to_string(index=False))
print("╚══════════════════════════════════════════════════════════════════════════╝")

print(f"\n5-fold CV Accuracy (LR L2 HVG): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"L1 Selected Genes: {n_nonzero} / {X_expr.shape[1]}")

# Clustering results
print("\n╔══════════════════════════════════════════════════════════════════════════╗")
print("║                     CLUSTERING METHOD RESULTS                          ║")
print("╠══════════════════════════════════════════════════════════════════════════╣")
clust_display = clust_compare[["method", "n_clusters", "ARI", "NMI"]].copy()
clust_display.columns = ["Method", "Clusters", "ARI", "NMI"]
for col in ["ARI", "NMI"]:
    clust_display[col] = clust_display[col].map(lambda x: f"{x:.4f}")
print(clust_display.to_string(index=False))
print("╚══════════════════════════════════════════════════════════════════════════╝")

# Save results to CSV
sup_df.to_csv("figures/supervised_results.csv", index=False)
clust_df.to_csv("figures/clustering_results.csv", index=False)
print("\nSaved: figures/supervised_results.csv")
print("Saved: figures/clustering_results.csv")

# ── Merge transformer results from 05_transformer_based.py if present ──
transformer_csv = "figures/transformer_results.csv"
if os.path.exists(transformer_csv):
    fm_df = pd.read_csv(transformer_csv)
    keep_cols = [c for c in ["method", "accuracy", "macro_f1", "weighted_f1"] if c in fm_df.columns]
    combined = pd.concat(
        [sup_df[keep_cols], fm_df[keep_cols]],
        ignore_index=True,
    )
    combined.to_csv("figures/all_methods_results.csv", index=False)
    print(f"Merged {len(fm_df)} transformer rows → figures/all_methods_results.csv")

    # Unified bar chart across baselines + foundation models
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(combined))
    width = 0.25
    ax.bar(x - width, combined["accuracy"], width, label="Accuracy", color="steelblue")
    ax.bar(x, combined["macro_f1"], width, label="Macro F1", color="darkorange")
    ax.bar(x + width, combined["weighted_f1"], width, label="Weighted F1", color="seagreen")
    ax.set_ylabel("Score")
    ax.set_title("All Methods: Baselines + Foundation Models")
    ax.set_xticks(x)
    ax.set_xticklabels(combined["method"], rotation=30, ha="right")
    ax.set_ylim(0, 1.08)
    ax.legend()
    plt.tight_layout()
    plt.savefig("figures/all_methods_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/all_methods_comparison.png")
else:
    print(f"\n(No {transformer_csv} found — run 05_transformer_based.py to add "
          f"foundation-model results to the unified comparison.)")

print("\n" + "=" * 60)
print("ALL DONE! Check the 'figures/' folder for all plots and CSVs.")
print("=" * 60)

# List all generated files
print("\nGenerated files:")
for f in sorted(os.listdir("figures")):
    size = os.path.getsize(f"figures/{f}") / 1024
    print(f"  figures/{f}  ({size:.1f} KB)")