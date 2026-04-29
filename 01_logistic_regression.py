# 01_logistic_regression.py
import scanpy as sc
import numpy as np
import anndata as ad
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, ConfusionMatrixDisplay)

# ── Load data ──
adata = ad.read_h5ad('data/pbmc3k_processed.h5ad')

# ── Prepare features and labels ──
# Option A: Use full gene expression matrix (1838 HVGs)
X_expr = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.array(adata.X)

# Option B: Use PCA-reduced features (50 components)
X_pca = adata.obsm['X_pca']

# Encode cell type labels
le = LabelEncoder()
y = le.fit_transform(adata.obs['louvain'].values)
cell_type_names = le.classes_

print(f"Features (expression): {X_expr.shape}")
print(f"Features (PCA): {X_pca.shape}")
print(f"Cell types ({len(cell_type_names)}): {list(cell_type_names)}")

# ── Train/test split (stratified to preserve cell type proportions) ──
X_train, X_test, y_train, y_test = train_test_split(
    X_expr, y, test_size=0.2, random_state=42, stratify=y
)
X_train_pca, X_test_pca, _, _ = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)

# ── Model 1a: Logistic Regression on full HVG expression ──
lr_expr = LogisticRegression(
    C=1.0,
    penalty='l2',
    solver='lbfgs',
    max_iter=2000,
    random_state=42,
    n_jobs=-1
)
lr_expr.fit(X_train, y_train)
y_pred_expr = lr_expr.predict(X_test)
acc_expr = accuracy_score(y_test, y_pred_expr)

# ── Model 1b: Logistic Regression on PCA features ──
lr_pca = LogisticRegression(
    C=1.0,
    penalty='l2',
    solver='lbfgs',
    max_iter=2000,
    random_state=42,
    n_jobs=-1
)
lr_pca.fit(X_train_pca, y_train)
y_pred_pca = lr_pca.predict(X_test_pca)
acc_pca = accuracy_score(y_test, y_pred_pca)

print(f"\n{'='*50}")
print(f"Logistic Regression (HVG expression): {acc_expr:.4f}")
print(f"Logistic Regression (PCA 50 components): {acc_pca:.4f}")
print(f"{'='*50}")

# ── Classification report ──
print("\n--- Classification Report (HVG expression) ---")
print(classification_report(y_test, y_pred_expr, target_names=cell_type_names))

# ── Cross-validation ──
cv_scores = cross_val_score(lr_expr, X_expr, y, cv=5, scoring='accuracy')
print(f"5-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Confusion matrix visualization ──
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for ax, y_pred, title in zip(axes,
                              [y_pred_expr, y_pred_pca],
                              ['LR (HVG Expression)', 'LR (PCA Features)']):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=cell_type_names)
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45, colorbar=False)
    ax.set_title(title)

plt.tight_layout()
plt.savefig('figures/lr_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Extract top marker genes per cell type from LR coefficients ──
# lr_expr.coef_ shape: (n_classes, n_features)
gene_names = adata.var_names
top_n = 10
print(f"\n--- Top {top_n} marker genes per cell type (by LR coefficient) ---")
for i, ct in enumerate(cell_type_names):
    top_idx = np.argsort(lr_expr.coef_[i])[::-1][:top_n]
    top_genes = gene_names[top_idx]
    print(f"{ct}: {', '.join(top_genes)}")

# ── Model comparison with L1 regularization (sparse feature selection) ──
lr_l1 = LogisticRegression(
    C=0.1,
    l1_ratio=1,        
    solver='saga',
    max_iter=3000,
    random_state=42
)
lr_l1.fit(X_train, y_train)
y_pred_l1 = lr_l1.predict(X_test)
n_nonzero = np.sum(np.any(lr_l1.coef_ != 0, axis=0))
print(f"\nL1 Logistic Regression accuracy: {accuracy_score(y_test, y_pred_l1):.4f}")
print(f"Non-zero genes selected by L1: {n_nonzero} / {X_train.shape[1]}")

lr_l1.fit(X_train_pca, y_train)
y_pred_l1_pca = lr_l1.predict(X_test_pca)
print(f"L1 Logistic Regression (PCA) accuracy: {accuracy_score(y_test, y_pred_l1_pca):.4f}")