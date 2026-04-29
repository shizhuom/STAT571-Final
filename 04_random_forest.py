# 04_random_forest.py
import scanpy as sc
import numpy as np
import anndata as ad
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, ConfusionMatrixDisplay)

# ── Load data ──
adata = ad.read_h5ad('data/pbmc3k_processed.h5ad')

# Auto-detect label column (matches the convention in 06_final_comparison.py)
label_col = 'louvain' if 'louvain' in adata.obs.columns else 'cell_type'

# ── Prepare features and labels ──
X_expr = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.array(adata.X)
X_pca = adata.obsm['X_pca']

le = LabelEncoder()
y = le.fit_transform(adata.obs[label_col].values)
cell_type_names = le.classes_

print(f"Features (expression): {X_expr.shape}")
print(f"Features (PCA): {X_pca.shape}")
print(f"Cell types ({len(cell_type_names)}): {list(cell_type_names)}")

# ── Train/test split (shared random_state with the rest of the pipeline) ──
X_train, X_test, y_train, y_test = train_test_split(
    X_expr, y, test_size=0.2, random_state=42, stratify=y
)
X_train_pca, X_test_pca, _, _ = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)

# ── Model A: Random Forest on full HVG expression ──
rf_expr = RandomForestClassifier(
    n_estimators=500,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
)
rf_expr.fit(X_train, y_train)
y_pred_expr = rf_expr.predict(X_test)
acc_expr = accuracy_score(y_test, y_pred_expr)

# ── Model B: Random Forest on PCA features ──
rf_pca = RandomForestClassifier(
    n_estimators=500,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
)
rf_pca.fit(X_train_pca, y_train)
y_pred_pca = rf_pca.predict(X_test_pca)
acc_pca = accuracy_score(y_test, y_pred_pca)

print(f"\n{'='*50}")
print(f"Random Forest (HVG expression): {acc_expr:.4f}")
print(f"Random Forest (PCA 50 components): {acc_pca:.4f}")
print(f"{'='*50}")

print("\n--- Classification Report (HVG expression) ---")
print(classification_report(y_test, y_pred_expr, target_names=cell_type_names))

# ── 5-fold cross-validation on HVG features ──
cv_scores = cross_val_score(rf_expr, X_expr, y, cv=5, scoring='accuracy', n_jobs=-1)
print(f"5-fold CV accuracy (HVG): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Confusion matrices ──
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, y_pred, title in zip(axes,
                             [y_pred_expr, y_pred_pca],
                             ['RF (HVG Expression)', 'RF (PCA Features)']):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=cell_type_names)
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45, colorbar=False)
    ax.set_title(title)
plt.tight_layout()
plt.savefig('figures/rf_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Top genes by RF feature importance (HVG model) ──
gene_names = adata.var_names
top_n = 20
importances = rf_expr.feature_importances_
top_idx = np.argsort(importances)[::-1][:top_n]
print(f"\n--- Top {top_n} genes by RF feature importance (HVG) ---")
for rank, idx in enumerate(top_idx, 1):
    print(f"  {rank:2d}. {gene_names[idx]:<12} importance={importances[idx]:.4f}")

# Bar plot of top importances
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(top_n), importances[top_idx][::-1], color='steelblue')
ax.set_yticks(range(top_n))
ax.set_yticklabels(gene_names[top_idx][::-1])
ax.set_xlabel('Feature Importance')
ax.set_title(f'Top {top_n} Genes by Random Forest Importance')
plt.tight_layout()
plt.savefig('figures/rf_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
