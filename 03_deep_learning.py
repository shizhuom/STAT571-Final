# 02_deep_learning.py
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, ConfusionMatrixDisplay)
from collections import Counter

# ── Load data ──
adata = sc.read('data/pbmc3k_processed.h5ad')
X = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.array(adata.X)

le = LabelEncoder()
y = le.fit_transform(adata.obs['louvain'].values)
cell_type_names = le.classes_
num_classes = len(cell_type_names)

# ── Scale features (important for neural networks) ──
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Train/val/test split (60/20/20) ──
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# ── PyTorch Dataset ──
class GeneExpressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ── Handle class imbalance with WeightedRandomSampler ──
counts = Counter(y_train)
class_weights = {c: len(y_train) / (num_classes * n) for c, n in counts.items()}
sample_weights = [class_weights[label] for label in y_train]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                replacement=True)

train_dataset = GeneExpressionDataset(X_train, y_train)
val_dataset = GeneExpressionDataset(X_val, y_val)
test_dataset = GeneExpressionDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# ── Neural Network Architecture ──
class CellTypeClassifier(nn.Module):
    """
    Single-hidden-layer MLP. PBMC 3k is small (~2700 cells, 8 classes);
    a deeper network overfits — one hidden layer is enough.
    """
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

# ── Initialize model, loss, optimizer ──
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CellTypeClassifier(X_train.shape[1], num_classes).to(device)

# Weighted cross-entropy loss for class imbalance
weight_tensor = torch.FloatTensor(
    [class_weights[i] for i in range(num_classes)]
).to(device)
criterion = nn.CrossEntropyLoss(weight=weight_tensor)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                  patience=10, factor=0.5)

# ── Training loop with early stopping ──
n_epochs = 100
best_val_acc = 0
patience_counter = 0
patience = 20
train_losses, val_accs = [], []

for epoch in range(n_epochs):
    # Training phase
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

    # Validation phase
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
        torch.save(model.state_dict(), 'data/best_model.pt')
        patience_counter = 0
    else:
        patience_counter += 1

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

# ── Load best model and evaluate on test set ──
model.load_state_dict(torch.load('data/best_model.pt'))
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
test_acc = accuracy_score(y_test_dl, y_pred_dl)

print(f"\n{'='*50}")
print(f"Deep Learning Test Accuracy: {test_acc:.4f}")
print(f"{'='*50}")
print(classification_report(y_test_dl, y_pred_dl, target_names=cell_type_names))

# ── Plot training curves ──
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(train_losses)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].set_title('Training Loss')
axes[1].plot(val_accs)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy'); axes[1].set_title('Validation Accuracy')
plt.tight_layout()
plt.savefig('figures/dl_training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Confusion matrix ──
cm = confusion_matrix(y_test_dl, y_pred_dl)
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(cm, display_labels=cell_type_names)
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
ax.set_title('Deep Learning Classifier')
plt.tight_layout()
plt.savefig('figures/dl_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()