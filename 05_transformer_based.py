"""
=============================================================================
  ADVANCED METHOD: Foundation Model Embeddings for Cell Type Prediction
  Using scGPT and Geneformer on PBMC 3k Data
=============================================================================

IMPORTANT NOTES:
  - Both scGPT and Geneformer benefit greatly from GPU access.
  - If you don't have a local GPU, use Google Colab (free T4 GPU).
  - This script is designed to run on Google Colab OR a local machine.
  - scGPT embeddings are 512-dimensional; Geneformer embeddings are 256/512-d.
  - We extract embeddings and then feed them into the same classifiers
    (Logistic Regression, MLP) from your existing pipeline for comparison.

SETUP (run in terminal or Colab cell BEFORE running this script):

  # ── For scGPT ──
  pip install scgpt scanpy gdown torch

  # ── For Geneformer ──
  # Geneformer is NOT on PyPI — install from HuggingFace:
  # (requires git-lfs: https://git-lfs.com)
  git lfs install
  git clone https://huggingface.co/ctheodoris/Geneformer
  cd Geneformer
  pip install .

  # ── Shared dependencies ──
  pip install scikit-learn matplotlib seaborn pandas numpy

=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import time
import anndata


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")
os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# PART A: scGPT EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════
#
# scGPT (Cui et al., Nature Methods 2024) is a generative pretrained
# transformer trained on 33 million single cells. It produces 512-dim
# cell embeddings in a zero-shot setting (no fine-tuning needed).
#
# Key concept for your report:
#   scGPT treats genes like "words" and cells like "sentences". It uses
#   a masked gene prediction objective during pretraining — analogous to
#   masked language modeling in NLP (like BERT). The resulting embeddings
#   capture rich biological context that standard PCA cannot.
# ══════════════════════════════════════════════════════════════════════════

def run_scgpt_embeddings():
    """
    Extract scGPT cell embeddings from PBMC 3k data.
    Returns an AnnData with embeddings in .obsm['X_scGPT'].
    """
    import scgpt as scg
    from pathlib import Path
    import torch

    print("=" * 60)
    print("PART A: Extracting scGPT Embeddings")
    print("=" * 60)

    # ── Step 1: Download the scGPT pretrained model checkpoint ──
    # The whole-human model (33M cells) is recommended for general use.
    # There is also a blood-specific model (10.3M blood/bone marrow cells).
    # We use the continual-pretrained (CP) model which works best for
    # cell embedding tasks in zero-shot mode.
    #
    # Google Drive folder ID for scGPT_CP model:
    #   1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y
    #
    # You can also use the whole-human model:
    #   1_GROJTzXiAV8HB4imruOTk6PEGuNOcgB  (not CP, slightly different)

    model_dir = Path("./scGPT_human")

    if not model_dir.exists():
        print("Downloading scGPT model checkpoint (~100 MB)...")
        import gdown
        folder_id = "1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y"
        gdown.download_folder(id=folder_id, output=str(model_dir), quiet=False)
    else:
        print(f"Model checkpoint found at {model_dir}")

    # ── Step 2: Load and prepare PBMC 3k data ──
    # scGPT needs RAW counts (not scaled), log-normalized, with gene names
    # matching the scGPT vocabulary (HGNC gene symbols).
    #
    # IMPORTANT: scGPT's embed_data function handles preprocessing internally.
    # We need to provide data with gene names as var_names.

    # Load from raw if possible (scGPT prefers minimally processed data)
    adata = sc.read("data/pbmc3k_processed.h5ad")

    # The processed data should have gene symbols as var_names already.
    # scGPT's embed_data will select HVGs and handle the rest.
    print(f"Loaded data: {adata.shape}")
    print(f"Gene name examples: {list(adata.var_names[:5])}")

    # Detect label column
    label_col = "louvain" if "louvain" in adata.obs.columns else "cell_type"

    # ── Step 3: Generate embeddings ──
    # embed_data() does the following internally:
    #   1. Selects top 3000 HVGs that overlap with scGPT vocabulary
    #   2. Tokenizes each cell (ranks genes by expression)
    #   3. Passes through the pretrained transformer
    #   4. Returns 512-dim cell embeddings
    #
    # gene_col: column in adata.var that has gene names
    #   For most scanpy datasets, gene names ARE the index, so use "index"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: CPU inference is slow (~5-10 min for 2700 cells).")
        print("Consider using Google Colab with a free T4 GPU for faster results.")

    t0 = time.time()
    embed_adata = scg.tasks.embed_data(
        adata,
        model_dir,
        gene_col="index",            # gene names are in adata.var_names (the index)
        obs_to_save=label_col,        # keep cell type labels in output
        batch_size=64,
        device=device,
        use_fast_transformer=False,   # set True only if flash-attn installed
        return_new_adata=True,        # return a new AnnData with embeddings as .X
    )
    t_scgpt = time.time() - t0
    print(f"scGPT embedding extraction took {t_scgpt:.1f} seconds")

    # embed_adata.X now contains 512-dim embeddings (cells × 512)
    # Also stored in the original adata.obsm["X_scGPT"] if return_new_adata=False
    print(f"Embedding shape: {embed_adata.X.shape}")

    # Save embeddings for reuse
    np.save("data/scgpt_embeddings.npy", embed_adata.X)
    print("Saved embeddings to data/scgpt_embeddings.npy")

    return embed_adata.X, adata.obs[label_col].values


def run_scgpt_classification(X_scgpt, labels):
    """
    Use scGPT embeddings as features for cell type classification.
    This directly compares with PCA features from your baseline methods.
    """
    print("\n--- scGPT Embedding Classification ---")

    le = LabelEncoder()
    y = le.fit_transform(labels)
    cell_type_names = le.classes_

    # Train/test split (same random state as main pipeline for fair comparison)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scgpt, y, test_size=0.2, random_state=42, stratify=y
    )

    results = []

    # Method A: Logistic Regression on scGPT embeddings
    lr = LogisticRegression(
        C=1.0, l1_ratio=0, solver="lbfgs", max_iter=2000, random_state=42
    )
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    report_lr = classification_report(
        y_test, y_pred_lr, target_names=cell_type_names, output_dict=True
    )
    print(f"\nLR on scGPT embeddings — Accuracy: {acc_lr:.4f}")
    print(classification_report(y_test, y_pred_lr, target_names=cell_type_names))
    results.append({
        "method": "LR + scGPT Embed",
        "accuracy": acc_lr,
        "macro_f1": report_lr["macro avg"]["f1-score"],
        "weighted_f1": report_lr["weighted avg"]["f1-score"],
    })

    # Method B: MLP on scGPT embeddings (using sklearn for simplicity)
    from sklearn.neural_network import MLPClassifier

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
    )
    mlp.fit(X_train_s, y_train)
    y_pred_mlp = mlp.predict(X_test_s)
    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    report_mlp = classification_report(
        y_test, y_pred_mlp, target_names=cell_type_names, output_dict=True
    )
    print(f"MLP on scGPT embeddings — Accuracy: {acc_mlp:.4f}")
    print(classification_report(y_test, y_pred_mlp, target_names=cell_type_names))
    results.append({
        "method": "MLP + scGPT Embed",
        "accuracy": acc_mlp,
        "macro_f1": report_mlp["macro avg"]["f1-score"],
        "weighted_f1": report_mlp["weighted avg"]["f1-score"],
    })

    # Confusion matrix for scGPT + LR
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, y_pred, title in zip(
        axes,
        [y_pred_lr, y_pred_mlp],
        ["LR + scGPT Embeddings", "MLP + scGPT Embeddings"],
    ):
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=cell_type_names)
        disp.plot(ax=ax, cmap="Greens", xticks_rotation=45, colorbar=False)
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig("figures/scgpt_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/scgpt_confusion_matrices.png")

    return results, y_test, y_pred_lr, cell_type_names


# ══════════════════════════════════════════════════════════════════════════
# PART B: GENEFORMER EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════
#
# Geneformer (Theodoris et al., Nature 2023) is a BERT-style transformer
# pretrained on ~30M (V1) or ~104M (V2) single-cell transcriptomes.
# It uses a unique "rank value encoding" — genes are ranked by their
# expression level in each cell, and the rank order is the input sequence.
#
# Key differences from scGPT for your report:
#   - Geneformer uses BIDIRECTIONAL attention (like BERT)
#   - scGPT uses UNIDIRECTIONAL masked attention (like GPT)
#   - Geneformer input: rank-ordered gene tokens (no expression values)
#   - scGPT input: gene tokens + expression value embeddings
#   - Geneformer requires a tokenization step that converts h5ad → .dataset
# ══════════════════════════════════════════════════════════════════════════

def run_geneformer_embeddings():
    """
    Extract Geneformer cell embeddings from PBMC 3k data.

    Geneformer requires more setup than scGPT:
      1. Data must have Ensembl IDs (not gene symbols)
      2. Raw counts must be available
      3. Data must be tokenized into a HuggingFace .dataset format
      4. Then embeddings are extracted via EmbExtractor
    """
    from geneformer import TranscriptomeTokenizer, EmbExtractor
    from pathlib import Path
    import torch

    print("\n" + "=" * 60)
    print("PART B: Extracting Geneformer Embeddings")
    print("=" * 60)

    # ── Step 1: Prepare data with Ensembl IDs ──
    # Geneformer requires Ensembl gene IDs, not HGNC symbols.
    # We need to convert gene names like "CD3D" → "ENSG00000167286".
    #
    # The easiest approach: load raw PBMC 3k and add Ensembl IDs.

    print("Loading and preparing data...")
    adata = sc.read("data/pbmc3k_processed.h5ad")
    label_col = "louvain" if "louvain" in adata.obs.columns else "cell_type"

    # We need a mapping from gene symbols to Ensembl IDs.
    # Geneformer provides a token dictionary; alternatively, use biomart.
    # Here we use a quick approach with the built-in Geneformer files.

    # The Geneformer repo includes a token dictionary mapping Ensembl → token.
    # We need the reverse: symbol → Ensembl.
    # The simplest approach is to use the gene_info.csv from scGPT or
    # query a mapping. For PBMC 3k, most genes have standard HGNC symbols.

    # Quick approach: try to load gene name mapping
    try:
        # If you have biomart/mygene installed:
        import mygene
        mg = mygene.MyGeneInfo()
        gene_symbols = list(adata.var_names)
        query = mg.querymany(
            gene_symbols, scopes="symbol", fields="ensembl.gene",
            species="human", returnall=True
        )
        symbol_to_ensembl = {}
        for hit in query["out"]:
            if "ensembl" in hit:
                ens = hit["ensembl"]
                if isinstance(ens, list):
                    ens = ens[0]
                symbol_to_ensembl[hit["query"]] = ens["gene"]
    except ImportError:
        print("mygene not installed. Using alternative mapping approach.")
        print("Install with: pip install mygene")
        print("Or download a gene mapping file.")

        # Alternative: hardcoded mapping for common PBMC genes
        # For a full run, you MUST install mygene or provide a mapping file.
        # Here we show the pattern; see instructions below.
        symbol_to_ensembl = None

    if symbol_to_ensembl is None:
        print("\nCannot proceed without Ensembl ID mapping.")
        print("Please install mygene: pip install mygene")
        print("Then re-run this script.")
        return None, None

    # Add Ensembl IDs to adata
    adata.var["ensembl_id"] = adata.var_names.map(symbol_to_ensembl)
    # Drop genes without Ensembl mapping
    mask = adata.var["ensembl_id"].notna()
    adata = adata[:, mask].copy()
    print(f"Genes with Ensembl IDs: {mask.sum()} / {len(mask)}")

    # ── Step 2: Tokenize ──
    # Geneformer's TranscriptomeTokenizer converts h5ad → HuggingFace dataset.
    # It expects: adata.X to contain raw or normalized counts,
    #             adata.var to have "ensembl_id" column.

    # Save processed adata for tokenizer
    token_input = "data/pbmc3k_for_geneformer.h5ad"
    import anndata
    anndata.settings.allow_write_nullable_strings = True
    adata.write(token_input)

    token_output = "data/geneformer_tokens"
    os.makedirs(token_output, exist_ok=True)

    tokenizer = TranscriptomeTokenizer(
        custom_attr_name_dict={label_col: label_col},  # keep cell type labels
        nproc=4,
        model_input_size=2048,  # max genes per cell
    )
    tokenizer.tokenize_data(
        data_directory=os.path.dirname(token_input),
        output_directory=token_output,
        output_prefix="pbmc",
        file_format="h5ad",
    )
    print(f"Tokenized data saved to {token_output}")

    # ── Step 3: Extract embeddings ──
    # Use the pretrained Geneformer model (not fine-tuned).
    # Model path: the cloned Geneformer repo contains model weights.

    model_dir = "Geneformer"  # path to cloned Geneformer repo
    if not os.path.exists(model_dir):
        print(f"\nGeneformer model not found at '{model_dir}'.")
        print("Please clone it first:")
        print("  git lfs install")
        print("  git clone https://huggingface.co/ctheodoris/Geneformer")
        return None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    emb_output = "data/geneformer_embeddings"
    os.makedirs(emb_output, exist_ok=True)

    t0 = time.time()
    embex = EmbExtractor(
        model_type="Pretrained",
        num_classes=0,
        emb_mode="cell",            # "cell" for cell embeddings, "gene" for gene emb
        cell_emb_style="mean_pool",  # average gene embeddings for cell embedding
        max_ncells=None,             # use all cells (default is 1000)
        emb_layer=-2,               # 2nd-to-last layer (more general features)
        emb_label=[label_col],       # keep labels in output
        forward_batch_size=64,
        nproc=4,
    )

    embs = embex.extract_embs(
        model_directory=model_dir,
        input_data_file=os.path.join(token_output, "pbmc.dataset"),
        output_directory=emb_output,
        output_prefix="pbmc_emb",
    )
    t_gf = time.time() - t0
    print(f"Geneformer embedding extraction took {t_gf:.1f} seconds")

    # embs is a pandas DataFrame with cell embeddings + labels
    # Embedding columns are numeric (0, 1, 2, ..., 255 or 511)
    emb_cols = [c for c in embs.columns if isinstance(c, int) or c.isdigit()]
    X_gf = embs[emb_cols].values
    labels = embs[label_col].values

    print(f"Geneformer embedding shape: {X_gf.shape}")
    np.save("data/geneformer_embeddings.npy", X_gf)
    print("Saved to data/geneformer_embeddings.npy")

    return X_gf, labels


def run_geneformer_classification(X_gf, labels):
    """Classify cell types using Geneformer embeddings."""
    print("\n--- Geneformer Embedding Classification ---")

    le = LabelEncoder()
    y = le.fit_transform(labels)
    cell_type_names = le.classes_

    X_train, X_test, y_train, y_test = train_test_split(
        X_gf, y, test_size=0.2, random_state=42, stratify=y
    )

    results = []

    # LR on Geneformer embeddings
    lr = LogisticRegression(
        C=1.0, l1_ratio=0, solver="lbfgs", max_iter=2000, random_state=42
    )
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=cell_type_names, output_dict=True
    )
    print(f"LR on Geneformer embeddings — Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=cell_type_names))
    results.append({
        "method": "LR + Geneformer Embed",
        "accuracy": acc,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    })

    return results


# ══════════════════════════════════════════════════════════════════════════
# PART C: SAVE TRANSFORMER RESULTS FOR FINAL COMPARISON
# ══════════════════════════════════════════════════════════════════════════

def save_transformer_results(scgpt_results, geneformer_results=None):
    """
    Persist foundation-model results to figures/transformer_results.csv so
    that 06_final_comparison.py can merge them into the unified benchmark.
    """
    print("\n" + "=" * 60)
    print("PART C: Saving transformer results")
    print("=" * 60)

    fm_results = list(scgpt_results)
    if geneformer_results:
        fm_results.extend(geneformer_results)

    if not fm_results:
        print("No foundation-model results to save.")
        return None

    results_df = pd.DataFrame(fm_results)

    print("\n" + "=" * 70)
    print("FOUNDATION-MODEL RESULTS")
    print("=" * 70)
    display_df = results_df.copy()
    for col in ["accuracy", "macro_f1", "weighted_f1"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda x: f"{x:.4f}")
    print(display_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(results_df))
    width = 0.25
    ax.bar(x - width, results_df["accuracy"], width, label="Accuracy", color="steelblue")
    ax.bar(x, results_df["macro_f1"], width, label="Macro F1", color="darkorange")
    ax.bar(x + width, results_df["weighted_f1"], width, label="Weighted F1", color="seagreen")
    ax.set_ylabel("Score")
    ax.set_title("Foundation-Model Embeddings + Linear Classifier")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["method"], rotation=20, ha="right")
    ax.set_ylim(0, 1.08)
    ax.legend()
    plt.tight_layout()
    plt.savefig("figures/transformer_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved: figures/transformer_comparison.png")

    results_df.to_csv("figures/transformer_results.csv", index=False)
    print("Saved: figures/transformer_results.csv")

    return results_df


# ══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 60)
    print("Foundation Model Embeddings for Cell Type Prediction")
    print("=" * 60)

    # ── Run scGPT ──
    scgpt_results = []
    try:
        X_scgpt, labels_scgpt = run_scgpt_embeddings()
        scgpt_results, _, _, _ = run_scgpt_classification(X_scgpt, labels_scgpt)
    except ImportError as e:
        print(f"\nscGPT not installed: {e}")
        print("Install with: pip install scgpt gdown")
        print("Skipping scGPT...\n")
    except Exception as e:
        print(f"\nscGPT failed: {e}")
        print("If embeddings were previously saved, loading from file...")
        if os.path.exists("data/scgpt_embeddings.npy"):
            X_scgpt = np.load("data/scgpt_embeddings.npy")
            adata = sc.read("data/pbmc3k_processed.h5ad")
            lc = "louvain" if "louvain" in adata.obs.columns else "cell_type"
            scgpt_results, _, _, _ = run_scgpt_classification(
                X_scgpt, adata.obs[lc].values
            )

    # ── Run Geneformer ──
    geneformer_results = []
    try:
        X_gf, labels_gf = run_geneformer_embeddings()
        if X_gf is not None:
            geneformer_results = run_geneformer_classification(X_gf, labels_gf)
    except ImportError as e:
        print(f"\nGeneformer not installed: {e}")
        print("Install from: https://huggingface.co/ctheodoris/Geneformer")
        print("Skipping Geneformer...\n")
    except Exception as e:
        print(f"\nGeneformer failed: {e}")
        if os.path.exists("data/geneformer_embeddings.npy"):
            X_gf = np.load("data/geneformer_embeddings.npy")
            adata = sc.read("data/pbmc3k_processed.h5ad")
            lc = "louvain" if "louvain" in adata.obs.columns else "cell_type"
            geneformer_results = run_geneformer_classification(
                X_gf, adata.obs[lc].values
            )

    # ── Save transformer results for 06_final_comparison.py ──
    if scgpt_results or geneformer_results:
        save_transformer_results(scgpt_results, geneformer_results)
    else:
        print("\nNo foundation model results generated.")
        print("Please install scGPT and/or Geneformer first.")

    print("\nDone!")
    