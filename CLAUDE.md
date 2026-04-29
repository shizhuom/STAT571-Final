# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Single-cell RNA-seq cell-type prediction study on the **PBMC 3k** dataset (course final project). Six numbered scripts compare four baseline paradigms — Logistic Regression, Clustering (K-means / Leiden), a PyTorch MLP, and Random Forest — against foundation-model cell embeddings (scGPT, Geneformer). Everything is flat Python scripts; there is no package, CLI, or test suite.

## Running the scripts

```bash
# One-time env setup (see requirements.txt header for rationale)
conda create -n scrna_full python=3.10 -y
conda activate scrna_full
pip install -r requirements.txt
pip uninstall -y torchtext torchdata   # REQUIRED: broken DLLs otherwise

# Full pipeline (run in order — each script reads data/pbmc3k_processed.h5ad)
python 00_preprocessing.py         # writes data/pbmc3k_processed.h5ad
python 01_logistic_regression.py   # LR (L2/L1) on HVG and PCA features
python 02_clustering.py            # K-means + Leiden sweep → figures/
python 03_deep_learning.py         # single-hidden-layer MLP → data/best_model.pt
python 04_random_forest.py         # RF on HVG + PCA → figures/rf_*.png
python 05_transformer_based.py     # scGPT + Geneformer → figures/transformer_results.csv
python 06_final_comparison.py      # self-contained re-run of 01–04 → figures/supervised_results.csv (+ merges transformer_results.csv if present)
```

Scripts 01–05 are standalone exploratory runs. `06_final_comparison.py` is a self-contained re-run of all four baselines (LR, Clustering, MLP, RF) in one process that writes `figures/supervised_results.csv` and `figures/clustering_results.csv`. If `figures/transformer_results.csv` exists from a prior `05_transformer_based.py` run, `06_` merges those rows and writes a unified `figures/all_methods_results.csv` plus `figures/all_methods_comparison.png`. Run `05_` before `06_` to get the unified benchmark.

No tests, linter, or build step exists. Running a script *is* the test — verify by inspecting `figures/` outputs.

## Architecture and conventions

**Shared inputs.** Every script after `00_` loads `data/pbmc3k_processed.h5ad` (Scanpy's built-in `pbmc3k_processed`, 2700 cells × 1838 HVGs, 8 Louvain cell-type labels). PCA (50 comps) and UMAP are cached in `adata.obsm` during preprocessing.

**Label column auto-detect.** Scripts accept either `adata.obs['louvain']` (from `pbmc3k_processed`) or `adata.obs['cell_type']` (if you swap in the raw preprocessing block at the top of `00_preprocessing.py`). When editing, preserve this dual path — see the `label_col = ...` detection in `06_final_comparison.py` (Section 0) and `05_transformer_based.py:126`.

**Reproducibility contract.** All supervised splits use `train_test_split(..., test_size=0.2, random_state=42, stratify=y)` so LR, MLP, and foundation-model classifiers are compared on the **same test cells**. Do not change `random_state=42` without updating every script in lockstep.

**Two feature paths for supervised methods.** `X_expr` = full HVG matrix (dense, 1838 dims, `adata.X.toarray()`); `X_pca` = `adata.obsm['X_pca']` (50 dims). LR and RF are trained on both as a fair comparison; the MLP uses `X_expr` with `StandardScaler`; clustering uses `X_pca`. Foundation model embeddings (512-d scGPT, 256/512-d Geneformer) are a third feature path cached to `data/*_embeddings.npy`.

**MLP architecture.** `03_deep_learning.py` and the MLP section of `06_final_comparison.py` use a single hidden layer (`Linear → BatchNorm → ReLU → Dropout → Linear`, hidden_dim=128). Earlier 3-layer versions overfit on this small dataset — keep the architecture simple unless you have a reason to add capacity.

**Class imbalance handling in the MLP.** `03_deep_learning.py` and the MLP section of `06_final_comparison.py` apply inverse-frequency weights *both* as a `WeightedRandomSampler` on the train loader **and** as the `weight=` arg to `CrossEntropyLoss` — these are two independent mechanisms, both intentional. Don't remove one assuming it duplicates the other.

**Outputs.** All figures go to `figures/*.png`; result tables go to `figures/*.csv` (note: CSVs live under `figures/`, not a separate `results/` dir). Model weights go to `data/best_model.pt`. The scGPT checkpoint is committed at `scGPT_human/` (args.json + best_model.pt + vocab.json) — do not re-download if present.

**Transformer hand-off.** `05_transformer_based.py` writes `figures/transformer_results.csv`; `06_final_comparison.py` reads it (if present) and produces the unified `figures/all_methods_results.csv` + `figures/all_methods_comparison.png`. The two scripts only communicate via that CSV — do not reintroduce a direct dependency in either direction.

**Foundation model quirks.** scGPT wants HGNC symbols as `var_names` and uses `gene_col="index"`. Geneformer wants **Ensembl IDs** (requires `mygene` for symbol→Ensembl mapping) plus a tokenization step that writes a HuggingFace `.dataset`. Both fall back to loading cached `.npy` embeddings in the `__main__` of `05_transformer_based.py` if the embedding extractor errors — preserve those except/fallback blocks when editing.

**CPU vs CUDA.** PyTorch and both foundation models auto-select `cuda` if available. scGPT CPU inference is documented as ~5–10 min for 2700 cells; flag CPU runs to the user before kicking off long extractions.
