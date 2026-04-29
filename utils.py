# utils.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, adjusted_rand_score,
                             normalized_mutual_info_score)

def evaluate_classifier(y_true, y_pred, class_names, method_name):
    """Evaluate a supervised classifier and return metrics dict."""
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names,
                                    output_dict=True)
    print(f"\n{'='*50}")
    print(f"{method_name} — Accuracy: {acc:.4f}")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=class_names))
    return {
        'method': method_name,
        'accuracy': acc,
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score'],
    }

def evaluate_clustering(y_true, y_pred, method_name):
    """Evaluate unsupervised clustering against ground truth."""
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    n_clusters = len(np.unique(y_pred))
    print(f"{method_name} — ARI: {ari:.4f}, NMI: {nmi:.4f}, "
          f"Clusters: {n_clusters}")
    return {
        'method': method_name, 'ARI': ari, 'NMI': nmi,
        'n_clusters': n_clusters
    }

def plot_method_comparison(results_df):
    """Bar chart comparing metrics across all methods."""
    fig, ax = plt.subplots(figsize=(10, 5))
    results_df.plot(x='Method', y=['Accuracy', 'Macro_f1', 'ARI', "NMI"],
                    kind='bar', ax=ax)
    ax.set_ylabel('Score')
    ax.set_title('Method Comparison')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig('figures/method_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()