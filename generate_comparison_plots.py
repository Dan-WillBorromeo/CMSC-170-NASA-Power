# Module: Unified 3-Model Comparison Figures (Neural Network, k-NN, K-Means)
# > generate_comparison_plots.py
#
# Disclosures:
#   - Gemini 3.1 Pro assisted in generating Matplotlib visualization code.

import csv
from pathlib import Path
import matplotlib.pyplot as plt


def read_csv(path):
    with Path(path).open() as fh:
        return list(csv.DictReader(fh))


def compute_roc_curve(scores, labels):
    pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    pos_total = sum(labels)
    neg_total = len(labels) - pos_total
    tp = fp = 0
    fprs, tprs = [0.0], [0.0]
    for _, y in pairs:
        if y == 1:
            tp += 1
        else:
            fp += 1
        tprs.append(tp / pos_total)
        fprs.append(fp / neg_total)
    return fprs, tprs


def compute_metrics(preds, actuals):
    tp = sum(1 for p, a in zip(preds, actuals) if p == 1 and a == 1)
    tn = sum(1 for p, a in zip(preds, actuals) if p == 0 and a == 0)
    fp = sum(1 for p, a in zip(preds, actuals) if p == 1 and a == 0)
    fn = sum(1 for p, a in zip(preds, actuals) if p == 0 and a == 1)
    acc = (tp + tn) / len(actuals)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return acc, prec, rec, f1


def main():
    repo_root = Path(__file__).resolve().parent
    metrics_dir = repo_root / "Outputs" / "metrics"
    plots_dir = repo_root / "Outputs" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Neural Network
    nn_row = read_csv(metrics_dir / "nn_cv_summary.csv")[0]
    nn_oof = read_csv(metrics_dir / "nn_oof_predictions.csv")
    nn_acc, nn_prec, nn_rec, nn_f1 = compute_metrics(
        [int(r["pred_label"]) for r in nn_oof], [int(r["actual"]) for r in nn_oof]
    )

    # 2. Best k-NN
    knn_summary = read_csv(metrics_dir / "knn_cv_summary.csv")
    knn_oof = read_csv(metrics_dir / "knn_oof_predictions.csv")
    best_knn = max(knn_summary, key=lambda r: (float(r["auc_mean"]), float(r["acc_mean"])))
    knn_acc, knn_prec, knn_rec, knn_f1 = compute_metrics(
        [int(r["pred_label"]) for r in knn_oof], [int(r["actual"]) for r in knn_oof]
    )

    # 3. K-Means Clustering (k=2)
    km_assign = read_csv(metrics_dir / "kmeans_assignments.csv")
    km_sum = read_csv(metrics_dir / "kmeans_summary.csv")
    best_km = max(km_sum, key=lambda r: float(r["silhouette_score"]))

    km_actuals = [int(r["actual_label"]) for r in km_assign]
    km_clusters = [int(r["cluster_id"]) for r in km_assign]

    # Map cluster with higher solar farm share -> 1 (Solar Farm)
    unique_c = sorted(set(km_clusters))
    c_rates = {
        c: sum(1 for cl, a in zip(km_clusters, km_actuals) if cl == c and a == 1)
        / max(1, sum(1 for cl in km_clusters if cl == c))
        for c in unique_c
    }
    pos_cluster = max(c_rates, key=c_rates.get)
    km_preds = [1 if c == pos_cluster else 0 for c in km_clusters]
    km_acc, km_prec, km_rec, km_f1 = compute_metrics(km_preds, km_actuals)

    print("\n=== 3-MODEL COMPARISON SUMMARY (78 GRID CELLS) ===")
    print(f"Neural Network (MLP)  | Acc: {nn_acc:.3f} | Prec: {nn_prec:.3f} | Rec: {nn_rec:.3f} | F1: {nn_f1:.3f} | 5-Fold AUC: {float(nn_row['auc_mean']):.3f}")
    print(f"k-NN (k=9, Euclidean) | Acc: {knn_acc:.3f} | Prec: {knn_prec:.3f} | Rec: {knn_rec:.3f} | F1: {knn_f1:.3f} | 5-Fold AUC: {float(best_knn['auc_mean']):.3f}")
    print(f"K-Means (k={best_km['k']}, Unsup.)  | Acc: {km_acc:.3f} | Prec: {km_prec:.3f} | Rec: {km_rec:.3f} | F1: {km_f1:.3f} | Silhouette: {float(best_km['silhouette_score']):.3f}\n")

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

    # --- Panel 1: All 3 Methods Comparison (Accuracy, F1, Recall) ---
    ax = axes[0]
    models = ["Neural Net\n(Supervised)", "k-NN (k=9)\n(Supervised)", f"K-Means (k={best_km['k']})\n(Unsupervised)"]
    accs = [nn_acc, knn_acc, km_acc]
    f1s = [nn_f1, knn_f1, km_f1]
    recs = [nn_rec, knn_rec, km_rec]

    x = list(range(len(models)))
    w = 0.25
    b1 = ax.bar([i - w for i in x], accs, width=w, label="Accuracy", color="#1f77b4")
    b2 = ax.bar(x, f1s, width=w, label="F1-Score", color="#2ca02c")
    b3 = ax.bar([i + w for i in x], recs, width=w, label="Solar Farm Recall", color="#ff7f0e")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.2f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score (78 Grid Cells)")
    ax.set_title("All 3 Methods: Classification Alignment")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # --- Panel 2: Supervised Out-of-Fold ROC Curves ---
    ax = axes[1]
    fpr_n, tpr_n = compute_roc_curve(
        [float(r["pred_score"]) for r in nn_oof], [int(r["actual"]) for r in nn_oof]
    )
    fpr_k, tpr_k = compute_roc_curve(
        [float(r["pred_score"]) for r in knn_oof], [int(r["actual"]) for r in knn_oof]
    )
    ax.plot(fpr_n, tpr_n, lw=2, color="#2ca02c", label=f"Neural Net (AUC = {float(nn_row['auc_mean']):.3f})")
    ax.plot(fpr_k, tpr_k, lw=2, color="#1f77b4", label=f"k-NN (AUC = {float(best_knn['auc_mean']):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Baseline (0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Supervised Out-of-Fold ROC Curves")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.4)

    # --- Panel 3: k-NN Hyperparameter Grid Search ---
    ax = axes[2]
    for p_val, weighted_val, style, label in [
        ("1", "True", "-o", "Manhattan (p=1, Weighted)"),
        ("1", "False", "--o", "Manhattan (p=1, Unweighted)"),
        ("2", "True", "-s", "Euclidean (p=2, Weighted)"),
        ("2", "False", "--s", "Euclidean (p=2, Unweighted)"),
    ]:
        subset = [
            r for r in knn_summary
            if f"p={p_val}," in r["params"] and f"weighted={weighted_val}" in r["params"]
        ]
        subset.sort(key=lambda r: int(r["params"].split(",")[0].split("=")[1]))
        ks = [int(r["params"].split(",")[0].split("=")[1]) for r in subset]
        vals = [float(r["auc_mean"]) for r in subset]
        ax.plot(ks, vals, style, label=label)

    ax.set_xticks([3, 5, 7, 9])
    ax.set_xlabel("Number of Neighbors (k)")
    ax.set_ylabel("5-Fold ROC-AUC")
    ax.set_title("k-NN Hyperparameter Sensitivity")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_file = plots_dir / "model_comparison.png"
    plt.savefig(out_file, dpi=220)
    print(f"Saved 3-model comparison figure: {out_file}")


if __name__ == "__main__":
    main()
