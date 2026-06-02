
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def plot_label_distribution(df, out_dir):
    counts = df["target_broad"].value_counts().sort_values()
    plt.figure(figsize=(8.2, 4.8))
    plt.barh(counts.index, counts.values)
    plt.xlabel("Number of cases")
    plt.title("Broad cause-of-death group distribution")
    for i, v in enumerate(counts.values):
        plt.text(v, i, f" {v}", va="center")
    savefig(out_dir / "fig_label_distribution_broad.png")

def plot_performance_bubble(metrics, out_dir):
    if metrics.empty:
        return
    m = metrics.dropna(subset=["accuracy", "macro_f1"]).copy()
    if "n_classes" not in m.columns:
        m["n_classes"] = 1
    m["size"] = 80 + 25 * m["n_classes"].fillna(1).astype(float)
    plt.figure(figsize=(8.5, 5.6))
    for task, g in m.groupby("task"):
        plt.scatter(g["accuracy"], g["macro_f1"], s=g["size"], alpha=0.65, label=task)
        for _, r in g.iterrows():
            label = str(r["method"]).replace("Structured_", "S-").replace("Narrative_", "N-").replace("_", " ")
            plt.text(r["accuracy"] + 0.003, r["macro_f1"] + 0.003, label[:24], fontsize=7)
    plt.xlabel("Accuracy")
    plt.ylabel("Macro-F1")
    plt.title("Model comparison across PHMRC-VA tasks")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    savefig(out_dir / "fig_performance_bubble.png")

def plot_metric_heatmap(metrics, out_dir):
    if metrics.empty:
        return
    pivot = metrics.pivot_table(index="method", columns="task", values="macro_f1", aggfunc="max")
    plt.figure(figsize=(8.5, max(4, 0.35 * len(pivot))))
    im = plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(im, label="Macro-F1")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title("Macro-F1 heatmap by method and task")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    savefig(out_dir / "fig_metric_heatmap.png")

def plot_boxplot_predictions(out_dir):
    pred_files = list(Path(out_dir.parent / "outputs").glob("pred_*_*.csv"))
    if not pred_files:
        return

def plot_selective_accuracy(out_dir):
    pred_path = out_dir.parent / "outputs" / "agent_ablation_predictions.csv"
    if not pred_path.exists():
        return
    df = pd.read_csv(pred_path)
    df = df.sort_values("evidence_confidence", ascending=False)
    xs, ys = [], []
    for cov in np.linspace(0.1, 1.0, 10):
        n = max(1, int(len(df) * cov))
        sub = df.head(n)
        acc = (sub["y_true"] == sub["final_pred"]).mean()
        xs.append(cov)
        ys.append(acc)
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Coverage retained by confidence threshold")
    plt.ylabel("Accuracy among retained cases")
    plt.title("Selective prediction curve for the triage agent")
    plt.grid(alpha=0.25)
    savefig(out_dir / "fig_selective_accuracy_curve.png")

def plot_review_reasons(out_dir):
    pred_path = out_dir.parent / "outputs" / "agent_ablation_predictions.csv"
    if not pred_path.exists():
        return
    df = pd.read_csv(pred_path)
    reasons = []
    for x in df["triage_reasons"].fillna(""):
        reasons.extend([r for r in str(x).split("|") if r])
    if not reasons:
        return
    counts = pd.Series(reasons).value_counts().sort_values()
    plt.figure(figsize=(8, 4.8))
    plt.barh(counts.index, counts.values)
    plt.xlabel("Number of triggered cases")
    plt.title("Human-review triage reasons")
    for i, v in enumerate(counts.values):
        plt.text(v, i, f" {v}", va="center")
    savefig(out_dir / "fig_triage_reasons.png")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.processed, low_memory=False)
    metrics = pd.read_csv(args.metrics) if Path(args.metrics).exists() else pd.DataFrame()

    plot_label_distribution(df, out_dir)
    plot_performance_bubble(metrics, out_dir)
    plot_metric_heatmap(metrics, out_dir)
    plot_selective_accuracy(out_dir)
    plot_review_reasons(out_dir)

    print(f"Figures saved to: {out_dir}")

if __name__ == "__main__":
    main()
