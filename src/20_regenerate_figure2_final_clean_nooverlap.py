import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 400,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


C = {
    "proposed": "#E84A5F",
    "structured": "#4C78A8",
    "narrative": "#2BB3A3",
    "ablation": "#F5A623",
    "llm": "#8E63CE",
    "blue_dark": "#315F8D",
}


def read_csv(path):
    path = Path(path)
    if not path.exists():
        print(f"[WARN] missing: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def short_method(method):
    method = str(method)
    mapping = {
        "ForensicVA-Agent-v2-rf-all_cases": "FVA-v2-RF-all",
        "ForensicVA-Agent-v2-rf-auto_decided": "FVA-v2-RF-auto",
        "ForensicVA-Agent-v2-logreg-all_cases": "FVA-v2-LR-all",
        "ForensicVA-Agent-v2-logreg-auto_decided": "FVA-v2-LR-auto",
    }
    return mapping.get(method, method)


def family_color(family):
    return {
        "Proposed FVA": C["proposed"],
        "Structured ML": C["structured"],
        "Narrative TF-IDF": C["narrative"],
        "Agent ablation": C["ablation"],
        "Local LLM": C["llm"],
    }.get(family, "#999999")


def load_latest_fva(outputs):
    rf = read_csv(outputs / "forensic_agent_v2_metrics_rf.csv")
    lr = read_csv(outputs / "forensic_agent_v2_metrics_logreg.csv")
    frames = [x for x in [rf, lr] if not x.empty]
    if not frames:
        raise FileNotFoundError("Missing forensic_agent_v2_metrics_rf/logreg.csv")

    df = pd.concat(frames, ignore_index=True)
    df["method_short"] = df["method"].map(short_method)
    df["family"] = "Proposed FVA"
    return df


def build_metrics(outputs):
    fva = load_latest_fva(outputs)

    rows = [
        # task, method_short, accuracy, macro_f1, family
        ("broad", "Struct-RF", 0.743, 0.692, "Structured ML"),
        ("broad", "Struct-LR", 0.712, 0.682, "Structured ML"),
        ("broad", "Narr-SVM", 0.700, 0.659, "Narrative TF-IDF"),
        ("broad", "Narr-LR", 0.690, 0.653, "Narrative TF-IDF"),
        ("broad", "Struct-SVM", 0.680, 0.646, "Structured ML"),
        ("broad", "Evidence", 0.460, 0.197, "Agent ablation"),
        ("broad", "Evid+Verify", 0.290, 0.197, "Agent ablation"),
        ("broad", "Prior", 0.290, 0.181, "Agent ablation"),
        ("broad", "Full triage auto", 0.430, 0.136, "Local LLM"),
        ("broad", "LLM-gemma3-1b", 0.000, 0.000, "Local LLM"),
        ("broad", "LLM-llama3.2-1b", 0.000, 0.000, "Local LLM"),

        ("external", "Struct-LR", 0.971, 0.930, "Structured ML"),
        ("external", "Struct-SVM", 0.966, 0.915, "Structured ML"),
        ("external", "Struct-RF", 0.958, 0.885, "Structured ML"),
        ("external", "Narr-SVM", 0.947, 0.864, "Narrative TF-IDF"),
        ("external", "Narr-LR", 0.938, 0.833, "Narrative TF-IDF"),

        ("fine34", "Struct-LR", 0.553, 0.484, "Structured ML"),
        ("fine34", "Struct-RF", 0.572, 0.471, "Structured ML"),
        ("fine34", "Struct-SVM", 0.517, 0.433, "Structured ML"),
        ("fine34", "Narr-SVM", 0.482, 0.369, "Narrative TF-IDF"),
        ("fine34", "Narr-LR", 0.458, 0.342, "Narrative TF-IDF"),
    ]

    metrics = pd.DataFrame(rows, columns=["task", "method_short", "accuracy", "macro_f1", "family"])

    fva_rows = []
    for _, r in fva.iterrows():
        fva_rows.append({
            "task": "broad",
            "method_short": r["method_short"],
            "accuracy": float(r["accuracy"]),
            "macro_f1": float(r["macro_f1"]),
            "family": "Proposed FVA",
            "coverage": float(r["coverage"]),
            "review_rate": float(r["review_rate"]),
            "error_capture_rate": float(r["error_capture_rate"]),
        })

    metrics = pd.concat([metrics, pd.DataFrame(fva_rows)], ignore_index=True)
    return metrics, fva


def load_label_distribution(outputs):
    pred = read_csv(outputs / "forensic_agent_v2_predictions_rf.csv")
    if not pred.empty and "y_true" in pred.columns:
        counts = pred["y_true"].value_counts().reset_index()
        counts.columns = ["label", "count"]
        return counts

    return pd.DataFrame({
        "label": [
            "Infectious/Respiratory",
            "Neonatal/Perinatal",
            "Cardiovascular",
            "External/Injury-related",
            "Chronic/Other medical",
            "Other",
            "Cancer",
            "Maternal",
        ],
        "count": [615, 553, 304, 292, 272, 198, 178, 94],
    })


def wrap_label(s):
    return (
        str(s)
        .replace("External/Injury-related", "External/Injury-\nrelated")
        .replace("Chronic/Other medical", "Chronic/Other\nmedical")
    )


def draw(outputs, figures):
    figures.mkdir(parents=True, exist_ok=True)

    metrics, latest_fva = build_metrics(outputs)
    dist = load_label_distribution(outputs)

    print("\nLatest FVA rows used:")
    print(latest_fva[["method_short", "accuracy", "macro_f1", "coverage", "review_rate", "error_capture_rate"]].to_string(index=False))

    fig = plt.figure(figsize=(15.6, 9.3))
    gs = fig.add_gridspec(
        2, 2,
        left=0.065,
        right=0.975,
        bottom=0.080,
        top=0.905,
        wspace=0.235,
        hspace=0.36,
        width_ratios=[1.02, 1.30],
        height_ratios=[1.0, 1.06],
    )

    fig.suptitle(
        "ForensicVA-Agent classification performance",
        fontsize=18,
        fontweight="bold",
        y=0.975
    )

    # ================= A =================
    ax = fig.add_subplot(gs[0, 0])
    dist = dist.copy()
    dist["label_wrapped"] = dist["label"].map(wrap_label)
    dist = dist.sort_values("count", ascending=True)

    bars = ax.barh(
        dist["label_wrapped"],
        dist["count"],
        color=C["structured"],
        alpha=0.90,
        edgecolor="white",
        linewidth=0.7,
    )

    for b in bars:
        w = b.get_width()
        ax.barh(
            b.get_y() + b.get_height() / 2,
            w * 0.28,
            height=b.get_height(),
            color=C["blue_dark"],
            alpha=0.28,
            edgecolor="none",
        )
        ax.text(
            w + max(dist["count"]) * 0.015,
            b.get_y() + b.get_height() / 2,
            f"{int(w)}",
            va="center",
            ha="left",
            fontsize=7.5,
        )

    ax.set_title("Dataset label distribution")
    ax.set_xlabel("Number of cases")
    ax.set_xlim(0, max(dist["count"]) * 1.18)
    ax.grid(axis="x", alpha=0.18)
    ax.text(-0.13, 1.05, "A", transform=ax.transAxes, fontsize=15, fontweight="bold")

    # ================= B =================
    ax = fig.add_subplot(gs[0, 1])

    broad = metrics[metrics["task"].eq("broad")].copy()

    keep = [
        "FVA-v2-RF-all",
        "Struct-RF",
        "FVA-v2-LR-all",
        "Struct-LR",
        "Narr-SVM",
        "Narr-LR",
        "Struct-SVM",
        "FVA-v2-LR-auto",
        "FVA-v2-RF-auto",
        "Evidence",
        "Evid+Verify",
        "Prior",
        "Full triage auto",
        "LLM-gemma3-1b",
        "LLM-llama3.2-1b",
    ]

    broad = broad[broad["method_short"].isin(keep)].copy()
    broad["order"] = broad["method_short"].map({m: i for i, m in enumerate(keep)})
    broad = broad.sort_values(["macro_f1", "order"], ascending=[True, False])

    y = np.arange(len(broad))
    ax.barh(
        y,
        broad["macro_f1"],
        color=[family_color(f) for f in broad["family"]],
        alpha=0.93,
        edgecolor="white",
        linewidth=0.6,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(broad["method_short"])
    ax.set_xlabel("Macro-F1")
    ax.set_title("Broad cause-of-death Macro-F1 ranking")
    ax.set_xlim(0, 0.75)
    ax.grid(axis="x", alpha=0.18)

    for yy, val in zip(y, broad["macro_f1"]):
        ax.text(val + 0.008, yy, f"{val:.3f}", va="center", ha="left", fontsize=7.3)

    ax.legend(
        handles=[
            Patch(facecolor=C["proposed"], label="Proposed FVA"),
            Patch(facecolor=C["structured"], label="Structured ML"),
            Patch(facecolor=C["narrative"], label="Narrative TF-IDF"),
            Patch(facecolor=C["ablation"], label="Agent ablation"),
            Patch(facecolor=C["llm"], label="Local LLM"),
        ],
        loc="lower right",
        frameon=False,
        ncol=2
    )
    ax.text(-0.08, 1.05, "B", transform=ax.transAxes, fontsize=15, fontweight="bold")

    # ================= C =================
    ax = fig.add_subplot(gs[1, 0])

    plot_df = broad.copy()
    plot_df["accuracy_plot"] = plot_df["accuracy"]
    plot_df.loc[plot_df["accuracy_plot"].isna(), "accuracy_plot"] = plot_df["macro_f1"] + 0.08

    review_rate_map = {
        r["method_short"]: float(r["review_rate"])
        for _, r in latest_fva.iterrows()
    }

    for fam in ["Structured ML", "Narrative TF-IDF", "Proposed FVA", "Agent ablation", "Local LLM"]:
        sub = plot_df[plot_df["family"].eq(fam)]
        if sub.empty:
            continue

        sizes = []
        for _, r in sub.iterrows():
            rr = review_rate_map.get(r["method_short"], 0.0)
            sizes.append(42 + 430 * rr)

        ax.scatter(
            sub["accuracy_plot"],
            sub["macro_f1"],
            s=sizes,
            color=family_color(fam),
            alpha=0.76,
            edgecolor="white",
            linewidth=0.9,
            label=fam,
            zorder=3,
        )

    # 用编号标注，避免文字重叠
    key_points = [
        ("1", "FVA-v2-RF-all", 0.022, 0.020),
        ("2", "Struct-RF", 0.016, -0.022),
        ("3", "FVA-v2-LR-all", 0.022, -0.020),
        ("4", "Struct-LR", 0.016, 0.020),
        ("5", "FVA-v2-RF-auto", 0.018, -0.030),
        ("6", "FVA-v2-LR-auto", 0.018, 0.025),
        ("7", "Narr-SVM", 0.015, 0.020),
    ]

    for num, name, dx, dy in key_points:
        row = plot_df[plot_df["method_short"].eq(name)]
        if row.empty:
            continue
        x = float(row["accuracy_plot"].iloc[0])
        yv = float(row["macro_f1"].iloc[0])
        ax.text(
            x + dx,
            yv + dy,
            num,
            fontsize=7.5,
            fontweight="bold",
            ha="center",
            va="center",
            color="white",
            bbox=dict(boxstyle="circle,pad=0.18", facecolor="#1F2D3D", edgecolor="white", linewidth=0.6),
            zorder=4,
        )

    # Panel C 右下角放编号说明，小而清楚
    legend_text = (
        "1 FVA-v2-RF-all\n"
        "2 Struct-RF\n"
        "3 FVA-v2-LR-all\n"
        "4 Struct-LR\n"
        "5 FVA-v2-RF-auto\n"
        "6 FVA-v2-LR-auto\n"
        "7 Narr-SVM"
    )

    ax.text(
        0.975,
        0.045,
        legend_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.0,
        linespacing=1.20,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#D9E1EA", alpha=0.96),
    )

    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Accuracy–F1–review trade-off")
    ax.set_xlim(0, 0.90)
    ax.set_ylim(0, 0.76)
    ax.grid(alpha=0.18)
    ax.legend(loc="upper left", frameon=False)
    ax.text(-0.13, 1.05, "C", transform=ax.transAxes, fontsize=15, fontweight="bold")

    # ================= D =================
    ax = fig.add_subplot(gs[1, 1])

    heat_methods = ["Struct-LR", "Struct-SVM", "Struct-RF", "Narr-SVM", "Narr-LR"]
    tasks = ["broad", "external", "fine34"]

    mat = []
    for m in heat_methods:
        row = []
        for t in tasks:
            vals = metrics[(metrics["method_short"].eq(m)) & (metrics["task"].eq(t))]["macro_f1"]
            row.append(float(vals.iloc[0]) if len(vals) else np.nan)
        mat.append(row)

    mat = np.array(mat, dtype=float)

    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=0.95)

    ax.set_xticks(np.arange(len(tasks)))
    ax.set_xticklabels(["broad", "external", "fine34"])
    ax.set_yticks(np.arange(len(heat_methods)))
    ax.set_yticklabels(heat_methods)
    ax.set_title("Task-wise Macro-F1 heatmap")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            color = "white" if val > 0.62 else "#202020"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.015)
    cbar.set_label("Macro-F1")
    ax.text(-0.08, 1.05, "D", transform=ax.transAxes, fontsize=15, fontweight="bold")

    # 关键修改：不再放右下角大 note，避免压住 colorbar
    # FVA broad values 已经在 Panel B 和 Panel C 里展示。

    out_png = figures / "Fig2_publication_classification_performance_FINAL_CLEAN.png"
    out_pdf = figures / "Fig2_publication_classification_performance_FINAL_CLEAN.pdf"

    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved:")
    print(out_png)
    print(out_pdf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--figures", default="figures")
    args = parser.parse_args()
    draw(Path(args.outputs), Path(args.figures))


if __name__ == "__main__":
    main()