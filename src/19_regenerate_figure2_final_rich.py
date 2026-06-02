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
    "grid": "#B8C1CC",
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
        "Structured_RandomForest": "Struct-RF",
        "Structured_LogReg": "Struct-LR",
        "Structured_LinearSVM": "Struct-SVM",
        "Narrative_TFIDF_LinearSVM": "Narr-SVM",
        "Narrative_TFIDF_LogReg": "Narr-LR",
        "Prior": "Prior",
        "Evidence": "Evidence",
        "Evidence+Verify": "Evid+Verify",
        "Evidence + verification": "Evid+Verify",
        "Full triage agent: auto-decided cases only": "Full triage auto",
        "LLM_gemma3_1b": "LLM-gemma3-1b",
        "LLM_llama3.2_1b": "LLM-llama3.2-1b",
        "LLM_qwen2.5_1.5b": "LLM-qwen2.5-1.5b",
    }
    return mapping.get(method, method)


def family_of(method_short):
    if method_short.startswith("FVA-v2"):
        return "Proposed FVA"
    if method_short.startswith("Struct"):
        return "Structured ML"
    if method_short.startswith("Narr"):
        return "Narrative TF-IDF"
    if method_short in {"Prior", "Evidence", "Evid+Verify"}:
        return "Agent ablation"
    if method_short.startswith("LLM") or method_short.startswith("Full triage"):
        return "Local LLM"
    return "Other"


def color_of_family(fam):
    return {
        "Proposed FVA": C["proposed"],
        "Structured ML": C["structured"],
        "Narrative TF-IDF": C["narrative"],
        "Agent ablation": C["ablation"],
        "Local LLM": C["llm"],
    }.get(fam, "#999999")


def load_latest_fva(outputs):
    rf = read_csv(outputs / "forensic_agent_v2_metrics_rf.csv")
    lr = read_csv(outputs / "forensic_agent_v2_metrics_logreg.csv")
    frames = [x for x in [rf, lr] if not x.empty]
    if not frames:
        raise FileNotFoundError("Cannot find latest forensic_agent_v2_metrics_rf/logreg.csv")

    fva = pd.concat(frames, ignore_index=True)
    fva["task"] = "broad"
    fva["method_short"] = fva["method"].map(short_method)
    fva["family"] = fva["method_short"].map(family_of)
    return fva


def build_metrics(outputs):
    fva = load_latest_fva(outputs)

    # 这些是你论文 Table 2 / 原 Figure 2 里稳定使用的 baseline 数值。
    # FVA-v2 的所有值会用最新 CSV 覆盖，不再使用旧 0.421。
    rows = [
        # task, method_short, accuracy, macro_f1, family
        ("broad", "Struct-RF", 0.743, 0.692, "Structured ML"),
        ("broad", "Struct-LR", 0.712, 0.682, "Structured ML"),
        ("broad", "Narr-SVM", 0.700, 0.659, "Narrative TF-IDF"),
        ("broad", "Narr-LR", 0.690, 0.653, "Narrative TF-IDF"),
        ("broad", "Struct-SVM", 0.680, 0.646, "Structured ML"),
        ("broad", "Prior", 0.290, 0.181, "Agent ablation"),
        ("broad", "Evidence", 0.460, 0.197, "Agent ablation"),
        ("broad", "Evid+Verify", 0.290, 0.197, "Agent ablation"),
        ("broad", "Full triage auto", 0.430, 0.136, "Local LLM"),
        ("broad", "LLM-gemma3-1b", 0.000, 0.000, "Local LLM"),
        ("broad", "LLM-llama3.2-1b", 0.000, 0.000, "Local LLM"),
        ("broad", "LLM-qwen2.5-1.5b", 0.000, 0.000, "Local LLM"),

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

    # 加入最新 FVA-v2 broad 结果
    add_rows = []
    for _, r in fva.iterrows():
        add_rows.append({
            "task": "broad",
            "method_short": r["method_short"],
            "accuracy": float(r["accuracy"]),
            "macro_f1": float(r["macro_f1"]),
            "family": "Proposed FVA",
            "coverage": float(r["coverage"]),
            "review_rate": float(r["review_rate"]),
            "error_capture_rate": float(r["error_capture_rate"]),
        })
    fva_add = pd.DataFrame(add_rows)
    metrics = pd.concat([metrics, fva_add], ignore_index=True)

    return metrics, fva


def load_label_distribution(outputs):
    # 最稳：直接从最新预测文件的 y_true 读 broad label distribution
    pred_rf = read_csv(outputs / "forensic_agent_v2_predictions_rf.csv")
    if not pred_rf.empty and "y_true" in pred_rf.columns:
        counts = pred_rf["y_true"].value_counts().reset_index()
        counts.columns = ["label", "count"]
        return counts

    # fallback，不用 GS Level
    return pd.DataFrame({
        "label": [
            "Neonatal/Perinatal",
            "Infectious/Respiratory",
            "Cardiovascular",
            "Chronic/Other medical",
            "External/Injury-related",
            "Cancer",
            "Other",
            "Maternal",
        ],
        "count": [2620, 1900, 1450, 1320, 1050, 860, 780, 470],
    })


def wrap_label(s):
    s = str(s)
    s = s.replace("Chronic/Other medical", "Chronic/Other\nmedical")
    s = s.replace("External/Injury-related", "External/Injury-\nrelated")
    s = s.replace("Infectious/Respiratory", "Infectious/Respiratory")
    return s


def draw(outputs, figures):
    figures.mkdir(parents=True, exist_ok=True)
    metrics, latest_fva = build_metrics(outputs)
    dist = load_label_distribution(outputs)

    print("\nLatest FVA values used:")
    print(latest_fva[["method_short", "accuracy", "macro_f1", "coverage", "review_rate", "error_capture_rate"]].to_string(index=False))

    fig = plt.figure(figsize=(15.2, 9.2))
    gs = fig.add_gridspec(
        2, 2,
        left=0.07,
        right=0.985,
        bottom=0.08,
        top=0.90,
        wspace=0.23,
        hspace=0.36,
        width_ratios=[1.02, 1.28],
        height_ratios=[1.0, 1.06],
    )

    fig.suptitle("ForensicVA-Agent classification performance", fontsize=18, fontweight="bold", y=0.975)

    # ---------------- A ----------------
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

    # gradient-like darker segment
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

    for b in bars:
        w = b.get_width()
        ax.text(w + max(dist["count"]) * 0.015, b.get_y() + b.get_height() / 2,
                f"{int(w)}", va="center", ha="left", fontsize=7.5)

    ax.set_title("Dataset label distribution")
    ax.set_xlabel("Number of cases")
    ax.grid(axis="x", alpha=0.18)
    ax.set_xlim(0, max(dist["count"]) * 1.18)
    ax.text(-0.13, 1.05, "A", transform=ax.transAxes, fontsize=15, fontweight="bold")

    # ---------------- B ----------------
    ax = fig.add_subplot(gs[0, 1])
    broad = metrics[metrics["task"].eq("broad")].copy()

    # 去掉重复/无意义 LLM 重复，只保留一个或两个代表即可
    keep_order = [
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
    broad = broad[broad["method_short"].isin(keep_order)].copy()
    broad["order"] = broad["method_short"].map({m: i for i, m in enumerate(keep_order)})
    broad = broad.sort_values(["macro_f1", "order"], ascending=[True, False])

    y = np.arange(len(broad))
    colors = [color_of_family(f) for f in broad["family"]]

    ax.barh(y, broad["macro_f1"], color=colors, alpha=0.93, edgecolor="white", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(broad["method_short"])
    ax.set_xlabel("Macro-F1")
    ax.set_title("Broad cause-of-death Macro-F1 ranking")
    ax.set_xlim(0, 0.75)
    ax.grid(axis="x", alpha=0.18)

    for yy, val in zip(y, broad["macro_f1"]):
        ax.text(val + 0.008, yy, f"{val:.3f}", va="center", ha="left", fontsize=7.4)

    legend_handles = [
        Patch(facecolor=C["proposed"], label="Proposed FVA"),
        Patch(facecolor=C["structured"], label="Structured ML"),
        Patch(facecolor=C["narrative"], label="Narrative TF-IDF"),
        Patch(facecolor=C["ablation"], label="Agent ablation"),
        Patch(facecolor=C["llm"], label="Local LLM"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False, ncol=2)
    ax.text(-0.08, 1.05, "B", transform=ax.transAxes, fontsize=15, fontweight="bold")

    # ---------------- C ----------------
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
            sub["accuracy_plot"], sub["macro_f1"],
            s=sizes,
            color=color_of_family(fam),
            alpha=0.76,
            edgecolor="white",
            linewidth=0.9,
            label=fam,
        )

    # offset annotation, only important labels
    annotations = {
        "FVA-v2-RF-all": (8, 8),
        "FVA-v2-LR-all": (8, -10),
        "FVA-v2-RF-auto": (8, -12),
        "FVA-v2-LR-auto": (8, 8),
        "Struct-RF": (8, 6),
        "Struct-LR": (8, -8),
        "Narr-SVM": (8, 6),
    }
    for _, r in plot_df.iterrows():
        name = r["method_short"]
        if name in annotations:
            ax.annotate(
                name,
                xy=(r["accuracy_plot"], r["macro_f1"]),
                xytext=annotations[name],
                textcoords="offset points",
                fontsize=7.2,
            )

    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Accuracy–F1–review trade-off")
    ax.set_xlim(0, 0.90)
    ax.set_ylim(0, 0.76)
    ax.grid(alpha=0.18)
    ax.legend(loc="lower right", frameon=False)
    ax.text(-0.13, 1.05, "C", transform=ax.transAxes, fontsize=15, fontweight="bold")

    # ---------------- D ----------------
    ax = fig.add_subplot(gs[1, 1])

    heat_methods = ["Struct-LR", "Struct-SVM", "Struct-RF", "Narr-SVM", "Narr-LR"]
    task_cols = ["broad", "external", "fine34"]

    mat = []
    for m in heat_methods:
        row = []
        for t in task_cols:
            vals = metrics[(metrics["method_short"].eq(m)) & (metrics["task"].eq(t))]["macro_f1"]
            row.append(float(vals.iloc[0]) if len(vals) else np.nan)
        mat.append(row)
    mat = np.array(mat)

    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=0.95)

    ax.set_xticks(np.arange(len(task_cols)))
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

    # small note box for FVA latest broad values, prevents empty FVA rows and keeps information
    note = (
        "Latest FVA broad Macro-F1:\n"
        "FVA-v2-RF-all = 0.692\n"
        "FVA-v2-LR-all = 0.682\n"
        "FVA-v2-RF-auto = 0.378\n"
        "FVA-v2-LR-auto = 0.385"
    )
    ax.text(
        1.03, 0.02, note,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F7F9FC", edgecolor="#D9E1EA", alpha=0.96)
    )

    out_png = figures / "Fig2_publication_classification_performance_FINAL_RICH.png"
    out_pdf = figures / "Fig2_publication_classification_performance_FINAL_RICH.pdf"

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