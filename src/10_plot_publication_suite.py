#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication-grade figure suite for ForensicVA-Agent.

Run from project root:
    python src/10_plot_publication_suite.py --outputs outputs --figures figures_pub

Inputs expected when available:
    outputs/all_metrics.csv
    outputs/label_distribution.csv
    outputs/agent_ablation_metrics.csv
    outputs/forensic_agent_v2_predictions_*.csv
    outputs/forensic_agent_v2_selective_curve_*.csv
    outputs/ollama_llm_metrics.csv
    outputs/cm_*.csv

Outputs:
    figures_pub/Fig1_main_performance_suite.png/.pdf
    figures_pub/Fig2_safety_triage_suite.png/.pdf
    figures_pub/Fig3_error_and_distribution_suite.png/.pdf
    figures_pub/Fig4_llm_diagnostics_suite.png/.pdf
    figures_pub/Fig5_3d_safety_landscape.png/.pdf
    figures_pub/tables/*.csv
"""

import argparse
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


warnings.filterwarnings("ignore", category=UserWarning)

# =========================
# Global visual style
# =========================
PALETTE = {
    "blue": "#3B82F6",
    "cyan": "#06B6D4",
    "teal": "#14B8A6",
    "green": "#22C55E",
    "orange": "#F59E0B",
    "red": "#F43F5E",
    "purple": "#8B5CF6",
    "gray": "#64748B",
    "lightgray": "#E5E7EB",
    "dark": "#0F172A",
}

METHOD_COLORS = {
    "FVA": PALETTE["red"],
    "Agent": PALETTE["red"],
    "Struct": PALETTE["blue"],
    "Narr": PALETTE["teal"],
    "LLM": PALETTE["purple"],
    "Evidence": PALETTE["orange"],
    "Prior": PALETTE["gray"],
    "Other": PALETTE["gray"],
}

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 450,
    "font.size": 9.5,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 11,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
    "legend.fontsize": 8.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.65,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =========================
# Utility functions
# =========================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def read_csv_optional(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def save_both(fig, out_path: Path):
    ensure_dir(out_path.parent)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def wrap_label(x, width=18):
    x = str(x)
    return "\n".join(textwrap.wrap(x, width=width, break_long_words=False))


def short_method_name(name):
    if pd.isna(name):
        return "NA"
    s = str(name)
    replacements = [
        ("ForensicVA-Agent-v2-", "FVA-v2-"),
        ("ForensicVA-Agent-", "FVA-"),
        ("Narrative_TFIDF_", "Narr-TFIDF-"),
        ("Narrative-TFIDF-", "Narr-TFIDF-"),
        ("Structured_", "Struct-"),
        ("Structured-", "Struct-"),
        ("RandomForest", "RF"),
        ("LogReg", "LR"),
        ("LinearSVM", "SVM"),
        ("logreg", "LR"),
        ("rf", "RF"),
        ("all_cases", "all"),
        ("auto_decided", "auto"),
        ("qwen2.5:", "qwen"),
        ("llama3.2:", "llama"),
        ("gemma3:", "gemma"),
        ("deepseek-r1:", "deepseek"),
        ("mistral:", "mistral"),
    ]
    for a, b in replacements:
        s = s.replace(a, b)
    s = s.replace("__", "_")
    return s


def method_group(name):
    s = str(name)
    if s.startswith("LLM") or "qwen" in s or "llama" in s or "gemma" in s or "mistral" in s or "deepseek" in s:
        return "LLM"
    if "FVA" in s or "ForensicVA" in s:
        return "FVA"
    if "Struct" in s or "Structured" in s:
        return "Struct"
    if "Narr" in s or "TFIDF" in s:
        return "Narr"
    if "Evidence" in s or "Prior" in s:
        return "Agent-v1"
    return "Other"


def color_for_method(name):
    g = method_group(name)
    return {
        "FVA": PALETTE["red"],
        "Struct": PALETTE["blue"],
        "Narr": PALETTE["teal"],
        "LLM": PALETTE["purple"],
        "Agent-v1": PALETTE["orange"],
        "Other": PALETTE["gray"],
    }.get(g, PALETTE["gray"])


def clean_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    for col in ["accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall",
                "coverage", "review_rate", "error_capture_rate", "parse_error_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "task" not in df.columns:
        df["task"] = "broad"
    if "method" not in df.columns:
        df["method"] = "unknown"

    df["method_short"] = df["method"].apply(short_method_name)
    df["method_group"] = df["method"].apply(method_group)

    # Avoid label collisions when the same method is appended multiple times.
    df["_dup_index"] = df.groupby(["task", "method_short"]).cumcount() + 1
    df["_dup_count"] = df.groupby(["task", "method_short"])["method_short"].transform("count")
    df["method_display"] = np.where(
        df["_dup_count"] > 1,
        df["method_short"] + " #" + df["_dup_index"].astype(str),
        df["method_short"]
    )

    return df


def add_panel_label(ax, label, x=-0.08, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", ha="left", color=PALETTE["dark"])


def add_clean_legend(ax, *args, **kwargs):
    leg = ax.legend(*args, frameon=False, **kwargs)
    return leg


def value_annotate_barh(ax, values, fmt="{:.3f}", pad=0.006):
    xmax = np.nanmax(values) if len(values) else 1
    for i, v in enumerate(values):
        if pd.notna(v):
            ax.text(v + pad * max(1, xmax), i, fmt.format(v), va="center", fontsize=7.5, color=PALETTE["dark"])


def find_first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


# =========================
# Plot 1: Main dashboard
# =========================
def plot_main_performance_suite(metrics, label_dist, out_dir: Path):
    if metrics.empty:
        print("[Skip] metrics empty: Fig1")
        return

    broad = metrics[(metrics["task"].astype(str) == "broad") & metrics["macro_f1"].notna()].copy()
    if broad.empty:
        broad = metrics[metrics["macro_f1"].notna()].copy()

    # Use top methods for readable dashboard.
    top = broad.sort_values("macro_f1", ascending=False).head(14).sort_values("macro_f1", ascending=True)

    fig = plt.figure(figsize=(16.8, 10.8), constrained_layout=True)
    gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1.05, 1.25, 1.05],
                           width_ratios=[1.05, 1.25, 1.20, 1.0])

    fig.suptitle("ForensicVA-Agent: performance, safety triage, and ablation dashboard",
                 fontsize=17, fontweight="bold", y=1.02)

    # A. Label distribution
    axA = fig.add_subplot(gs[0, 0])
    add_panel_label(axA, "A")
    if not label_dist.empty:
        ld = label_dist.copy()
        # try to infer label/count columns
        label_col = next((c for c in ld.columns if "label" in c.lower() or "target" in c.lower() or "broad" in c.lower()), ld.columns[0])
        count_col = next((c for c in ld.columns if "count" in c.lower() or "n" == c.lower()), ld.columns[-1])
        ld[count_col] = pd.to_numeric(ld[count_col], errors="coerce")
        ld = ld.dropna(subset=[count_col]).sort_values(count_col, ascending=True).tail(10)
        axA.barh([wrap_label(x, 16) for x in ld[label_col]], ld[count_col], color=PALETTE["blue"], alpha=0.85)
        axA.set_xlabel("Cases")
        axA.set_title("Dataset label distribution")
    else:
        axA.text(0.5, 0.5, "label_distribution.csv\nnot found", ha="center", va="center")
        axA.set_axis_off()

    # B. Ranked Macro-F1 lollipop/bar
    axB = fig.add_subplot(gs[0, 1:3])
    add_panel_label(axB, "B")
    colors = [color_for_method(m) for m in top["method_short"]]
    axB.barh(top["method_display"], top["macro_f1"], color=colors, alpha=0.88)
    value_annotate_barh(axB, top["macro_f1"].values)
    axB.set_xlabel("Macro-F1")
    axB.set_xlim(0, min(1.0, max(0.75, top["macro_f1"].max() + 0.08)))
    axB.set_title("Broad cause-of-death classification ranking")
    axB.tick_params(axis="y", labelsize=7.8)

    legend_elems = [
        Patch(facecolor=PALETTE["red"], label="FVA / proposed"),
        Patch(facecolor=PALETTE["blue"], label="Structured ML"),
        Patch(facecolor=PALETTE["teal"], label="Narrative TF-IDF"),
        Patch(facecolor=PALETTE["purple"], label="LLM"),
        Patch(facecolor=PALETTE["orange"], label="Agent-v1 / ablation"),
    ]
    axB.legend(handles=legend_elems, loc="lower right", ncol=2, frameon=False, fontsize=7.5)

    # C. Pareto-like scatter
    axC = fig.add_subplot(gs[0, 3])
    add_panel_label(axC, "C")
    sc_df = broad.dropna(subset=["accuracy", "macro_f1"]).copy()
    if not sc_df.empty:
        rv = sc_df["review_rate"] if "review_rate" in sc_df.columns else pd.Series(np.zeros(len(sc_df)))
        rv = rv.fillna(0)
        sizes = 60 + 520 * rv.clip(0, 1)
        colors = [color_for_method(m) for m in sc_df["method_short"]]
        axC.scatter(sc_df["accuracy"], sc_df["macro_f1"], s=sizes, c=colors,
                    alpha=0.72, edgecolor="white", linewidth=0.8)
        # annotate only best few to avoid clutter
        best_idx = sc_df.sort_values("macro_f1", ascending=False).head(6).index
        for idx in best_idx:
            r = sc_df.loc[idx]
            axC.annotate(short_method_name(r["method_short"]), (r["accuracy"], r["macro_f1"]),
                         xytext=(4, 4), textcoords="offset points", fontsize=7.3)
        axC.set_xlabel("Accuracy")
        axC.set_ylabel("Macro-F1")
        axC.set_title("Accuracy–F1–review trade-off")
    else:
        axC.text(0.5, 0.5, "No accuracy/F1 data", ha="center", va="center")
        axC.set_axis_off()

    # D. Task-wise heatmap
    axD = fig.add_subplot(gs[1, 0:2])
    add_panel_label(axD, "D")
    hm = metrics.dropna(subset=["macro_f1"]).copy()
    if not hm.empty:
        pivot = hm.pivot_table(index="method_display", columns="task", values="macro_f1", aggfunc="max")
        pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).head(18).index]
        im = axD.imshow(pivot.values, aspect="auto", cmap="YlGnBu", vmin=np.nanmin(pivot.values), vmax=np.nanmax(pivot.values))
        axD.set_xticks(np.arange(len(pivot.columns)))
        axD.set_xticklabels(pivot.columns, rotation=25, ha="right")
        axD.set_yticks(np.arange(len(pivot.index)))
        axD.set_yticklabels([wrap_label(x, 18) for x in pivot.index], fontsize=7.5)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if pd.notna(val):
                    axD.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                             color="white" if val > np.nanmean(pivot.values) else PALETTE["dark"])
        axD.set_title("Task-wise Macro-F1 heatmap")
        cbar = fig.colorbar(im, ax=axD, shrink=0.78, pad=0.01)
        cbar.set_label("Macro-F1")
    else:
        axD.set_axis_off()

    # E. Selective prediction curves
    axE = fig.add_subplot(gs[1, 2])
    add_panel_label(axE, "E")
    curve_files = sorted((out_dir.parent / "outputs").glob("forensic_agent_v2_selective_curve_*.csv"))
    # if script called with different output dir, pass via global? fallback handled below in main by copying path.
    if not curve_files:
        curve_files = sorted(Path("outputs").glob("forensic_agent_v2_selective_curve_*.csv"))
    if curve_files:
        for f in curve_files:
            cd = pd.read_csv(f)
            label = f.stem.replace("forensic_agent_v2_selective_curve_", "")
            if "coverage" in cd.columns and "accuracy" in cd.columns:
                axE.plot(cd["coverage"], cd["accuracy"], marker="o", linewidth=2.0, label=f"{label} Acc.")
            if "macro_f1" in cd.columns:
                axE.plot(cd["coverage"], cd["macro_f1"], marker="s", linewidth=1.6, linestyle="--", label=f"{label} F1")
        axE.set_xlabel("Coverage retained")
        axE.set_ylabel("Score")
        axE.set_ylim(0, 1.05)
        axE.set_title("Selective prediction curves")
        axE.legend(loc="lower left", frameon=False, fontsize=7)
    else:
        axE.text(0.5, 0.5, "Selective curves\nnot found", ha="center", va="center")
        axE.set_axis_off()

    # F. Confidence boxplot
    axF = fig.add_subplot(gs[1, 3])
    add_panel_label(axF, "F")
    pred_files = sorted((out_dir.parent / "outputs").glob("forensic_agent_v2_predictions_*.csv"))
    if not pred_files:
        pred_files = sorted(Path("outputs").glob("forensic_agent_v2_predictions_*.csv"))
    box_data, labels, box_colors = [], [], []
    if pred_files:
        pdf = pd.concat([pd.read_csv(f).assign(model=f.stem.replace("forensic_agent_v2_predictions_", "")) for f in pred_files], ignore_index=True)
        pdf["correct"] = pdf["y_true"].astype(str) == pdf["final_pred"].astype(str)
        for model in ["logreg", "rf"]:
            for status in [True, False]:
                sub = pdf[(pdf["model"] == model) & (pdf["correct"] == status)]
                if len(sub) and "confidence" in sub.columns:
                    box_data.append(pd.to_numeric(sub["confidence"], errors="coerce").dropna().values)
                    labels.append(f"{model}\n{'correct' if status else 'wrong'}")
                    box_colors.append(PALETTE["green"] if status else PALETTE["red"])
    if box_data:
        bp = axF.boxplot(box_data, labels=labels, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        axF.set_ylabel("Classifier confidence")
        axF.set_title("Confidence separation")
        axF.tick_params(axis="x", labelrotation=20)
    else:
        axF.text(0.5, 0.5, "Prediction files\nnot found", ha="center", va="center")
        axF.set_axis_off()

    # G. Ablation bars
    axG = fig.add_subplot(gs[2, 0:2])
    add_panel_label(axG, "G")
    ab_path = find_first_existing([Path("outputs/agent_ablation_metrics.csv"), out_dir.parent / "outputs/agent_ablation_metrics.csv"])
    ab = read_csv_optional(ab_path) if ab_path else pd.DataFrame()
    if not ab.empty:
        method_col = "method" if "method" in ab.columns else ab.columns[0]
        x = np.arange(len(ab))
        width = 0.35
        if "accuracy" in ab.columns:
            axG.bar(x - width/2, ab["accuracy"], width, label="Accuracy", color=PALETTE["blue"], alpha=0.85)
        if "macro_f1" in ab.columns:
            axG.bar(x + width/2, ab["macro_f1"], width, label="Macro-F1", color=PALETTE["orange"], alpha=0.85)
        axG.set_xticks(x)
        axG.set_xticklabels([wrap_label(short_method_name(v), 14) for v in ab[method_col]], rotation=15, ha="right")
        axG.set_ylim(0, min(1, max(0.5, np.nanmax(ab[[c for c in ["accuracy", "macro_f1"] if c in ab.columns]].values) + 0.12)))
        axG.set_title("Agent-v1 ablation study")
        axG.legend(frameon=False)
    else:
        axG.text(0.5, 0.5, "Ablation file\nnot found", ha="center", va="center")
        axG.set_axis_off()

    # H. Compact summary table
    axH = fig.add_subplot(gs[2, 2:4])
    add_panel_label(axH, "H")
    summary = broad.sort_values("macro_f1", ascending=False).head(8).copy()
    cols = ["method_short", "accuracy", "macro_f1", "coverage", "review_rate", "error_capture_rate"]
    cols = [c for c in cols if c in summary.columns]
    if not summary.empty:
        disp = summary[cols].copy()
        for c in disp.columns:
            if c != "method_short":
                disp[c] = disp[c].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        disp["method_short"] = disp["method_short"].map(lambda x: wrap_label(x, 22))
        axH.axis("off")
        table = axH.table(cellText=disp.values,
                          colLabels=["Method" if c=="method_short" else c.replace("_", "\n").title() for c in disp.columns],
                          cellLoc="center", colLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7.6)
        table.scale(1.0, 1.35)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(fontweight="bold", color="white")
                cell.set_facecolor(PALETTE["dark"])
            else:
                cell.set_edgecolor("#CBD5E1")
                cell.set_linewidth(0.4)
    else:
        axH.set_axis_off()

    save_both(fig, out_dir / "Fig1_main_performance_suite")


# =========================
# Plot 2: Safety/triage suite
# =========================
def plot_safety_triage_suite(metrics, out_dir: Path):
    if metrics.empty:
        print("[Skip] metrics empty: Fig2")
        return

    sm = metrics[(metrics["accuracy"].notna()) & (metrics["macro_f1"].notna())].copy()
    if sm.empty:
        print("[Skip] no safety metrics: Fig2")
        return

    fig = plt.figure(figsize=(16.8, 9.8), constrained_layout=True)
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1.05, 1.10, 1.25], height_ratios=[1, 1])
    fig.suptitle("Safety-oriented evaluation: coverage, review burden, and error capture",
                 fontsize=16.5, fontweight="bold", y=1.02)

    # A. Coverage vs error capture bubble
    axA = fig.add_subplot(gs[0, 0])
    add_panel_label(axA, "A")
    sdata = sm.copy()
    if "coverage" not in sdata.columns:
        sdata["coverage"] = 1.0
    if "error_capture_rate" not in sdata.columns:
        sdata["error_capture_rate"] = 0.0
    if "review_rate" not in sdata.columns:
        sdata["review_rate"] = 0.0
    sdata[["coverage", "error_capture_rate", "review_rate"]] = sdata[["coverage", "error_capture_rate", "review_rate"]].fillna(0)
    sizes = 70 + 600 * sdata["review_rate"].clip(0, 1)
    colors = [color_for_method(m) for m in sdata["method_short"]]
    axA.scatter(sdata["coverage"], sdata["error_capture_rate"], s=sizes, c=colors, alpha=0.72,
                edgecolor="white", linewidth=0.8)
    # annotate selected FVA points and top error capture points
    ann = sdata[(sdata["method_short"].str.contains("FVA", case=False, na=False)) |
                (sdata["error_capture_rate"].rank(ascending=False) <= 4)]
    for _, r in ann.head(10).iterrows():
        axA.annotate(short_method_name(r["method_short"]), (r["coverage"], r["error_capture_rate"]),
                     xytext=(4, 4), textcoords="offset points", fontsize=7.2)
    axA.set_xlabel("Automatic coverage")
    axA.set_ylabel("Error capture rate")
    axA.set_xlim(-0.03, 1.05)
    axA.set_ylim(-0.03, 1.05)
    axA.set_title("Coverage–error-capture trade-off")

    # B. Review burden vs auto accuracy
    axB = fig.add_subplot(gs[0, 1])
    add_panel_label(axB, "B")
    axB.scatter(sdata["review_rate"], sdata["accuracy"], s=70 + 500*sdata["coverage"].clip(0,1),
                c=sdata["macro_f1"], cmap="viridis", alpha=0.75, edgecolor="white", linewidth=0.8)
    for _, r in sdata.sort_values("accuracy", ascending=False).head(6).iterrows():
        axB.annotate(short_method_name(r["method_short"]), (r["review_rate"], r["accuracy"]),
                     xytext=(4, 4), textcoords="offset points", fontsize=7.2)
    axB.set_xlabel("Review rate")
    axB.set_ylabel("Accuracy")
    axB.set_xlim(-0.03, 1.05)
    axB.set_ylim(0, 1.05)
    axB.set_title("Review burden and accuracy")
    cbar = fig.colorbar(axB.collections[0], ax=axB, shrink=0.8, pad=0.01)
    cbar.set_label("Macro-F1")

    # C. Selective curves area/line
    axC = fig.add_subplot(gs[0, 2])
    add_panel_label(axC, "C")
    curve_files = sorted(Path("outputs").glob("forensic_agent_v2_selective_curve_*.csv"))
    if not curve_files:
        curve_files = sorted((out_dir.parent / "outputs").glob("forensic_agent_v2_selective_curve_*.csv"))
    if curve_files:
        for f in curve_files:
            cd = pd.read_csv(f)
            label = f.stem.replace("forensic_agent_v2_selective_curve_", "")
            x = pd.to_numeric(cd["coverage"], errors="coerce")
            for metric, marker, alpha in [("accuracy", "o", 0.18), ("macro_f1", "s", 0.10)]:
                if metric in cd.columns:
                    y = pd.to_numeric(cd[metric], errors="coerce")
                    axC.plot(x, y, marker=marker, linewidth=2, label=f"{label} {metric}")
                    axC.fill_between(x, y, alpha=alpha)
        axC.set_xlabel("Coverage retained")
        axC.set_ylabel("Score")
        axC.set_ylim(0, 1.05)
        axC.set_title("Selective prediction area curves")
        axC.legend(loc="lower left", frameon=False, ncol=2, fontsize=7.5)
    else:
        axC.text(0.5, 0.5, "Selective curves not found", ha="center", va="center")
        axC.set_axis_off()

    # D. Review triage reasons matrix/bubble
    axD = fig.add_subplot(gs[1, 0])
    add_panel_label(axD, "D")
    pred_files = sorted(Path("outputs").glob("forensic_agent_v2_predictions_*.csv"))
    if not pred_files:
        pred_files = sorted((out_dir.parent / "outputs").glob("forensic_agent_v2_predictions_*.csv"))
    rows = []
    for f in pred_files:
        pdf = pd.read_csv(f)
        model = f.stem.replace("forensic_agent_v2_predictions_", "")
        if "triage_reasons" in pdf.columns:
            for reason in pdf["triage_reasons"].fillna(""):
                for r in str(reason).split("|"):
                    if r.strip():
                        rows.append({"model": model, "reason": r.strip()})
    if rows:
        rr = pd.DataFrame(rows).groupby(["model", "reason"]).size().reset_index(name="n")
        xs = list(rr["model"].unique())
        ys = list(rr["reason"].unique())
        xmap = {v:i for i, v in enumerate(xs)}
        ymap = {v:i for i, v in enumerate(ys)}
        rr["x"] = rr["model"].map(xmap)
        rr["y"] = rr["reason"].map(ymap)
        axD.scatter(rr["x"], rr["y"], s=rr["n"]*0.55, c=PALETTE["purple"], alpha=0.55, edgecolor="white")
        for _, r in rr.iterrows():
            axD.text(r["x"], r["y"], str(int(r["n"])), ha="center", va="center", fontsize=7)
        axD.set_xticks(range(len(xs)))
        axD.set_xticklabels(xs)
        axD.set_yticks(range(len(ys)))
        axD.set_yticklabels([wrap_label(y, 18) for y in ys])
        axD.set_title("Human-review triage reason map")
    else:
        axD.text(0.5, 0.5, "Triage reason data not found", ha="center", va="center")
        axD.set_axis_off()

    # E. Auto vs all-case paired bars
    axE = fig.add_subplot(gs[1, 1])
    add_panel_label(axE, "E")
    fva = sm[sm["method_short"].str.contains("FVA-v2", case=False, na=False)].copy()
    if not fva.empty:
        fva = fva.sort_values("accuracy", ascending=False).head(10)
        x = np.arange(len(fva))
        width = 0.38
        axE.bar(x - width/2, fva["accuracy"], width, color=PALETTE["blue"], alpha=0.85, label="Accuracy")
        axE.bar(x + width/2, fva["macro_f1"], width, color=PALETTE["orange"], alpha=0.85, label="Macro-F1")
        axE.set_xticks(x)
        axE.set_xticklabels([wrap_label(x, 12) for x in fva["method_short"]], rotation=25, ha="right")
        axE.set_ylim(0, 1.05)
        axE.set_title("FVA-v2 performance variants")
        axE.legend(frameon=False)
    else:
        axE.text(0.5, 0.5, "FVA-v2 rows not found", ha="center", va="center")
        axE.set_axis_off()

    # F. Decision policy table
    axF = fig.add_subplot(gs[1, 2])
    add_panel_label(axF, "F")
    table_df = sm.copy()
    table_df = table_df[table_df["method_short"].str.contains("FVA|Struct-RF|Narr", case=False, na=False)]
    table_df = table_df.sort_values(["error_capture_rate", "macro_f1"], ascending=False).head(8)
    show_cols = [c for c in ["method_short", "coverage", "review_rate", "error_capture_rate", "accuracy", "macro_f1"] if c in table_df.columns]
    if not table_df.empty:
        disp = table_df[show_cols].copy()
        for c in disp.columns:
            if c != "method_short":
                disp[c] = disp[c].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        disp["method_short"] = disp["method_short"].map(lambda x: wrap_label(x, 18))
        axF.axis("off")
        tb = axF.table(cellText=disp.values,
                       colLabels=["Method" if c=="method_short" else c.replace("_", "\n").title() for c in disp.columns],
                       loc="center", cellLoc="center", colLoc="center")
        tb.auto_set_font_size(False)
        tb.set_fontsize(7.6)
        tb.scale(1, 1.4)
        for (row, col), cell in tb.get_celld().items():
            if row == 0:
                cell.set_facecolor(PALETTE["dark"])
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_edgecolor("#CBD5E1")
                cell.set_linewidth(0.4)
        axF.set_title("Safety-policy summary", pad=8)
    else:
        axF.set_axis_off()

    save_both(fig, out_dir / "Fig2_safety_triage_suite")


# =========================
# Plot 3: Error/distribution suite
# =========================
def plot_error_distribution_suite(metrics, label_dist, out_dir: Path):
    fig = plt.figure(figsize=(15.8, 9.2), constrained_layout=True)
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1.1], height_ratios=[1, 1])
    fig.suptitle("Distributional structure and error-analysis views", fontsize=16.5, fontweight="bold", y=1.02)

    # A. Label distribution pareto
    axA = fig.add_subplot(gs[0, 0])
    add_panel_label(axA, "A")
    if not label_dist.empty:
        ld = label_dist.copy()
        label_col = next((c for c in ld.columns if "label" in c.lower() or "target" in c.lower() or "broad" in c.lower()), ld.columns[0])
        count_col = next((c for c in ld.columns if "count" in c.lower() or "n" == c.lower()), ld.columns[-1])
        ld[count_col] = pd.to_numeric(ld[count_col], errors="coerce")
        ld = ld.dropna(subset=[count_col]).sort_values(count_col, ascending=False).head(12)
        axA.bar(np.arange(len(ld)), ld[count_col], color=PALETTE["blue"], alpha=0.85)
        axA.set_xticks(np.arange(len(ld)))
        axA.set_xticklabels([wrap_label(x, 12) for x in ld[label_col]], rotation=35, ha="right")
        axA.set_ylabel("Cases")
        axA.set_title("Class-size distribution")
        axA2 = axA.twinx()
        cum = ld[count_col].cumsum() / ld[count_col].sum()
        axA2.plot(np.arange(len(ld)), cum, color=PALETTE["red"], marker="o", linewidth=2)
        axA2.set_ylabel("Cumulative fraction")
        axA2.set_ylim(0, 1.05)
        axA2.grid(False)
    else:
        axA.text(0.5, 0.5, "No label distribution", ha="center", va="center")
        axA.set_axis_off()

    # B. Broad confusion matrix if available
    axB = fig.add_subplot(gs[0, 1])
    add_panel_label(axB, "B")
    cm_files = list(Path("outputs").glob("cm_broad_*.csv")) + list((out_dir.parent / "outputs").glob("cm_broad_*.csv"))
    if cm_files:
        # choose a strong-looking RF/FVA file if exists else first
        choice = next((p for p in cm_files if "RandomForest" in p.name or "RF" in p.name), cm_files[0])
        cm = pd.read_csv(choice, index_col=0)
        # normalize rows
        cmn = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        im = axB.imshow(cmn.values, cmap="Blues", vmin=0, vmax=max(0.01, cmn.values.max()))
        axB.set_xticks(np.arange(len(cmn.columns)))
        axB.set_yticks(np.arange(len(cmn.index)))
        axB.set_xticklabels([wrap_label(c, 10) for c in cmn.columns], rotation=45, ha="right", fontsize=7)
        axB.set_yticklabels([wrap_label(i, 10) for i in cmn.index], fontsize=7)
        axB.set_title(f"Row-normalized confusion matrix\n{choice.stem.replace('cm_broad_', '')}")
        cbar = fig.colorbar(im, ax=axB, shrink=0.75, pad=0.01)
        cbar.set_label("Row-normalized")
    else:
        axB.text(0.5, 0.5, "Confusion matrix file\nnot found", ha="center", va="center")
        axB.set_axis_off()

    # C. Method family summary with error bars-like min/max
    axC = fig.add_subplot(gs[0, 2])
    add_panel_label(axC, "C")
    if not metrics.empty and "method_group" in metrics.columns:
        mf = metrics.dropna(subset=["macro_f1"]).groupby(["method_group", "task"]).agg(
            mean_f1=("macro_f1", "mean"),
            max_f1=("macro_f1", "max"),
            min_f1=("macro_f1", "min")
        ).reset_index()
        groups = list(mf["method_group"].unique())
        tasks = list(mf["task"].unique())
        x = np.arange(len(groups))
        width = 0.8 / max(1, len(tasks))
        for j, task in enumerate(tasks):
            sub = mf[mf["task"] == task].set_index("method_group").reindex(groups)
            vals = sub["mean_f1"].values
            axC.bar(x + (j - len(tasks)/2)*width + width/2, vals, width=width,
                    label=str(task), alpha=0.85)
        axC.set_xticks(x)
        axC.set_xticklabels(groups, rotation=20, ha="right")
        axC.set_ylabel("Mean Macro-F1")
        axC.set_ylim(0, 1.0)
        axC.set_title("Performance by method family")
        axC.legend(frameon=False, ncol=2)
    else:
        axC.set_axis_off()

    # D. External task dedicated ranking
    axD = fig.add_subplot(gs[1, 0:2])
    add_panel_label(axD, "D")
    if not metrics.empty:
        ext = metrics[(metrics["task"].astype(str) == "external") & metrics["macro_f1"].notna()].copy()
        if ext.empty:
            ext = metrics[(metrics["task"].astype(str).str.contains("external", case=False, na=False)) & metrics["macro_f1"].notna()].copy()
        if not ext.empty:
            top = ext.sort_values("macro_f1", ascending=False).head(12).sort_values("macro_f1")
            axD.barh(top["method_display"], top["macro_f1"], color=[color_for_method(m) for m in top["method_short"]], alpha=0.88)
            value_annotate_barh(axD, top["macro_f1"].values)
            axD.set_xlabel("Macro-F1")
            axD.set_xlim(0, 1.02)
            axD.set_title("Medico-legal external death detection performance")
            axD.tick_params(axis="y", labelsize=7.8)
        else:
            axD.text(0.5, 0.5, "External-task rows not found", ha="center", va="center")
            axD.set_axis_off()
    else:
        axD.set_axis_off()

    # E. Top method detailed table
    axE = fig.add_subplot(gs[1, 2])
    add_panel_label(axE, "E")
    if not metrics.empty:
        top = metrics.dropna(subset=["macro_f1"]).sort_values("macro_f1", ascending=False).head(8)
        cols = [c for c in ["task", "method_short", "accuracy", "macro_f1", "weighted_f1"] if c in top.columns]
        disp = top[cols].copy()
        for c in disp.columns:
            if c not in ["task", "method_short"]:
                disp[c] = disp[c].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        disp["method_short"] = disp["method_short"].map(lambda x: wrap_label(x, 16))
        axE.axis("off")
        tb = axE.table(cellText=disp.values,
                       colLabels=["Task" if c=="task" else "Method" if c=="method_short" else c.replace("_", "\n").title() for c in disp.columns],
                       loc="center", cellLoc="center", colLoc="center")
        tb.auto_set_font_size(False)
        tb.set_fontsize(7.5)
        tb.scale(1, 1.45)
        for (row, col), cell in tb.get_celld().items():
            if row == 0:
                cell.set_facecolor(PALETTE["dark"])
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_edgecolor("#CBD5E1")
                cell.set_linewidth(0.4)
        axE.set_title("Top-performing rows", pad=8)
    else:
        axE.set_axis_off()

    save_both(fig, out_dir / "Fig3_error_distribution_suite")


# =========================
# Plot 4: LLM diagnostics
# =========================
def plot_llm_diagnostics_suite(outputs_dir: Path, out_dir: Path):
    llm = read_csv_optional(outputs_dir / "ollama_llm_metrics.csv")
    if llm.empty:
        print("[Skip] no ollama_llm_metrics.csv: Fig4")
        return
    llm = clean_metrics(llm)

    fig = plt.figure(figsize=(15.8, 5.8), constrained_layout=True)
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.05, 1.05, 1.2])
    fig.suptitle("Local LLM prompting diagnostics", fontsize=15.5, fontweight="bold", y=1.03)

    axA = fig.add_subplot(gs[0, 0])
    add_panel_label(axA, "A")
    x = np.arange(len(llm))
    parse = llm["parse_error_rate"].fillna(0) if "parse_error_rate" in llm.columns else pd.Series(np.zeros(len(llm)))
    axA.bar(x, parse, color=PALETTE["red"], alpha=0.75)
    axA.set_xticks(x)
    axA.set_xticklabels([wrap_label(short_method_name(m), 14) for m in llm["method"]], rotation=30, ha="right")
    axA.set_ylabel("Parse error rate")
    axA.set_ylim(0, 1.05)
    axA.set_title("Output parseability")

    axB = fig.add_subplot(gs[0, 1])
    add_panel_label(axB, "B")
    if "n_test" in llm.columns and "valid_n" in llm.columns:
        n_test = pd.to_numeric(llm["n_test"], errors="coerce").fillna(0)
        valid = pd.to_numeric(llm["valid_n"], errors="coerce").fillna(0)
        axB.bar(x, n_test, color=PALETTE["lightgray"], label="Generated")
        axB.bar(x, valid, color=PALETTE["blue"], label="Valid parsed")
        axB.set_xticks(x)
        axB.set_xticklabels([wrap_label(short_method_name(m), 14) for m in llm["method"]], rotation=30, ha="right")
        axB.set_ylabel("Cases")
        axB.set_title("Generated vs valid parsed cases")
        axB.legend(frameon=False)
    else:
        axB.text(0.5, 0.5, "n_test / valid_n missing", ha="center", va="center")
        axB.set_axis_off()

    axC = fig.add_subplot(gs[0, 2])
    add_panel_label(axC, "C")
    width = 0.36
    if "accuracy" in llm.columns:
        axC.bar(x - width/2, llm["accuracy"].fillna(0), width, color=PALETTE["green"], alpha=0.8, label="Accuracy")
    if "macro_f1" in llm.columns:
        axC.bar(x + width/2, llm["macro_f1"].fillna(0), width, color=PALETTE["orange"], alpha=0.8, label="Macro-F1")
    axC.set_xticks(x)
    axC.set_xticklabels([wrap_label(short_method_name(m), 14) for m in llm["method"]], rotation=30, ha="right")
    axC.set_ylim(0, 1.05)
    axC.set_title("LLM baseline on parsed subset")
    axC.legend(frameon=False)

    save_both(fig, out_dir / "Fig4_llm_diagnostics_suite")


# =========================
# Plot 5: 3D safety landscape
# =========================
def plot_3d_safety_landscape(metrics, out_dir: Path):
    if metrics.empty:
        return
    df = metrics.dropna(subset=["accuracy", "macro_f1"]).copy()
    if df.empty:
        return
    for c, default in [("coverage", 1.0), ("review_rate", 0.0), ("error_capture_rate", 0.0)]:
        if c not in df.columns:
            df[c] = default
        df[c] = df[c].fillna(default)

    fig = plt.figure(figsize=(9.2, 7.4), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    colors = df["macro_f1"].fillna(0).values
    sizes = 45 + 350 * df["error_capture_rate"].clip(0, 1)
    sc = ax.scatter(df["coverage"], df["review_rate"], df["accuracy"],
                    c=colors, s=sizes, cmap="plasma", alpha=0.78,
                    edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Coverage", labelpad=8)
    ax.set_ylabel("Review rate", labelpad=8)
    ax.set_zlabel("Accuracy", labelpad=8)
    ax.set_title("3D safety-performance landscape", pad=18, fontsize=13)
    ax.view_init(elev=25, azim=-55)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.08)
    cbar.set_label("Macro-F1")

    # annotate top FVA methods only
    ann = df[df["method_short"].str.contains("FVA-v2|Struct-RF", case=False, na=False)].sort_values("accuracy", ascending=False).head(7)
    for _, r in ann.iterrows():
        ax.text(r["coverage"], r["review_rate"], r["accuracy"], short_method_name(r["method_short"]), fontsize=7)

    save_both(fig, out_dir / "Fig5_3d_safety_landscape")


# =========================
# Tables
# =========================
def export_tables(metrics, out_dir: Path):
    table_dir = out_dir / "tables"
    ensure_dir(table_dir)
    if metrics.empty:
        return
    metrics.to_csv(table_dir / "all_metrics_cleaned_for_paper.csv", index=False)
    if "macro_f1" in metrics.columns:
        metrics.sort_values("macro_f1", ascending=False).head(30).to_csv(table_dir / "top30_methods_by_macro_f1.csv", index=False)
    if "task" in metrics.columns:
        for task, sub in metrics.groupby("task"):
            sub.sort_values("macro_f1", ascending=False).head(20).to_csv(table_dir / f"top_methods_{task}.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default="outputs", help="Path to outputs directory")
    parser.add_argument("--figures", default="figures_pub", help="Path to output figure directory")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs)
    out_dir = Path(args.figures)
    ensure_dir(out_dir)

    metrics = clean_metrics(read_csv_optional(outputs_dir / "all_metrics.csv"))
    label_dist = read_csv_optional(outputs_dir / "label_distribution.csv")

    print(f"[Info] Loaded metrics: {metrics.shape if not metrics.empty else 'EMPTY'}")
    print(f"[Info] Loaded label distribution: {label_dist.shape if not label_dist.empty else 'EMPTY'}")

    plot_main_performance_suite(metrics, label_dist, out_dir)
    plot_safety_triage_suite(metrics, out_dir)
    plot_error_distribution_suite(metrics, label_dist, out_dir)
    plot_llm_diagnostics_suite(outputs_dir, out_dir)
    plot_3d_safety_landscape(metrics, out_dir)
    export_tables(metrics, out_dir)

    print(f"[Done] Publication figures saved to: {out_dir.resolve()}")
    print("[Files]")
    for p in sorted(out_dir.glob("*.png")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
