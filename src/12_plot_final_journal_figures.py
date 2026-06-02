#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINAL journal-grade figures for ForensicVA-Agent.

Design rules in this final version:
1) No point labels inside scatter plots.
2) Panel letters are placed outside axes with fixed coordinates.
3) No tables inside figures. Tables are exported separately as CSV.
4) Long text is shortened aggressively.
5) Each figure uses only 2–4 panels.
6) Layout uses large margins and constrained_layout.
7) LLM diagnostics are supplementary only and are visually separated.

Run from project root:
    python src/12_plot_final_journal_figures.py --outputs outputs --figures figures_final

Generated:
    figures_final/Fig1_final_main_results.png/.pdf
    figures_final/Fig2_final_safety_triage.png/.pdf
    figures_final/Fig3_final_error_external.png/.pdf
    figures_final/FigS1_final_llm_diagnostics.png/.pdf
    figures_final/FigS2_final_selective_curves_only.png/.pdf
    figures_final/tables/*.csv
"""

import argparse
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import gridspec

warnings.filterwarnings("ignore")

# ======================
# Global style
# ======================
COL = {
    "fva": "#E64B5D",
    "struct": "#4C78A8",
    "narr": "#2CB7A4",
    "agent": "#F2A93B",
    "llm": "#8A63D2",
    "gray": "#6B7280",
    "light": "#E5E7EB",
    "dark": "#111827",
    "green": "#38A169",
    "blue": "#3B82F6",
    "red": "#E64B5D",
    "orange": "#F59E0B",
    "teal": "#14B8A6",
}

plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 500,
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.titlesize": 12.5,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.16,
    "grid.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ======================
# Helpers
# ======================
def mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def savefig(fig, path: Path):
    mkdir(path.parent)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def wrap(s, width=14):
    return "\n".join(textwrap.wrap(str(s), width=width, break_long_words=False))


def safe_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def simple_method(s):
    s = str(s)
    repl = {
        "ForensicVA-Agent-v2-": "FVA-v2-",
        "ForensicVA-Agent-": "FVA-",
        "Narrative_TFIDF_": "Narr-",
        "Narrative-TFIDF-": "Narr-",
        "Structured_": "Struct-",
        "Structured-": "Struct-",
        "RandomForest": "RF",
        "LinearSVM": "SVM",
        "LogReg": "LR",
        "logreg": "LR",
        "rf": "RF",
        "all_cases": "all",
        "auto_decided": "auto",
        "Direct prior baseline": "Prior",
        "Evidence agent": "Evidence",
        "Evidence + verification": "Evid+Verify",
        "Full triage agent: baseline auto-decided cases only": "FVA-v1-auto",
        "LLM_": "LLM-",
        "qwen2.5:": "qwen",
        "llama3.2:": "llama",
        "gemma3:": "gemma",
        "deepseek-r1:": "deepseek",
        "mistral:": "mistral",
    }
    for a, b in repl.items():
        s = s.replace(a, b)
    s = s.replace("_", "-")
    return s


def family(method_short):
    m = str(method_short)
    if "FVA" in m:
        return "Proposed FVA"
    if "Struct" in m:
        return "Structured ML"
    if "Narr" in m:
        return "Narrative TF-IDF"
    if "LLM" in m or "qwen" in m or "llama" in m or "gemma" in m:
        return "Local LLM"
    if "Prior" in m or "Evidence" in m or "Evid" in m:
        return "Agent ablation"
    return "Other"


def color_family(method_short):
    fam = family(method_short)
    return {
        "Proposed FVA": COL["fva"],
        "Structured ML": COL["struct"],
        "Narrative TF-IDF": COL["narr"],
        "Agent ablation": COL["agent"],
        "Local LLM": COL["llm"],
        "Other": COL["gray"],
    }.get(fam, COL["gray"])


def load_metrics(outputs: Path) -> pd.DataFrame:
    df = read_csv(outputs / "all_metrics.csv")
    if df.empty:
        raise FileNotFoundError(f"Missing {outputs / 'all_metrics.csv'}")
    df = safe_numeric(df, ["accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall",
                           "coverage", "review_rate", "error_capture_rate", "n_test",
                           "valid_n", "parse_error_rate"])
    if "task" not in df.columns:
        df["task"] = "broad"
    if "method" not in df.columns:
        df["method"] = "unknown"
    df["method_short"] = df["method"].map(simple_method)
    df["family"] = df["method_short"].map(family)
    return df


def best_rows(df, keys=("task", "method_short"), metric="macro_f1"):
    if df.empty:
        return df
    tmp = df.copy()
    tmp["_metric"] = tmp[metric].fillna(-1)
    idx = tmp.sort_values("_metric", ascending=False).groupby(list(keys), as_index=False).head(1).index
    return tmp.loc[idx].drop(columns=["_metric"], errors="ignore")


def load_label_dist(outputs: Path) -> pd.DataFrame:
    ld = read_csv(outputs / "label_distribution.csv")
    if not ld.empty:
        return ld
    processed = outputs / "phmrc_forensicva_processed.csv"
    if processed.exists():
        df = pd.read_csv(processed)
        for c in ["gold_broad", "broad", "target_broad", "broad_label"]:
            if c in df.columns:
                return df[c].value_counts().rename_axis("label").reset_index(name="count")
    return pd.DataFrame()


def label_count_cols(ld):
    label_col = next((c for c in ld.columns if "label" in c.lower() or "broad" in c.lower()), ld.columns[0])
    count_col = next((c for c in ld.columns if "count" in c.lower() or c.lower() in ["n", "cases"]), ld.columns[-1])
    return label_col, count_col


def add_panel_letters(fig, axes, letters):
    """
    Put panel letters in the figure coordinate system, outside each axes.
    This prevents collision with titles and plot content.
    """
    fig.canvas.draw()
    for ax, letter in zip(axes, letters):
        pos = ax.get_position()
        fig.text(
            pos.x0 - 0.018,
            pos.y1 + 0.010,
            letter,
            fontsize=15,
            fontweight="bold",
            ha="left",
            va="bottom",
            color=COL["dark"],
        )


def add_family_legend(fig, y=0.01, ncol=5):
    handles = [
        Patch(facecolor=COL["fva"], label="Proposed FVA"),
        Patch(facecolor=COL["struct"], label="Structured ML"),
        Patch(facecolor=COL["narr"], label="Narrative TF-IDF"),
        Patch(facecolor=COL["agent"], label="Agent ablation"),
        Patch(facecolor=COL["llm"], label="Local LLM"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y),
               ncol=ncol, frameon=False)


def set_axis_clean(ax):
    ax.grid(True, alpha=0.16)
    ax.tick_params(axis="both", length=3, width=0.8)


# ======================
# Fig1: Main result
# ======================
def plot_fig1(metrics, label_dist, figures):
    broad = metrics[(metrics["task"].astype(str) == "broad") & metrics["macro_f1"].notna()].copy()
    broad = best_rows(broad, ("task", "method_short"), "macro_f1")

    fig, axes = plt.subplots(2, 2, figsize=(16.8, 10.4), constrained_layout=True)
    fig.suptitle("ForensicVA-Agent classification performance", fontsize=18, fontweight="bold")

    # A dataset
    ax = axes[0, 0]
    if not label_dist.empty:
        label_col, count_col = label_count_cols(label_dist)
        ld = label_dist.copy()
        ld[count_col] = pd.to_numeric(ld[count_col], errors="coerce")
        ld = ld.dropna(subset=[count_col]).sort_values(count_col, ascending=True)
        ax.barh([wrap(x, 18) for x in ld[label_col]], ld[count_col], color=COL["struct"], alpha=0.88)
        ax.set_xlabel("Number of cases")
        ax.set_title("Dataset label distribution", pad=12)
    else:
        ax.text(0.5, 0.5, "Label distribution not found", ha="center", va="center")
    set_axis_clean(ax)

    # B broad ranking
    ax = axes[0, 1]
    top = broad.sort_values("macro_f1", ascending=False).head(12).sort_values("macro_f1", ascending=True)
    y = np.arange(len(top))
    ax.barh(y, top["macro_f1"], color=[color_family(m) for m in top["method_short"]], alpha=0.92)
    ax.set_yticks(y)
    ax.set_yticklabels([wrap(m, 20) for m in top["method_short"]])
    for i, v in enumerate(top["macro_f1"]):
        ax.text(v + 0.006, i, f"{v:.3f}", va="center", fontsize=8.5)
    ax.set_xlim(0, max(0.76, min(1.0, top["macro_f1"].max() + 0.08)))
    ax.set_xlabel("Macro-F1")
    ax.set_title("Broad cause-of-death ranking", pad=12)
    set_axis_clean(ax)

    # C scatter WITHOUT labels
    ax = axes[1, 0]
    sc = broad.dropna(subset=["accuracy", "macro_f1"]).copy()
    if not sc.empty:
        review = sc["review_rate"].fillna(0) if "review_rate" in sc.columns else pd.Series(np.zeros(len(sc)))
        sizes = 70 + 420 * review.clip(0, 1)
        ax.scatter(sc["accuracy"], sc["macro_f1"],
                   s=sizes,
                   c=[color_family(m) for m in sc["method_short"]],
                   alpha=0.74, edgecolor="white", linewidth=0.9)
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("Macro-F1")
        ax.set_title("Accuracy–F1–review trade-off", pad=12)
        ax.set_xlim(max(0, sc["accuracy"].min() - 0.07), min(1, sc["accuracy"].max() + 0.07))
        ax.set_ylim(max(0, sc["macro_f1"].min() - 0.07), min(1, sc["macro_f1"].max() + 0.07))
        # Size legend only, no text labels inside plot.
        example_handles = [
            Line2D([0], [0], marker="o", color="w", label="low review",
                   markerfacecolor=COL["gray"], markersize=7, alpha=0.7),
            Line2D([0], [0], marker="o", color="w", label="high review",
                   markerfacecolor=COL["gray"], markersize=13, alpha=0.7),
        ]
        ax.legend(handles=example_handles, title="Bubble size", frameon=False, loc="lower right")
    set_axis_clean(ax)

    # D heatmap
    ax = axes[1, 1]
    hm = metrics.dropna(subset=["macro_f1"]).copy()
    hm = best_rows(hm, ("task", "method_short"), "macro_f1")
    # Only show strongest clean rows, no LLM rows in main heatmap.
    allow = hm[~hm["family"].isin(["Local LLM", "Other"])].copy()
    # Keep top by broad plus all structured/FVA/narr.
    keep = set(allow.sort_values("macro_f1", ascending=False).head(10)["method_short"])
    keep.update(allow[allow["method_short"].str.contains("FVA-v2|Struct|Narr", case=False, na=False)]["method_short"].head(10))
    hm = allow[allow["method_short"].isin(keep)]
    pivot = hm.pivot_table(index="method_short", columns="task", values="macro_f1", aggfunc="max")
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(0.01, np.nanmax(pivot.values)))
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([wrap(x, 18) for x in pivot.index])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if val > 0.56 else COL["dark"])
    ax.set_title("Task-wise Macro-F1 heatmap", pad=12)
    cbar = fig.colorbar(im, ax=ax, shrink=0.86, pad=0.012)
    cbar.set_label("Macro-F1")

    add_panel_letters(fig, axes.flat, ["A", "B", "C", "D"])
    add_family_legend(fig, y=-0.015, ncol=5)

    savefig(fig, figures / "Fig1_final_main_results")


# ======================
# Fig2: Safety
# ======================
def plot_fig2(metrics, outputs, figures):
    df = metrics.dropna(subset=["accuracy", "macro_f1"]).copy()
    for c, default in [("coverage", 1.0), ("review_rate", 0.0), ("error_capture_rate", 0.0)]:
        if c not in df.columns:
            df[c] = default
        df[c] = df[c].fillna(default)
    df = best_rows(df, ("task", "method_short"), "macro_f1")
    broad = df[df["task"].astype(str) == "broad"].copy()
    if broad.empty:
        broad = df.copy()

    fig, axes = plt.subplots(2, 2, figsize=(16.8, 10.2), constrained_layout=True)
    fig.suptitle("Safety-oriented human-review triage analysis", fontsize=18, fontweight="bold")

    # A no internal labels, external legend
    ax = axes[0, 0]
    sdf = broad[broad["method_short"].str.contains("FVA-v2|Struct-RF", case=False, na=False)].copy()
    if sdf.empty:
        sdf = broad.copy()
    ax.scatter(sdf["coverage"], sdf["error_capture_rate"],
               s=90 + 420 * sdf["review_rate"].clip(0, 1),
               c=[color_family(m) for m in sdf["method_short"]],
               alpha=0.76, edgecolor="white", linewidth=0.9)
    ax.set_xlabel("Automatic coverage")
    ax.set_ylabel("Error capture rate")
    ax.set_xlim(-0.04, 1.06)
    ax.set_ylim(-0.04, 1.06)
    ax.set_title("Coverage–error-capture trade-off", pad=12)
    # clean marker legend outside plot content
    legend_rows = sdf.sort_values(["error_capture_rate", "coverage"], ascending=False).head(6)
    handles = [Line2D([0], [0], marker="o", color="w", label=m,
                      markerfacecolor=color_family(m), markersize=8)
               for m in legend_rows["method_short"]]
    ax.legend(handles=handles, frameon=False, loc="lower left", bbox_to_anchor=(0.02, 0.02))
    set_axis_clean(ax)

    # B selective curves, legend outside
    ax = axes[0, 1]
    curve_files = sorted(outputs.glob("forensic_agent_v2_selective_curve_*.csv"))
    if curve_files:
        styles = {
            "logreg accuracy": (COL["struct"], "o", "-"),
            "logreg macro_f1": (COL["agent"], "s", "--"),
            "rf accuracy": (COL["green"], "o", "-"),
            "rf macro_f1": (COL["red"], "s", "--"),
        }
        for f in curve_files:
            cd = pd.read_csv(f)
            name = f.stem.replace("forensic_agent_v2_selective_curve_", "")
            if "coverage" not in cd.columns:
                continue
            x = pd.to_numeric(cd["coverage"], errors="coerce")
            for met in ["accuracy", "macro_f1"]:
                if met in cd.columns:
                    y = pd.to_numeric(cd[met], errors="coerce")
                    key = f"{name} {met}"
                    color, marker, ls = styles.get(key, (COL["gray"], "o", "-"))
                    ax.plot(x, y, marker=marker, linestyle=ls, color=color, linewidth=2.1, label=key.replace("_", "-"))
                    ax.fill_between(x, y, alpha=0.055, color=color)
        ax.set_xlabel("Coverage retained")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.set_title("Selective prediction curves", pad=12)
        ax.legend(frameon=False, loc="lower left", ncol=2)
    else:
        ax.text(0.5, 0.5, "Selective curve files not found", ha="center", va="center")
    set_axis_clean(ax)

    # C confidence boxplot
    ax = axes[1, 0]
    pred_files = sorted(outputs.glob("forensic_agent_v2_predictions_*.csv"))
    rows = []
    for f in pred_files:
        pdf = pd.read_csv(f)
        model = f.stem.replace("forensic_agent_v2_predictions_", "")
        if {"y_true", "final_pred", "confidence"}.issubset(pdf.columns):
            temp = pdf[["y_true", "final_pred", "confidence"]].copy()
            temp["model"] = model
            temp["correct"] = temp["y_true"].astype(str) == temp["final_pred"].astype(str)
            rows.append(temp)
    if rows:
        pdx = pd.concat(rows, ignore_index=True)
        box_data, labels, colors = [], [], []
        for model in ["logreg", "rf"]:
            for correct in [True, False]:
                sub = pdx[(pdx["model"] == model) & (pdx["correct"] == correct)]["confidence"].dropna()
                if len(sub):
                    box_data.append(sub.values)
                    labels.append(f"{model}\n{'correct' if correct else 'wrong'}")
                    colors.append(COL["green"] if correct else COL["red"])
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True, showfliers=False,
                        medianprops=dict(color=COL["orange"], linewidth=1.5))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.50)
        ax.set_ylabel("Classifier confidence")
        ax.set_title("Confidence separation by correctness", pad=12)
    else:
        ax.text(0.5, 0.5, "Prediction files not found", ha="center", va="center")
    set_axis_clean(ax)

    # D reason map, no axis crowd
    ax = axes[1, 1]
    reason_rows = []
    for f in pred_files:
        pdf = pd.read_csv(f)
        model = f.stem.replace("forensic_agent_v2_predictions_", "")
        if "triage_reasons" in pdf.columns:
            for rs in pdf["triage_reasons"].fillna(""):
                for r in str(rs).split("|"):
                    r = r.strip()
                    if r:
                        reason_rows.append({"model": model, "reason": r})
    if reason_rows:
        rr = pd.DataFrame(reason_rows).groupby(["model", "reason"]).size().reset_index(name="n")
        models = [m for m in ["logreg", "rf"] if m in rr["model"].unique()]
        if not models:
            models = sorted(rr["model"].unique())
        reason_order = ["high_medico_legal_risk", "low_confidence", "small_probability_margin", "weak_textual_evidence"]
        reasons = [r for r in reason_order if r in rr["reason"].unique()]
        xmap = {m: i for i, m in enumerate(models)}
        ymap = {r: i for i, r in enumerate(reasons)}
        rr = rr[rr["model"].isin(models) & rr["reason"].isin(reasons)].copy()
        rr["x"] = rr["model"].map(xmap)
        rr["y"] = rr["reason"].map(ymap)
        ax.scatter(rr["x"], rr["y"], s=np.maximum(80, rr["n"] * 0.45),
                   color=COL["llm"], alpha=0.55, edgecolor="white", linewidth=0.9)
        for _, r in rr.iterrows():
            ax.text(r["x"], r["y"], str(int(r["n"])), ha="center", va="center", fontsize=8.5)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models)
        ax.set_yticks(range(len(reasons)))
        pretty = {
            "high_medico_legal_risk": "High medico-legal risk",
            "low_confidence": "Low confidence",
            "small_probability_margin": "Small probability margin",
            "weak_textual_evidence": "Weak textual evidence",
        }
        ax.set_yticklabels([pretty.get(r, r) for r in reasons])
        ax.set_xlim(-0.25, len(models)-0.75)
        ax.set_ylim(-0.35, len(reasons)-0.65)
        ax.set_title("Human-review triage reason map", pad=12)
    else:
        ax.text(0.5, 0.5, "Triage reason data not found", ha="center", va="center")
    set_axis_clean(ax)

    add_panel_letters(fig, axes.flat, ["A", "B", "C", "D"])
    savefig(fig, figures / "Fig2_final_safety_triage")


# ======================
# Fig3: External/error
# ======================
def plot_fig3(metrics, label_dist, outputs, figures):
    fig, axes = plt.subplots(2, 2, figsize=(16.8, 10.2), constrained_layout=True)
    fig.suptitle("Distributional and medico-legal error analysis", fontsize=18, fontweight="bold")

    # A top 10 only to avoid x label crowd
    ax = axes[0, 0]
    if not label_dist.empty:
        label_col, count_col = label_count_cols(label_dist)
        ld = label_dist.copy()
        ld[count_col] = pd.to_numeric(ld[count_col], errors="coerce")
        ld = ld.dropna(subset=[count_col]).sort_values(count_col, ascending=False).head(10)
        x = np.arange(len(ld))
        ax.bar(x, ld[count_col], color=COL["struct"], alpha=0.88)
        ax.set_xticks(x)
        ax.set_xticklabels([wrap(v, 11) for v in ld[label_col]], rotation=35, ha="right")
        ax.set_ylabel("Cases")
        ax.set_title("Top class-size distribution", pad=12)
        ax2 = ax.twinx()
        cum = ld[count_col].cumsum() / ld[count_col].sum()
        ax2.plot(x, cum, color=COL["red"], marker="o", linewidth=2)
        ax2.set_ylabel("Cumulative fraction")
        ax2.set_ylim(0, 1.05)
        ax2.grid(False)
    else:
        ax.text(0.5, 0.5, "Label distribution not found", ha="center", va="center")
    set_axis_clean(ax)

    # B external ranking
    ax = axes[0, 1]
    ext = metrics[(metrics["task"].astype(str) == "external") & metrics["macro_f1"].notna()].copy()
    ext = best_rows(ext, ("task", "method_short"), "macro_f1")
    if not ext.empty:
        top = ext.sort_values("macro_f1", ascending=False).head(8).sort_values("macro_f1")
        y = np.arange(len(top))
        ax.barh(y, top["macro_f1"], color=[color_family(m) for m in top["method_short"]], alpha=0.92)
        ax.set_yticks(y)
        ax.set_yticklabels([wrap(m, 18) for m in top["method_short"]])
        for i, v in enumerate(top["macro_f1"]):
            ax.text(v + 0.007, i, f"{v:.3f}", va="center", fontsize=8.5)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("Macro-F1")
        ax.set_title("External death detection performance", pad=12)
    else:
        ax.text(0.5, 0.5, "External metrics not found", ha="center", va="center")
    set_axis_clean(ax)

    # C confusion matrix
    ax = axes[1, 0]
    cm_files = sorted(outputs.glob("cm_broad_*.csv"))
    choice = None
    for f in cm_files:
        if "RandomForest" in f.name or "RF" in f.name or "Struct" in f.name:
            choice = f
            break
    if choice is None and cm_files:
        choice = cm_files[0]
    if choice is not None:
        cm = pd.read_csv(choice, index_col=0)
        cmn = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        im = ax.imshow(cmn.values, cmap="Blues", vmin=0, vmax=max(0.01, cmn.values.max()))
        ax.set_xticks(np.arange(len(cmn.columns)))
        ax.set_yticks(np.arange(len(cmn.index)))
        ax.set_xticklabels([wrap(c, 10) for c in cmn.columns], rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels([wrap(i, 12) for i in cmn.index], fontsize=8)
        ax.set_title("Row-normalized broad-task confusion matrix", pad=12)
        cb = fig.colorbar(im, ax=ax, shrink=0.84, pad=0.012)
        cb.set_label("Row-normalized value")
    else:
        ax.text(0.5, 0.5, "Confusion matrix not found", ha="center", va="center")
    set_axis_clean(ax)

    # D family performance
    ax = axes[1, 1]
    mm = metrics.dropna(subset=["macro_f1"]).copy()
    mm = best_rows(mm, ("task", "family"), "macro_f1")
    fam_order = ["Structured ML", "Narrative TF-IDF", "Proposed FVA", "Agent ablation", "Local LLM"]
    tasks = [t for t in ["broad", "external", "fine34"] if t in mm["task"].unique()]
    fams = [f for f in fam_order if f in mm["family"].unique()]
    x = np.arange(len(fams))
    width = 0.72 / max(1, len(tasks))
    task_colors = {"broad": COL["blue"], "external": COL["teal"], "fine34": COL["red"]}
    for j, task in enumerate(tasks):
        sub = mm[mm["task"] == task].set_index("family").reindex(fams)
        vals = sub["macro_f1"].fillna(0).values
        ax.bar(x + (j - (len(tasks)-1)/2)*width, vals, width=width,
               label=task, color=task_colors.get(task, COL["gray"]), alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels([wrap(f, 13) for f in fams], rotation=15, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Best Macro-F1")
    ax.set_title("Best performance by method family", pad=12)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    set_axis_clean(ax)

    add_panel_letters(fig, axes.flat, ["A", "B", "C", "D"])
    savefig(fig, figures / "Fig3_final_error_external")


# ======================
# Supp LLM
# ======================
def plot_figS1_llm(outputs, figures):
    llm = read_csv(outputs / "ollama_llm_metrics.csv")
    if llm.empty:
        return
    llm = safe_numeric(llm, ["parse_error_rate", "n_test", "valid_n", "accuracy", "macro_f1"])
    if "method" not in llm.columns:
        llm["method"] = "LLM"
    llm["method_short"] = llm["method"].map(simple_method)

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.2), constrained_layout=True)
    fig.suptitle("Supplementary local LLM prompting diagnostics", fontsize=17, fontweight="bold")

    x = np.arange(len(llm))
    labels = [wrap(m, 12) for m in llm["method_short"]]

    ax = axes[0]
    ax.bar(x, llm["parse_error_rate"].fillna(0), color=COL["red"], alpha=0.82)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Parse error rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Output parseability", pad=12)
    set_axis_clean(ax)

    ax = axes[1]
    ax.bar(x, llm["n_test"].fillna(0), color=COL["light"], label="Generated")
    ax.bar(x, llm["valid_n"].fillna(0), color=COL["blue"], label="Valid parsed")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Cases")
    ax.set_title("Generated vs valid parsed cases", pad=12)
    ax.legend(frameon=False)
    set_axis_clean(ax)

    ax = axes[2]
    width = 0.36
    ax.bar(x - width/2, llm["accuracy"].fillna(0), width=width, color=COL["green"], alpha=0.85, label="Accuracy")
    ax.bar(x + width/2, llm["macro_f1"].fillna(0), width=width, color=COL["orange"], alpha=0.85, label="Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Score on parsed subset", pad=12)
    ax.legend(frameon=False)
    set_axis_clean(ax)

    add_panel_letters(fig, axes, ["A", "B", "C"])
    savefig(fig, figures / "FigS1_final_llm_diagnostics")


# ======================
# Standalone selective curve, if user wants a simple clean figure
# ======================
def plot_figS2_selective(outputs, figures):
    curve_files = sorted(outputs.glob("forensic_agent_v2_selective_curve_*.csv"))
    if not curve_files:
        return
    fig, ax = plt.subplots(figsize=(8.8, 6.2), constrained_layout=True)
    for f in curve_files:
        cd = pd.read_csv(f)
        name = f.stem.replace("forensic_agent_v2_selective_curve_", "")
        x = pd.to_numeric(cd["coverage"], errors="coerce")
        if "accuracy" in cd.columns:
            y = pd.to_numeric(cd["accuracy"], errors="coerce")
            ax.plot(x, y, marker="o", linewidth=2.3, label=f"{name} accuracy")
        if "macro_f1" in cd.columns:
            y = pd.to_numeric(cd["macro_f1"], errors="coerce")
            ax.plot(x, y, marker="s", linestyle="--", linewidth=2.1, label=f"{name} Macro-F1")
    ax.set_xlabel("Coverage retained")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Selective prediction under retained coverage")
    ax.legend(frameon=False, loc="lower left")
    set_axis_clean(ax)
    savefig(fig, figures / "FigS2_final_selective_curves_only")


def export_tables(metrics, outputs, figures):
    table_dir = figures / "tables"
    mkdir(table_dir)
    metrics.to_csv(table_dir / "all_metrics_for_paper.csv", index=False)

    for task in sorted(metrics["task"].dropna().unique()):
        sub = metrics[(metrics["task"] == task) & metrics["macro_f1"].notna()].copy()
        sub = best_rows(sub, ("task", "method_short"), "macro_f1")
        sub.sort_values("macro_f1", ascending=False).to_csv(table_dir / f"table_{task}_metrics.csv", index=False)

    safety_cols = [c for c in ["task", "method", "method_short", "family", "accuracy", "macro_f1", "weighted_f1",
                               "coverage", "review_rate", "error_capture_rate", "n_test"] if c in metrics.columns]
    metrics[safety_cols].sort_values(["task", "macro_f1"], ascending=[True, False]).to_csv(
        table_dir / "table_safety_metrics.csv", index=False
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--figures", default="figures_final")
    args = parser.parse_args()

    outputs = Path(args.outputs)
    figures = Path(args.figures)
    mkdir(figures)

    metrics = load_metrics(outputs)
    label_dist = load_label_dist(outputs)

    plot_fig1(metrics, label_dist, figures)
    plot_fig2(metrics, outputs, figures)
    plot_fig3(metrics, label_dist, outputs, figures)
    plot_figS1_llm(outputs, figures)
    plot_figS2_selective(outputs, figures)
    export_tables(metrics, outputs, figures)

    print(f"Done. Final journal-grade figures saved to: {figures.resolve()}")
    for p in sorted(figures.glob("*.png")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
