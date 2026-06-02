#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra-final no-empty-panel publication figures for ForensicVA-Agent.

This script is designed for the exact problem you encountered:
- If LLM parsing is poor, it does NOT draw an empty score panel.
- Instead, it turns parse failure into meaningful analysis:
  parseability, valid/invalid composition, cost-efficiency, and robustness summary.
- No point labels in crowded regions.
- No long text inside dense scatter plots.
- Panel letters are outside axes.
- All figures export PNG + PDF.

Run from project root:
    python src/14_plot_ultrafinal_noempty_figures.py --outputs outputs --figures figures_ultrafinal
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

# =========================
# Style
# =========================
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
    "purple": "#8A63D2",
    "paper": "#F8FAFC",
}

plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 500,
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.titlesize": 12.3,
    "axes.labelsize": 10.3,
    "xtick.labelsize": 8.8,
    "ytick.labelsize": 8.8,
    "legend.fontsize": 8.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.16,
    "grid.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =========================
# Helpers
# =========================
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


def to_num(df, cols):
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
        "_": "-",
    }
    for a, b in repl.items():
        s = s.replace(a, b)
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
    return {
        "Proposed FVA": COL["fva"],
        "Structured ML": COL["struct"],
        "Narrative TF-IDF": COL["narr"],
        "Agent ablation": COL["agent"],
        "Local LLM": COL["llm"],
        "Other": COL["gray"],
    }.get(family(method_short), COL["gray"])


def load_metrics(outputs: Path) -> pd.DataFrame:
    df = read_csv(outputs / "all_metrics.csv")
    if df.empty:
        raise FileNotFoundError(f"Missing {outputs / 'all_metrics.csv'}")
    df = to_num(df, ["accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall",
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
    tmp["_score"] = tmp[metric].fillna(-1)
    idx = tmp.sort_values("_score", ascending=False).groupby(list(keys), as_index=False).head(1).index
    return tmp.loc[idx].drop(columns=["_score"], errors="ignore")


def load_label_dist(outputs: Path) -> pd.DataFrame:
    ld = read_csv(outputs / "label_distribution.csv")
    if not ld.empty:
        return ld
    p = outputs / "phmrc_forensicva_processed.csv"
    if p.exists():
        df = pd.read_csv(p)
        for c in ["gold_broad", "broad", "target_broad", "broad_label"]:
            if c in df.columns:
                return df[c].value_counts().rename_axis("label").reset_index(name="count")
    return pd.DataFrame()


def label_count_cols(ld):
    label_col = next((c for c in ld.columns if "label" in c.lower() or "broad" in c.lower()), ld.columns[0])
    count_col = next((c for c in ld.columns if "count" in c.lower() or c.lower() in ["n", "cases"]), ld.columns[-1])
    return label_col, count_col


def add_panel_letters(fig, axes, letters):
    fig.canvas.draw()
    for ax, letter in zip(axes, letters):
        pos = ax.get_position()
        fig.text(pos.x0 - 0.018, pos.y1 + 0.012, letter,
                 fontsize=15, fontweight="bold", ha="left", va="bottom", color=COL["dark"])


def clean_axis(ax):
    ax.grid(True, alpha=0.16)
    ax.tick_params(axis="both", length=3, width=0.8)


def export_table(df, path):
    mkdir(path.parent)
    df.to_csv(path, index=False)


# =========================
# Main Figure 1
# =========================
def fig1_main(metrics, label_dist, figures):
    broad = metrics[(metrics["task"].astype(str) == "broad") & metrics["macro_f1"].notna()].copy()
    broad = best_rows(broad, ("task", "method_short"), "macro_f1")

    fig, axes = plt.subplots(2, 2, figsize=(16.8, 10.4), constrained_layout=True)
    fig.suptitle("ForensicVA-Agent classification performance", fontsize=18, fontweight="bold")

    # A
    ax = axes[0, 0]
    if not label_dist.empty:
        label_col, count_col = label_count_cols(label_dist)
        ld = label_dist.copy()
        ld[count_col] = pd.to_numeric(ld[count_col], errors="coerce")
        ld = ld.dropna(subset=[count_col]).sort_values(count_col, ascending=True)
        ax.barh([wrap(x, 17) for x in ld[label_col]], ld[count_col], color=COL["struct"], alpha=0.9)
        ax.set_xlabel("Number of cases")
        ax.set_title("Dataset label distribution", pad=12)
    else:
        ax.text(0.5, 0.5, "Label distribution not found", ha="center", va="center")
    clean_axis(ax)

    # B
    ax = axes[0, 1]
    top = broad.sort_values("macro_f1", ascending=False).head(12).sort_values("macro_f1", ascending=True)
    y = np.arange(len(top))
    ax.barh(y, top["macro_f1"], color=[color_family(m) for m in top["method_short"]], alpha=0.92)
    ax.set_yticks(y)
    ax.set_yticklabels([wrap(m, 19) for m in top["method_short"]])
    for i, v in enumerate(top["macro_f1"]):
        ax.text(v + 0.006, i, f"{v:.3f}", va="center", fontsize=8.5)
    ax.set_xlim(0, max(0.76, min(1.0, top["macro_f1"].max() + 0.08)))
    ax.set_xlabel("Macro-F1")
    ax.set_title("Broad cause-of-death ranking", pad=12)
    clean_axis(ax)

    # C no labels
    ax = axes[1, 0]
    sc = broad.dropna(subset=["accuracy", "macro_f1"]).copy()
    if not sc.empty:
        review = sc["review_rate"].fillna(0) if "review_rate" in sc.columns else pd.Series(np.zeros(len(sc)))
        sizes = 70 + 420 * review.clip(0, 1)
        ax.scatter(sc["accuracy"], sc["macro_f1"],
                   s=sizes, c=[color_family(m) for m in sc["method_short"]],
                   alpha=0.75, edgecolor="white", linewidth=0.9)
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("Macro-F1")
        ax.set_title("Accuracy–F1–review trade-off", pad=12)
        ax.set_xlim(max(0, sc["accuracy"].min() - 0.07), min(1, sc["accuracy"].max() + 0.07))
        ax.set_ylim(max(0, sc["macro_f1"].min() - 0.07), min(1, sc["macro_f1"].max() + 0.07))
        ax.legend(
            handles=[
                Patch(facecolor=COL["fva"], label="Proposed FVA"),
                Patch(facecolor=COL["struct"], label="Structured ML"),
                Patch(facecolor=COL["narr"], label="Narrative TF-IDF"),
                Patch(facecolor=COL["agent"], label="Agent ablation"),
                Patch(facecolor=COL["llm"], label="Local LLM"),
            ],
            frameon=False, loc="lower right", ncol=1
        )
    clean_axis(ax)

    # D heatmap
    ax = axes[1, 1]
    hm = metrics.dropna(subset=["macro_f1"]).copy()
    hm = best_rows(hm, ("task", "method_short"), "macro_f1")
    hm = hm[~hm["family"].isin(["Local LLM", "Other"])]
    keep = set(hm.sort_values("macro_f1", ascending=False).head(10)["method_short"])
    keep.update(hm[hm["method_short"].str.contains("FVA-v2|Struct|Narr", case=False, na=False)]["method_short"].head(10))
    hm = hm[hm["method_short"].isin(keep)]
    pivot = hm.pivot_table(index="method_short", columns="task", values="macro_f1", aggfunc="max")
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(0.01, np.nanmax(pivot.values)))
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([wrap(x, 17) for x in pivot.index])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if val > 0.56 else COL["dark"])
    ax.set_title("Task-wise Macro-F1 heatmap", pad=12)
    cb = fig.colorbar(im, ax=ax, shrink=0.86, pad=0.012)
    cb.set_label("Macro-F1")

    add_panel_letters(fig, axes.flat, ["A", "B", "C", "D"])
    savefig(fig, figures / "Fig1_ultrafinal_main_results")


# =========================
# Main Figure 2
# =========================
def fig2_safety(metrics, outputs, figures):
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

    # A
    ax = axes[0, 0]
    sdf = broad[broad["method_short"].str.contains("FVA-v2|Struct-RF", case=False, na=False)].copy()
    if sdf.empty:
        sdf = broad.copy()
    ax.scatter(sdf["coverage"], sdf["error_capture_rate"],
               s=95 + 430 * sdf["review_rate"].clip(0, 1),
               c=[color_family(m) for m in sdf["method_short"]],
               alpha=0.76, edgecolor="white", linewidth=0.9)
    ax.set_xlabel("Automatic coverage")
    ax.set_ylabel("Error capture rate")
    ax.set_xlim(-0.04, 1.06)
    ax.set_ylim(-0.04, 1.06)
    ax.set_title("Coverage–error-capture trade-off", pad=12)
    # legend, not direct labels
    legend_rows = sdf.sort_values(["error_capture_rate", "coverage"], ascending=False).head(6)
    handles = [Line2D([0], [0], marker="o", color="w", label=m,
                      markerfacecolor=color_family(m), markersize=8)
               for m in legend_rows["method_short"]]
    ax.legend(handles=handles, frameon=False, loc="lower left", bbox_to_anchor=(0.02, 0.02))
    clean_axis(ax)

    # B
    ax = axes[0, 1]
    curve_files = sorted(outputs.glob("forensic_agent_v2_selective_curve_*.csv"))
    if curve_files:
        palette = {
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
                    color, marker, ls = palette.get(key, (COL["gray"], "o", "-"))
                    ax.plot(x, y, marker=marker, linestyle=ls, color=color, linewidth=2.2, label=key.replace("_", "-"))
                    ax.fill_between(x, y, color=color, alpha=0.055)
        ax.set_xlabel("Coverage retained")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.set_title("Selective prediction curves", pad=12)
        ax.legend(frameon=False, loc="lower left", ncol=2)
    else:
        ax.text(0.5, 0.5, "Selective curve files not found", ha="center", va="center")
    clean_axis(ax)

    # C
    ax = axes[1, 0]
    pred_files = sorted(outputs.glob("forensic_agent_v2_predictions_*.csv"))
    rows = []
    for f in pred_files:
        pdf = pd.read_csv(f)
        model = f.stem.replace("forensic_agent_v2_predictions_", "")
        if {"y_true", "final_pred", "confidence"}.issubset(pdf.columns):
            tmp = pdf[["y_true", "final_pred", "confidence"]].copy()
            tmp["model"] = model
            tmp["correct"] = tmp["y_true"].astype(str) == tmp["final_pred"].astype(str)
            rows.append(tmp)
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
    clean_axis(ax)

    # D
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
        order = ["high_medico_legal_risk", "low_confidence", "small_probability_margin", "weak_textual_evidence"]
        reasons = [r for r in order if r in rr["reason"].unique()]
        xmap = {m: i for i, m in enumerate(models)}
        ymap = {r: i for i, r in enumerate(reasons)}
        rr = rr[rr["model"].isin(models) & rr["reason"].isin(reasons)].copy()
        rr["x"] = rr["model"].map(xmap)
        rr["y"] = rr["reason"].map(ymap)
        ax.scatter(rr["x"], rr["y"], s=np.maximum(80, rr["n"] * 0.45),
                   color=COL["llm"], alpha=0.55, edgecolor="white", linewidth=0.9)
        for _, r in rr.iterrows():
            ax.text(r["x"], r["y"], str(int(r["n"])), ha="center", va="center", fontsize=8.5)
        pretty = {
            "high_medico_legal_risk": "High medico-legal risk",
            "low_confidence": "Low confidence",
            "small_probability_margin": "Small probability margin",
            "weak_textual_evidence": "Weak textual evidence",
        }
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models)
        ax.set_yticks(range(len(reasons)))
        ax.set_yticklabels([pretty.get(r, r) for r in reasons])
        ax.set_xlim(-0.25, len(models)-0.75)
        ax.set_ylim(-0.35, len(reasons)-0.65)
        ax.set_title("Human-review triage reason map", pad=12)
    else:
        ax.text(0.5, 0.5, "Triage reason data not found", ha="center", va="center")
    clean_axis(ax)

    add_panel_letters(fig, axes.flat, ["A", "B", "C", "D"])
    savefig(fig, figures / "Fig2_ultrafinal_safety_triage")


# =========================
# Fig3
# =========================
def fig3_error(metrics, label_dist, outputs, figures):
    fig, axes = plt.subplots(2, 2, figsize=(16.8, 10.2), constrained_layout=True)
    fig.suptitle("Distributional and medico-legal error analysis", fontsize=18, fontweight="bold")

    # A
    ax = axes[0, 0]
    if not label_dist.empty:
        label_col, count_col = label_count_cols(label_dist)
        ld = label_dist.copy()
        ld[count_col] = pd.to_numeric(ld[count_col], errors="coerce")
        ld = ld.dropna(subset=[count_col]).sort_values(count_col, ascending=False).head(10)
        x = np.arange(len(ld))
        ax.bar(x, ld[count_col], color=COL["struct"], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([wrap(v, 10) for v in ld[label_col]], rotation=35, ha="right")
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
    clean_axis(ax)

    # B
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
    clean_axis(ax)

    # C
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
    clean_axis(ax)

    # D
    ax = axes[1, 1]
    mm = metrics.dropna(subset=["macro_f1"]).copy()
    mm = best_rows(mm, ("task", "family"), "macro_f1")
    fam_order = ["Structured ML", "Narrative TF-IDF", "Proposed FVA", "Agent ablation", "Local LLM"]
    tasks = [t for t in ["broad", "external", "fine34"] if t in mm["task"].unique()]
    fams = [f for f in fam_order if f in mm["family"].unique()]
    x = np.arange(len(fams))
    width = 0.72 / max(1, len(tasks))
    tcols = {"broad": COL["blue"], "external": COL["teal"], "fine34": COL["red"]}
    for j, task in enumerate(tasks):
        sub = mm[mm["task"] == task].set_index("family").reindex(fams)
        vals = sub["macro_f1"].fillna(0).values
        ax.bar(x + (j - (len(tasks)-1)/2)*width, vals, width=width,
               label=task, color=tcols.get(task, COL["gray"]), alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels([wrap(f, 13) for f in fams], rotation=15, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Best Macro-F1")
    ax.set_title("Best performance by method family", pad=12)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    clean_axis(ax)

    add_panel_letters(fig, axes.flat, ["A", "B", "C", "D"])
    savefig(fig, figures / "Fig3_ultrafinal_error_external")


# =========================
# LLM figure with content, no empty panels
# =========================
def figS1_llm_content(outputs, figures):
    llm = read_csv(outputs / "ollama_llm_metrics.csv")
    if llm.empty:
        return
    llm = to_num(llm, ["parse_error_rate", "n_test", "valid_n", "accuracy", "macro_f1"])
    if "method" not in llm.columns:
        llm["method"] = "LLM"
    llm["method_short"] = llm["method"].map(simple_method)

    # Construct useful fields even when parsed subset is empty.
    if "n_test" not in llm.columns:
        llm["n_test"] = 0
    if "valid_n" not in llm.columns:
        llm["valid_n"] = 0
    if "parse_error_rate" not in llm.columns:
        llm["parse_error_rate"] = 1 - (llm["valid_n"] / llm["n_test"].replace(0, np.nan))
    llm["parse_success_rate"] = 1 - llm["parse_error_rate"].fillna(1.0)
    llm["invalid_n"] = (llm["n_test"].fillna(0) - llm["valid_n"].fillna(0)).clip(lower=0)
    llm["valid_rate"] = llm["valid_n"].fillna(0) / llm["n_test"].replace(0, np.nan).fillna(1)
    llm["model_size_hint"] = llm["method_short"].str.extract(r"(\d+(?:\.\d+)?)b", expand=False).astype(float)

    fig = plt.figure(figsize=(16.8, 8.8), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.05, 1.0], width_ratios=[1.1, 1.0])
    fig.suptitle("Supplementary local LLM prompting robustness analysis", fontsize=18, fontweight="bold")

    labels = [wrap(x, 13) for x in llm["method_short"]]
    x = np.arange(len(llm))

    # A stacked valid/invalid composition
    axA = fig.add_subplot(gs[0, 0])
    axA.bar(x, llm["invalid_n"], color=COL["red"], alpha=0.75, label="Invalid / unparsable")
    axA.bar(x, llm["valid_n"], bottom=llm["invalid_n"], color=COL["blue"], alpha=0.82, label="Valid parsed")
    axA.set_xticks(x)
    axA.set_xticklabels(labels, rotation=25, ha="right")
    axA.set_ylabel("Generated cases")
    axA.set_title("Output validity composition", pad=12)
    axA.legend(frameon=False)
    clean_axis(axA)

    # B parse success/error paired bars
    axB = fig.add_subplot(gs[0, 1])
    width = 0.38
    axB.bar(x - width/2, llm["parse_success_rate"].fillna(0), width=width,
            color=COL["green"], alpha=0.82, label="Parse success")
    axB.bar(x + width/2, llm["parse_error_rate"].fillna(1), width=width,
            color=COL["red"], alpha=0.72, label="Parse error")
    axB.set_xticks(x)
    axB.set_xticklabels(labels, rotation=25, ha="right")
    axB.set_ylim(0, 1.05)
    axB.set_ylabel("Rate")
    axB.set_title("Parse success versus parse error", pad=12)
    axB.legend(frameon=False)
    clean_axis(axB)

    # C model size vs parse success, still meaningful even if all fail
    axC = fig.add_subplot(gs[1, 0])
    if llm["model_size_hint"].notna().any():
        sx = llm["model_size_hint"].fillna(pd.Series(np.arange(len(llm)) + 1, index=llm.index))
        sy = llm["parse_success_rate"].fillna(0)
        ss = 120 + 4 * llm["n_test"].fillna(0)
        axC.scatter(sx, sy, s=ss, color=COL["purple"], alpha=0.7, edgecolor="white", linewidth=0.9)
        for i, r in llm.iterrows():
            # Place label with small offsets; few points only, no overlap risk.
            axC.annotate(r["method_short"], (sx.iloc[i], sy.iloc[i]),
                         xytext=(5, 6), textcoords="offset points", fontsize=8)
        axC.set_xlabel("Model size hint (B parameters)")
        axC.set_ylabel("Parse success rate")
        axC.set_ylim(-0.03, 1.05)
        axC.set_title("Model size and parseability", pad=12)
    else:
        axC.bar(x, llm["parse_success_rate"].fillna(0), color=COL["purple"], alpha=0.75)
        axC.set_xticks(x)
        axC.set_xticklabels(labels, rotation=25, ha="right")
        axC.set_ylabel("Parse success rate")
        axC.set_ylim(0, 1.05)
        axC.set_title("Parseability by local LLM", pad=12)
    clean_axis(axC)

    # D robustness summary card, no empty panel
    axD = fig.add_subplot(gs[1, 1])
    axD.set_axis_off()
    total_generated = int(llm["n_test"].fillna(0).sum())
    total_valid = int(llm["valid_n"].fillna(0).sum())
    overall_success = total_valid / total_generated if total_generated else 0
    best_idx = llm["parse_success_rate"].fillna(-1).idxmax()
    best_name = llm.loc[best_idx, "method_short"]
    best_rate = llm.loc[best_idx, "parse_success_rate"]

    lines = [
        "Robustness summary",
        "",
        f"Generated cases: {total_generated}",
        f"Valid parsed cases: {total_valid}",
        f"Overall parse success: {overall_success:.1%}",
        f"Best parseability: {best_name} ({best_rate:.1%})",
        "",
        "Interpretation:",
        "Direct local LLM prompting produced limited",
        "machine-readable outputs under the current",
        "strict JSON parsing protocol.",
        "",
        "This supports the need for structured,",
        "verifiable, and human-review-aware pipelines."
    ]
    axD.text(
        0.02, 0.98, "\n".join(lines),
        ha="left", va="top",
        fontsize=10.6,
        color=COL["dark"],
        bbox=dict(boxstyle="round,pad=0.6", facecolor=COL["paper"], edgecolor=COL["light"], linewidth=1.0)
    )
    axD.set_title("Interpretive summary", pad=12)

    add_panel_letters(fig, [axA, axB, axC, axD], ["A", "B", "C", "D"])
    savefig(fig, figures / "FigS1_ultrafinal_llm_robustness")


# =========================
# One extra beautiful combined figure:
# safety-performance narrative in 3 panels, no text overlap
# =========================
def fig4_policy_story(metrics, outputs, figures):
    df = metrics.dropna(subset=["accuracy", "macro_f1"]).copy()
    for c, default in [("coverage", 1.0), ("review_rate", 0.0), ("error_capture_rate", 0.0)]:
        if c not in df.columns:
            df[c] = default
        df[c] = df[c].fillna(default)
    df = best_rows(df, ("task", "method_short"), "macro_f1")
    broad = df[df["task"].astype(str) == "broad"].copy()
    if broad.empty:
        broad = df.copy()

    # Keep main comparable methods only.
    keep = broad[broad["family"].isin(["Structured ML", "Narrative TF-IDF", "Proposed FVA", "Agent ablation"])].copy()
    keep = keep.sort_values("macro_f1", ascending=False).head(12)

    fig = plt.figure(figsize=(16.8, 5.8), constrained_layout=True)
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.1, 1.15, 1.0])
    fig.suptitle("Safety-performance policy view for medico-legal decision support", fontsize=17, fontweight="bold")

    # A ordered macro-f1 dot-line
    axA = fig.add_subplot(gs[0, 0])
    order = keep.sort_values("macro_f1", ascending=True)
    y = np.arange(len(order))
    axA.hlines(y, 0, order["macro_f1"], color=COL["light"], linewidth=2)
    axA.scatter(order["macro_f1"], y, s=80, c=[color_family(m) for m in order["method_short"]],
                edgecolor="white", linewidth=0.8)
    axA.set_yticks(y)
    axA.set_yticklabels([wrap(m, 17) for m in order["method_short"]])
    axA.set_xlabel("Macro-F1")
    axA.set_xlim(0, max(0.75, order["macro_f1"].max()+0.07))
    axA.set_title("Performance ranking", pad=12)
    clean_axis(axA)

    # B coverage-review map
    axB = fig.add_subplot(gs[0, 1])
    axB.scatter(keep["coverage"], keep["review_rate"],
                s=90 + 400 * keep["error_capture_rate"].clip(0,1),
                c=[color_family(m) for m in keep["method_short"]],
                alpha=0.76, edgecolor="white", linewidth=0.9)
    axB.set_xlabel("Automatic coverage")
    axB.set_ylabel("Review rate")
    axB.set_xlim(-0.04, 1.06)
    axB.set_ylim(-0.04, 1.06)
    axB.set_title("Coverage versus review burden", pad=12)
    axB.legend(
        handles=[
            Patch(facecolor=COL["fva"], label="Proposed FVA"),
            Patch(facecolor=COL["struct"], label="Structured ML"),
            Patch(facecolor=COL["narr"], label="Narrative TF-IDF"),
            Patch(facecolor=COL["agent"], label="Agent ablation"),
        ],
        frameon=False, loc="upper left"
    )
    clean_axis(axB)

    # C normalized policy score heatmap
    axC = fig.add_subplot(gs[0, 2])
    cols = ["accuracy", "macro_f1", "coverage", "review_rate", "error_capture_rate"]
    avail = [c for c in cols if c in keep.columns]
    mat = keep.sort_values("macro_f1", ascending=False).head(8)[["method_short"] + avail].copy()
    # For review_rate, lower is not always better; keep actual value but label clearly.
    vals = mat[avail].fillna(0).values
    im = axC.imshow(vals, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axC.set_xticks(np.arange(len(avail)))
    axC.set_xticklabels([c.replace("_", "\n") for c in avail], rotation=0)
    axC.set_yticks(np.arange(len(mat)))
    axC.set_yticklabels([wrap(m, 15) for m in mat["method_short"]], fontsize=8)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            axC.text(j, i, f"{vals[i,j]:.2f}", ha="center", va="center",
                     fontsize=7.5, color="white" if vals[i,j] > 0.55 else COL["dark"])
    axC.set_title("Policy-relevant metrics", pad=12)
    cb = fig.colorbar(im, ax=axC, shrink=0.78, pad=0.012)
    cb.set_label("Value")

    add_panel_letters(fig, [axA, axB, axC], ["A", "B", "C"])
    savefig(fig, figures / "Fig4_ultrafinal_policy_story")


def export_tables(metrics, outputs, figures):
    table_dir = figures / "tables"
    mkdir(table_dir)
    metrics.to_csv(table_dir / "all_metrics_for_paper.csv", index=False)
    for task in sorted(metrics["task"].dropna().unique()):
        sub = metrics[(metrics["task"] == task) & metrics["macro_f1"].notna()].copy()
        sub = best_rows(sub, ("task", "method_short"), "macro_f1")
        sub.sort_values("macro_f1", ascending=False).to_csv(table_dir / f"table_{task}_metrics.csv", index=False)

    llm = read_csv(outputs / "ollama_llm_metrics.csv")
    if not llm.empty:
        llm.to_csv(table_dir / "table_llm_robustness.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--figures", default="figures_ultrafinal")
    args = parser.parse_args()

    outputs = Path(args.outputs)
    figures = Path(args.figures)
    mkdir(figures)

    metrics = load_metrics(outputs)
    label_dist = load_label_dist(outputs)

    fig1_main(metrics, label_dist, figures)
    fig2_safety(metrics, outputs, figures)
    fig3_error(metrics, label_dist, outputs, figures)
    figS1_llm_content(outputs, figures)
    fig4_policy_story(metrics, outputs, figures)
    export_tables(metrics, outputs, figures)

    print(f"Done. Ultra-final figures saved to: {figures.resolve()}")
    for p in sorted(figures.glob("*.png")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
