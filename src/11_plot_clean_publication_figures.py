#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean publication figures for ForensicVA-Agent.

Why this version:
- Fewer panels per figure; no overcrowded 8-panel dashboard.
- Panel letters are placed in a fixed corner with white background.
- Long method names are aggressively shortened.
- Labels are only added to the top few points to avoid overlap.
- Tables are exported as CSV instead of being forced inside figures.
- PDF + PNG are both exported for Overleaf.

Run from project root:
    python src/11_plot_clean_publication_figures.py --outputs outputs --figures figures_clean
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

# -------------------------
# Style
# -------------------------
plt.rcParams.update({
    "figure.dpi": 170,
    "savefig.dpi": 450,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COL = {
    "proposed": "#E84A5F",
    "struct": "#3B82F6",
    "narr": "#14B8A6",
    "agent": "#F59E0B",
    "llm": "#8B5CF6",
    "gray": "#64748B",
    "light": "#E5E7EB",
    "dark": "#111827",
    "green": "#22C55E",
    "blue2": "#06B6D4",
}


# -------------------------
# Helpers
# -------------------------
def mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def savefig(fig, path: Path):
    mkdir(path.parent)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def wrap(x, width=16):
    return "\n".join(textwrap.wrap(str(x), width=width, break_long_words=False))


def panel(ax, letter):
    # Put panel letters INSIDE the axes to avoid collision with titles.
    ax.text(
        0.012, 0.985, letter,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=14, fontweight="bold",
        color=COL["dark"],
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=2.5),
        zorder=20
    )


def simplify_method(s):
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
        "Evidence + verification": "Evidence+Verify",
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
    return s


def group_color(method):
    m = str(method)
    if "FVA" in m or "ForensicVA" in m:
        return COL["proposed"]
    if "Struct" in m or "Structured" in m:
        return COL["struct"]
    if "Narr" in m or "TFIDF" in m:
        return COL["narr"]
    if "LLM" in m or "qwen" in m or "llama" in m or "gemma" in m:
        return COL["llm"]
    if "Evidence" in m or "Prior" in m:
        return COL["agent"]
    return COL["gray"]


def load_metrics(outputs: Path) -> pd.DataFrame:
    df = read_csv(outputs / "all_metrics.csv")
    if df.empty:
        raise FileNotFoundError(f"Cannot find {outputs / 'all_metrics.csv'}")

    for c in ["accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall",
              "coverage", "review_rate", "error_capture_rate", "n_test"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "task" not in df.columns:
        df["task"] = "broad"
    if "method" not in df.columns:
        df["method"] = "unknown"

    df["method_short"] = df["method"].map(simplify_method)

    # For duplicate rows, keep a suffix only in tables. For plotting, we often deduplicate by max score.
    df["_row_id"] = np.arange(len(df))
    return df


def dedup_best(df, key_cols=("task", "method_short"), score="macro_f1"):
    if df.empty:
        return df
    tmp = df.copy()
    tmp["_score"] = tmp[score].fillna(-1)
    idx = tmp.sort_values("_score", ascending=False).groupby(list(key_cols), as_index=False).head(1).index
    return tmp.loc[idx].drop(columns=["_score"], errors="ignore")


def load_label_distribution(outputs: Path, processed_path: Path = None) -> pd.DataFrame:
    ld = read_csv(outputs / "label_distribution.csv")
    if not ld.empty:
        return ld

    # fallback from processed data
    if processed_path and processed_path.exists():
        df = pd.read_csv(processed_path)
        label_col = None
        for c in ["broad_label", "gold_broad", "target_broad", "broad"]:
            if c in df.columns:
                label_col = c
                break
        if label_col:
            return df[label_col].value_counts().rename_axis("label").reset_index(name="count")
    return pd.DataFrame()


def get_label_count_cols(ld: pd.DataFrame):
    label_col = next((c for c in ld.columns if "label" in c.lower() or "broad" in c.lower() or "target" in c.lower()), ld.columns[0])
    count_col = next((c for c in ld.columns if "count" in c.lower() or c.lower() in ["n", "cases"]), ld.columns[-1])
    return label_col, count_col


# -------------------------
# Figure 1: main results
# -------------------------
def fig1_main_results(metrics, label_dist, outputs, figures):
    broad = metrics[(metrics["task"].astype(str) == "broad") & metrics["macro_f1"].notna()].copy()
    broad = dedup_best(broad, ("task", "method_short"), "macro_f1")

    fig = plt.figure(figsize=(15.8, 9.2), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[0.95, 1.25], height_ratios=[1.0, 1.05])
    fig.suptitle("ForensicVA-Agent classification performance", fontsize=17, fontweight="bold", y=1.02)

    # A label distribution
    ax = fig.add_subplot(gs[0, 0])
    panel(ax, "A")
    if not label_dist.empty:
        label_col, count_col = get_label_count_cols(label_dist)
        ld = label_dist.copy()
        ld[count_col] = pd.to_numeric(ld[count_col], errors="coerce")
        ld = ld.dropna(subset=[count_col]).sort_values(count_col, ascending=True)
        ax.barh([wrap(x, 18) for x in ld[label_col]], ld[count_col], color=COL["struct"], alpha=0.86)
        ax.set_xlabel("Number of cases")
        ax.set_title("Dataset label distribution", pad=8)
    else:
        ax.text(0.5, 0.5, "Label distribution not found", ha="center", va="center")
        ax.set_axis_off()

    # B top broad ranking
    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "B")
    top = broad.sort_values("macro_f1", ascending=False).head(12).sort_values("macro_f1", ascending=True)
    colors = [group_color(m) for m in top["method_short"]]
    y = np.arange(len(top))
    ax.barh(y, top["macro_f1"], color=colors, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([wrap(m, 23) for m in top["method_short"]])
    for i, v in enumerate(top["macro_f1"]):
        ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, min(1.0, max(0.76, top["macro_f1"].max() + 0.08)))
    ax.set_xlabel("Macro-F1")
    ax.set_title("Broad cause-of-death Macro-F1 ranking", pad=8)
    ax.legend(
        handles=[
            Patch(facecolor=COL["proposed"], label="Proposed FVA"),
            Patch(facecolor=COL["struct"], label="Structured ML"),
            Patch(facecolor=COL["narr"], label="Narrative TF-IDF"),
            Patch(facecolor=COL["agent"], label="Agent ablation"),
            Patch(facecolor=COL["llm"], label="Local LLM"),
        ],
        loc="lower right", frameon=False, ncol=2
    )

    # C scatter accuracy vs F1
    ax = fig.add_subplot(gs[1, 0])
    panel(ax, "C")
    sc = broad.dropna(subset=["accuracy", "macro_f1"]).copy()
    if not sc.empty:
        review = sc["review_rate"].fillna(0) if "review_rate" in sc.columns else pd.Series(np.zeros(len(sc)))
        sizes = 65 + 480 * review.clip(0, 1)
        ax.scatter(sc["accuracy"], sc["macro_f1"], s=sizes,
                   c=[group_color(m) for m in sc["method_short"]],
                   alpha=0.72, edgecolor="white", linewidth=0.9)
        # annotate only best 5 to avoid clutter
        for _, r in sc.sort_values("macro_f1", ascending=False).head(5).iterrows():
            ax.annotate(r["method_short"], (r["accuracy"], r["macro_f1"]),
                        xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("Macro-F1")
        ax.set_title("Accuracy–F1–review trade-off", pad=8)
        ax.set_xlim(max(0, sc["accuracy"].min() - 0.08), min(1, sc["accuracy"].max() + 0.08))
        ax.set_ylim(max(0, sc["macro_f1"].min() - 0.08), min(1, sc["macro_f1"].max() + 0.08))
    else:
        ax.set_axis_off()

    # D task heatmap
    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "D")
    hm = metrics.dropna(subset=["macro_f1"]).copy()
    hm = dedup_best(hm, ("task", "method_short"), "macro_f1")
    # choose readable subset: top broad + top external + FVA
    keep_methods = set(
        hm.sort_values("macro_f1", ascending=False).head(12)["method_short"].tolist()
        + hm[hm["method_short"].str.contains("FVA|Struct", case=False, na=False)]["method_short"].head(8).tolist()
    )
    hm = hm[hm["method_short"].isin(keep_methods)]
    if not hm.empty:
        pivot = hm.pivot_table(index="method_short", columns="task", values="macro_f1", aggfunc="max")
        pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(0.01, np.nanmax(pivot.values)))
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=0)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([wrap(x, 20) for x in pivot.index], fontsize=8)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.4,
                            color="white" if val > 0.55 else COL["dark"])
        ax.set_title("Task-wise Macro-F1 heatmap", pad=8)
        cbar = fig.colorbar(im, ax=ax, shrink=0.84, pad=0.015)
        cbar.set_label("Macro-F1")
    else:
        ax.set_axis_off()

    savefig(fig, figures / "Fig1_clean_main_results")


# -------------------------
# Figure 2 safety
# -------------------------
def fig2_safety(metrics, outputs, figures):
    df = metrics.dropna(subset=["accuracy", "macro_f1"]).copy()
    for c, default in [("coverage", 1.0), ("review_rate", 0.0), ("error_capture_rate", 0.0)]:
        if c not in df.columns:
            df[c] = default
        df[c] = df[c].fillna(default)

    # Reduce duplicate clutter by keeping best per method_short.
    df = dedup_best(df, ("task", "method_short"), "macro_f1")
    broad = df[df["task"].astype(str) == "broad"].copy()
    if broad.empty:
        broad = df.copy()

    fig = plt.figure(figsize=(15.8, 8.9), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 1])
    fig.suptitle("Safety-oriented human-review triage analysis", fontsize=17, fontweight="bold", y=1.02)

    # A coverage error capture
    ax = fig.add_subplot(gs[0, 0])
    panel(ax, "A")
    sdf = broad.copy()
    ax.scatter(sdf["coverage"], sdf["error_capture_rate"],
               s=70 + 420 * sdf["review_rate"].clip(0, 1),
               c=[group_color(m) for m in sdf["method_short"]],
               alpha=0.72, edgecolor="white", linewidth=0.9)
    ann = sdf[sdf["method_short"].str.contains("FVA-v2|Struct-RF", case=False, na=False)]
    for _, r in ann.iterrows():
        ax.annotate(r["method_short"], (r["coverage"], r["error_capture_rate"]),
                    xytext=(6, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Automatic coverage")
    ax.set_ylabel("Error capture rate")
    ax.set_xlim(-0.04, 1.06)
    ax.set_ylim(-0.04, 1.06)
    ax.set_title("Coverage–error-capture trade-off", pad=8)

    # B selective curves
    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "B")
    curve_files = sorted(outputs.glob("forensic_agent_v2_selective_curve_*.csv"))
    if curve_files:
        for f in curve_files:
            cd = pd.read_csv(f)
            name = f.stem.replace("forensic_agent_v2_selective_curve_", "")
            if "coverage" not in cd.columns:
                continue
            x = pd.to_numeric(cd["coverage"], errors="coerce")
            if "accuracy" in cd.columns:
                y = pd.to_numeric(cd["accuracy"], errors="coerce")
                ax.plot(x, y, marker="o", linewidth=2.0, label=f"{name} accuracy")
                ax.fill_between(x, y, alpha=0.08)
            if "macro_f1" in cd.columns:
                y = pd.to_numeric(cd["macro_f1"], errors="coerce")
                ax.plot(x, y, marker="s", linestyle="--", linewidth=1.8, label=f"{name} Macro-F1")
        ax.set_xlabel("Coverage retained")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.set_title("Selective prediction curves", pad=8)
        ax.legend(frameon=False, loc="lower left", ncol=2)
    else:
        ax.text(0.5, 0.5, "Selective curve files not found", ha="center", va="center")
        ax.set_axis_off()

    # C confidence boxplot
    ax = fig.add_subplot(gs[1, 0])
    panel(ax, "C")
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
                    colors.append(COL["green"] if correct else COL["proposed"])
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        ax.set_ylabel("Classifier confidence")
        ax.set_title("Confidence separation by correctness", pad=8)
    else:
        ax.text(0.5, 0.5, "Prediction files not found", ha="center", va="center")
        ax.set_axis_off()

    # D triage reason map
    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "D")
    reason_rows = []
    for f in pred_files:
        pdf = pd.read_csv(f)
        model = f.stem.replace("forensic_agent_v2_predictions_", "")
        if "triage_reasons" in pdf.columns:
            for reasons in pdf["triage_reasons"].fillna(""):
                for r in str(reasons).split("|"):
                    r = r.strip()
                    if r:
                        reason_rows.append({"model": model, "reason": r})
    if reason_rows:
        rr = pd.DataFrame(reason_rows).groupby(["model", "reason"]).size().reset_index(name="n")
        # ordered axes
        models = [m for m in ["logreg", "rf"] if m in rr["model"].unique()]
        if not models:
            models = sorted(rr["model"].unique())
        reasons = ["high_medico_legal_risk", "low_confidence", "small_probability_margin", "weak_textual_evidence"]
        reasons = [r for r in reasons if r in rr["reason"].unique()] + [r for r in sorted(rr["reason"].unique()) if r not in reasons]
        xmap = {m: i for i, m in enumerate(models)}
        ymap = {r: i for i, r in enumerate(reasons)}
        rr = rr[rr["model"].isin(models)]
        rr["x"] = rr["model"].map(xmap)
        rr["y"] = rr["reason"].map(ymap)
        ax.scatter(rr["x"], rr["y"], s=np.maximum(60, rr["n"] * 0.42),
                   color=COL["llm"], alpha=0.58, edgecolor="white", linewidth=0.9)
        for _, r in rr.iterrows():
            ax.text(r["x"], r["y"], str(int(r["n"])), ha="center", va="center", fontsize=8)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models)
        ax.set_yticks(range(len(reasons)))
        ax.set_yticklabels([wrap(r.replace("_", " "), 24) for r in reasons])
        ax.set_title("Human-review triage reason map", pad=8)
    else:
        ax.text(0.5, 0.5, "Triage reason data not found", ha="center", va="center")
        ax.set_axis_off()

    savefig(fig, figures / "Fig2_clean_safety_triage")


# -------------------------
# Figure 3 error/external
# -------------------------
def fig3_error_external(metrics, label_dist, outputs, figures):
    fig = plt.figure(figsize=(15.8, 8.8), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[0.95, 1.05], height_ratios=[1.0, 1.0])
    fig.suptitle("Distributional and medico-legal error analysis", fontsize=17, fontweight="bold", y=1.02)

    # A class size + cumulative
    ax = fig.add_subplot(gs[0, 0])
    panel(ax, "A")
    if not label_dist.empty:
        label_col, count_col = get_label_count_cols(label_dist)
        ld = label_dist.copy()
        ld[count_col] = pd.to_numeric(ld[count_col], errors="coerce")
        ld = ld.dropna(subset=[count_col]).sort_values(count_col, ascending=False)
        x = np.arange(len(ld))
        ax.bar(x, ld[count_col], color=COL["struct"], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([wrap(v, 12) for v in ld[label_col]], rotation=35, ha="right")
        ax.set_ylabel("Cases")
        ax.set_title("Class-size distribution", pad=8)
        ax2 = ax.twinx()
        cum = ld[count_col].cumsum() / ld[count_col].sum()
        ax2.plot(x, cum, color=COL["proposed"], marker="o", linewidth=2)
        ax2.set_ylabel("Cumulative fraction")
        ax2.set_ylim(0, 1.05)
        ax2.grid(False)
    else:
        ax.set_axis_off()

    # B external ranking
    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "B")
    ext = metrics[(metrics["task"].astype(str) == "external") & metrics["macro_f1"].notna()].copy()
    ext = dedup_best(ext, ("task", "method_short"), "macro_f1")
    if not ext.empty:
        top = ext.sort_values("macro_f1", ascending=False).head(10).sort_values("macro_f1")
        y = np.arange(len(top))
        ax.barh(y, top["macro_f1"], color=[group_color(m) for m in top["method_short"]], alpha=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels([wrap(m, 22) for m in top["method_short"]])
        for i, v in enumerate(top["macro_f1"]):
            ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=8)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("Macro-F1")
        ax.set_title("Medico-legal external death detection", pad=8)
    else:
        ax.text(0.5, 0.5, "External task metrics not found", ha="center", va="center")
        ax.set_axis_off()

    # C confusion matrix
    ax = fig.add_subplot(gs[1, 0])
    panel(ax, "C")
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
        ax.set_xticklabels([wrap(c, 10) for c in cmn.columns], rotation=45, ha="right", fontsize=7.5)
        ax.set_yticklabels([wrap(i, 12) for i in cmn.index], fontsize=7.5)
        ax.set_title("Row-normalized broad-task confusion matrix", pad=8)
        cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.015)
        cb.set_label("Row-normalized value")
    else:
        ax.text(0.5, 0.5, "Confusion matrix not found", ha="center", va="center")
        ax.set_axis_off()

    # D method family grouped bars
    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "D")
    mm = metrics.dropna(subset=["macro_f1"]).copy()
    mm["family"] = mm["method_short"].map(lambda m:
        "Proposed FVA" if "FVA" in m else
        "Structured ML" if "Struct" in m else
        "Narrative TF-IDF" if "Narr" in m else
        "Local LLM" if "LLM" in m else
        "Agent ablation" if ("Prior" in m or "Evidence" in m) else
        "Other"
    )
    fam = mm.groupby(["family", "task"], as_index=False)["macro_f1"].max()
    families = ["Structured ML", "Narrative TF-IDF", "Proposed FVA", "Agent ablation", "Local LLM"]
    families = [f for f in families if f in fam["family"].unique()]
    tasks = [t for t in ["broad", "external", "fine34"] if t in fam["task"].unique()]
    x = np.arange(len(families))
    width = 0.75 / max(1, len(tasks))
    colors = [COL["struct"], COL["narr"], COL["proposed"]]
    for j, task in enumerate(tasks):
        sub = fam[fam["task"] == task].set_index("family").reindex(families)
        vals = sub["macro_f1"].fillna(0).values
        ax.bar(x + (j - (len(tasks)-1)/2)*width, vals, width=width, label=task,
               alpha=0.9, color=colors[j % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels([wrap(f, 13) for f in families], rotation=15, ha="right")
    ax.set_ylabel("Best Macro-F1")
    ax.set_ylim(0, 1.02)
    ax.set_title("Best performance by method family", pad=8)
    ax.legend(frameon=False, ncol=3, loc="upper left")

    savefig(fig, figures / "Fig3_clean_error_external")


# -------------------------
# Figure 4 LLM, separated and honest
# -------------------------
def fig4_llm(metrics, outputs, figures):
    llm = read_csv(outputs / "ollama_llm_metrics.csv")
    if llm.empty:
        return
    for c in ["parse_error_rate", "n_test", "valid_n", "accuracy", "macro_f1"]:
        if c in llm.columns:
            llm[c] = pd.to_numeric(llm[c], errors="coerce")
    if "method" not in llm.columns:
        llm["method"] = "LLM"
    llm["method_short"] = llm["method"].map(simplify_method)

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.8), constrained_layout=True)
    fig.suptitle("Local lightweight LLM prompting diagnostics", fontsize=16, fontweight="bold", y=1.05)

    # A parse error
    ax = axes[0]
    panel(ax, "A")
    x = np.arange(len(llm))
    pe = llm["parse_error_rate"].fillna(0) if "parse_error_rate" in llm.columns else np.zeros(len(llm))
    ax.bar(x, pe, color=COL["proposed"], alpha=0.78)
    ax.set_xticks(x)
    ax.set_xticklabels([wrap(m, 14) for m in llm["method_short"]], rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Parse error rate")
    ax.set_title("Output parseability", pad=8)

    # B valid cases
    ax = axes[1]
    panel(ax, "B")
    if {"n_test", "valid_n"}.issubset(llm.columns):
        ax.bar(x, llm["n_test"].fillna(0), color=COL["light"], label="Generated")
        ax.bar(x, llm["valid_n"].fillna(0), color=COL["struct"], label="Valid parsed")
        ax.set_xticks(x)
        ax.set_xticklabels([wrap(m, 14) for m in llm["method_short"]], rotation=25, ha="right")
        ax.set_ylabel("Cases")
        ax.set_title("Generated vs valid parsed cases", pad=8)
        ax.legend(frameon=False)

    # C scores only if valid
    ax = axes[2]
    panel(ax, "C")
    width = 0.36
    acc = llm["accuracy"].fillna(0) if "accuracy" in llm.columns else np.zeros(len(llm))
    f1 = llm["macro_f1"].fillna(0) if "macro_f1" in llm.columns else np.zeros(len(llm))
    ax.bar(x - width/2, acc, width=width, color=COL["green"], alpha=0.82, label="Accuracy")
    ax.bar(x + width/2, f1, width=width, color=COL["agent"], alpha=0.82, label="Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels([wrap(m, 14) for m in llm["method_short"]], rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Score on parsed subset", pad=8)
    ax.legend(frameon=False)

    savefig(fig, figures / "Fig4_clean_llm_diagnostics")


def export_tables(metrics, outputs, figures):
    table_dir = figures / "tables"
    mkdir(table_dir)
    metrics.to_csv(table_dir / "all_metrics_cleaned.csv", index=False)

    # Main table: broad top methods
    broad = metrics[(metrics["task"].astype(str) == "broad") & metrics["macro_f1"].notna()].copy()
    broad = dedup_best(broad, ("task", "method_short"), "macro_f1")
    broad.sort_values("macro_f1", ascending=False).head(20).to_csv(table_dir / "main_table_broad_top20.csv", index=False)

    # External table
    ext = metrics[(metrics["task"].astype(str) == "external") & metrics["macro_f1"].notna()].copy()
    ext = dedup_best(ext, ("task", "method_short"), "macro_f1")
    ext.sort_values("macro_f1", ascending=False).head(20).to_csv(table_dir / "main_table_external_top20.csv", index=False)

    # Safety table
    safety_cols = [c for c in ["task", "method", "method_short", "accuracy", "macro_f1", "coverage", "review_rate", "error_capture_rate", "n_test"] if c in metrics.columns]
    metrics[safety_cols].sort_values(["task", "macro_f1"], ascending=[True, False]).to_csv(table_dir / "safety_metrics_for_paper.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--figures", default="figures_clean")
    args = parser.parse_args()

    outputs = Path(args.outputs)
    figures = Path(args.figures)
    mkdir(figures)

    metrics = load_metrics(outputs)
    label_dist = load_label_distribution(outputs, outputs / "phmrc_forensicva_processed.csv")

    fig1_main_results(metrics, label_dist, outputs, figures)
    fig2_safety(metrics, outputs, figures)
    fig3_error_external(metrics, label_dist, outputs, figures)
    fig4_llm(metrics, outputs, figures)
    export_tables(metrics, outputs, figures)

    print(f"Done. Clean publication figures saved to: {figures.resolve()}")
    for p in sorted(figures.glob("*.png")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
