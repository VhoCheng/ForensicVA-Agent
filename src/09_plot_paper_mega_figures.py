#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09_plot_paper_mega_figures.py

Generate publication-style composite figures for ForensicVA-Agent experiments.
Inputs expected:
  outputs/all_metrics.csv
  outputs/label_distribution.csv
  outputs/agent_ablation_metrics.csv
  outputs/forensic_agent_v2_predictions_*.csv
  outputs/forensic_agent_v2_selective_curve_*.csv

Usage:
  python src/09_plot_paper_mega_figures.py --outputs outputs --figures figures_paper
"""

import argparse
from pathlib import Path
import textwrap
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


warnings.filterwarnings("ignore", category=UserWarning)

# ---------- Global style ----------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 11,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.65,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PALETTE = {
    "blue": "#2F6BFF",
    "cyan": "#00A6D6",
    "green": "#2EAD67",
    "orange": "#F59E0B",
    "red": "#E84A5F",
    "purple": "#7C3AED",
    "gray": "#64748B",
    "dark": "#0F172A",
    "light": "#F8FAFC",
}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def savefig(fig, outpath: Path):
    fig.savefig(outpath.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

def short_method(name: str, max_len: int = 30) -> str:
    if not isinstance(name, str):
        return str(name)
    rep = {
        "Structured_": "Struct-",
        "Narrative_TFIDF_": "Narr-TFIDF-",
        "ForensicVA-Agent-v2-": "FVA-v2-",
        "Full triage agent: auto-decided cases only": "FVA-v1-auto",
        "Direct prior baseline": "Prior",
        "Evidence + verification": "Evidence+Verify",
        "Evidence agent": "Evidence",
        "RandomForest": "RF",
        "LinearSVM": "SVM",
        "LogReg": "LR",
        "auto_decided": "auto",
        "all_cases": "all",
    }
    s = name
    for k, v in rep.items():
        s = s.replace(k, v)
    if len(s) > max_len:
        return s[:max_len-1] + "…"
    return s

def panel_label(ax, label):
    # Axes3D.text has a different signature, so use text2D when available.
    if hasattr(ax, "text2D"):
        ax.text2D(-0.08, 1.08, label, transform=ax.transAxes,
                  fontsize=13, fontweight="bold", va="top", ha="left",
                  color=PALETTE["dark"])
    else:
        ax.text(-0.08, 1.08, label, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top", ha="left",
                color=PALETTE["dark"])

def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

def deduplicate_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows but add display name suffix for duplicate method names."""
    if df.empty or "method" not in df.columns:
        return df
    out = df.copy()
    counts = {}
    display = []
    for m in out["method"].astype(str):
        counts[m] = counts.get(m, 0) + 1
        display.append(short_method(m) if counts[m] == 1 else f"{short_method(m)} #{counts[m]}")
    out["method_display"] = display
    return out

def load_all(outputs: Path):
    all_metrics = read_csv_if_exists(outputs / "all_metrics.csv")
    label_dist = read_csv_if_exists(outputs / "label_distribution.csv")
    ablation = read_csv_if_exists(outputs / "agent_ablation_metrics.csv")
    return all_metrics, label_dist, ablation

# ---------- Figure 1: Full experimental dashboard ----------
def plot_mega_dashboard(outputs: Path, figures: Path):
    all_metrics, label_dist, ablation = load_all(outputs)
    all_metrics = deduplicate_for_display(all_metrics)

    fig = plt.figure(figsize=(17.5, 11.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.05, 1.0], width_ratios=[1.1, 1.0, 1.1])

    # A. Label distribution
    ax = fig.add_subplot(gs[0, 0])
    if not label_dist.empty:
        label_col = "target_broad" if "target_broad" in label_dist.columns else label_dist.columns[0]
        count_col = "count" if "count" in label_dist.columns else label_dist.columns[-1]
        d = label_dist.copy()
        d = d.sort_values(count_col, ascending=True).tail(10)
        labels = [textwrap.fill(str(x), 16) for x in d[label_col]]
        ax.barh(labels, d[count_col], color=PALETTE["blue"], alpha=0.85)
        ax.set_xlabel("Number of cases")
        ax.set_title("Dataset label distribution")
    else:
        ax.text(0.5, 0.5, "label_distribution.csv not found", ha="center", va="center")
        ax.set_axis_off()
    panel_label(ax, "A")

    # B. Method ranking by macro-F1
    ax = fig.add_subplot(gs[0, 1:])
    if not all_metrics.empty:
        m = all_metrics.dropna(subset=["macro_f1"]).copy()
        m = m[m["task"].astype(str).eq("broad")]
        # remove severe LLM parse failures for visual cleanliness but keep valid LLM rows
        if "parse_error_rate" in m.columns:
            m = m[(m["parse_error_rate"].isna()) | (m["parse_error_rate"] < 0.98)]
        m = m.sort_values("macro_f1", ascending=True).tail(16)
        colors = np.where(m["method"].astype(str).str.contains("ForensicVA-Agent|FVA", case=False, regex=True),
                          PALETTE["red"], PALETTE["gray"])
        ax.barh(m["method_display"], m["macro_f1"], color=colors, alpha=0.88)
        for y, v in enumerate(m["macro_f1"]):
            ax.text(v + 0.006, y, f"{v:.3f}", va="center", fontsize=8)
        ax.set_xlim(0, min(1.0, max(0.05, m["macro_f1"].max() + 0.10)))
        ax.set_xlabel("Macro-F1")
        ax.set_title("Broad cause-of-death classification performance")
    else:
        ax.text(0.5, 0.5, "all_metrics.csv not found", ha="center", va="center")
        ax.set_axis_off()
    panel_label(ax, "B")

    # C. Bubble: accuracy vs macro-F1
    ax = fig.add_subplot(gs[1, 0])
    if not all_metrics.empty:
        m = all_metrics.dropna(subset=["accuracy", "macro_f1"]).copy()
        m = m[m["task"].astype(str).eq("broad")]
        if "parse_error_rate" in m.columns:
            m = m[(m["parse_error_rate"].isna()) | (m["parse_error_rate"] < 0.98)]
        if len(m) > 0:
            size_base = m["n_test"].fillna(m["n_test"].median() if "n_test" in m else 100)
            sizes = 80 + 420 * (size_base / max(size_base.max(), 1))
            cvals = m["review_rate"].fillna(0) if "review_rate" in m else np.zeros(len(m))
            sc = ax.scatter(m["accuracy"], m["macro_f1"], s=sizes, c=cvals, cmap="viridis",
                            alpha=0.72, edgecolors="white", linewidths=0.7)
            for _, r in m.sort_values("macro_f1", ascending=False).head(6).iterrows():
                ax.annotate(short_method(r["method"], 16), (r["accuracy"], r["macro_f1"]),
                            xytext=(4, 4), textcoords="offset points", fontsize=7)
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label("Review rate")
            ax.set_xlabel("Accuracy")
            ax.set_ylabel("Macro-F1")
            ax.set_title("Accuracy–F1–review trade-off")
        else:
            ax.text(0.5, 0.5, "No valid rows", ha="center", va="center")
    panel_label(ax, "C")

    # D. Task heatmap
    ax = fig.add_subplot(gs[1, 1])
    if not all_metrics.empty:
        m = all_metrics.dropna(subset=["macro_f1"]).copy()
        if "parse_error_rate" in m.columns:
            m = m[(m["parse_error_rate"].isna()) | (m["parse_error_rate"] < 0.98)]
        p = m.pivot_table(index="method_display", columns="task", values="macro_f1", aggfunc="max")
        if not p.empty:
            p = p.loc[p.max(axis=1).sort_values(ascending=False).head(14).index]
            im = ax.imshow(p.values, aspect="auto", cmap="YlGnBu", vmin=np.nanmin(p.values), vmax=np.nanmax(p.values))
            ax.set_xticks(np.arange(len(p.columns)))
            ax.set_xticklabels(p.columns, rotation=25, ha="right")
            ax.set_yticks(np.arange(len(p.index)))
            ax.set_yticklabels(p.index)
            for i in range(p.shape[0]):
                for j in range(p.shape[1]):
                    val = p.values[i, j]
                    if np.isfinite(val):
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                                color="black" if val < np.nanmean(p.values) else "white")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label("Macro-F1")
            ax.set_title("Task-wise Macro-F1 heatmap")
        else:
            ax.text(0.5, 0.5, "No heatmap data", ha="center", va="center")
    panel_label(ax, "D")

    # E. Selective curves
    ax = fig.add_subplot(gs[1, 2])
    curve_files = sorted(outputs.glob("forensic_agent_v2_selective_curve_*.csv"))
    if curve_files:
        for f in curve_files:
            d = pd.read_csv(f)
            lab = f.stem.replace("forensic_agent_v2_selective_curve_", "")
            if "coverage" in d and "accuracy" in d:
                ax.plot(d["coverage"], d["accuracy"], marker="o", linewidth=2.0, label=lab)
        ax.set_xlabel("Coverage retained")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.02)
        ax.set_title("Selective prediction curves")
        ax.legend(frameon=False, loc="lower right")
    else:
        ax.text(0.5, 0.5, "Selective curves not found", ha="center", va="center")
    panel_label(ax, "E")

    # F. Triage reasons bubble
    ax = fig.add_subplot(gs[2, 0])
    pred_files = sorted(outputs.glob("forensic_agent_v2_predictions_*.csv"))
    if pred_files:
        rows = []
        for f in pred_files:
            d = pd.read_csv(f)
            source = f.stem.replace("forensic_agent_v2_predictions_", "")
            if "triage_reasons" in d:
                for reason in d["triage_reasons"].fillna(""):
                    for r in str(reason).split("|"):
                        if r:
                            rows.append({"model": source, "reason": r})
        if rows:
            rr = pd.DataFrame(rows).groupby(["model", "reason"]).size().reset_index(name="n")
            xs = list(rr["model"].unique())
            ys = list(rr["reason"].unique())
            rr["x"] = rr["model"].map({v: i for i, v in enumerate(xs)})
            rr["y"] = rr["reason"].map({v: i for i, v in enumerate(ys)})
            ax.scatter(rr["x"], rr["y"], s=rr["n"] * 0.65, color=PALETTE["purple"], alpha=0.65)
            for _, r in rr.iterrows():
                ax.text(r["x"], r["y"], str(int(r["n"])), ha="center", va="center", fontsize=7, color="white")
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels(xs)
            ax.set_yticks(range(len(ys)))
            ax.set_yticklabels([textwrap.fill(y, 18) for y in ys])
            ax.set_title("Human-review triage reason map")
        else:
            ax.text(0.5, 0.5, "No triage reasons", ha="center", va="center")
    else:
        ax.text(0.5, 0.5, "Prediction files not found", ha="center", va="center")
    panel_label(ax, "F")

    # G. Confidence boxplot
    ax = fig.add_subplot(gs[2, 1])
    if pred_files:
        frames = []
        for f in pred_files:
            d = pd.read_csv(f)
            d["model"] = f.stem.replace("forensic_agent_v2_predictions_", "")
            frames.append(d)
        d = pd.concat(frames, ignore_index=True)
        if {"confidence", "y_true", "final_pred", "model"}.issubset(d.columns):
            d["correct"] = np.where(d["y_true"] == d["final_pred"], "Correct", "Wrong")
            labels, data = [], []
            for model in sorted(d["model"].unique()):
                for status in ["Correct", "Wrong"]:
                    vals = d[(d["model"] == model) & (d["correct"] == status)]["confidence"].dropna().values
                    if len(vals):
                        data.append(vals)
                        labels.append(f"{model}\n{status}")
            if data:
                bp = ax.boxplot(data, patch_artist=True, showfliers=False)
                for patch, lab in zip(bp["boxes"], labels):
                    patch.set_facecolor(PALETTE["green"] if "Correct" in lab else PALETTE["red"])
                    patch.set_alpha(0.62)
                ax.set_xticklabels(labels, rotation=20, ha="right")
                ax.set_ylabel("Classifier confidence")
                ax.set_title("Confidence separation by correctness")
            else:
                ax.text(0.5, 0.5, "No confidence values", ha="center", va="center")
    panel_label(ax, "G")

    # H. Ablation
    ax = fig.add_subplot(gs[2, 2])
    if not ablation.empty:
        d = ablation.copy()
        d["method_short"] = d["method"].map(lambda x: short_method(x, 24))
        x = np.arange(len(d))
        width = 0.36
        ax.bar(x - width/2, d["accuracy"], width, label="Accuracy", color=PALETTE["blue"], alpha=0.85)
        ax.bar(x + width/2, d["macro_f1"], width, label="Macro-F1", color=PALETTE["orange"], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(d["method_short"], rotation=25, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title("Ablation study")
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "agent_ablation_metrics.csv not found", ha="center", va="center")
    panel_label(ax, "H")

    fig.suptitle("ForensicVA-Agent experimental dashboard", fontsize=16, fontweight="bold", y=1.01)
    savefig(fig, figures / "fig_paper_mega_dashboard")


# ---------- Figure 2: Safety-performance landscape ----------
def plot_safety_landscape(outputs: Path, figures: Path):
    all_metrics = deduplicate_for_display(read_csv_if_exists(outputs / "all_metrics.csv"))
    if all_metrics.empty:
        return
    m = all_metrics.copy()
    m = m[m["task"].astype(str).eq("broad")]
    m = m.dropna(subset=["accuracy", "macro_f1"], how="all")
    if m.empty:
        return

    # Use ForensicVA rows + top baselines
    keep = m["method"].astype(str).str.contains("ForensicVA-Agent|Structured|Narrative|LLM", case=False, regex=True)
    m = m[keep].copy()
    if "parse_error_rate" in m.columns:
        m = m[(m["parse_error_rate"].isna()) | (m["parse_error_rate"] < 0.98)]
    m = m.tail(24)

    fig = plt.figure(figsize=(16, 5.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 1.05])

    # 3D scatter
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    cov = m["coverage"].fillna(1.0)
    rev = m["review_rate"].fillna(0.0)
    errcap = m["error_capture_rate"].fillna(0.0)
    acc = m["accuracy"].fillna(0.0)
    colors = m["macro_f1"].fillna(0.0)
    sc = ax.scatter(cov, rev, acc, c=colors, cmap="plasma", s=70, depthshade=True, alpha=0.88)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Review rate")
    ax.set_zlabel("Accuracy")
    ax.set_title("3D safety-performance landscape")
    cb = fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.08)
    cb.set_label("Macro-F1")
    panel_label(ax, "A")

    # Pareto-like plot: coverage vs error capture, color accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    sc2 = ax2.scatter(cov, errcap, c=acc, cmap="viridis", s=90, alpha=0.82,
                      edgecolor="white", linewidth=0.7)
    for _, r in m.sort_values("accuracy", ascending=False).head(6).iterrows():
        ax2.annotate(short_method(r["method"], 15),
                     (r.get("coverage", 1.0) if pd.notna(r.get("coverage", np.nan)) else 1.0,
                      r.get("error_capture_rate", 0.0) if pd.notna(r.get("error_capture_rate", np.nan)) else 0.0),
                     xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax2.set_xlim(-0.02, 1.04)
    ax2.set_ylim(-0.02, 1.04)
    ax2.set_xlabel("Automatic coverage")
    ax2.set_ylabel("Error capture rate")
    ax2.set_title("Coverage–error-capture trade-off")
    cb2 = fig.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.02)
    cb2.set_label("Accuracy")
    panel_label(ax2, "B")

    # Area plot from selective curves
    ax3 = fig.add_subplot(gs[0, 2])
    curve_files = sorted(outputs.glob("forensic_agent_v2_selective_curve_*.csv"))
    if curve_files:
        for f in curve_files:
            d = pd.read_csv(f)
            lab = f.stem.replace("forensic_agent_v2_selective_curve_", "")
            ax3.plot(d["coverage"], d["macro_f1"], marker="o", linewidth=2.2, label=f"{lab} Macro-F1")
            ax3.fill_between(d["coverage"], d["macro_f1"], alpha=0.16)
        ax3.set_xlabel("Coverage retained")
        ax3.set_ylabel("Macro-F1")
        ax3.set_ylim(0, 1.02)
        ax3.set_title("Selective Macro-F1 area curves")
        ax3.legend(frameon=False, loc="best")
    else:
        ax3.text(0.5, 0.5, "Selective curve files not found", ha="center", va="center")
    panel_label(ax3, "C")

    savefig(fig, figures / "fig_paper_safety_landscape")


# ---------- Figure 3: LLM parseability / local deployment diagnostic ----------
def plot_llm_diagnostics(outputs: Path, figures: Path):
    llm = read_csv_if_exists(outputs / "ollama_llm_metrics.csv")
    if llm.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)

    d = llm.copy()
    d["model_short"] = d["method"].map(lambda x: short_method(x.replace("LLM_", ""), 18))

    # Parse error rate
    ax = axes[0]
    ax.bar(d["model_short"], d["parse_error_rate"], color=PALETTE["red"], alpha=0.78)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Parse error rate")
    ax.set_title("LLM output parseability")
    ax.tick_params(axis="x", rotation=25)
    panel_label(ax, "A")

    # Valid N vs N
    ax = axes[1]
    x = np.arange(len(d))
    ax.bar(x, d["n_test"], color=PALETTE["gray"], alpha=0.28, label="Generated")
    ax.bar(x, d["valid_n"], color=PALETTE["blue"], alpha=0.88, label="Valid parsed")
    ax.set_xticks(x)
    ax.set_xticklabels(d["model_short"], rotation=25, ha="right")
    ax.set_ylabel("Cases")
    ax.set_title("Generated vs valid parsed cases")
    ax.legend(frameon=False)
    panel_label(ax, "B")

    # Accuracy/F1 for valid rows
    ax = axes[2]
    width = 0.36
    acc = d["accuracy"].fillna(0)
    f1 = d["macro_f1"].fillna(0)
    ax.bar(x - width/2, acc, width, color=PALETTE["green"], alpha=0.82, label="Accuracy")
    ax.bar(x + width/2, f1, width, color=PALETTE["orange"], alpha=0.82, label="Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(d["model_short"], rotation=25, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_title("LLM baseline on parsed subset")
    ax.legend(frameon=False)
    panel_label(ax, "C")

    savefig(fig, figures / "fig_paper_llm_diagnostics")


# ---------- Figure 4: Export paper tables ----------
def export_tables(outputs: Path, figures: Path):
    all_metrics = read_csv_if_exists(outputs / "all_metrics.csv")
    if all_metrics.empty:
        return
    tables_dir = figures / "tables"
    ensure_dir(tables_dir)

    cols = [c for c in ["task", "method", "accuracy", "macro_f1", "weighted_f1", "coverage", "review_rate",
                        "error_capture_rate", "n_test", "valid_n", "parse_error_rate"] if c in all_metrics.columns]
    tab = all_metrics[cols].copy()
    for c in ["accuracy", "macro_f1", "weighted_f1", "coverage", "review_rate", "error_capture_rate", "parse_error_rate"]:
        if c in tab.columns:
            tab[c] = tab[c].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    tab.to_csv(tables_dir / "table_all_metrics_for_paper.csv", index=False)

    # Top broad table
    top = all_metrics.copy()
    top = top[top["task"].astype(str).eq("broad")]
    top = top.dropna(subset=["macro_f1"])
    if "parse_error_rate" in top.columns:
        top = top[(top["parse_error_rate"].isna()) | (top["parse_error_rate"] < 0.98)]
    top = top.sort_values("macro_f1", ascending=False).head(15)
    top = top[[c for c in cols if c in top.columns]]
    top.to_csv(tables_dir / "table_top_broad_methods.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs", help="Directory containing result CSV files.")
    ap.add_argument("--figures", default="figures_paper", help="Output directory for paper figures.")
    args = ap.parse_args()

    outputs = Path(args.outputs)
    figures = Path(args.figures)
    ensure_dir(figures)

    plot_mega_dashboard(outputs, figures)
    plot_safety_landscape(outputs, figures)
    plot_llm_diagnostics(outputs, figures)
    export_tables(outputs, figures)

    print(f"\nDone. Paper figures saved to: {figures.resolve()}")
    print("Generated files:")
    for p in sorted(figures.rglob("*")):
        if p.is_file():
            print(" -", p)

if __name__ == "__main__":
    main()
