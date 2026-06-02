#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart LLM diagnostics figure.

This script avoids the embarrassing empty right panel:
- If valid parsed cases exist and scores are available: draw 3 panels.
- If no valid parsed cases / no usable scores: draw only 2 panels + a compact note.
Run:
    python src/13_plot_llm_diagnostics_smart.py --outputs outputs --figures figures_final
"""

import argparse
from pathlib import Path
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COL = {
    "red": "#E64B5D",
    "blue": "#3B82F6",
    "gray": "#E5E7EB",
    "dark": "#111827",
    "green": "#38A169",
    "orange": "#F59E0B",
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

def wrap(s, width=12):
    return "\n".join(textwrap.wrap(str(s), width=width, break_long_words=False))

def simple_method(s):
    s = str(s)
    repl = {
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

def panel(fig, ax, letter):
    fig.canvas.draw()
    pos = ax.get_position()
    fig.text(pos.x0 - 0.015, pos.y1 + 0.010, letter,
             fontsize=15, fontweight="bold", ha="left", va="bottom", color=COL["dark"])

def savefig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--figures", default="figures_final")
    args = parser.parse_args()

    outputs = Path(args.outputs)
    figures = Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)

    path = outputs / "ollama_llm_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

    df = pd.read_csv(path)
    if "method" not in df.columns:
        df["method"] = "LLM"

    for c in ["parse_error_rate", "n_test", "valid_n", "accuracy", "macro_f1"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["method_short"] = df["method"].map(simple_method)
    labels = [wrap(x, 13) for x in df["method_short"]]
    x = np.arange(len(df))

    valid_sum = df["valid_n"].fillna(0).sum() if "valid_n" in df.columns else 0
    score_available = (
        valid_sum > 0 and
        (("accuracy" in df.columns and df["accuracy"].notna().any()) or
         ("macro_f1" in df.columns and df["macro_f1"].notna().any()))
    )

    if score_available:
        fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.2), constrained_layout=True)
        fig.suptitle("Supplementary local LLM prompting diagnostics", fontsize=17, fontweight="bold")

        ax = axes[0]
        pe = df["parse_error_rate"].fillna(0) if "parse_error_rate" in df.columns else np.zeros(len(df))
        ax.bar(x, pe, color=COL["red"], alpha=0.82)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Parse error rate")
        ax.set_ylim(0, 1.05)
        ax.set_title("Output parseability", pad=12)

        ax = axes[1]
        ax.bar(x, df["n_test"].fillna(0), color=COL["gray"], label="Generated")
        ax.bar(x, df["valid_n"].fillna(0), color=COL["blue"], label="Valid parsed")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Cases")
        ax.set_title("Generated vs valid parsed cases", pad=12)
        ax.legend(frameon=False)

        ax = axes[2]
        width = 0.36
        ax.bar(x - width/2, df["accuracy"].fillna(0), width=width, color=COL["green"], alpha=0.85, label="Accuracy")
        ax.bar(x + width/2, df["macro_f1"].fillna(0), width=width, color=COL["orange"], alpha=0.85, label="Macro-F1")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title("Score on parsed subset", pad=12)
        ax.legend(frameon=False)

        for ax, letter in zip(axes, ["A", "B", "C"]):
            panel(fig, ax, letter)

    else:
        # No empty third panel. The note is intentional and compact.
        fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2), constrained_layout=True)
        fig.suptitle("Supplementary local LLM prompting diagnostics", fontsize=17, fontweight="bold")

        ax = axes[0]
        pe = df["parse_error_rate"].fillna(1.0) if "parse_error_rate" in df.columns else np.ones(len(df))
        ax.bar(x, pe, color=COL["red"], alpha=0.82)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Parse error rate")
        ax.set_ylim(0, 1.05)
        ax.set_title("Output parseability", pad=12)

        ax = axes[1]
        generated = df["n_test"].fillna(0) if "n_test" in df.columns else np.zeros(len(df))
        valid = df["valid_n"].fillna(0) if "valid_n" in df.columns else np.zeros(len(df))
        ax.bar(x, generated, color=COL["gray"], label="Generated")
        ax.bar(x, valid, color=COL["blue"], label="Valid parsed")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Cases")
        ax.set_title("Generated vs valid parsed cases", pad=12)
        ax.legend(frameon=False)

        # Put an explanation as a figure-level note, not as an empty panel.
        fig.text(
            0.5, -0.015,
            "Note: no separate score panel is shown because the parsed subset was empty or too small for reliable Accuracy/Macro-F1 estimation.",
            ha="center", va="top", fontsize=9.5, color=COL["dark"]
        )

        for ax, letter in zip(axes, ["A", "B"]):
            panel(fig, ax, letter)

    savefig(fig, figures / "FigS1_final_llm_diagnostics_smart")
    print(f"Saved: {figures / 'FigS1_final_llm_diagnostics_smart.png'}")
    print(f"Saved: {figures / 'FigS1_final_llm_diagnostics_smart.pdf'}")

if __name__ == "__main__":
    main()
