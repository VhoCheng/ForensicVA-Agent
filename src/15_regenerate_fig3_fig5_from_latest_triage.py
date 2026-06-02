import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


COLORS = {
    "proposed": "#E84A5F",
    "structured": "#4C78A8",
    "narrative": "#2A9D8F",
    "ablation": "#F4A261",
    "llm": "#8E63CE",
    "gray": "#8D99AE",
}


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find file: {path}")
    return pd.read_csv(path)


def normalize_method_name(x: str) -> str:
    x = str(x)
    x = x.replace("ForensicVA-Agent-v2-rf", "FVA-v2-RF")
    x = x.replace("ForensicVA-Agent-v2-logreg", "FVA-v2-LR")
    x = x.replace("all_cases", "all")
    x = x.replace("auto_decided", "auto")
    return x


def collect_latest_triage_metrics(outputs: Path) -> pd.DataFrame:
    rf = read_csv_safe(outputs / "forensic_agent_v2_metrics_rf.csv")
    lr = read_csv_safe(outputs / "forensic_agent_v2_metrics_logreg.csv")

    triage = pd.concat([rf, lr], ignore_index=True)
    triage["method_short"] = triage["method"].map(normalize_method_name)

    required = [
        "method", "method_short", "accuracy", "macro_f1",
        "coverage", "review_rate", "error_capture_rate"
    ]
    missing = [c for c in required if c not in triage.columns]
    if missing:
        raise ValueError(f"Missing columns in triage metrics: {missing}")

    return triage[required].copy()


def load_all_metrics(outputs: Path) -> pd.DataFrame:
    path = outputs / "all_metrics.csv"
    if not path.exists():
        print("[Warning] outputs/all_metrics.csv not found. Figure 5 ranking will use triage metrics only.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df


def load_selective_curves(outputs: Path) -> pd.DataFrame:
    files = [
        outputs / "forensic_agent_v2_selective_curve_rf.csv",
        outputs / "forensic_agent_v2_selective_curve_logreg.csv",
    ]
    frames = []
    for f in files:
        if f.exists():
            tmp = pd.read_csv(f)
            model = "rf" if "rf" in f.name else "logreg"
            tmp["base_model"] = model
            frames.append(tmp)
    if not frames:
        print("[Warning] No selective curve files found.")
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_predictions(outputs: Path) -> pd.DataFrame:
    files = [
        outputs / "forensic_agent_v2_predictions_rf.csv",
        outputs / "forensic_agent_v2_predictions_logreg.csv",
    ]
    frames = []
    for f in files:
        if f.exists():
            tmp = pd.read_csv(f)
            model = "rf" if "rf" in f.name else "logreg"
            tmp["base_model"] = model
            frames.append(tmp)
    if not frames:
        print("[Warning] No prediction files found.")
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def pretty_method(x: str) -> str:
    mapping = {
        "FVA-v2-RF-all": "FVA-v2-RF-all",
        "FVA-v2-RF-auto": "FVA-v2-RF-auto",
        "FVA-v2-LR-all": "FVA-v2-LR-all",
        "FVA-v2-LR-auto": "FVA-v2-LR-auto",
    }
    return mapping.get(str(x), str(x))


def fig3_safety_triage(outputs: Path, figures: Path, triage: pd.DataFrame):
    figures.mkdir(parents=True, exist_ok=True)
    curves = load_selective_curves(outputs)
    preds = load_predictions(outputs)

    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(
        2, 2,
        left=0.06, right=0.98, bottom=0.08, top=0.90,
        wspace=0.28, hspace=0.38
    )
    fig.suptitle(
        "Safety-oriented human-review triage analysis",
        fontsize=16, fontweight="bold", y=0.965
    )

    # A. coverage-error-capture tradeoff
    ax = fig.add_subplot(gs[0, 0])
    plot_df = triage.copy()
    plot_df["size"] = 120 + 360 * plot_df["review_rate"].fillna(0)
    for _, r in plot_df.iterrows():
        color = COLORS["proposed"]
        ax.scatter(
            r["coverage"], r["error_capture_rate"],
            s=r["size"], color=color, alpha=0.78,
            edgecolor="white", linewidth=1.2
        )
        ax.annotate(
            pretty_method(r["method_short"]),
            xy=(r["coverage"], r["error_capture_rate"]),
            xytext=(7, 5), textcoords="offset points",
            fontsize=8
        )

    # add reference structured RF point if meaningful
    ax.scatter(1.0, 0.0, s=80, color=COLORS["structured"], alpha=0.85,
               edgecolor="white", linewidth=1.0)
    ax.annotate("Struct-RF\n(no triage)", xy=(1.0, 0.0),
                xytext=(-62, 9), textcoords="offset points", fontsize=8)

    ax.set_xlim(-0.05, 1.08)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("Automatic coverage")
    ax.set_ylabel("Error capture rate")
    ax.set_title("Coverage–error-capture trade-off")
    ax.grid(alpha=0.18)
    ax.text(-0.10, 1.05, "A", transform=ax.transAxes,
            fontsize=14, fontweight="bold")

    # B. selective prediction curves
    ax = fig.add_subplot(gs[0, 1])
    if not curves.empty:
        # Try flexible column names
        coverage_col = "coverage" if "coverage" in curves.columns else "coverage_retained"
        acc_col = "accuracy" if "accuracy" in curves.columns else None
        f1_col = "macro_f1" if "macro_f1" in curves.columns else None

        for model, color in [("logreg", "#4C78A8"), ("rf", "#2CA02C")]:
            sub = curves[curves["base_model"] == model].copy()
            if coverage_col in sub.columns:
                sub = sub.sort_values(coverage_col)
                if acc_col:
                    ax.plot(sub[coverage_col], sub[acc_col], marker="o",
                            linewidth=2, label=f"{model} accuracy", color=color)
                if f1_col:
                    ax.plot(sub[coverage_col], sub[f1_col], marker="s",
                            linewidth=2, linestyle="--", label=f"{model} Macro-F1",
                            color=color, alpha=0.72)
    else:
        ax.text(0.5, 0.5, "Selective-curve files not found",
                ha="center", va="center", transform=ax.transAxes)

    ax.set_xlim(0.05, 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Coverage retained")
    ax.set_ylabel("Score")
    ax.set_title("Selective prediction curves")
    ax.grid(alpha=0.18)
    ax.legend(loc="lower left", frameon=False, ncol=2)
    ax.text(-0.10, 1.05, "B", transform=ax.transAxes,
            fontsize=14, fontweight="bold")

    # C. confidence separation by correctness
    ax = fig.add_subplot(gs[1, 0])

    if not preds.empty and "confidence" in preds.columns:
        # 当前 prediction 文件真实列名是 y_true / final_pred
        if "is_correct" not in preds.columns:
            if {"y_true", "final_pred"}.issubset(preds.columns):
                preds["is_correct"] = preds["y_true"].astype(str) == preds["final_pred"].astype(str)
            elif {"gold_broad", "pred_broad"}.issubset(preds.columns):
                preds["is_correct"] = preds["gold_broad"].astype(str) == preds["pred_broad"].astype(str)
            elif {"y_true", "y_pred"}.issubset(preds.columns):
                preds["is_correct"] = preds["y_true"].astype(str) == preds["y_pred"].astype(str)
            else:
                preds["is_correct"] = np.nan

        groups = []
        labels = []
        colors = []

        for model in ["logreg", "rf"]:
            for correct, label2, color in [
                (True, "correct", "#6CC08B"),
                (False, "wrong", "#E76F7A"),
            ]:
                sub = preds[
                    (preds["base_model"] == model) &
                    (preds["is_correct"] == correct)
                ]["confidence"]

                sub = pd.to_numeric(sub, errors="coerce").dropna()

                if len(sub) >= 2:
                    groups.append(sub.values)
                    labels.append(f"{model}\n{label2}")
                    colors.append(color)

        if len(groups) > 0 and len(groups) == len(labels):
            bp = ax.boxplot(
                groups,
                tick_labels=labels,
                patch_artist=True,
                showfliers=False
            )

            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.72)
                patch.set_edgecolor("#333333")

            for median in bp["medians"]:
                median.set_color("#F4A261")
                median.set_linewidth(2)

            ax.set_ylabel("Classifier confidence")
        else:
            ax.text(
                0.5, 0.5,
                "Confidence values not available\nfor boxplot display",
                ha="center", va="center",
                transform=ax.transAxes,
                fontsize=9
            )
            ax.set_xticks([])
            ax.set_yticks([])

    else:
        ax.text(
            0.5, 0.5,
            "Prediction confidence files not found",
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=9
        )
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_title("Confidence separation by correctness")
    ax.grid(axis="y", alpha=0.18)
    ax.text(
        -0.10, 1.05, "C",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold"
    )

    # D. triage reason map
    ax = fig.add_subplot(gs[1, 1])

    if not preds.empty and "triage_reasons" in preds.columns:
        reason_order = [
            "weak_textual_evidence",
            "small_probability_margin",
            "low_confidence",
            "high_medico_legal_risk",
        ]

        reason_label_map = {
            "weak_textual_evidence": "Weak textual evidence",
            "small_probability_margin": "Small probability margin",
            "low_confidence": "Low confidence",
            "high_medico_legal_risk": "High medico-legal risk",
        }

        models = ["logreg", "rf"]

        for x_i, model in enumerate(models):
            sub = preds[preds["base_model"] == model].copy()
            reasons = sub["triage_reasons"].fillna("").astype(str)

            for y_i, reason in enumerate(reason_order):
                count = int(reasons.str.contains(reason, regex=False).sum())
                size = 35 + count * 0.40

                ax.scatter(
                    x_i,
                    y_i,
                    s=size,
                    color="#8E63CE",
                    alpha=0.58,
                    edgecolor="white",
                    linewidth=1.0
                )

                ax.text(
                    x_i,
                    y_i,
                    str(count),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#222222"
                )

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models)
        ax.set_yticks(range(len(reason_order)))
        ax.set_yticklabels([reason_label_map[r] for r in reason_order])
        ax.set_xlim(-0.5, len(models) - 0.5)
        ax.set_ylim(-0.5, len(reason_order) - 0.5)

    else:
        ax.text(
            0.5, 0.5,
            "Triage reason column not found",
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=9
        )
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_title("Human-review triage reason map")
    ax.grid(alpha=0.12)
    ax.text(
        -0.10, 1.05, "D",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold"
    )

    out_png = figures / "Fig2_ultrafinal_safety_triage.png"
    out_pdf = figures / "Fig2_ultrafinal_safety_triage.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


def classify_family(method: str) -> str:
    m = str(method)
    if "FVA-v2" in m or "ForensicVA-Agent" in m:
        return "Proposed FVA"
    if "Struct" in m or "Structured" in m:
        return "Structured ML"
    if "Narr" in m or "TFIDF" in m or "TF--IDF" in m:
        return "Narrative TF-IDF"
    if "Evidence" in m or "Prior" in m:
        return "Agent ablation"
    if "LLM" in m or "ollama" in m.lower():
        return "Local LLM"
    return "Other"


def fig5_policy_view(outputs: Path, figures: Path, triage: pd.DataFrame):
    figures.mkdir(parents=True, exist_ok=True)
    all_metrics = load_all_metrics(outputs)

    # Build ranking dataframe from all_metrics if possible, otherwise triage
    rows = []

    if not all_metrics.empty and {"method", "macro_f1"}.issubset(all_metrics.columns):
        tmp = all_metrics.copy()
        if "task" not in tmp.columns:
            tmp["task"] = "broad"
        broad = tmp[tmp["task"].astype(str).str.contains("broad", case=False, na=False)].copy()
        if broad.empty:
            broad = tmp.copy()
        for _, r in broad.iterrows():
            rows.append({
                "method": normalize_method_name(r.get("method", "")),
                "macro_f1": r.get("macro_f1", np.nan),
                "accuracy": r.get("accuracy", np.nan),
                "family": classify_family(normalize_method_name(r.get("method", ""))),
                "coverage": r.get("coverage", 1.0),
                "review_rate": r.get("review_rate", 0.0),
                "error_capture_rate": r.get("error_capture_rate", 0.0),
            })

    # overwrite / append latest triage metrics
    for _, r in triage.iterrows():
        rows.append({
            "method": pretty_method(r["method_short"]),
            "macro_f1": r["macro_f1"],
            "accuracy": r["accuracy"],
            "family": "Proposed FVA",
            "coverage": r["coverage"],
            "review_rate": r["review_rate"],
            "error_capture_rate": r["error_capture_rate"],
        })

    rank = pd.DataFrame(rows)
    rank = rank.dropna(subset=["macro_f1"])
    # De-duplicate by method, keeping latest appended rows for FVA
    rank = rank.drop_duplicates(subset=["method"], keep="last")
    rank = rank.sort_values("macro_f1", ascending=False).head(12)

    fig = plt.figure(figsize=(14, 5.4))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1.15, 1.0, 1.25],
        left=0.06, right=0.98, bottom=0.16, top=0.82,
        wspace=0.35
    )
    fig.suptitle(
        "Safety-performance policy view for medico-legal decision support",
        fontsize=15, fontweight="bold", y=0.96
    )

    # A. lollipop ranking
    ax = fig.add_subplot(gs[0, 0])
    order = rank.sort_values("macro_f1", ascending=True)
    y = np.arange(len(order))
    family_colors = {
        "Proposed FVA": COLORS["proposed"],
        "Structured ML": COLORS["structured"],
        "Narrative TF-IDF": COLORS["narrative"],
        "Agent ablation": COLORS["ablation"],
        "Local LLM": COLORS["llm"],
        "Other": COLORS["gray"],
    }
    for yi, (_, r) in zip(y, order.iterrows()):
        color = family_colors.get(r["family"], COLORS["gray"])
        ax.hlines(yi, 0, r["macro_f1"], color="#D8DEE9", linewidth=2)
        ax.scatter(r["macro_f1"], yi, s=55, color=color, edgecolor="white", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(order["method"])
    ax.set_xlabel("Macro-F1")
    ax.set_title("Performance ranking")
    ax.set_xlim(0, max(0.75, order["macro_f1"].max() + 0.05))
    ax.grid(axis="x", alpha=0.18)
    ax.text(-0.12, 1.06, "A", transform=ax.transAxes,
            fontsize=14, fontweight="bold")

    # B. coverage vs review burden
    ax = fig.add_subplot(gs[0, 1])
    for _, r in triage.iterrows():
        ax.scatter(
            r["coverage"], r["review_rate"],
            s=120 + 350 * r["error_capture_rate"],
            color=COLORS["proposed"], alpha=0.78,
            edgecolor="white", linewidth=1.0
        )
        ax.annotate(
            pretty_method(r["method_short"]),
            xy=(r["coverage"], r["review_rate"]),
            xytext=(6, 6), textcoords="offset points", fontsize=8
        )
    ax.scatter(1.0, 0.0, s=80, color=COLORS["structured"],
               edgecolor="white", linewidth=1.0, label="Structured baseline")
    ax.set_xlim(-0.05, 1.08)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("Automatic coverage")
    ax.set_ylabel("Review rate")
    ax.set_title("Coverage versus review burden")
    ax.grid(alpha=0.18)
    ax.text(-0.12, 1.06, "B", transform=ax.transAxes,
            fontsize=14, fontweight="bold")

    # C. policy heatmap
    ax = fig.add_subplot(gs[0, 2])
    # Use top methods + latest triage rows
    heat_methods = [
        "Struct-RF",
        "FVA-v2-RF-all",
        "Struct-LR",
        "FVA-v2-LR-all",
        "Narr-SVM",
        "Narr-LR",
        "Struct-SVM",
        "FVA-v2-RF-auto",
    ]

    # create lookup from rank
    lookup = {str(r["method"]): r for _, r in rank.iterrows()}
    # flexible name mappings from all_metrics
    def find_row(name):
        if name in lookup:
            return lookup[name]
        for k, v in lookup.items():
            if name.lower().replace("-", "") in k.lower().replace("-", ""):
                return v
        return None

    heat_rows = []
    ylabels = []
    for m in heat_methods:
        r = find_row(m)
        if r is None and m.startswith("FVA"):
            sub = triage[triage["method_short"].map(pretty_method) == m]
            if not sub.empty:
                s = sub.iloc[0]
                heat_rows.append([
                    s["accuracy"], s["macro_f1"], s["coverage"],
                    s["review_rate"], s["error_capture_rate"]
                ])
                ylabels.append(m)
                continue
        if r is not None:
            heat_rows.append([
                r.get("accuracy", np.nan),
                r.get("macro_f1", np.nan),
                r.get("coverage", 1.0),
                r.get("review_rate", 0.0),
                r.get("error_capture_rate", 0.0),
            ])
            ylabels.append(m)

    heat = np.array(heat_rows, dtype=float)
    im = ax.imshow(heat, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(["accuracy", "macro\nf1", "coverage", "review\nrate", "error\ncapture\nrate"])
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_title("Policy-relevant metrics")

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat[i, j]
            if np.isfinite(val):
                color = "white" if val > 0.55 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.015)
    cbar.set_label("Value")
    ax.text(-0.12, 1.06, "C", transform=ax.transAxes,
            fontsize=14, fontweight="bold")

    out_png = figures / "Fig4_ultrafinal_policy_story.png"
    out_pdf = figures / "Fig4_ultrafinal_policy_story.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--figures", default="figures")
    args = parser.parse_args()

    outputs = Path(args.outputs)
    figures = Path(args.figures)

    triage = collect_latest_triage_metrics(outputs)

    print("\nLatest triage metrics used for Figure 3 and Figure 5:")
    print(triage.to_string(index=False))

    fig3_safety_triage(outputs, figures, triage)
    fig5_policy_view(outputs, figures, triage)

    print("\nDone. Please re-upload/refresh these files in Overleaf:")
    print(figures / "Fig2_ultrafinal_safety_triage.pdf")
    print(figures / "Fig4_ultrafinal_policy_story.pdf")


if __name__ == "__main__":
    main()