
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

EXTERNAL_KEYWORDS = [
    "road traffic", "accident", "injury", "injured", "fall", "falls", "drowning",
    "burn", "fire", "homicide", "suicide", "violence", "violent", "poison", "bite", "venom"
]
CARDIO_KEYWORDS = ["chest pain", "heart", "stroke", "paralysis", "blood pressure", "myocardial"]
INFECTION_KEYWORDS = ["fever", "cough", "difficulty breathing", "fast breathing", "diarrhea", "sputum", "tb", "aids", "malaria", "sepsis"]
MATERNAL_KEYWORDS = ["pregnancy", "delivery", "vaginal bleeding", "labor", "postpartum", "maternal"]
NEONATAL_KEYWORDS = ["baby", "birth", "newborn", "suckle", "cry", "fontanelle", "umbilical", "neonatal"]

def score_keywords(text, keywords):
    text = str(text).lower()
    return sum(1 for k in keywords if k in text)

def direct_prior_predict(row):
    # Deliberately weak baseline: broad module prior.
    module = str(row.get("module", "")).lower()
    if "neonate" in module:
        return "Neonatal/Perinatal", 0.45
    return "Infectious/Respiratory", 0.35

def evidence_agent_predict(row):
    text = row["narrative"]
    scores = {
        "External/Injury-related": score_keywords(text, EXTERNAL_KEYWORDS),
        "Cardiovascular": score_keywords(text, CARDIO_KEYWORDS),
        "Infectious/Respiratory": score_keywords(text, INFECTION_KEYWORDS),
        "Maternal": score_keywords(text, MATERNAL_KEYWORDS),
        "Neonatal/Perinatal": score_keywords(text, NEONATAL_KEYWORDS),
    }
    best = max(scores, key=scores.get)
    total = sum(scores.values())
    conf = scores[best] / max(total, 1)
    if scores[best] == 0:
        return "Other", 0.20
    return best, float(conf)

def verification_agent(row, pred, conf):
    text = row["narrative"].lower()
    contradictions = []
    weak_evidence = False

    if pred == "External/Injury-related" and score_keywords(text, EXTERNAL_KEYWORDS) == 0:
        contradictions.append("external_prediction_without_external_evidence")
    if pred == "Maternal" and score_keywords(text, MATERNAL_KEYWORDS) == 0:
        contradictions.append("maternal_prediction_without_maternal_evidence")
    if pred == "Neonatal/Perinatal" and score_keywords(text, NEONATAL_KEYWORDS) == 0:
        contradictions.append("neonatal_prediction_without_neonatal_evidence")

    if conf < 0.45:
        weak_evidence = True

    verified = len(contradictions) == 0 and not weak_evidence
    return verified, contradictions, weak_evidence

def human_review_triage(row, pred, conf, verified, contradictions, weak_evidence):
    high_risk = pred in {"External/Injury-related", "Maternal", "Neonatal/Perinatal"}
    review = (not verified) or weak_evidence or high_risk or conf < 0.50
    reasons = []
    if high_risk:
        reasons.append("high_medico_legal_risk_group")
    if weak_evidence:
        reasons.append("weak_evidence")
    reasons.extend(contradictions)
    if conf < 0.50:
        reasons.append("low_confidence")
    return review, reasons

def evaluate(y_true, y_pred, mask=None):
    if mask is not None:
        y_true = np.array(y_true)[mask]
        y_pred = np.array(y_pred)[mask]
    if len(y_true) == 0:
        return {"accuracy": np.nan, "macro_f1": np.nan, "coverage": 0}
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "coverage": len(y_true)
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input, low_memory=False)

    records = []
    for _, row in df.iterrows():
        pred0, conf0 = direct_prior_predict(row)
        pred1, conf1 = evidence_agent_predict(row)
        verified, contradictions, weak = verification_agent(row, pred1, conf1)
        review, reasons = human_review_triage(row, pred1, conf1, verified, contradictions, weak)

        records.append({
            "case_id": row["case_id"],
            "y_true": row["target_broad"],
            "direct_prior_pred": pred0,
            "evidence_pred": pred1,
            "evidence_confidence": conf1,
            "verified": verified,
            "human_review": review,
            "triage_reasons": "|".join(reasons),
            "final_pred": pred1,
        })

    out = pd.DataFrame(records)
    out.to_csv(out_dir / "agent_ablation_predictions.csv", index=False)

    rows = []
    y = out["y_true"].values
    for name, col in [
        ("Direct prior baseline", "direct_prior_pred"),
        ("Evidence agent", "evidence_pred"),
        ("Evidence + verification", "evidence_pred"),
        ("Full triage agent: auto-decided cases only", "final_pred")
    ]:
        if "auto-decided" in name:
            mask = ~out["human_review"].values
        else:
            mask = np.ones(len(out), dtype=bool)
        m = evaluate(y, out[col].values, mask)
        m["method"] = name
        m["task"] = "broad"
        if "auto-decided" in name:
            m["review_rate"] = float(out["human_review"].mean())
        else:
            m["review_rate"] = 0.0
        rows.append(m)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "agent_ablation_metrics.csv", index=False)

    # Append to all_metrics.csv if it exists.
    all_metrics_path = out_dir / "all_metrics.csv"
    if all_metrics_path.exists():
        old = pd.read_csv(all_metrics_path)
        common_cols = sorted(set(old.columns).union(metrics.columns))
        old = old.reindex(columns=common_cols)
        metrics = metrics.reindex(columns=common_cols)
        pd.concat([old, metrics], ignore_index=True).to_csv(all_metrics_path, index=False)

    print(metrics.to_string(index=False))
    print(f"\nSaved: {out_dir / 'agent_ablation_predictions.csv'}")

if __name__ == "__main__":
    main()
