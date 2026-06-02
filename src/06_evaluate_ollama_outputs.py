
import argparse, json, re
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ALLOWED = ["External/Injury-related","Cardiovascular","Infectious/Respiratory","Maternal","Neonatal/Perinatal","Cancer","Chronic/Other medical","Other"]

def extract_json(text):
    if not isinstance(text, str): return {}
    if "Client Error" in text or "404" in text[:120]: return {"_parse_error": text[:200]}
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.I).strip()
    try:
        obj = json.loads(t); return obj if isinstance(obj, dict) else {}
    except Exception: pass
    m = re.search(r"\{.*\}", t, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0)); return obj if isinstance(obj, dict) else {}
        except Exception: pass
    low = t.lower()
    for lab in ALLOWED:
        if lab.lower() in low: return {"predicted_broad_group": lab, "_fallback": True}
    return {"_parse_error": t[:200]}

def normalize_label(x):
    if not isinstance(x, str): return "PARSE_ERROR"
    xl = x.strip().lower()
    for lab in ALLOWED:
        if xl == lab.lower(): return lab
    rules = [
        ("External/Injury-related", ["external","injury","traffic","drowning","poison","accident","suicide","homicide","fall","fire"]),
        ("Cardiovascular", ["cardio","heart","stroke","myocardial"]),
        ("Infectious/Respiratory", ["infect","resp","pneumonia","fever","malaria","sepsis","cough"]),
        ("Maternal", ["maternal","pregnan","delivery"]),
        ("Neonatal/Perinatal", ["neonatal","perinatal","newborn","birth"]),
        ("Cancer", ["cancer","tumor"]),
        ("Chronic/Other medical", ["chronic","medical","diabetes","renal","copd"]),
        ("Other", ["other"]),
    ]
    for lab, keys in rules:
        if any(k in xl for k in keys): return lab
    return "PARSE_ERROR"

def safe_bool(x):
    if isinstance(x, bool): return x
    if isinstance(x, str): return x.strip().lower() in {"true","yes","1"}
    return False

def evaluate_file(path):
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        obj = extract_json(r.get("raw_response", ""))
        pred = normalize_label(obj.get("predicted_broad_group", obj.get("prediction", obj.get("label", ""))))
        rows.append({"case_id": r.get("case_id"), "gold_broad": r.get("gold_broad"), "gold_fine34": r.get("gold_fine34"),
                     "model": r.get("model"), "pred_broad": pred, "confidence": obj.get("confidence", None),
                     "human_review_required": safe_bool(obj.get("human_review_required", obj.get("requires_human_review", False))),
                     "parse_error": pred == "PARSE_ERROR", "raw_response": r.get("raw_response", "")})
    out = pd.DataFrame(rows)
    valid = out[~out["parse_error"]].copy()
    method = f"LLM_{str(df['model'].iloc[0]).replace(':','_')}" if "model" in df.columns and len(df) else f"LLM_{path.stem}"
    if len(valid) == 0:
        metrics = {"method": method, "task": "broad", "n_test": len(out), "valid_n": 0, "parse_error_rate": 1.0,
                   "accuracy": None, "macro_f1": None, "weighted_f1": None, "macro_precision": None, "macro_recall": None, "review_rate": None}
    else:
        metrics = {"method": method, "task": "broad", "n_test": len(out), "valid_n": len(valid),
                   "parse_error_rate": float(out["parse_error"].mean()),
                   "accuracy": accuracy_score(valid["gold_broad"], valid["pred_broad"]),
                   "macro_f1": f1_score(valid["gold_broad"], valid["pred_broad"], average="macro", zero_division=0),
                   "weighted_f1": f1_score(valid["gold_broad"], valid["pred_broad"], average="weighted", zero_division=0),
                   "macro_precision": precision_score(valid["gold_broad"], valid["pred_broad"], average="macro", zero_division=0),
                   "macro_recall": recall_score(valid["gold_broad"], valid["pred_broad"], average="macro", zero_division=0),
                   "review_rate": float(valid["human_review_required"].mean())}
    return out, metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pattern", default="ollama_agent_sample_*.csv")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    files = sorted(out_dir.glob(args.pattern))
    if not files: raise FileNotFoundError(f"No files found: {out_dir}/{args.pattern}")
    metrics = []
    for p in files:
        parsed, m = evaluate_file(p)
        parsed.to_csv(out_dir / f"parsed_{p.name}", index=False)
        metrics.append(m)
        print(f"{p.name}: acc={m['accuracy']} macro_f1={m['macro_f1']} parse_error={m['parse_error_rate']}")
    new = pd.DataFrame(metrics)
    new.to_csv(out_dir / "ollama_llm_metrics.csv", index=False)
    all_path = out_dir / "all_metrics.csv"
    if all_path.exists():
        old = pd.read_csv(all_path)
        common = sorted(set(old.columns).union(new.columns))
        pd.concat([old.reindex(columns=common), new.reindex(columns=common)], ignore_index=True).to_csv(all_path, index=False)
    print("Saved:", out_dir / "ollama_llm_metrics.csv")

if __name__ == "__main__":
    main()
