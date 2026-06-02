
import argparse
from pathlib import Path
import re
import pandas as pd
import numpy as np

LABEL_COLS = {
    "site", "module",
    "gs_code34", "gs_text34", "va34",
    "gs_code46", "gs_text46", "va46",
    "gs_code55", "gs_text55", "va55",
    "gs_comorbid1", "gs_comorbid2", "gs_level",
    "newid", "gs_diagnosis"
}

EXTERNAL_CAUSES = {
    "Road Traffic", "Falls", "Drowning", "Fires", "Homicide", "Suicide",
    "Poisonings", "Bite of Venomous Animal", "Violent Death"
}

MATERNAL_CAUSES = {"Maternal"}
NEONATAL_HINTS = {"Birth Asphyxia", "Preterm Delivery", "Stillbirth", "Neonatal Pneumonia", "Sepsis"}

def read_csv_safely(path: Path):
    for enc in ["utf-8", "latin1", "ISO-8859-1"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding_errors="replace", low_memory=False)

def load_codebook(data_dir: Path):
    xlsx = list(data_dir.glob("*CODEBOOK*.xlsx"))
    if not xlsx:
        return {}
    cb = pd.read_excel(xlsx[0])
    cb.columns = [str(c).strip().lower() for c in cb.columns]
    if "variable" not in cb.columns or "question" not in cb.columns:
        return {}
    mapping = {}
    for _, r in cb.iterrows():
        var = str(r["variable"]).strip()
        q = str(r["question"]).strip()
        if var and var.lower() != "nan" and q and q.lower() != "nan":
            q = re.sub(r"\s+", " ", q)
            mapping[var] = q
    return mapping

def normalize_answer(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", ""}:
        return ""
    return s

def is_informative_answer(x):
    s = normalize_answer(x)
    if not s:
        return False
    sl = s.lower()
    # Exclude non-informative negative or unknown answers.
    if sl in {"no", "0", "false", "don't know", "dont know", "refused to answer", "refused", "dk", "unknown", "not applicable"}:
        return False
    # Keep yes, numeric durations, positive categories, free text.
    return True

def cause_to_broad(cause, module):
    cause = str(cause)
    module = str(module).lower()
    if cause in EXTERNAL_CAUSES:
        return "External/Injury-related"
    if cause in MATERNAL_CAUSES:
        return "Maternal"
    if "neonate" in module or any(h.lower() in cause.lower() for h in NEONATAL_HINTS):
        return "Neonatal/Perinatal"
    if any(k in cause.lower() for k in ["pneumonia", "tb", "aids", "malaria", "sepsis", "infectious", "diarrhea", "dysentery", "meningitis", "encephalitis", "measles", "hemorrhagic fever"]):
        return "Infectious/Respiratory"
    if any(k in cause.lower() for k in ["stroke", "myocardial", "cardiovascular", "heart"]):
        return "Cardiovascular"
    if any(k in cause.lower() for k in ["cancer", "leukemia", "lymphomas", "cervical", "breast"]):
        return "Cancer"
    if any(k in cause.lower() for k in ["diabetes", "renal", "cirrhosis", "digestive", "copd"]):
        return "Chronic/Other medical"
    return "Other"

def external_binary(cause):
    return "External/Injury-related" if str(cause) in EXTERNAL_CAUSES else "Non-external"

def row_to_narrative(row, question_map, max_items=80):
    module = normalize_answer(row.get("module", "unknown"))
    site = normalize_answer(row.get("site", "unknown"))
    sex = normalize_answer(row.get("g1_05", "")) or normalize_answer(row.get("g5_02", ""))
    age_y = normalize_answer(row.get("g5_04a", "")) or normalize_answer(row.get("g1_07a", ""))
    parts = [f"This is a {module} verbal autopsy record from site {site}."]
    if sex:
        parts.append(f"The recorded sex field is {sex}.")
    if age_y:
        parts.append(f"The recorded age-related field is {age_y}.")

    evidence = []
    for col, val in row.items():
        if col in LABEL_COLS:
            continue
        ans = normalize_answer(val)
        if not is_informative_answer(ans):
            continue
        q = question_map.get(col, col)
        # Avoid long IDs or date-like boilerplate dominating the narrative.
        if any(skip in q.lower() for skip in ["date", "name", "address", "interviewer", "study id"]):
            continue
        evidence.append(f"{q}: {ans}")
        if len(evidence) >= max_items:
            break

    if evidence:
        parts.append("Reported findings include: " + "; ".join(evidence) + ".")
    else:
        parts.append("No positive symptom or circumstance fields were identified after preprocessing.")
    return " ".join(parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    question_map = load_codebook(data_dir)

    files = sorted(data_dir.glob("*ADULT*.csv")) + sorted(data_dir.glob("*CHILD*.csv")) + sorted(data_dir.glob("*NEONATE*.csv"))
    if not files:
        files = sorted(data_dir.glob("*.csv"))

    frames = []
    for p in files:
        df = read_csv_safely(p)
        df["source_file"] = p.name
        frames.append(df)

    data = pd.concat(frames, ignore_index=True, sort=False)

    if "case_id" not in data.columns:
        data.insert(0, "case_id", [f"PHMRC_{i:05d}" for i in range(len(data))])

    data["target_fine_34"] = data["gs_text34"].astype(str)
    data["target_broad"] = [cause_to_broad(c, m) for c, m in zip(data["gs_text34"], data["module"])]
    data["target_external"] = data["gs_text34"].apply(external_binary)

    data["narrative"] = data.apply(lambda r: row_to_narrative(r, question_map), axis=1)

    # Keep all raw columns for structured models plus narrative and targets.
    out_csv = out_dir / "phmrc_forensicva_processed.csv"
    data.to_csv(out_csv, index=False)

    dist = data.groupby(["module", "target_broad"]).size().reset_index(name="n")
    dist.to_csv(out_dir / "label_distribution.csv", index=False)

    print(f"Saved: {out_csv}")
    print(f"Shape: {data.shape}")
    print("\nBroad label distribution:")
    print(data["target_broad"].value_counts().to_string())
    print("\nExternal legal-medicine task:")
    print(data["target_external"].value_counts().to_string())

if __name__ == "__main__":
    main()
