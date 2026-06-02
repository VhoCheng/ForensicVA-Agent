
import argparse
from pathlib import Path
import json
import time
import requests
import pandas as pd

SYSTEM_PROMPT = """You are a cautious forensic medicine assistant.
Your task is cause-of-death classification from verbal autopsy narratives.
You must not invent evidence. Use only the provided narrative.
Return strict JSON with keys:
evidence, predicted_broad_group, confidence, verification_comment, human_review_required.
Allowed predicted_broad_group values:
External/Injury-related, Cardiovascular, Infectious/Respiratory, Maternal, Neonatal/Perinatal, Cancer, Chronic/Other medical, Other.
"""

def call_ollama(model, prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 60, "num_ctx": 1024}
    }
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    return r.json().get("response", "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, low_memory=False).sample(n=args.n, random_state=42)
    rows = []
    for _, r in df.iterrows():
        user_prompt = f"""{SYSTEM_PROMPT}

Gold label is hidden from you.

Verbal autopsy narrative:
{r['narrative']}

Return JSON only.
"""
        try:
            resp = call_ollama(args.model, user_prompt)
        except Exception as e:
            resp = json.dumps({"error": str(e)})
        rows.append({
            "case_id": r["case_id"],
            "gold_broad": r["target_broad"],
            "gold_fine34": r["target_fine_34"],
            "model": args.model,
            "raw_response": resp
        })
        print(f"Done {r['case_id']}")
        time.sleep(0.2)

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / f"ollama_agent_sample_{args.model.replace(':','_')}.csv", index=False)
    print(f"Saved: {out_dir}")

if __name__ == "__main__":
    main()
