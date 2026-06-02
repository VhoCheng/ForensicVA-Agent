
import argparse
from pathlib import Path
import pandas as pd

def read_csv_safely(path: Path, nrows=None):
    for enc in ["utf-8", "latin1", "ISO-8859-1"]:
        try:
            return pd.read_csv(path, encoding=enc, nrows=nrows, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding_errors="replace", nrows=nrows, low_memory=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    args = ap.parse_args()
    data_dir = Path(args.data_dir)

    csv_files = sorted(data_dir.glob("*PHMRC*VA*DATA*Y2013M09D11_0.csv"))
    if not csv_files:
        csv_files = sorted(data_dir.glob("*.csv"))

    print("\n=== Found CSV files ===")
    for p in csv_files:
        df = read_csv_safely(p)
        print(f"\n{p.name}")
        print(f"shape: {df.shape}")
        print("first 20 columns:", list(df.columns[:20]))

        label_cols = [c for c in ["site", "module", "gs_text34", "gs_text46", "gs_text55", "gs_level"] if c in df.columns]
        print("key columns:", label_cols)

        if "gs_text34" in df.columns:
            print("\nTop gs_text34 labels:")
            print(df["gs_text34"].value_counts(dropna=False).head(20).to_string())

if __name__ == "__main__":
    main()
