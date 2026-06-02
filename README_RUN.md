
# ForensicVA-Agent Starter Pipeline

This starter package is designed for the PHMRC Verbal Autopsy CSV files.

## 0. Recommended folder structure

Put your downloaded PHMRC files here:

```text
ForensicVA-Agent/
├── data/
│   └── raw/
│       ├── IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv
│       ├── IHME_PHMRC_VA_DATA_CHILD_Y2013M09D11_0.csv
│       ├── IHME_PHMRC_VA_DATA_NEONATE_Y2013M09D11_0.csv
│       └── IHME_PHMRC_VA_DATA_CODEBOOK_Y2013M09D11_0.xlsx
├── src/
├── outputs/
└── figures/
```

## 1. Create and activate environment on macOS

```bash
cd /path/to/ForensicVA-Agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Inspect data

```bash
python src/00_inspect_data.py --data_dir data/raw
```

## 3. Prepare processed dataset

```bash
python src/01_prepare_dataset.py --data_dir data/raw --out_dir outputs
```

This generates:

```text
outputs/phmrc_forensicva_processed.csv
outputs/label_distribution.csv
```

## 4. Run baseline experiments

```bash
python src/02_run_ml_baselines.py --input outputs/phmrc_forensicva_processed.csv --out_dir outputs
```

This runs:
- structured Logistic Regression
- structured Linear SVM
- structured Random Forest
- narrative TF-IDF + Logistic Regression
- narrative TF-IDF + Linear SVM

It evaluates:
- broad cause group
- external vs non-external legal-medicine task
- fine-grained gs_text34 cause-of-death task

## 5. Run rule-based ForensicVA-Agent ablation prototype

```bash
python src/03_run_forensic_agent_ablation.py --input outputs/phmrc_forensicva_processed.csv --out_dir outputs
```

This is not the final LLM version. It is a reproducible prototype for the paper's ablation design:
- direct prior baseline
- evidence-only agent
- evidence + verification
- evidence + verification + human-review triage

## 6. Draw figures

```bash
python src/04_plot_results.py --processed outputs/phmrc_forensicva_processed.csv --metrics outputs/all_metrics.csv --out_dir figures
```

## 7. Optional: run local Ollama LLM agent on a small subset

Install Ollama first, then pull a model:

```bash
ollama pull qwen2.5:7b
python src/05_run_ollama_agent_sample.py --input outputs/phmrc_forensicva_processed.csv --out_dir outputs --model qwen2.5:7b --n 20
```

## Suggested first manuscript title

Toward Auditable AI in Legal Medicine: Evidence-Grounded Large Language Model Agents for Cause-of-Death Classification from Verbal Autopsy Data

