
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi":180,"savefig.dpi":300,"font.size":10,"axes.titlesize":13,"axes.labelsize":11,"xtick.labelsize":9,"ytick.labelsize":9,"legend.fontsize":9,"axes.spines.top":False,"axes.spines.right":False})
def savefig(p): plt.tight_layout(); plt.savefig(p,bbox_inches="tight"); plt.close()
def top_methods(metrics,out):
    m=metrics.dropna(subset=["macro_f1"]).copy(); m=m[m["task"].astype(str).eq("broad")].sort_values("macro_f1",ascending=True).tail(16)
    plt.figure(figsize=(9,5.6)); plt.barh(m["method"],m["macro_f1"]); plt.xlabel("Macro-F1"); plt.title("Broad cause-of-death classification: method comparison")
    for i,v in enumerate(m["macro_f1"]): plt.text(v,i,f" {v:.3f}",va="center")
    savefig(out/"fig_advanced_broad_method_ranking.png")
def heatmap(metrics,out):
    m=metrics.dropna(subset=["macro_f1"]).copy(); p=m.pivot_table(index="method",columns="task",values="macro_f1",aggfunc="max")
    p=p.loc[p.max(axis=1).sort_values(ascending=False).head(20).index]
    plt.figure(figsize=(9.5,max(5,0.38*len(p)))); im=plt.imshow(p.values,aspect="auto"); plt.colorbar(im,label="Macro-F1")
    plt.xticks(range(len(p.columns)),p.columns,rotation=30,ha="right"); plt.yticks(range(len(p.index)),p.index); plt.title("Macro-F1 across tasks and methods")
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            val=p.values[i,j]
            if not np.isnan(val): plt.text(j,i,f"{val:.2f}",ha="center",va="center",fontsize=8)
    savefig(out/"fig_advanced_task_heatmap.png")
def triage_bubble(out,od):
    files=list(od.glob("forensic_agent_v2_predictions_*.csv"))
    if not files: return
    df=pd.concat([pd.read_csv(f).assign(source=f.stem.replace("forensic_agent_v2_predictions_","")) for f in files],ignore_index=True)
    rows=[]
    for s,x in zip(df["source"],df["triage_reasons"].fillna("")):
        for r in str(x).split("|"):
            if r: rows.append({"source":s,"reason":r})
    if not rows: return
    rr=pd.DataFrame(rows).groupby(["source","reason"]).size().reset_index(name="n")
    xs=list(rr["source"].unique()); ys=list(rr["reason"].unique())
    rr["x"]=rr["source"].map({v:i for i,v in enumerate(xs)}); rr["y"]=rr["reason"].map({v:i for i,v in enumerate(ys)})
    plt.figure(figsize=(8.5,4.8)); plt.scatter(rr["x"],rr["y"],s=rr["n"]*0.7,alpha=0.65)
    for _,r in rr.iterrows(): plt.text(r["x"],r["y"],str(r["n"]),ha="center",va="center",fontsize=8)
    plt.xticks(range(len(xs)),xs); plt.yticks(range(len(ys)),ys); plt.title("Human-review triage reasons in ForensicVA-Agent-v2"); plt.xlabel("Base classifier"); plt.ylabel("Triage reason"); plt.grid(alpha=0.2)
    savefig(out/"fig_advanced_triage_reason_bubble.png")
def selective(out,od):
    files=sorted(od.glob("forensic_agent_v2_selective_curve_*.csv"))
    if not files: return
    plt.figure(figsize=(7.6,4.8))
    for f in files:
        df=pd.read_csv(f); lab=f.stem.replace("forensic_agent_v2_selective_curve_","")
        plt.plot(df["coverage"],df["accuracy"],marker="o",label=lab)
    plt.xlabel("Coverage retained by confidence"); plt.ylabel("Accuracy"); plt.title("Selective prediction curves"); plt.legend(frameon=False); plt.grid(alpha=0.25)
    savefig(out/"fig_advanced_selective_curves.png")
def boxplot(out,od):
    files=sorted(od.glob("forensic_agent_v2_predictions_*.csv"))
    if not files: return
    df=pd.concat([pd.read_csv(f).assign(model=f.stem.replace("forensic_agent_v2_predictions_","")) for f in files],ignore_index=True)
    df["correct"]=df["y_true"]==df["final_pred"]; labels=[]; data=[]
    for model,g in df.groupby("model"):
        for status,sub in g.groupby("correct"):
            labels.append(f"{model}\n{'correct' if status else 'wrong'}"); data.append(sub["confidence"].dropna().values)
    plt.figure(figsize=(9,5)); plt.boxplot(data,tick_labels=labels,showfliers=False); plt.ylabel("Predicted probability / confidence"); plt.title("Confidence separation between correct and incorrect cases"); plt.xticks(rotation=25,ha="right"); plt.grid(axis="y",alpha=0.25)
    savefig(out/"fig_advanced_confidence_boxplot.png")
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--metrics",required=True); ap.add_argument("--outputs_dir",required=True); ap.add_argument("--out_dir",required=True); a=ap.parse_args()
    out=Path(a.out_dir); od=Path(a.outputs_dir); out.mkdir(parents=True,exist_ok=True); metrics=pd.read_csv(a.metrics)
    top_methods(metrics,out); heatmap(metrics,out); triage_bubble(out,od); selective(out,od); boxplot(out,od)
    print("Advanced figures saved to:",out)
if __name__=="__main__": main()
