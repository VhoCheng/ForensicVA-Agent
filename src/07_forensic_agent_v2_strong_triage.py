
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

META_COLS = {"case_id","narrative","source_file","target_fine_34","target_broad","target_external","gs_code34","gs_text34","va34","gs_code46","gs_text46","va46","gs_code55","gs_text55","va55","gs_comorbid1","gs_comorbid2","gs_level"}
HIGH_RISK_GROUPS = {"External/Injury-related","Maternal","Neonatal/Perinatal"}
EVIDENCE_HINTS = {
 "External/Injury-related":["accident","injury","fall","drowning","fire","burn","homicide","suicide","poison","bite","venom","traffic","violent"],
 "Cardiovascular":["chest pain","heart","stroke","paralysis","blood pressure","myocardial"],
 "Infectious/Respiratory":["fever","cough","difficulty breathing","fast breathing","sputum","diarrhea","tb","aids","malaria","sepsis","pneumonia"],
 "Maternal":["pregnancy","delivery","labor","vaginal bleeding","postpartum","maternal"],
 "Neonatal/Perinatal":["baby","birth","newborn","suckle","cry","fontanelle","umbilical","neonatal"],
 "Cancer":["cancer","tumor","mass","breast","cervical","leukemia","lymphoma"],
 "Chronic/Other medical":["diabetes","renal","kidney","cirrhosis","copd","asthma","chronic"], "Other":[]}

def clean_X(df):
    X = df.copy()
    for c in X.columns: X[c] = X[c].astype(str).replace({"nan":"","None":""})
    return X

def make_model(kind):
    clf = RandomForestClassifier(n_estimators=300,class_weight="balanced_subsample",n_jobs=-1,random_state=42) if kind=="rf" else LogisticRegression(max_iter=3000,class_weight="balanced",n_jobs=-1)
    return Pipeline([("clean",FunctionTransformer(clean_X,validate=False)),("onehot",OneHotEncoder(handle_unknown="ignore",min_frequency=3)),("clf",clf)])

def evidence_score(txt,label):
    t = str(txt).lower()
    return sum(1 for h in EVIDENCE_HINTS.get(label,[]) if h in t)

def eval_basic(y,p):
    return {"accuracy":accuracy_score(y,p),"macro_f1":f1_score(y,p,average="macro",zero_division=0),"weighted_f1":f1_score(y,p,average="weighted",zero_division=0),"macro_precision":precision_score(y,p,average="macro",zero_division=0),"macro_recall":recall_score(y,p,average="macro",zero_division=0)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True); ap.add_argument("--out_dir",required=True)
    ap.add_argument("--base_model",choices=["logreg","rf"],default="logreg")
    ap.add_argument("--confidence_threshold",type=float,default=0.55)
    ap.add_argument("--margin_threshold",type=float,default=0.15)
    args=ap.parse_args()
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.input,low_memory=False).dropna(subset=["target_broad"]).copy()
    y=df["target_broad"].astype(str)
    cols=[c for c in df.columns if c not in META_COLS and c!="narrative"]
    tr,te,ytr,yte=train_test_split(df,y,test_size=0.2,random_state=42,stratify=y)
    model=make_model(args.base_model); model.fit(tr[cols],ytr)
    labels=list(model.classes_); proba=model.predict_proba(te[cols]); order=np.argsort(-proba,axis=1)
    top1,top2=order[:,0],order[:,1]
    pred=np.array(labels)[top1]; pred2=np.array(labels)[top2]
    conf=proba[np.arange(len(te)),top1]; margin=conf-proba[np.arange(len(te)),top2]
    rows=[]
    for i,(_,r) in enumerate(te.iterrows()):
        es=evidence_score(r["narrative"],pred[i])
        reasons=[]
        if pred[i] in HIGH_RISK_GROUPS: reasons.append("high_medico_legal_risk")
        if conf[i] < args.confidence_threshold: reasons.append("low_confidence")
        if margin[i] < args.margin_threshold: reasons.append("small_probability_margin")
        if es == 0 and pred[i] != "Other": reasons.append("weak_textual_evidence")
        rows.append({"case_id":r["case_id"],"y_true":yte.loc[r.name],"final_pred":pred[i],"top2_pred":pred2[i],"confidence":conf[i],"margin":margin[i],"evidence_score":es,"human_review":bool(reasons),"triage_reasons":"|".join(reasons),"narrative":r["narrative"]})
    p=pd.DataFrame(rows)
    p.to_csv(out_dir/f"forensic_agent_v2_predictions_{args.base_model}.csv",index=False)
    err=p["y_true"]!=p["final_pred"]
    mets=[]
    full=eval_basic(p["y_true"],p["final_pred"]); full.update({"task":"broad","method":f"ForensicVA-Agent-v2-{args.base_model}-all_cases","n_test":len(p),"coverage":1.0,"review_rate":float(p["human_review"].mean()),"error_capture_rate":float((p["human_review"]&err).sum()/max(err.sum(),1))}); mets.append(full)
    auto=p[~p["human_review"]]
    am=eval_basic(auto["y_true"],auto["final_pred"]) if len(auto) else {k:np.nan for k in full}
    am.update({"task":"broad","method":f"ForensicVA-Agent-v2-{args.base_model}-auto_decided","n_test":len(auto),"coverage":float(len(auto)/len(p)),"review_rate":float(p["human_review"].mean()),"error_capture_rate":float((p["human_review"]&err).sum()/max(err.sum(),1))}); mets.append(am)
    m=pd.DataFrame(mets); m.to_csv(out_dir/f"forensic_agent_v2_metrics_{args.base_model}.csv",index=False)
    curve=[]; tmp=p.sort_values("confidence",ascending=False)
    for cov in np.linspace(0.1,1.0,10):
        n=max(1,int(len(tmp)*cov)); sub=tmp.head(n)
        curve.append({"coverage":cov,"accuracy":accuracy_score(sub["y_true"],sub["final_pred"]),"macro_f1":f1_score(sub["y_true"],sub["final_pred"],average="macro",zero_division=0)})
    pd.DataFrame(curve).to_csv(out_dir/f"forensic_agent_v2_selective_curve_{args.base_model}.csv",index=False)
    all_path=out_dir/"all_metrics.csv"
    if all_path.exists():
        old=pd.read_csv(all_path); common=sorted(set(old.columns).union(m.columns))
        pd.concat([old.reindex(columns=common),m.reindex(columns=common)],ignore_index=True).to_csv(all_path,index=False)
    print(m.to_string(index=False))

if __name__=="__main__":
    main()
