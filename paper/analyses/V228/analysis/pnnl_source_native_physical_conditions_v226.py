#!/usr/bin/env python3
"""Exploratory leave-one-experiment-out condition benchmark on printed PNNL XRD rows.

This is a source-native physical endpoint analysis; it is not a PATH-Mat evaluation.
All preprocessing and fitting are repeated inside each LOOCV training fold.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

ROOT=Path(__file__).resolve().parents[2]
DATA=Path(__file__).with_name("ironoxide_pnnl_validation_23.csv")
OUT=ROOT/"PATH-Mat_Evidence_Strengthening_V226"/"results"
OUT.mkdir(parents=True,exist_ok=True)
NUM_BASE=[]
CAT_BASE=["additive"]
NUM_PLUS=["fecl3_mM","pH","temp_C","duration_h","additive_mM"]

def sha256(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def predict_loocv(df, numeric, categorical):
    y=df.observed_nonhematite.to_numpy(int)
    cols=numeric+categorical
    X=df[cols].copy()
    numpipe=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())])
    catpipe=Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))])
    tr=ColumnTransformer([("num",numpipe,numeric),("cat",catpipe,categorical)],remainder="drop")
    model=Pipeline([("features",tr),("logit",LogisticRegression(C=1.0,solver="liblinear",max_iter=500))])
    scores=np.zeros(len(y),float)
    for train,test in LeaveOneOut().split(X,y):
        model.fit(X.iloc[train],y[train])
        scores[test]=model.predict_proba(X.iloc[test])[:,1]
    return scores

def predict_logo(df,numeric,categorical,groups):
    y=df.observed_nonhematite.to_numpy(int); X=df[numeric+categorical].copy(); scores=np.zeros(len(y),float)
    numpipe=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())])
    catpipe=Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))])
    tr=ColumnTransformer([("num",numpipe,numeric),("cat",catpipe,categorical)],remainder="drop")
    model=Pipeline([("features",tr),("logit",LogisticRegression(C=1.0,solver="liblinear",max_iter=500))])
    for train,test in LeaveOneGroupOut().split(X,y,groups=groups):
        model.fit(X.iloc[train],y[train]); scores[test]=model.predict_proba(X.iloc[test])[:,1]
    return scores

def metrics(y,s):
    return {"AP":float(average_precision_score(y,s)),"AUROC":float(roc_auc_score(y,s)),"Brier":float(brier_score_loss(y,s)),"score_sd":float(np.std(s,ddof=1)),"unique_scores":int(np.unique(np.round(s,12)).size)}

def main():
    d=pd.read_csv(DATA)
    d["additive"]=d["additive"].fillna("none").astype(str)
    y=d.observed_nonhematite.to_numpy(int)
    scores={
        "composition_identity_only":predict_loocv(d,NUM_BASE,CAT_BASE),
        "composition_plus_measured_conditions":predict_loocv(d,NUM_PLUS,CAT_BASE),
    }
    results={name:metrics(y,s) for name,s in scores.items()}
    results["prevalence_baseline"]={"AP":float(y.mean()),"AUROC":0.5,"Brier":float(np.mean((y-y.mean())**2))}
    # Descriptive paired row bootstrap; not an independent-lab uncertainty interval.
    rng=np.random.default_rng(22618); delta=[]
    for _ in range(20000):
        ix=rng.integers(0,len(y),len(y))
        if len(np.unique(y[ix]))<2: continue
        delta.append(float(average_precision_score(y[ix],scores["composition_plus_measured_conditions"][ix])-average_precision_score(y[ix],scores["composition_identity_only"][ix])))
    results["condition_minus_composition_delta_AP_bootstrap_rows_descriptive"]={"mean":float(np.mean(delta)),"CI95":[float(np.quantile(delta,.025)),float(np.quantile(delta,.975))],"valid_draws":len(delta)}
    scored=d.copy()
    for name,s in scores.items(): scored[name+"_LOO_score"]=s
    scored.to_csv(OUT/"V226_PNNL_SOURCE_NATIVE_LOO_PREDICTIONS.csv",index=False)
    base=d[d.additive.eq("none")].copy().reset_index(drop=True)
    yb=base.observed_nonhematite.to_numpy(int)
    sb=predict_loocv(base,["fecl3_mM","pH","temp_C"],[])
    base_results={"n":len(base),"positive":int(yb.sum()),"prevalence":float(yb.mean()),"same_precursor_identity":True,"LOOCV_conditions":metrics(yb,sb),"prevalence_baseline":{"AP":float(yb.mean()),"AUROC":0.5,"Brier":float(np.mean((yb-yb.mean())**2))}}
    # Label-free three-band concentration grouping approximates the source's
    # low/intermediate/high FeCl3 design levels; every prediction is out of band.
    groups=pd.qcut(base.fecl3_mM,3,labels=False,duplicates="drop").to_numpy()
    sg=predict_logo(base,["fecl3_mM","pH","temp_C"],[],groups)
    base_results["leave_one_concentration_band_out_conditions"]=metrics(yb,sg)
    base_results["concentration_band_sizes"]={str(k):int((groups==k).sum()) for k in np.unique(groups)}
    base_scores=base.copy();base_scores["conditions_LOCO_concentration_band_score"]=sg
    base_scores.to_csv(OUT/"V226_PNNL_SAME_COMPOSITION_LOCO_PREDICTIONS.csv",index=False)
    out={"schema":"pathmat_v226_pnnl_source_native_physical_conditions_loocv","status":"exploratory_retrospective_source_native_analysis; labels were public before analysis","source":"Liu et al. (2023), arXiv:2303.11244 Tables 1-2; only 23 printed rows; Exp. 18 absent","n":len(d),"positive":int(y.sum()),"prevalence":float(y.mean()),"endpoint":"non-Hem phase vs Hem as reported by source XRD table","unit":"one table row per designed experiment; no replicate-level raw XRD data","models":{"composition_identity_only":"fixed C=1 logistic regression; one-hot additive identity; no target or precursor identity variation beyond additive","composition_plus_measured_conditions":"same fixed logistic regression plus FeCl3 concentration, pH, temperature, reaction time, additive concentration; all conditions reported before synthesis","same_composition_control":"17 no-additive rows share FeCl3/NaOH precursor identities and Fe2O3 target; fixed C=1 logistic regression with FeCl3 concentration, pH, temperature; both leave-one-row-out and leave-one-label-free-FeCl3-concentration-tertile-out predictions","preprocessing":"median imputation, standardization and one-hot encoding fit separately within each training fold","selection":"no hyperparameter tuning; no threshold selection"},"metrics":results,"same_composition_results":base_results,"source_data_sha256":sha256(DATA),"claim_boundary":"This can test whether measured conditions associate with the XRD phase label within this single iron-oxide source. It is not a frozen PATH-Mat transfer test, not a multi-material generalization estimate, and not an intervention-effect estimate. The small set is a designed experiment table with no replicate-level error bars; row bootstrap is descriptive only."}
    (OUT/"V226_PNNL_SOURCE_NATIVE_LOO_AUDIT.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps({"n":len(d),"positive":int(y.sum()),"prevalence":float(y.mean()),"results":results},indent=2))

if __name__=="__main__": main()
