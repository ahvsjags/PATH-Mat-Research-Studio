#!/usr/bin/env python3
"""Nested model-family benchmark for held-metal condition selection.

This is an exploratory sensitivity analysis of whether a more flexible model
family can improve the existing source-adapted rare-earth MOF condition policy.
Outer test metals and associated run blocks are excluded from all fitting and
selection. Candidate class/complexity is selected inside grouped inner folds.
The held-out comparison is against both random selection and a training-only
global best condition. It cannot establish zero-shot transfer or causality.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "inputs" / "rare_earth_mof_campaign.csv"
OUT_JSON = ROOT / "results" / "V233_RE_MOF_NESTED_MODEL_FAMILY_POLICY.json"
OUT_CSV = ROOT / "results" / "V233_RE_MOF_NESTED_MODEL_FAMILY_POLICY_BY_GRID.csv"
SEED = 20261016
BOOT = 20000
NO_PRODUCT = {"no solid product", "no crystalline product"}
RADIUS = {"Sc": .870, "Y": 1.019, "La": 1.160, "Ce": 1.143, "Pr": 1.126, "Nd": 1.109, "Sm": 1.079, "Eu": 1.066, "Gd": 1.053, "Tb": 1.040, "Dy": 1.027, "Ho": 1.015, "Er": 1.004, "Tm": .994, "Yb": .985, "Lu": .977}
Z = {"Sc": 21, "Y": 39, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71}
CARBOXYLATES = {"BDC": 2, "Phthalic": 2, "Isophthalic": 2, "BTC": 3, "Trimellitic": 3, "Pyromellitic": 4, "Mellitic": 6}

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def load():
    d=pd.read_csv(INPUT)
    d=d[~d.Metal.astype(str).str.contains(r"\+",regex=True)].copy()
    d["Primary Phase"]=d["Primary Phase"].fillna("").astype(str).str.strip()
    p=d["Primary Phase"].str.lower(); d=d[p.isin(NO_PRODUCT)|(~p.isin({"","unknown phase"}))].copy()
    d["y"]=(~d["Primary Phase"].str.lower().isin(NO_PRODUCT)).astype(int)
    d["run_id"]=d.SampleID.astype(str).str.extract(r"MG2-(\d+)-",expand=False)
    d["radius"]=d.Metal.map(RADIUS); d["atomic_number"]=d.Metal.map(Z); d["topology"]=d.Linker.map(CARBOXYLATES)
    if d[["run_id","radius","atomic_number","topology"]].isna().any().any(): raise ValueError("unmapped row")
    d["Temp"]=d.Temp.astype(float); d["AA Conc"]=d["AA Conc"].astype(float)
    return d.reset_index(drop=True)

def features(d,variant):
    x=pd.DataFrame({"radius":d.radius,"atomic_number":d.atomic_number},index=d.index)
    for l in sorted(CARBOXYLATES): x[f"linker={l}"]=(d.Linker==l).astype(float)
    t=(d.Temp-60)/20; a=d["AA Conc"]/5; r=(d.radius-1.03)/.1; q=(d.topology-3)/2
    if variant in {"poly","categorical"}:
        if variant=="poly":
            x["temp"]=t; x["temp2"]=t*t; x["acid"]=a; x["acid2"]=a*a
            x["radius_temp"]=r*t; x["radius_acid"]=r*a; x["topology_temp"]=q*t; x["topology_acid"]=q*a
        else:
            for v in sorted(d.Temp.unique())[1:]:
                z=(d.Temp==v).astype(float); x[f"temp={v:g}"]=z; x[f"radius_temp={v:g}"]=r*z; x[f"topology_temp={v:g}"]=q*z
            for v in sorted(d["AA Conc"].unique())[1:]:
                z=(d["AA Conc"]==v).astype(float); x[f"acid={v:g}"]=z; x[f"radius_acid={v:g}"]=r*z; x[f"topology_acid={v:g}"]=q*z
            for tv in sorted(d.Temp.unique())[1:]:
                for av in sorted(d["AA Conc"].unique())[1:]: x[f"temp={tv:g}:acid={av:g}"]=((d.Temp==tv)&(d["AA Conc"]==av)).astype(float)
    return x.astype(float)

SPECS=[]
for v in ["composition","poly","categorical"]:
    SPECS.append({"name":f"logistic-{v}-C1", "family":"logistic", "variant":v, "param":1.0})
for cls, fam in [(RandomForestClassifier,"random_forest"),(ExtraTreesClassifier,"extra_trees")]:
    for leaf in [2,5]: SPECS.append({"name":f"{fam}-leaf{leaf}","family":fam,"variant":"categorical","param":leaf})
for leaf in [7,15]: SPECS.append({"name":f"hist_gradient_leaf{leaf}","family":"hist_gradient","variant":"categorical","param":leaf})

def fit_predict(train,test,spec):
    Xtr=features(train,spec["variant"]); Xte=features(test,spec["variant"]); y=train.y.astype(int)
    if spec["family"]=="logistic":
        model=make_pipeline(StandardScaler(),LogisticRegression(C=spec["param"],max_iter=5000,solver="lbfgs"))
    elif spec["family"]=="random_forest":
        model=RandomForestClassifier(n_estimators=100,min_samples_leaf=spec["param"],max_features="sqrt",class_weight=None,random_state=SEED,n_jobs=1)
    elif spec["family"]=="extra_trees":
        model=ExtraTreesClassifier(n_estimators=100,min_samples_leaf=spec["param"],max_features="sqrt",class_weight=None,random_state=SEED,n_jobs=1)
    elif spec["family"]=="hist_gradient":
        model=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=spec["param"],learning_rate=.08,l2_regularization=1.,random_state=SEED)
    model.fit(Xtr,y); return model.predict_proba(Xte)[:,1]

def top1(g,col):
    s=g[col].to_numpy(float); m=np.isclose(s,s.max(),rtol=0,atol=1e-12); return float(g.y.to_numpy()[m].mean())

def boot(v,seed):
    a=np.asarray(v,float); rng=np.random.default_rng(seed); z=a[rng.integers(0,len(a),size=(BOOT,len(a)))].mean(axis=1)
    return [float(np.quantile(z,.025)),float(np.quantile(z,.975))]

def main():
    d=load(); sizes=d.groupby(["Metal","Linker"]).size(); complete=set(sizes[sizes==12].index)
    d["complete_grid"]=[(m,l) in complete for m,l in zip(d.Metal,d.Linker)]
    rows=[]; folds=[]
    for outer in sorted(d.Metal.unique()):
        print(f"outer {outer}", flush=True)
        test_all=d[d.Metal==outer].copy(); runs=set(test_all.run_id); test=test_all[test_all.complete_grid].copy()
        tr=d[(d.Metal!=outer)&(~d.run_id.isin(runs))].copy()
        if not len(test) or tr.y.nunique()!=2: continue
        scores={s["name"]:[] for s in SPECS}
        for inner in sorted(tr.Metal.unique()):
            va_all=tr[tr.Metal==inner].copy(); va=va_all[va_all.complete_grid].copy(); ir=set(va_all.run_id)
            ti=tr[(tr.Metal!=inner)&(~tr.run_id.isin(ir))].copy()
            if not len(va) or ti.y.nunique()!=2: continue
            for s in SPECS:
                va["score"]=fit_predict(ti,va,s)
                gs=[top1(g,"score")-float(g.y.mean()) for _,g in va.groupby(["Metal","Linker"],sort=True)]
                if gs: scores[s["name"]].append(float(np.mean(gs)))
        means={n:(float(np.mean(v)) if v else float("-inf")) for n,v in scores.items()}
        chosen=max(SPECS,key=lambda s:(means[s["name"]],-SPECS.index(s)))
        folds.append({"held_out_metal":outer,"selected":chosen,"inner_mean_top1_gain":means[chosen["name"]]})
        test["score_selected"]=fit_predict(tr,test,chosen)
        test["score_composition"]=fit_predict(tr,test,SPECS[0])
        cnt=tr.groupby(["Temp","AA Conc"]).y.agg(["sum","count"]); rates=((cnt["sum"]+1)/(cnt["count"]+2)).sort_values(ascending=False,kind="stable")
        t0,a0=rates.index[0]
        for (m,l),g in test.groupby(["Metal","Linker"],sort=True):
            mask=(g.Temp==t0)&(g["AA Conc"]==a0)
            rows.append({"held_out_metal":outer,"linker":l,"selected_model":chosen["name"],"selected_top1":top1(g,"score_selected"),
                         "composition_top1":top1(g,"score_composition"),"global_recipe_top1":float(g.loc[mask,"y"].iloc[0]),"random_expected_top1":float(g.y.mean()),"prevalence":float(g.y.mean())})
    out=pd.DataFrame(rows); out.to_csv(OUT_CSV,index=False)
    metrics=["selected_top1","composition_top1","global_recipe_top1","random_expected_top1"]
    bm={m:out.groupby("held_out_metal")[m].mean().sort_index() for m in metrics}
    summary={m:float(bm[m].mean()) for m in metrics}; deltas={}
    for b in metrics[1:]:
        dlt=bm["selected_top1"]-bm[b]
        deltas[b]={"mean":float(dlt.mean()),"metal_bootstrap_95ci":boot(dlt.to_numpy(),SEED+len(b)),"positive_metals":int((dlt>0).sum()),"ties":int((dlt==0).sum()),"negative_metals":int((dlt<0).sum())}
    result={"schema":"pathmat_v233_nested_model_family_condition_policy","status":"post hoc nested source-adapted offline policy replay; not zero-shot transfer, prospective validation or causal effect",
            "source":"Gaidimas et al., Chemical Science 2026, DOI 10.1039/D5SC09992G; Zenodo 10.5281/zenodo.17902549","input_sha256":sha256(INPUT),
            "cohort":{"grids":len(out),"metals":int(out.held_out_metal.nunique()),"linkers":int(out.linker.nunique()),"conditions_per_grid":12,"rows":len(out)*12,"prevalence":float(out.random_expected_top1.mean())},
            "candidate_models":SPECS,"folds":folds,"selected_model_counts":out.selected_model.value_counts().to_dict(),"macro_metal_means":summary,"selected_minus_baseline":deltas,
            "limits":["Complete 12-condition grids only; unknown endpoints excluded.","Primary-phase label is source-assigned product identity, not target purity or yield.","One campaign; models are refit within the source.","Nested selection controls held-out-metal outcome reuse, but this candidate family analysis was designed post hoc.","Offline replay does not establish prospective utility or causal effects."]}
    result["output_csv_sha256"]=sha256(OUT_CSV); OUT_JSON.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"cohort":result["cohort"],"macro_metal_means":summary,"deltas":deltas,"selected_model_counts":result["selected_model_counts"]},indent=2))

if __name__=="__main__": main()
