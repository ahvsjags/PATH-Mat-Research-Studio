#!/usr/bin/env python3
"""Leave-one-metal-out evaluation of condition information on a PXRD MOF grid.

This is a source-adapted, within-source analysis (not frozen cross-source
validation). It uses a complete-factorial rare-earth/linker/temperature/modulator
campaign and a source-reported crystalline-product endpoint.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, balanced_accuracy_score, brier_score_loss

ROOT=Path(__file__).resolve().parents[1]; INP=ROOT/"inputs"; OUT=ROOT/"results"; OUT.mkdir(exist_ok=True)
RADIUS={"Sc":0.870,"Y":1.019,"La":1.160,"Ce":1.143,"Pr":1.126,"Nd":1.109,"Sm":1.079,"Eu":1.066,"Gd":1.053,"Tb":1.040,"Dy":1.027,"Ho":1.015,"Er":1.004,"Tm":0.994,"Yb":0.985,"Lu":0.977}
Z={"Sc":21,"Y":39,"La":57,"Ce":58,"Pr":59,"Nd":60,"Sm":62,"Eu":63,"Gd":64,"Tb":65,"Dy":66,"Ho":67,"Er":68,"Tm":69,"Yb":70,"Lu":71}
LINKER_CARBOXYL={"BDC":2,"Phthalic":2,"Isophthalic":2,"BTC":3,"Trimellitic":3,"Pyromellitic":4,"Mellitic":6}
NO_PRODUCT={"no solid product","no crystalline product"}

def sha(p):
 h=hashlib.sha256();
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def load():
 d=pd.read_csv(INP/"rare_earth_mof_campaign.csv")
 d["Metal"]=d.Metal.astype(str).str.strip(); d["Linker"]=d.Linker.astype(str).str.strip()
 d["Primary Phase"]=d["Primary Phase"].fillna("").astype(str).str.strip()
 d=d[~d.Metal.str.contains("\\+",regex=True)].copy()
 d=d[d["Primary Phase"].str.lower().isin(NO_PRODUCT)|(~d["Primary Phase"].str.lower().isin({"","unknown phase"}))].copy()
 d["y_crystalline_product"]=(~d["Primary Phase"].str.lower().isin(NO_PRODUCT)).astype(int)
 d["ionic_radius_8coord"]=d.Metal.map(RADIUS); d["atomic_number"]=d.Metal.map(Z)
 d["linker_carboxylate_count"]=d.Linker.map(LINKER_CARBOXYL)
 if d[["ionic_radius_8coord","atomic_number","linker_carboxylate_count"]].isna().any().any(): raise ValueError("Missing declared physical descriptor")
 if d.Metal.nunique()!=16: raise ValueError("Expected 16 held-out metal groups")
 return d.reset_index(drop=True)

def features(d, include_process):
 x=pd.DataFrame(index=d.index)
 x["ionic_radius_8coord"]=d.ionic_radius_8coord
 x["atomic_number"]=d.atomic_number
 for l in sorted(d.Linker.unique()): x["linker="+l]=(d.Linker==l).astype(float)
 if include_process:
  temp=(d.Temp.astype(float)-60)/20
  acid=d["AA Conc"].astype(float)/5
  radius=(d.ionic_radius_8coord-1.03)/0.1
  carb=(d.linker_carboxylate_count-3)/2
  x["temperature_linear"]=temp; x["temperature_squared"]=temp**2
  x["acetic_acid_linear"]=acid; x["acetic_acid_squared"]=acid**2
  x["radius_x_temperature"]=radius*temp
  x["radius_x_acid"]=radius*acid
  x["linker_carboxylates_x_temperature"]=carb*temp
  x["linker_carboxylates_x_acid"]=carb*acid
 return x

def fit_lomo(d, cval=1.0):
 y=d.y_crystalline_product.to_numpy(int); groups=d.Metal.to_numpy(str)
 outputs={}
 for label,include_process in [("composition","no"),("composition_plus_process","yes")]:
  X=features(d,include_process=="yes"); pred=np.full(len(d),np.nan)
  for metal in sorted(np.unique(groups)):
   tr=groups!=metal; te=~tr
   model=make_pipeline(StandardScaler(),LogisticRegression(C=cval,max_iter=5000,solver="lbfgs"))
   model.fit(X.loc[tr],y[tr]); pred[te]=model.predict_proba(X.loc[te])[:,1]
  outputs[label]=pred
 return outputs

def cluster_ci(d, scores, draws=10000, seed=20261016):
 rng=np.random.default_rng(seed); groups=sorted(d.Metal.unique()); y=d.y_crystalline_product.to_numpy(int)
 idx={g:np.flatnonzero(d.Metal.to_numpy()==g) for g in groups}
 vals={k:{"ap":[],"auc":[]} for k in scores}
 diff={"ap":[],"auc":[]}
 for _ in range(draws):
  pick=rng.integers(0,len(groups),len(groups)); ii=np.concatenate([idx[groups[j]] for j in pick]); yy=y[ii]
  if np.unique(yy).size<2:continue
  mm={}
  for name,s in scores.items():
   ss=s[ii]; mm[name]={"ap":average_precision_score(yy,ss),"auc":roc_auc_score(yy,ss)}
   for met,v in mm[name].items():vals[name][met].append(float(v))
  for met in diff:diff[met].append(float(mm["composition_plus_process"][met]-mm["composition"][met]))
 out={"bootstrap_draws_requested":draws,"cluster":"metal identity (16 groups)","models":{},"paired_process_minus_composition":{}}
 for name,met in vals.items():out["models"][name]={m:[float(np.quantile(v,.025)),float(np.quantile(v,.975))] for m,v in met.items()}
 out["paired_process_minus_composition"]={m:[float(np.quantile(v,.025)),float(np.quantile(v,.975))] for m,v in diff.items()}
 return out

def main():
 d=load(); pred=fit_lomo(d,1.0); y=d.y_crystalline_product.to_numpy(int)
 for k,v in pred.items():d["oof_"+k]=v
 metrics={k:{"auprc":float(average_precision_score(y,s)),"auroc":float(roc_auc_score(y,s)),"brier":float(brier_score_loss(y,s)),"balanced_accuracy_at_0.5":float(balanced_accuracy_score(y,(s>=0.5).astype(int)))} for k,s in pred.items()}
 by_metal={}
 for metal,g in d.groupby("Metal",sort=True):
  row={"n":int(len(g)),"positive":int(g.y_crystalline_product.sum()),"prevalence":float(g.y_crystalline_product.mean())}
  for name,s in pred.items():
   row[name+"_auprc"]=float(average_precision_score(g.y_crystalline_product,s[g.index])) if g.y_crystalline_product.nunique()==2 else None
   row[name+"_auroc"]=float(roc_auc_score(g.y_crystalline_product,s[g.index])) if g.y_crystalline_product.nunique()==2 else None
  by_metal[metal]=row
 boot=cluster_ci(d,pred)
 summary={"source":"Gaidimas et al. Chem. Sci. 2026, doi:10.1039/D5SC09992G; Zenodo 10.5281/zenodo.17902549",
  "source_rows":1488,"single_metal_factorial_rows":int(len(d)),"positive_crystalline_product":int(y.sum()),"negative_no_product":int((1-y).sum()),"positive_rate":float(y.mean()),
  "primary_estimand":"Difference in pooled OOF ranking performance when temperature and acetic-acid concentration (and prespecified interactions with metal radius/linker topology) are added to metal and linker descriptors, with each metal held out in turn.",
  "split":"leave-one-metal-out; all rows of the test metal are held out; no row from the held-out metal is used for fitting",
  "features":{"composition":"Shannon 8-coordinate ionic radius, atomic number, linker identity one-hot","process":"temperature, acetic-acid modulator concentration, quadratic terms, radius×condition and linker-topology×condition terms"},
  "models":"L2 logistic regression, C=1 fixed before evaluation; no tuning; process block specified from source design variables",
  "endpoint":"source-reported PXRD primary phase formation: named crystalline primary phase=1; explicit no-solid or no-crystalline-product=0; unknown primary phase excluded",
  "metrics":metrics,"metalwise":by_metal,"cluster_bootstrap":boot,
  "interpretation_boundary":"Independent source and physically characterized endpoint, but model is refit within this source; supports generalization to held-out metals in this campaign, not zero-shot transfer of the Lee-trained PATH-Mat model, cross-source generalization, or causal claims unless experimental allocation is verified.",
  "input_sha256":sha(INP/"rare_earth_mof_campaign.csv")}
 outcsv=OUT/"V229_RARE_EARTH_MOF_LEAVE_ONE_METAL_OUT_PREDICTIONS.csv"; d.to_csv(outcsv,index=False)
 summary["prediction_csv_sha256"]=sha(outcsv)
 (OUT/"V229_RARE_EARTH_MOF_LEAVE_ONE_METAL_OUT_AUDIT.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
 print(json.dumps({"n":len(d),"prevalence":float(y.mean()),"metrics":metrics,"paired_CI":boot["paired_process_minus_composition"],"by_metal":by_metal},indent=2,ensure_ascii=False))

if __name__=="__main__":main()
