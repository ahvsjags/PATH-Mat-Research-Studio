#!/usr/bin/env python3
"""Sensitivity suite for held-out design axes in the 2026 RE-MOF PXRD grid."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

ROOT=Path(__file__).resolve().parents[1]; INP=ROOT/"inputs"; OUT=ROOT/"results"
RADIUS={"Sc":0.870,"Y":1.019,"La":1.160,"Ce":1.143,"Pr":1.126,"Nd":1.109,"Sm":1.079,"Eu":1.066,"Gd":1.053,"Tb":1.040,"Dy":1.027,"Ho":1.015,"Er":1.004,"Tm":0.994,"Yb":0.985,"Lu":0.977}
Z={"Sc":21,"Y":39,"La":57,"Ce":58,"Pr":59,"Nd":60,"Sm":62,"Eu":63,"Gd":64,"Tb":65,"Dy":66,"Ho":67,"Er":68,"Tm":69,"Yb":70,"Lu":71}
LINKER={"BDC":(2,8,2),"Phthalic":(2,8,0),"Isophthalic":(2,8,1),"BTC":(3,9,3),"Trimellitic":(3,9,2),"Pyromellitic":(4,10,4),"Mellitic":(6,12,6)}
NO_PRODUCT={"no solid product","no crystalline product"}

def load(endpoint="crystalline_product"):
 d=pd.read_csv(INP/"rare_earth_mof_campaign.csv"); d.Metal=d.Metal.astype(str).str.strip(); d.Linker=d.Linker.astype(str).str.strip()
 d["Primary Phase"]=d["Primary Phase"].fillna("").astype(str).str.strip(); d["Secondary Phase"]=d["Secondary Phase"].fillna("").astype(str).str.strip(); d=d[~d.Metal.str.contains("\\+",regex=True)].copy()
 if endpoint=="crystalline_product":
  p=d["Primary Phase"].str.lower(); d=d[p.isin(NO_PRODUCT)|(~p.isin({"","unknown phase"}))].copy(); d["y"]=(~d["Primary Phase"].str.lower().isin(NO_PRODUCT)).astype(int)
 elif endpoint=="secondary_phase":
  sec=d["Secondary Phase"].str.upper(); primary=d["Primary Phase"].str.lower()
  pos=(d["Secondary Phase"]!="")&(sec!="NCP")&(sec!="UNKNOWN PHASE")
  neg=(d["Secondary Phase"]=="")&(~primary.isin({"","no solid product","no crystalline product","unknown phase"}))
  d=d[pos|neg].copy(); d["y"]=pos.loc[d.index].astype(int)
 else: raise ValueError(endpoint)
 d["radius"]=d.Metal.map(RADIUS); d["Z"]=d.Metal.map(Z)
 desc=np.array([LINKER[x] for x in d.Linker]); d["n_carboxylates"]=desc[:,0]; d["n_carbons"]=desc[:,1]
 return d.reset_index(drop=True)

def xmat(d,process,linker_onehot):
 x=pd.DataFrame(index=d.index)
 for c in ["radius","Z"]: x[c]=d[c].astype(float)
 if linker_onehot:
  for l in sorted(d.Linker.unique()):x["linker_"+l]=(d.Linker==l).astype(float)
 else:
  for c in ["n_carboxylates","n_carbons"]:x[c]=d[c].astype(float)
 if process:
  t=(d.Temp.astype(float)-60)/20; a=d["AA Conc"].astype(float)/5; r=(d.radius-1.03)/.1; q=(d.n_carboxylates-3)/2
  for k,v in {"temp":t,"temp2":t*t,"acid":a,"acid2":a*a,"radius_temp":r*t,"radius_acid":r*a,"carboxylate_temp":q*t,"carboxylate_acid":q*a}.items():x[k]=v
 return x

def eval_axis(d,group,linker_onehot=True):
 y=d.y.to_numpy(int); g=d[group].to_numpy(); pred={"composition":np.full(len(d),np.nan),"composition_plus_process":np.full(len(d),np.nan)}
 for v in pred:
  X=xmat(d,v!="composition",linker_onehot)
  for val in sorted(pd.unique(g)):
   te=(g==val); tr=~te; model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=5000))
   model.fit(X.loc[tr],y[tr]); pred[v][te]=model.predict_proba(X.loc[te])[:,1]
 per={};
 for val in sorted(pd.unique(g)):
  te=(g==val); gy=y[te]; per[str(val)]={"n":int(te.sum()),"positive":int(gy.sum()),"prevalence":float(gy.mean())}
  for name,s in pred.items():
   per[str(val)][name+"_ap"]=float(average_precision_score(gy,s[te])) if len(np.unique(gy))==2 else None
   per[str(val)][name+"_auc"]=float(roc_auc_score(gy,s[te])) if len(np.unique(gy))==2 else None
 pooled={name:{"ap":float(average_precision_score(y,s)),"auc":float(roc_auc_score(y,s)),"brier":float(brier_score_loss(y,s))} for name,s in pred.items()}
 diffs={"ap":pooled["composition_plus_process"]["ap"]-pooled["composition"]["ap"],"auc":pooled["composition_plus_process"]["auc"]-pooled["composition"]["auc"],"brier_improvement":pooled["composition"]["brier"]-pooled["composition_plus_process"]["brier"]}
 diffs["groups_with_process_AP_gain"]=int(sum((v["composition_plus_process_ap"] or 0)>(v["composition_ap"] or 0) for v in per.values() if v["composition_ap"] is not None))
 diffs["groups_total_scoreable"]=int(sum(v["composition_ap"] is not None for v in per.values()))
 return {"grouping":group,"n_groups":int(len(per)),"linker_encoding":"identity one-hot" if linker_onehot else "held-out molecular descriptors","pooled_oof":pooled,"pooled_process_minus_composition":diffs,"per_group":per}

def main():
 res={"source":"Gaidimas et al. Chem. Sci. 2026, doi:10.1039/D5SC09992G; Zenodo 10.5281/zenodo.17902549","analysis_status":"exploratory, source-adapted cross-axis transfer; no Lee-model transfer is implied","endpoints":{}}
 for ep in ["crystalline_product","secondary_phase"]:
  d=load(ep); block={"n":len(d),"positive_rate":float(d.y.mean()),"axes":{}}
  for name,group,onehot in [("held_out_metal","Metal",True),("held_out_linker","Linker",False),("held_out_temperature","Temp",True),("held_out_acetic_acid_level","AA Conc",True)]:
   block["axes"][name]=eval_axis(d,group,onehot)
  res["endpoints"][ep]=block
 p=OUT/"V229_RARE_EARTH_MOF_CROSS_AXIS_AUDIT.json"; p.write_text(json.dumps(res,indent=2,ensure_ascii=False),encoding="utf-8")
 concise={ep:{k:{"pooled_oof":v["pooled_oof"],"delta":v["pooled_process_minus_composition"],"groups":v["n_groups"]} for k,v in b["axes"].items()} for ep,b in res["endpoints"].items()}
 print(json.dumps({"endpoints":{ep:{"n":b["n"],"prevalence":b["positive_rate"],"axes":concise[ep]} for ep,b in res["endpoints"].items()}},indent=2,ensure_ascii=False))

if __name__=="__main__":main()
