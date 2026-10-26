#!/usr/bin/env python3
"""Frozen Lee-model transfer to an independent 2026 rare-earth MOF PXRD campaign.

No fit, recalibration, threshold selection or feature selection is performed.
The endpoint is a source-reported named secondary phase among single-metal
trials with a named primary phase; unassigned/unknown secondary entries are excluded.
All rows from the public campaign are retained if endpoint-eligible.
"""
from __future__ import annotations
import hashlib, json, math, pickle, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "inputs"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(INP))
from PATH_FEATURES import FEATURES

MODELS = {
 "B0_composition": "B0_composition.pkl",
 "B1_process_bag": "B1_process_bag.pkl",
 "B2_ordered_process": "B2_ordered_process.pkl",
}
EXPECTED_MODEL_SHA256 = {
 "B0_composition": "4a1ccd2ae6f6a51cf743a41da5b79ff7cec6f8688dbc47e001ab4dbfdd59d19e",
 "B1_process_bag": "364c85b24bc0c4d6d3a5651969bfc6bf3e0ad59e4af4a1fc905e5e3a9ae0ad3e",
 "B2_ordered_process": "f401a42cdfe20e71d3a879d961312934b13edcd82848071a7d2817ad36b1e0f6",
}
LINKER_FORMULA = {
 "BDC":"C8H6O4", "Phthalic":"C8H6O4", "Isophthalic":"C8H6O4",
 "BTC":"C9H6O6", "Trimellitic":"C9H6O6", "Pyromellitic":"C10H6O8",
 "Mellitic":"C12H6O12",
}
EXCLUDED_PRIMARY = {"", "no solid product", "no crystalline product", "unknown phase"}

def sha(p: Path) -> str:
 h=hashlib.sha256();
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()

def source_cohort() -> pd.DataFrame:
 d=pd.read_csv(INP/"rare_earth_mof_campaign.csv")
 d["Metal"]=d["Metal"].astype(str).str.strip(); d["Linker"]=d["Linker"].astype(str).str.strip()
 d["Primary Phase"]=d["Primary Phase"].fillna("").astype(str).str.strip()
 d["Secondary Phase"]=d["Secondary Phase"].fillna("").astype(str).str.strip()
 single=~d["Metal"].str.contains("\\+",regex=True)
 sec=d["Secondary Phase"]
 # Strict endpoint: named secondary-phase assignments are positive.
 # Unassigned/unknown secondary entries and NCP are excluded; negatives require
 # an identified primary phase and an empty secondary-phase field.
 pos=(sec!="")&(sec.str.upper()!="NCP")&(sec.str.lower()!="unknown phase")
 neg=(sec=="")&(~d["Primary Phase"].str.lower().isin(EXCLUDED_PRIMARY))
 keep=single&(pos|neg)
 d=d.loc[keep].copy(); d["y_secondary_phase"]=pos.loc[d.index].astype(int)
 if d["Metal"].nunique()!=16: raise ValueError(f"Expected 16 single-metal groups, observed {d['Metal'].nunique()}")
 if d["y_secondary_phase"].nunique()!=2: raise ValueError("Primary endpoint lacks both classes")
 return d.reset_index(drop=True)

def record(r) -> dict:
 m=str(r.Metal); lf=LINKER_FORMULA[str(r.Linker)]
 # Target element inventory derives from the intended metal + organic linker;
 # stoichiometry is not inferred from the phase outcome.
 target=m+lf
 # Source describes a rare-earth nitrate + organic carboxylate linker; acetic
 # acid is a modulator and is not treated as a target precursor. The model's
 # generic operation representation is the documented mix-then-heat sequence.
 prec=[f"{m}(NO3)3",lf]
 acts=[{"name":"mix"},{"name":"heat","temperature_C":float(r.Temp),"duration_h":18.0}]
 return {"target_formula":target,"precursor_formulas":prec,"actions":acts}

def score(obj: dict, rows: list[dict]):
 names=list(obj["feature_names"]); ix={n:i for i,n in enumerate(names)}
 fn=FEATURES[obj["feature_function"]]
 X=np.zeros((len(rows),len(names)),float); rowcov=[]; unknown=set(); seen=set(); matched=set()
 for i,r in enumerate(rows):
  raw=fn(r); seen.update(raw)
  for k,v in raw.items():
   if k in ix: X[i,ix[k]]+=float(v); matched.add(k)
   else: unknown.add(k)
  rowcov.append(sum(k in ix for k in raw)/max(1,len(raw)))
 model=obj["model"]; logit=X@model.coef_[0]+float(model.intercept_[0])
 rawp=1/(1+np.exp(-np.clip(logit,-700,700)))
 cal=obj["calibrator"]
 calibrated=np.interp(rawp,cal.X_thresholds_,cal.y_thresholds_)
 return rawp,calibrated,{"feature_count":len(names),"distinct_raw_keys":len(seen),
  "distinct_matched_keys":len(matched),"distinct_key_coverage":len(matched)/max(1,len(seen)),
  "mean_row_key_coverage":float(np.mean(rowcov)),"unmatched_keys":sorted(unknown),
  "unique_raw_scores":int(np.unique(rawp).size),"raw_score_sd":float(np.std(rawp)),
  "raw_score_min":float(np.min(rawp)),"raw_score_max":float(np.max(rawp))}

def ap_auc(y,s):
 return {"auprc":float(average_precision_score(y,s)),"auroc":float(roc_auc_score(y,s)),
  "brier":float(brier_score_loss(y,s)),"prevalence":float(np.mean(y))}

def within_metal_auc(d, s):
 tmp=d[["Metal","y_secondary_phase"]].copy(); tmp["score"]=s
 per={}
 for m,g in tmp.groupby("Metal",sort=True):
  if g.y_secondary_phase.nunique()==2: per[m]=float(roc_auc_score(g.y_secondary_phase,g.score))
 return per

def clustered_bootstrap(d, scores, draws=5000, seed=20261016):
 rng=np.random.default_rng(seed); groups=sorted(d.Metal.unique())
 ix={g:np.flatnonzero(d.Metal.to_numpy()==g) for g in groups}
 y=d.y_secondary_phase.to_numpy(int); model_names=list(scores)
 fixed_group_auc={}
 for name,sall in scores.items():
  fixed_group_auc[name]={}
  for g in groups:
   ii=ix[g]; gy=y[ii]
   fixed_group_auc[name][g]=float(roc_auc_score(gy,sall[ii])) if np.unique(gy).size==2 else None
 vals={n:{"ap":[],"auc":[],"within_metal_auc":[]} for n in model_names}
 diffs={f"{a}-{b}":{"ap":[],"auc":[],"within_metal_auc":[]} for a,b in [("B1_process_bag","B0_composition"),("B2_ordered_process","B1_process_bag"),("B2_ordered_process","B0_composition")]}
 for _ in range(draws):
  picked=rng.integers(0,len(groups),size=len(groups)); idx=np.concatenate([ix[groups[k]] for k in picked]); yy=y[idx]
  if np.unique(yy).size<2: continue
  mvals={}
  for n,sall in scores.items():
   ss=sall[idx]; mvals[n]={"ap":average_precision_score(yy,ss),"auc":roc_auc_score(yy,ss)}
   # Cluster-macro within-metal AUROC; precomputed once per element group.
   arr=[fixed_group_auc[n][groups[k]] for k in picked if fixed_group_auc[n][groups[k]] is not None]
   mvals[n]["within_metal_auc"]=float(np.mean(arr)) if arr else np.nan
   for metric,v in mvals[n].items():
    if np.isfinite(v): vals[n][metric].append(float(v))
  for key,out in diffs.items():
   a,b=key.split("-");
   for metric in out:
    va,vb=mvals[a][metric],mvals[b][metric]
    if np.isfinite(va) and np.isfinite(vb): out[metric].append(float(va-vb))
 ci={"models":{},"paired_differences":{},"valid_draws":None,"requested_draws":draws,"cluster":"single-metal element (16 groups)"}
 for n,ms in vals.items():
  ci["models"][n]={k:[float(np.quantile(v,.025)),float(np.quantile(v,.975))] for k,v in ms.items() if v}
 for key,ms in diffs.items():
  ci["paired_differences"][key]={k:[float(np.quantile(v,.025)),float(np.quantile(v,.975))] for k,v in ms.items() if v}
 ci["valid_draws"]=len(next(iter(vals.values()))["ap"])
 return ci

def main():
 warnings.filterwarnings("ignore",category=Warning)
 d=source_cohort(); recs=[record(r) for r in d.itertuples(index=False)]
 scores={}; calibrated={}; support={}; models={}
 for name,fn in MODELS.items():
  p=INP/fn; h=sha(p)
  if h!=EXPECTED_MODEL_SHA256[name]: raise ValueError(f"Frozen model digest mismatch: {name} {h}")
  obj=pickle.load(p.open("rb")); models[name]=obj
  scores[name],calibrated[name],support[name]=score(obj,recs)
  d[name+"_raw_score"]=scores[name]; d[name+"_calibrated_score"]=calibrated[name]
 y=d.y_secondary_phase.to_numpy(int)
 summary={n:{"raw":ap_auc(y,s),"calibrated":ap_auc(y,calibrated[n]),"support":support[n],"threshold_is_not_used":True} for n,s in scores.items()}
 group_auc={n:within_metal_auc(d,s) for n,s in scores.items()}
 summary["within_metal_auc"]={n:{"n_groups":len(v),"macro_auc":float(np.mean(list(v.values()))),"per_metal":v} for n,v in group_auc.items()}
 boot=clustered_bootstrap(d,scores)
 summary["cluster_bootstrap"]=boot
 summary["design"]={"independent_source":"Gaidimas et al., Chemical Science 2026; DOI 10.1039/D5SC09992G; Zenodo 10.5281/zenodo.17902549",
  "source_rows":1488,"single_metal_endpoint_eligible":int(len(d)),"positive_secondary_phase":int(y.sum()),"negative_known_single_phase":int((1-y).sum()),
  "single_metal_groups":int(d.Metal.nunique()),"linker_families":int(d.Linker.nunique()),
  "label_rule":"positive: any nonempty non-NCP secondary-phase field; negative: empty secondary-phase field plus identified primary crystalline phase; exclude no-solid/no-crystalline/unknown-primary rows and NCP failures; single-metal rows only",
  "feature_rule":"frozen model receives intended element inventory from metal+linker, RE nitrate and linker as two precursor formula sets, and documented mix->heat sequence with source temperature and 18 h; no labels, phase names, AC concentration or row ID enter features",
  "fit_or_calibration":"none; frozen Lee models used unchanged; no thresholds transferred",
  "sequence_test":"not estimable; all external rows share the same documented mix-then-heat representation; B2 order block is not independently varied",
  "limitations":"Retrospective source-published PXRD classifications in one automated RE-MOF campaign; external endpoint is secondary crystalline-phase presence, not Lee's text-mined impurity label; source schema mapping is explicit but differs from the inorganic-oxide Lee domain."}
 out=OUT/"V229_RARE_EARTH_MOF_FROZEN_TRANSFER_PREDICTIONS.csv"; d.to_csv(out,index=False)
 j={"schema":"pathmat_v229_frozen_transfer_rare_earth_mof","input_sha256":sha(INP/"rare_earth_mof_campaign.csv"),
    "model_sha256":{n:sha(INP/f) for n,f in MODELS.items()},"feature_script_sha256":sha(INP/"PATH_FEATURES.py"),
    "metrics":summary,"prediction_csv_sha256":sha(out)}
 (OUT/"V229_RARE_EARTH_MOF_FROZEN_TRANSFER_AUDIT.json").write_text(json.dumps(j,indent=2,ensure_ascii=False),encoding="utf-8")
 print(json.dumps({"cohort":summary["design"],"models":{n:{"raw":v["raw"],"support":v["support"]} for n,v in summary.items() if n in MODELS},"within_metal_auc":summary["within_metal_auc"],"paired_CI":boot["paired_differences"]},indent=2,ensure_ascii=False))

if __name__=="__main__": main()
