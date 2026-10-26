#!/usr/bin/env python3
"""Within-formulation matched condition contrasts in public RE-MOF experiments."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; INP=ROOT/"inputs"; OUT=ROOT/"results"
NO_PRODUCT={"no solid product","no crystalline product"}

def load():
 d=pd.read_csv(INP/"rare_earth_mof_campaign.csv")
 for c in ["Metal","Linker"]:d[c]=d[c].astype(str).str.strip()
 d["Primary Phase"]=d["Primary Phase"].fillna("").astype(str).str.strip()
 d["Secondary Phase"]=d["Secondary Phase"].fillna("").astype(str).str.strip()
 d=d[~d.Metal.str.contains("\\+",regex=True)].copy()
 p=d["Primary Phase"].str.lower()
 d["known_crystal_outcome"]=np.where(p.isin(NO_PRODUCT),0,np.where(~p.isin({"","unknown phase"}),1,np.nan))
 sec=d["Secondary Phase"].str.upper(); pri=p
 pos=(d["Secondary Phase"]!="")&(sec!="NCP")&(sec!="UNKNOWN PHASE")
 neg=(d["Secondary Phase"]=="")&(~pri.isin({"","no solid product","no crystalline product","unknown phase"}))
 d["known_secondary_outcome"]=np.where(pos,1,np.where(neg,0,np.nan))
 return d

def contrasts(d, ycol, factor, low, high, fixed):
 z=d[d[ycol].notna()].copy(); z[ycol]=z[ycol].astype(float)
 piv=z.pivot_table(index=fixed,columns=factor,values=ycol,aggfunc="first")
 if low not in piv or high not in piv:raise ValueError("missing factor level")
 q=piv[[low,high]].dropna().copy(); q["delta"]=q[high]-q[low]
 q["direction"]=np.select([q.delta>0,q.delta<0],["high_level_positive","high_level_negative"],default="tie")
 by_metal=q.groupby(level="Metal").delta.mean().dropna()
 rng=np.random.default_rng(20261016); metals=by_metal.index.to_list(); draws=[]
 for _ in range(20000):
  pick=rng.integers(0,len(metals),len(metals)); draws.append(float(by_metal.iloc[pick].mean()))
 draws=np.asarray(draws)
 return {"endpoint":ycol,"contrast":f"{high} minus {low}","pairing_strata":str(fixed),"n_matched_cells":int(len(q)),
  "n_up":int((q.delta>0).sum()),"n_down":int((q.delta<0).sum()),"n_ties":int((q.delta==0).sum()),
  "mean_cell_risk_difference":float(q.delta.mean()),"median_cell_risk_difference":float(q.delta.median()),
  "equal_metal_mean_difference":float(by_metal.mean()),"metal_cluster_bootstrap_95ci_equal_metal_mean":[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))],
  "direction_by_metal":{str(m):{"n":int((q.loc[(q.index.get_level_values("Metal")==m),"delta"]>0).sum()),"down":int((q.loc[(q.index.get_level_values("Metal")==m),"delta"]<0).sum()),"mean_delta":float(v)} for m,v in by_metal.items()},
  "interpretation":"matched observational contrast in a systematic high-throughput experimental grid; no treatment randomization or independent sample replication is claimed."}

def main():
 d=load(); res={"source":"Gaidimas et al. Chem. Sci. 2026; dataset DOI 10.5281/zenodo.17902549","n_single_metal_grid_rows":int(len(d)),"contrasts":{}}
 for ycol in ["known_crystal_outcome","known_secondary_outcome"]:
  res["contrasts"][ycol]={
   "temperature_80_vs_40C":contrasts(d,ycol,"Temp",40,80,["Metal","Linker","AA Conc"]),
   "acetic_acid_10_vs_0M":contrasts(d,ycol,"AA Conc",0,10,["Metal","Linker","Temp"])}
 out=OUT/"V229_RE_MOF_MATCHED_CONDITION_CONTRASTS.json"; out.write_text(json.dumps(res,indent=2,ensure_ascii=False),encoding="utf-8")
 print(json.dumps(res,indent=2,ensure_ascii=False))

if __name__=="__main__":main()
