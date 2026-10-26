from __future__ import annotations
import hashlib, json, math, pathlib, platform, time
from collections import Counter, defaultdict
import numpy as np
from pymatgen.core import Composition, Element
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

ROOT=pathlib.Path(__file__).resolve().parents[1]
LEDGER=ROOT/'PATH-Mat_Evidence_Expansion_V136'/'online_sources'/'ledger_precursor_genome.json'
OUT=pathlib.Path(__file__).resolve().parent
SEED=14829
N_BOOT=3000
ELEMENTS=[Element.from_Z(i).symbol for i in range(1,119)]

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def reduced(formula):
 try: return Composition(str(formula)).reduced_formula
 except Exception: return None

def frac(formula):
 out=np.zeros(len(ELEMENTS),dtype=float)
 if not formula:return out
 try:
  c=Composition(str(formula)); total=float(c.num_atoms)
  for e,a in c.get_el_amt_dict().items():out[Element(e).Z-1]=float(a)/total
 except Exception: pass
 return out

def get_profile(s):
 pr=((s.get('synthesis') or {}).get('temperature_profile') or {})
 t=np.asarray(pr.get('time_minutes') or [],dtype=float); y=np.asarray(pr.get('temperature_celsius') or [],dtype=float)
 ok=np.isfinite(t)&np.isfinite(y);t,y=t[ok],y[ok]
 if len(t)<2 or np.ptp(t)<=0:return None
 ix=np.argsort(t,kind='stable');t,y=t[ix],y[ix]
 tu,inv=np.unique(t,return_inverse=True)
 if len(tu)<2:return None
 yu=np.array([np.mean(y[inv==i]) for i in range(len(tu))])
 tau=(tu-tu[0])/(tu[-1]-tu[0]); grid=np.linspace(0,1,32)
 ordered=np.interp(grid,tau,yu); sorted_values=np.quantile(yu,grid)
 peak=float(np.max(yu)); pi=int(np.argmax(yu))
 ramp=(peak-float(yu[0]))/(float(tu[pi]-tu[0])+1e-9) if pi else 0.
 cool=(peak-float(yu[-1]))/(float(tu[-1]-tu[pi])+1e-9) if pi<len(tu)-1 else 0.
 summary=np.array([float(tu[-1]-tu[0]),float(np.mean(yu)),float(np.std(yu)),float(np.median(yu)),peak,
                   float(yu[0]),float(yu[-1]),float(np.mean(yu>=.9*peak)),float(np.mean(yu>=.95*peak)),
                   ramp,cool,float(np.trapezoid(yu,tau))],float)
 peak_pos=float(np.argmax(ordered)/max(1,len(ordered)-1))
 chronology=np.r_[peak_pos,[float(np.mean(ordered[i*8:(i+1)*8])) for i in range(4)]]
 return ordered,sorted_values,summary,chronology

def active_case(s):
 x=((s.get('characterization') or {}).get('xrd') or {})
 scans=x.get('scans') or []; si=s.get('active_scan_index')
 if si is None or int(si)>=len(scans):return None
 sc=scans[int(si)];ci=sc.get('active_case_index');cases=sc.get('refinement_cases') or []
 if ci is None or int(ci)>=len(cases):return None
 return cases[int(ci)]

def phase_metrics(s, case):
 pfs=case.get('phase_weights') or {}
 if not pfs:return None
 precs=s.get('precursors') or []
 precursor_formulas={reduced(p.get('formula_clean') or p.get('formula')) for p in precs}
 precursor_formulas.discard(None)
 residual=0.; matched_names=[]; unknown_weight=0.
 for name,w0 in pfs.items():
  try:w=float(w0)
  except Exception:continue
  f=reduced(str(name).split('_')[0])
  if f is None:unknown_weight+=w
  elif f in precursor_formulas:residual+=w;matched_names.append(str(name))
 total=float(sum(float(v) for v in pfs.values()))
 if total<=0:return None
 return min(1.,max(0.,residual/total)),total,unknown_weight/total,matched_names

def input_chemistry(s):
 syn=s.get('synthesis') or {}; precs=s.get('precursors') or []
 tmap={str(x.get('name')):x.get('target_mass') for x in (syn.get('precursor_target_masses') or [])}
 amap={str(x.get('name')):x.get('actual_mass') for x in (syn.get('precursor_actual_masses') or [])}
 # Formula-keyed arrays are sorted, so precursor list order cannot masquerade as route order.
 entries=[]
 for p in precs:
  name=str(p.get('sa_product_number') or p.get('name') or p.get('formula_clean') or p.get('formula'))
  # The synthesis mass keys normally use the stock name, which need not equal formula.
  candidates=[x.get('name') for x in (syn.get('precursor_target_masses') or []) if str(x.get('name','')).split('_PG')[0] in {str(p.get('formula_clean') or p.get('formula')),str(p.get('name'))}]
  masskey=str(candidates[0]) if candidates else None
  tm=tmap.get(masskey) if masskey else None; am=amap.get(masskey) if masskey else None
  if tm is None or am is None:
   # Fall back to the same input list positions only if the explicit stock names cannot be aligned.
   j=len(entries); titems=syn.get('precursor_target_masses') or []; aitems=syn.get('precursor_actual_masses') or []
   if j<len(titems) and j<len(aitems):tm=titems[j].get('target_mass');am=aitems[j].get('actual_mass')
  try:tm=float(tm) if tm is not None else np.nan
  except Exception:tm=np.nan
  try:am=float(am) if am is not None else np.nan
  except Exception:am=np.nan
  entries.append((str(p.get('formula_clean') or p.get('formula') or ''),tm,am,
                  float(p['sa_purity']) if p.get('sa_purity') is not None else np.nan))
 entries.sort(key=lambda z:z[0])
 tv=np.array([z[1] for z in entries],float);av=np.array([z[2] for z in entries],float)
 tsum=np.nansum(tv);asum=np.nansum(av)
 tfrac=tv/tsum if tsum>0 else np.full(len(tv),np.nan)
 afrac=av/asum if asum>0 else np.full(len(av),np.nan)
 massdev=(av-tv)/tv if len(tv) and np.all(np.isfinite(tv)) and np.all(tv>0) and np.all(np.isfinite(av)) else np.full(len(tv),np.nan)
 purge=np.array([z[3] for z in entries],float)
 vals=[s.get('reaction_energy_ev_per_atom'),tsum if tsum>0 else np.nan,asum if asum>0 else np.nan,
       (asum/tsum) if tsum>0 and asum>0 else np.nan]
 for a in (tfrac,afrac,massdev,purge): vals.extend(list(a[:2])+[np.nan]*(2-len(a[:2])))
 return np.asarray(vals,float)

def fit_predict(X,y,folds,model_name):
 p=np.full(len(y),np.nan)
 for tr,te in folds:
  if model_name=='ridge':
   model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),Ridge(alpha=10.0))
  else:
   model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),ExtraTreesRegressor(n_estimators=500,min_samples_leaf=8,max_features=.8,n_jobs=1,random_state=SEED))
  model.fit(X[tr],y[tr]);p[te]=np.clip(model.predict(X[te]),0.,1.)
 if np.isnan(p).any():raise RuntimeError('OOF incomplete')
 return p

def bootstrap_mae_gain(y,group,p_control,p_extended,seed):
 ids=np.unique(group); idx={g:np.flatnonzero(group==g) for g in ids};rg=np.random.default_rng(seed);v=[]
 for _ in range(N_BOOT):
  draw=rg.choice(ids,size=len(ids),replace=True);ix=np.concatenate([idx[g] for g in draw])
  v.append(mean_absolute_error(y[ix],p_control[ix])-mean_absolute_error(y[ix],p_extended[ix]))
 return {'mae_gain_control_minus_extended':float(mean_absolute_error(y,p_control)-mean_absolute_error(y,p_extended)),
         'ci95_formula_group_bootstrap':[float(x) for x in np.quantile(v,[.025,.975])],'draws':len(v)}
