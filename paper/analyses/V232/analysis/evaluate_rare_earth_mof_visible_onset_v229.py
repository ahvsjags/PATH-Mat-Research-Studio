from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
ROOT=Path(__file__).resolve().parents[1]
INP=ROOT/'inputs'/'rare_earth_mof_campaign.csv'; OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
RADIUS={'Sc':.870,'Y':1.019,'La':1.160,'Ce':1.143,'Pr':1.126,'Nd':1.109,'Sm':1.079,'Eu':1.066,'Gd':1.053,'Tb':1.040,'Dy':1.027,'Ho':1.015,'Er':1.004,'Tm':.994,'Yb':.985,'Lu':.977}
Z={'Sc':21,'Y':39,'La':57,'Ce':58,'Pr':59,'Nd':60,'Sm':62,'Eu':63,'Gd':64,'Tb':65,'Dy':66,'Ho':67,'Er':68,'Tm':69,'Yb':70,'Lu':71}
CARBOXYLATES={'BDC':2,'Phthalic':2,'Isophthalic':2,'BTC':3,'Trimellitic':3,'Pyromellitic':4,'Mellitic':6}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def load():
 d=pd.read_csv(INP)
 d=d[~d.Metal.astype(str).str.contains(r'\+',regex=True)].copy()
 d['Metal']=d.Metal.astype(str).str.strip(); d['Linker']=d.Linker.astype(str).str.strip()
 d['event_time_h']=pd.to_numeric(d['first_hetero_or_solid_h'],errors='coerce')
 d['y_visible_event_18h']=d.event_time_h.notna().astype(int)
 d['radius_8coord']=d.Metal.map(RADIUS); d['atomic_number']=d.Metal.map(Z); d['linker_carboxylates']=d.Linker.map(CARBOXYLATES)
 d['run_id']=d.SampleID.astype(str).str.extract(r'MG2-(\d+)-',expand=False)
 if d[['radius_8coord','atomic_number','linker_carboxylates','run_id']].isna().any().any(): raise ValueError('Unmapped descriptor/run id')
 return d.reset_index(drop=True)
def features(d, variant):
 x=pd.DataFrame({'radius_8coord':d.radius_8coord,'atomic_number':d.atomic_number},index=d.index)
 for linker in sorted(d.Linker.unique()): x['linker='+linker]=(d.Linker==linker).astype(float)
 t=(d.Temp.astype(float)-60)/20; a=d['AA Conc'].astype(float)/5; r=(d.radius_8coord-1.03)/.1; q=(d.linker_carboxylates-3)/2
 if variant in {'temperature','full'}: x['temperature']=t; x['temperature_squared']=t*t
 if variant in {'acid','full'}: x['acetic_acid']=a; x['acetic_acid_squared']=a*a
 if variant=='full':
  x['radius_x_temperature']=r*t; x['radius_x_acid']=r*a; x['linker_topology_x_temperature']=q*t; x['linker_topology_x_acid']=q*a
 return x.astype(float)
def fit_oof(d,scheme):
 groups=sorted(d.Metal.unique()) if scheme=='metal_and_run_purged' else sorted(d.run_id.unique())
 variants=['composition','temperature','acid','full']; pred={v:np.full(len(d),np.nan) for v in variants}; folds=[]
 for group in groups:
  if scheme=='metal_and_run_purged':
   test=(d.Metal==group).to_numpy(); test_runs=set(d.loc[test,'run_id']); train=(~test)&(~d.run_id.isin(test_runs).to_numpy())
   fold={'held_out_metal':str(group),'held_out_run_ids':sorted(test_runs)}
  else:
   test=(d.run_id==group).to_numpy(); train=~test; fold={'held_out_run_id':str(group)}
  if np.any(train&test) or not np.any(train) or not np.any(test): raise ValueError('bad fold')
  if scheme=='metal_and_run_purged' and (set(d.loc[train,'run_id'])&set(d.loc[test,'run_id']) or (d.loc[train,'Metal']==group).any()): raise ValueError('overlap')
  for v in variants:
   m=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=5000,solver='lbfgs'))
   m.fit(features(d.loc[train],v),d.y_visible_event_18h.to_numpy()[train])
   pred[v][test]=m.predict_proba(features(d.loc[test],v))[:,1]
  fold.update(n_train=int(train.sum()),n_test=int(test.sum())); folds.append(fold)
 if any(np.isnan(s).any() for s in pred.values()): raise ValueError('incomplete oof')
 return pred,folds
def metr(y,p): return {k:{'average_precision':float(average_precision_score(y,s)),'auroc':float(roc_auc_score(y,s)),'brier':float(brier_score_loss(y,s))} for k,s in p.items()}
def boot(d,p,cluster,draws=5000,seed=20261016):
 rng=np.random.default_rng(seed); groups=sorted(d[cluster].unique()); ids={g:np.flatnonzero(d[cluster].to_numpy()==g) for g in groups}; y0=d.y_visible_event_18h.to_numpy(); out={k:{m:[] for m in ['delta_ap','delta_auc','brier_improvement']} for k in ['temperature','acid','full']}
 for _ in range(draws):
  idx=np.concatenate([ids[groups[j]] for j in rng.integers(0,len(groups),size=len(groups))]); y=y0[idx]
  if np.unique(y).size<2: continue
  ap0=average_precision_score(y,p['composition'][idx]); auc0=roc_auc_score(y,p['composition'][idx]); br0=brier_score_loss(y,p['composition'][idx])
  for k in out:
   out[k]['delta_ap'].append(float(average_precision_score(y,p[k][idx])-ap0)); out[k]['delta_auc'].append(float(roc_auc_score(y,p[k][idx])-auc0)); out[k]['brier_improvement'].append(float(br0-brier_score_loss(y,p[k][idx])))
 return {k:{m:[float(np.quantile(v,.025)),float(np.quantile(v,.975))] for m,v in vals.items()} for k,vals in out.items()}
def concordance(d,p,draws=20000):
 z=d.copy(); z['p_comp']=p['composition']; z['p_full']=p['full']; cells=[]
 for (metal,linker),g in z.groupby(['Metal','Linker']):
  y=g.y_visible_event_18h.to_numpy(int); pos=np.flatnonzero(y==1); neg=np.flatnonzero(y==0)
  if not len(pos) or not len(neg): continue
  row={'metal':metal,'linker':linker,'n_pairs':int(len(pos)*len(neg))}
  for name,col in [('comp','p_comp'),('full','p_full')]:
   s=g[col].to_numpy(float); dd=s[pos[:,None]]-s[neg][None,:]; row[name]=float(((dd>1e-10).sum()+.5*(np.abs(dd)<=1e-10).sum())/dd.size)
  cells.append(row)
 c=pd.DataFrame(cells); metals=sorted(c.metal.unique()); by=np.array([c.loc[c.metal==m,'full'].mean()-c.loc[c.metal==m,'comp'].mean() for m in metals]); rng=np.random.default_rng(20261017); b=by[rng.integers(0,len(by),size=(draws,len(by)))].mean(axis=1)
 return {'mixed_metal_linker_cells':len(c),'positive_negative_pairs':int(c.n_pairs.sum()),'metal_clusters':len(metals),'composition_macro_concordance':float(c.comp.mean()),'conditions_macro_concordance':float(c.full.mean()),'difference_ci_metal_bootstrap':[float(np.quantile(b,.025)),float(np.quantile(b,.975))]}
def main():
 d=load(); y=d.y_visible_event_18h.to_numpy(); out={'source':'Gaidimas et al. Chemical Science 2026, DOI 10.1039/D5SC09992G; Zenodo DOI 10.5281/zenodo.17902549','analysis_status':'secondary, post hoc endpoint analysis; model form and split definitions reused unchanged from the crystalline-product analysis; no tuning','endpoint':'first_hetero_or_solid_h non-missing = visible heterogeneous liquid or solid detected during the 18-h imaging window; missing = no such event recorded during observation; this is not a PXRD crystallinity label','missingness_interpretation':'The endpoint treats rows with no recorded event time as event-free over the source observation window; this assumes all source rows were observed through the full 18-h window.','cohort':{'n':len(d),'events':int(y.sum()),'no_event':int((1-y).sum()),'prevalence':float(y.mean()),'metals':int(d.Metal.nunique()),'linkers':int(d.Linker.nunique()),'run_ids':int(d.run_id.nunique())},'model':'fixed L2 logistic regression C=1; same feature definitions as strict PXRD analysis; no tuning','splits':{},'input_sha256':sha(INP)}
 for scheme in ['metal_and_run_purged','leave_one_run_out']:
  p,folds=fit_oof(d,scheme); cluster='Metal' if scheme=='metal_and_run_purged' else 'run_id'; rec={'metrics':metr(y,p),'cluster_bootstrap_95ci_vs_composition':boot(d,p,cluster),'folds':folds}
  if scheme=='metal_and_run_purged':
   rec['within_exact_formulation_concordance']=concordance(d,p)
   outcsv=OUT/'V229_RE_MOF_VISIBLE_ONSET_PREDICTIONS.csv'; f=d[['Sample#','SampleID','Metal','Linker','AA Conc','Temp','first_hetero_or_solid_h','run_id','y_visible_event_18h']].copy()
   for k,v in p.items():f['oof_'+k]=v
   f.to_csv(outcsv,index=False); rec['prediction_csv_sha256']=sha(outcsv)
  out['splits'][scheme]=rec
 path=OUT/'V229_RE_MOF_VISIBLE_ONSET_AUDIT.json'; out['script_sha256']=sha(Path(__file__)); path.write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
 print(json.dumps({'cohort':out['cohort'],'splits':{k:{'metrics':v['metrics'],'ci':v['cluster_bootstrap_95ci_vs_composition'],'within':v.get('within_exact_formulation_concordance')} for k,v in out['splits'].items()}},indent=2))
if __name__=='__main__': main()
