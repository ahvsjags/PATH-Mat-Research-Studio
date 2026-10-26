#!/usr/bin/env python3
"""Inner-selected confidence guard for a cross-fitted condition policy.

The policy either uses the model's top-ranked condition or a single training-
selected recipe. Both the predictive model and the score-margin guard are
selected in inner metal/run-purged folds; outer test metals and their runs are
not used for fitting or selection. The analysis is retrospective and not a
causal or prospective intervention.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
INP=ROOT/'inputs'/'rare_earth_mof_campaign.csv'
OUT=ROOT/'results'/'V230_RE_MOF_GUARDED_POLICY.json'
CSV=ROOT/'results'/'V230_RE_MOF_GUARDED_POLICY_BY_FORMULATION.csv'
SEED=20261016; BOOT=20000
NO={'no solid product','no crystalline product'}
RADIUS={'Sc':.870,'Y':1.019,'La':1.160,'Ce':1.143,'Pr':1.126,'Nd':1.109,'Sm':1.079,'Eu':1.066,'Gd':1.053,'Tb':1.040,'Dy':1.027,'Ho':1.015,'Er':1.004,'Tm':.994,'Yb':.985,'Lu':.977}
Z={'Sc':21,'Y':39,'La':57,'Ce':58,'Pr':59,'Nd':60,'Sm':62,'Eu':63,'Gd':64,'Tb':65,'Dy':66,'Ho':67,'Er':68,'Tm':69,'Yb':70,'Lu':71}
Q={'BDC':2,'Phthalic':2,'Isophthalic':2,'BTC':3,'Trimellitic':3,'Pyromellitic':4,'Mellitic':6}
CS=[.1,1.,10.]
VARS=['composition','poly','categorical']
MARGINS=[0.,.005,.01,.02,.05,.1,.2,.5]

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def load():
 d=pd.read_csv(INP); d=d[~d.Metal.astype(str).str.contains(r'\+',regex=True)].copy()
 d['Primary Phase']=d['Primary Phase'].fillna('').astype(str).str.strip()
 p=d['Primary Phase'].str.lower(); d=d[p.isin(NO)|(~p.isin({'','unknown phase'}))].copy()
 p=d['Primary Phase'].str.lower(); d['y']=(~p.isin(NO)).astype(int)
 d['run_id']=d.SampleID.astype(str).str.extract(r'MG2-(\d+)-',expand=False)
 d['radius']=d.Metal.map(RADIUS); d['z']=d.Metal.map(Z); d['topology']=d.Linker.map(Q)
 if d[['run_id','radius','z','topology']].isna().any().any(): raise ValueError('unmapped rows')
 d['Temp']=d.Temp.astype(float); d['AA Conc']=d['AA Conc'].astype(float)
 return d.reset_index(drop=True)

def X(d,v):
 x=pd.DataFrame({'radius':d.radius,'z':d.z},index=d.index)
 for l in sorted(Q): x['L='+l]=(d.Linker==l).astype(float)
 t=(d.Temp-60)/20; a=d['AA Conc']/5; r=(d.radius-1.03)/.1; q=(d.topology-3)/2
 if v=='poly':
  for n,z in [('t',t),('t2',t*t),('a',a),('a2',a*a),('rt',r*t),('ra',r*a),('qt',q*t),('qa',q*a)]: x[n]=z
 elif v=='categorical':
  tc={}; ac={}
  for z0 in sorted(d.Temp.unique())[1:]:
   tc[z0]=(d.Temp==z0).astype(float); x[f't{z0}']=tc[z0]; x[f'rt{z0}']=r*tc[z0]; x[f'qt{z0}']=q*tc[z0]
  for z0 in sorted(d['AA Conc'].unique())[1:]:
   ac[z0]=(d['AA Conc']==z0).astype(float); x[f'a{z0}']=ac[z0]; x[f'ra{z0}']=r*ac[z0]; x[f'qa{z0}']=q*ac[z0]
  for tv,tb in tc.items():
   for av,ab in ac.items(): x[f't{tv}a{av}']=tb*ab
 elif v!='composition': raise ValueError(v)
 return x.astype(float)

def predict(tr,te,v,c):
 m=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=5000,solver='lbfgs'))
 m.fit(X(tr,v),tr.y.astype(int)); return m.predict_proba(X(te,v))[:,1]

def global_recipe(tr):
 s=tr.groupby(['Temp','AA Conc'],sort=True).y.mean().sort_values(ascending=False,kind='stable')
 return tuple(map(float,s.index[0]))

def top(g,scores):
 mx=float(np.max(scores)); ix=np.isclose(scores,mx,rtol=0,atol=1e-12)
 return float(g.y.to_numpy()[ix].mean()),mx

def ci(x,seed):
 x=np.asarray(x,float); rng=np.random.default_rng(seed)
 b=x[rng.integers(0,len(x),size=(BOOT,len(x)))].mean(axis=1)
 return [float(np.quantile(b,.025)),float(np.quantile(b,.975))]

def main():
 d=load(); sizes=d.groupby(['Metal','Linker']).size(); complete=set(sizes[sizes==12].index)
 d['complete']=[(m,l) in complete for m,l in zip(d.Metal,d.Linker)]
 specs=[(v,c) for v in VARS for c in CS]
 fold_sel=[]; out=[]
 for outer in sorted(d.Metal.unique()):
  test_all=d[d.Metal==outer]; test_runs=set(test_all.run_id)
  tr_outer=d[(d.Metal!=outer)&(~d.run_id.isin(test_runs))].copy()
  te=d[(d.Metal==outer)&d.complete].copy()
  # inner held-out metal predictions used only for selecting model + margin
  inner_rows={s:[] for s in specs}
  for im in sorted(tr_outer.Metal.unique()):
   va_all=tr_outer[tr_outer.Metal==im]; va=va_all[va_all.complete].copy()
   if not len(va): continue
   iruns=set(va_all.run_id)
   tri=tr_outer[(tr_outer.Metal!=im)&(~tr_outer.run_id.isin(iruns))].copy()
   if tri.y.nunique()!=2: continue
   gr=global_recipe(tri)
   for s in specs:
    p=predict(tri,va,s[0],s[1]); va2=va.copy(); va2['p']=p
    for (m,l),g in va2.groupby(['Metal','Linker'],sort=True):
     top_y,top_p=top(g,g.p.to_numpy())
     base=g[(g.Temp==gr[0])&(g['AA Conc']==gr[1])]
     if len(base)!=1: raise RuntimeError('global recipe absent from complete grid')
     inner_rows[s].append({'metal':m,'linker':l,'top_y':top_y,'top_p':top_p,'base_y':float(base.y.iloc[0]),'base_p':float(base.p.iloc[0])})
  # Choose model and threshold exclusively by inner folds, with equal weight by formulation.
  candidates=[]
  for s,rs in inner_rows.items():
   if not rs: continue
   g=pd.DataFrame(rs)
   for margin in MARGINS:
    use=g.top_y.where((g.top_p-g.base_p)>=margin,g.base_y)
    candidates.append((float(use.mean()),s,margin))
  if not candidates: raise RuntimeError('no valid inner policy candidates')
  candidates.sort(key=lambda z:(z[0],-specs.index(z[1]),-MARGINS.index(z[2])),reverse=True)
  best_utility,best_spec,best_margin=candidates[0]
  gr=global_recipe(tr_outer)
  for s in specs:
   te['p_'+s[0]+'_C'+str(s[1])]=predict(tr_outer,te,s[0],s[1])
  te['p_selected']=predict(tr_outer,te,best_spec[0],best_spec[1])
  fold_sel.append({'outer_metal':outer,'selected_variant':best_spec[0],'selected_C':best_spec[1],
                   'selected_score_margin':best_margin,'inner_policy_value':best_utility,
                   'training_global_recipe':f'{gr[0]:g}C/{gr[1]:g}M'})
  for (m,l),g in te.groupby(['Metal','Linker'],sort=True):
   ty,tp=top(g,g.p_selected.to_numpy())
   base=g[(g.Temp==gr[0])&(g['AA Conc']==gr[1])]
   if len(base)!=1: raise RuntimeError('outer global recipe absent')
   gap=float(tp-base.p_selected.iloc[0])
   use_model=gap>=best_margin
   out.append({'metal':m,'linker':l,'n_conditions':len(g),'random_outcome_rate':float(g.y.mean()),
               'recommended_temp_C':float(g.loc[g.p_selected.idxmax(),'Temp']),
               'recommended_acid_M':float(g.loc[g.p_selected.idxmax(),'AA Conc']),
               'model_top_expected_outcome':ty,'model_top_score_margin_vs_global':gap,
               'training_global_recipe':f'{gr[0]:g}C/{gr[1]:g}M',
               'global_recipe_outcome':float(base.y.iloc[0]),
               'guarded_policy_outcome':ty if use_model else float(base.y.iloc[0]),
               'used_model_recommendation':bool(use_model),'selected_variant':best_spec[0], 'selected_C':best_spec[1], 'selected_margin':best_margin})
 od=pd.DataFrame(out)
 def summarize(col,base):
  dd=od[col]-od[base]; md=od.assign(d=dd).groupby('metal').d.mean()
  return {'mean_rate':float(od[col].mean()),'mean_difference_vs_'+base+'_percentage_points':float(dd.mean()*100),
          'metal_cluster_bootstrap_95ci_difference_percentage_points':[100*x for x in ci(md,SEED+3)],
          'metals_positive':int((md>0).sum()),'metals_tied':int((md==0).sum()),'metals_negative':int((md<0).sum())}
 result={'analysis':'Nested model-and-margin selection for condition recommendation under metal/run-block holdout',
  'status':'post hoc, cross-fitted, source-adapted decision-value estimate; not randomized, prospective, frozen PATH-Mat or causal',
  'source':'Gaidimas et al., Chemical Science 2026, DOI 10.1039/D5SC09992G; Zenodo 10.5281/zenodo.17902549',
  'input_sha256':sha(INP),'cohort':{'complete_grids':len(od),'conditions_per_grid':12,'metals':int(od.metal.nunique()),'rows':int(12*len(od)),'prevalence':float(od.random_outcome_rate.mean())},
  'policy':'Inner-held-out-metal folds select the fixed logistic feature family, C and score-margin guard; the outer policy uses the model top-ranked condition only when its score exceeds the train-selected single global recipe by the selected margin, otherwise uses the global recipe.',
  'inner_selection':{'candidate_models':[{'variant':v,'C':c} for v,c in specs],'margin_candidates':MARGINS,'selected_per_outer_fold':fold_sel},
  'policies':{'nested_guarded_vs_random':summarize('guarded_policy_outcome','random_outcome_rate'),
              'nested_guarded_vs_global':summarize('guarded_policy_outcome','global_recipe_outcome'),
              'unguarded_nested_model_vs_global':summarize('model_top_expected_outcome','global_recipe_outcome')},
  'guarded_policy_fraction_using_model':float(od.used_model_recommendation.mean()),
  'selected_recipe_counts':od.training_global_recipe.value_counts().to_dict(),
  'limitations':['The source endpoint is a named crystalline primary phase, not target-specific purity/yield.','Complete-grid evaluation excludes 21 of 112 formulations with unknown primary labels.','The public grid was already run; outcomes were not prospectively acquired after predictions.','The global recipe is selected from training outcomes and may be as effective as or better than the individualized policy.','Treatment allocation was not randomized; no causal effect is identified.'],
  'output_csv':CSV.name}
 od.to_csv(CSV,index=False); result['output_csv_sha256']=sha(CSV); result['script_sha256']=sha(Path(__file__))
 OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
 print(json.dumps({'cohort':result['cohort'],'policies':result['policies'],'fraction_model':result['guarded_policy_fraction_using_model'],'outer_choices':pd.Series([x['selected_variant']+str(x['selected_C'])+'@'+str(x['selected_score_margin']) for x in fold_sel]).value_counts().to_dict()},indent=2,ensure_ascii=False))

if __name__=='__main__': main()
