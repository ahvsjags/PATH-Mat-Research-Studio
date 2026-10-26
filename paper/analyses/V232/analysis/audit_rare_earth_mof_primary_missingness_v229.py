from pathlib import Path
import sys, json
import pandas as pd
import numpy as np
A=Path(r'D:\Downloads\材料agent\PATH-Mat_Evidence_Strengthening_V229\analysis')
sys.path.insert(0,str(A))
import evaluate_rare_earth_mof_double_holdout_v229 as core
ROOT=A.parent; src=pd.read_csv(ROOT/'inputs'/'rare_earth_mof_campaign.csv')
single=src[~src.Metal.astype(str).str.contains(r'\+',regex=True)].copy()
single['Primary Phase']=single['Primary Phase'].fillna('').astype(str).str.strip()
p=single['Primary Phase'].str.lower()
no=p.isin(core.NO_PRODUCT); named=~p.isin({'','unknown phase'}) & ~no; unknown=~(no|named)
# Selection audit
single['label_status']=np.select([named,no,unknown],['named_crystalline_product','explicit_no_product','unknown_or_missing'],default='unknown')
counts={'single_metal_n':int(len(single)),'named_product':int(named.sum()),'explicit_no_product':int(no.sum()),'unknown_or_missing':int(unknown.sum()),'labelled_primary_cohort_n':int((named|no).sum())}
breakdowns={}
for col in ['Metal','Linker','Temp','AA Conc']:
 q=single.groupby(col,dropna=False)['label_status'].agg(n='size',unknown=lambda x:int((x=='unknown_or_missing').sum())).reset_index()
 q['unknown_fraction']=q.unknown/q.n
 breakdowns[col]=q.to_dict(orient='records')
# extreme endpoint sensitivity using exact same features/splits/bootstrap as primary
base=single.copy(); base['Metal']=base.Metal.astype(str).str.strip(); base['Linker']=base.Linker.astype(str).str.strip()
base['radius_8coord']=base.Metal.map(core.RADIUS); base['atomic_number']=base.Metal.map(core.Z); base['linker_carboxylates']=base.Linker.map(core.CARBOXYLATES); base['run_id']=base.SampleID.astype(str).str.extract(r'MG2-(\d+)-',expand=False)
base=base.reset_index(drop=True)
if base[['radius_8coord','atomic_number','linker_carboxylates','run_id']].isna().any().any(): raise ValueError('descriptor missing')
res={}
for assumption in ['unknown_as_negative','unknown_as_positive']:
 y=(named if assumption=='unknown_as_negative' else (named|unknown)).astype(int).to_numpy()
 d=base.copy(); d['y']=y
 for scheme in ['metal_and_run_purged','leave_one_run_out']:
  pred,folds=core.fit_oof(d,scheme); cluster='Metal' if scheme=='metal_and_run_purged' else 'run_id'
  ci=core.paired_cluster_bootstrap(d,pred,cluster,draws=5000,seed=20261016)
  res.setdefault(assumption,{})[scheme]={'prevalence':float(y.mean()),'metrics':core.metrics(y,pred),'full_vs_composition_ci':ci['full']}
result={'source_csv_sha256':core.sha256(ROOT/'inputs'/'rare_earth_mof_campaign.csv'),'cohort_and_status_counts':counts,'unknown_by_factor':breakdowns,'extreme_label_assignment_sensitivity':res,
'interpretation':'Unknown outcomes are reassigned in two extreme directions only as robustness bounds; they are not imputed truth. These sensitivities retain all 1,344 single-metal rows and use the same metal/run-purged folds, fixed model, features and cluster bootstrap as the primary analysis.'}
out=ROOT/'results'/'V229_RE_MOF_PRIMARY_LABEL_MISSINGNESS_AUDIT.json'
out.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'counts':counts,'unknown_by_factor':breakdowns,'extremes':res},indent=2,ensure_ascii=False))

