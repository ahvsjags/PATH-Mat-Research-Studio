#!/usr/bin/env python3
"""Exploratory chronological validation and stage-assignment ablation in ZeoSyn.

The temporal cutoff is fixed at publication year 2015/2016. This is a
source-adapted test, not PATH-Mat frozen cross-source transfer. Outcomes were
already available when this secondary analysis was devised, so it is not a
preregistered confirmatory test.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'inputs'/'ZEOSYN.xlsx'; OUT=ROOT/'results'; SEED=20261016; BOOT=2000
raw=pd.read_excel(INPUT,sheet_name='Sheet1')
df=raw.loc[raw.doi.notna() & raw.normed.astype(str).str.lower().eq('yes')].copy()
prod=df.product1.astype(str).str.strip().str.lower()
pos=df.Code1.notna() & df.Code1.astype(str).str.strip().ne('')
neg=prod.isin({'amorphous','dense'})
eligible=(pos|neg) & df.year.notna()
df=df.loc[eligible].copy().reset_index(drop=True)
df['y']=pos[eligible].astype(int).to_numpy()
df['doi']=df.doi.astype(str); df['year']=pd.to_numeric(df.year)
cutoff=2015
train=df.year<=cutoff; test=df.year>cutoff
assert set(df.loc[train,'doi']).isdisjoint(set(df.loc[test,'doi']))

comp_num=[c for c in raw.columns[1:52] if c!='solvent' and pd.api.types.is_numeric_dtype(raw[c])]
comp_cat=[c for c in ['solvent','osda1 smiles','osda2 smiles','osda3 smiles'] if c in df]
stage=['aging_time','aging_temp','cryst_time','cryst_temp']
other=[c for c in ['rotation','react_vol','pH','seed'] if c in df and pd.api.types.is_numeric_dtype(df[c])]
report_fields=stage+other
df['_condition_count']=df[report_fields].notna().sum(axis=1)
# Preserve the same multiset of reported stage values, including missingness,
# while erasing which value was reported for aging versus crystallization.
for a,b,prefix in [('aging_time','cryst_time','time_bag'),('aging_temp','cryst_temp','temp_bag')]:
    arr=np.sort(df[[a,b]].to_numpy(dtype=float),axis=1)
    df[prefix+'_low']=arr[:,0]
    df[prefix+'_high']=arr[:,1]
nonstage=other
feature_sets={
 'chemistry':['__CHEM__'],
 'chemistry_count':['__CHEM__','_condition_count'],
 'count_plus_nonstage_plus_stage':['__CHEM__','_condition_count']+nonstage+stage,
 'count_plus_nonstage_plus_unordered_bag':['__CHEM__','_condition_count']+nonstage+['time_bag_low','time_bag_high','temp_bag_low','temp_bag_high'],
}
def cols_for(key):
    cols=feature_sets[key]
    return comp_num+comp_cat if key=='chemistry' else comp_num+comp_cat+cols[1:]
def model_for(cols):
    num=[c for c in cols if c in df and pd.api.types.is_numeric_dtype(df[c])]
    cat=[c for c in cols if c in df and c not in num]
    prep=ColumnTransformer([
      ('num',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler(with_mean=False))]),num),
      ('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat),
    ])
    return Pipeline([('prep',prep),('clf',LogisticRegression(C=1.0,max_iter=2000,solver='liblinear',random_state=SEED))])

y=df.y.to_numpy(); tr=np.flatnonzero(train.to_numpy()); te=np.flatnonzero(test.to_numpy())
preds={}; feature_audit={}
for name in feature_sets:
    cols=cols_for(name); feature_audit[name]=cols
    model=model_for(cols); model.fit(df.iloc[tr][cols],y[tr]); preds[name]=model.predict_proba(df.iloc[te][cols])[:,1]

yt=y[te]; gt=df.iloc[te].doi.to_numpy(); unique=np.unique(gt)
def metrics(ix):
    return {name:{'average_precision':float(average_precision_score(yt[ix],pred[ix])),'auroc':float(roc_auc_score(yt[ix],pred[ix])),'brier':float(brier_score_loss(yt[ix],pred[ix]))} for name,pred in preds.items()}
point=metrics(np.arange(len(te)))
by_group={g:np.flatnonzero(gt==g) for g in unique}; rng=np.random.default_rng(SEED)
keys=[('count_plus_nonstage_plus_stage','count_plus_nonstage_plus_unordered_bag'),('count_plus_nonstage_plus_stage','chemistry_count'),('count_plus_nonstage_plus_unordered_bag','chemistry_count')]
boot={f'{a}-{b}':{'ap':[],'auroc':[],'brier':[]} for a,b in keys}
for _ in range(BOOT):
    draw=rng.choice(unique,size=len(unique),replace=True); ix=np.concatenate([by_group[g] for g in draw])
    if len(np.unique(yt[ix]))<2: continue
    for a,b in keys:
        boot[f'{a}-{b}']['ap'].append(average_precision_score(yt[ix],preds[a][ix])-average_precision_score(yt[ix],preds[b][ix]))
        boot[f'{a}-{b}']['auroc'].append(roc_auc_score(yt[ix],preds[a][ix])-roc_auc_score(yt[ix],preds[b][ix]))
        boot[f'{a}-{b}']['brier'].append(brier_score_loss(yt[ix],preds[a][ix])-brier_score_loss(yt[ix],preds[b][ix]))
boot_out={k:{m:{'mean':float(np.mean(v)),'ci95':[float(np.quantile(v,.025)),float(np.quantile(v,.975))]} for m,v in mv.items()} for k,mv in boot.items()}
payload={
 'schema':'v238_zeosyn_temporal_stage_assignment',
 'source':'ZeoSyn public literature corpus; source-adapted temporal evaluation',
 'input_sha256':hashlib.sha256(INPUT.read_bytes()).hexdigest(),
 'endpoint':'standardized IZA framework present vs product explicitly amorphous/dense; ambiguous outcomes excluded',
 'split':{'train_year_max':cutoff,'test_year_min_exclusive':cutoff,'n_train':int(train.sum()),'n_test':int(test.sum()),'n_train_doi':int(df.loc[train,'doi'].nunique()),'n_test_doi':int(df.loc[test,'doi'].nunique()),'doi_overlap':False,'test_prevalence':float(yt.mean())},
 'features':feature_audit,
 'model':'Fixed L2 logistic regression C=1.0; train only through 2015, evaluate 2016 onward; no tuning or test-derived preprocessing; same test set for all models.',
 'metrics':point,
 'paired_test_DOI_bootstrap':{'draws':BOOT,'seed':SEED,'unit':'test DOI','contrasts':boot_out},
 'interpretation_boundary':'Exploratory temporal within-source validation. It does not test a frozen model from a different source. The stage-vs-bag contrast isolates assignment of values to aging/crystallization fields; it is not a randomized reversal of operations or a causal effect.',
}
(OUT/'V238_ZEOSYN_TEMPORAL_STAGE_AUDIT.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
pd.DataFrame({'doi':gt,'year':df.iloc[te].year.to_numpy(),'y':yt,**preds}).to_csv(OUT/'V238_ZEOSYN_TEMPORAL_STAGE_PREDICTIONS.csv',index=False)
print(json.dumps({'split':payload['split'],'metrics':point,'contrasts':boot_out},indent=2,ensure_ascii=False))
