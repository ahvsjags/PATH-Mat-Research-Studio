from __future__ import annotations
import json, hashlib, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

ROOT=Path(__file__).resolve().parents[1]
INP=ROOT/'inputs'/'ZEOSYN.xlsx'
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
SEED=20261016
N_BOOT=1000

raw=pd.read_excel(INP, sheet_name='Sheet1')
# The published route table is the 23,961 DOI-linked rows; rows beyond it are spreadsheet padding/descriptor material.
df=raw.loc[raw['doi'].notna() & raw['normed'].astype(str).str.lower().eq('yes')].copy()
# Strict primary endpoint: named IZA framework vs explicit amorphous/dense product.
# Unknown or unstandardized outcomes are excluded rather than imputed.
negative_labels={'amorphous','dense'}
prod=df['product1'].astype(str).str.strip().str.lower()
code_present=df['Code1'].notna() & df['Code1'].astype(str).str.strip().ne('')
y=np.where(code_present,1,np.where(prod.isin(negative_labels),0,-1))
df=df.loc[y>=0].copy(); df['y']=y[y>=0].astype(int)

# Composition block = reported inorganic/gel recipe fractions, water, OSDA amounts and acid/modulator amounts,
# plus solvent and identity of the reported OSDAs. Process block = time/temperature/aging/rotation/volume/pH/seed.
composition_num=[c for c in raw.columns[1:52] if c not in {'solvent'} and pd.api.types.is_numeric_dtype(raw[c])]
composition_cat=[c for c in ['solvent','osda1 smiles','osda2 smiles','osda3 smiles'] if c in df.columns]
process_num=[c for c in ['aging_time','aging_temp','cryst_time','cryst_temp','rotation','react_vol','pH','seed'] if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
# A single preregistered-style granularity control counts which process-condition fields were reported.
df['_condition_reporting_count']=df[process_num].notna().sum(axis=1)

# Fixed DOI-grouped out-of-fold protocol; no tuning on the held-out predictions.
groups=df['doi'].astype(str).to_numpy(); y=df['y'].to_numpy()
cv=GroupKFold(n_splits=5)
models={'C_chemistry':composition_num+composition_cat,
        'C_plus_reporting':composition_num+composition_cat+['_condition_reporting_count'],
        'C_plus_reporting_plus_process':composition_num+composition_cat+['_condition_reporting_count']+process_num}

def make_model(cols):
    num=[c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    cat=[c for c in cols if c in df.columns and c not in num]
    pre=ColumnTransformer([
        ('num',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler(with_mean=False))]),num),
        ('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat)
    ],remainder='drop')
    return Pipeline([('pre',pre),('clf',LogisticRegression(C=1.0,max_iter=2000,solver='liblinear',class_weight=None,random_state=SEED))])

preds={name:np.full(len(df),np.nan) for name in models}
fold_rows=[]
for fold,(tr,te) in enumerate(cv.split(df,y,groups),1):
    assert set(groups[tr]).isdisjoint(set(groups[te]))
    for name,cols in models.items():
        model=make_model(cols)
        model.fit(df.iloc[tr][cols],y[tr])
        p=model.predict_proba(df.iloc[te][cols])[:,1]
        preds[name][te]=p
        fold_rows.append({'fold':fold,'model':name,'n_train':len(tr),'n_test':len(te),'n_test_doi':len(set(groups[te])),'ap':average_precision_score(y[te],p),'auroc':roc_auc_score(y[te],p),'brier':brier_score_loss(y[te],p)})

metrics={}
for name,p in preds.items():
    metrics[name]={'average_precision':float(average_precision_score(y,p)),'auroc':float(roc_auc_score(y,p)),'brier':float(brier_score_loss(y,p)),'unique_oof_scores':int(len(np.unique(np.round(p,12))))}

# Paired DOI-cluster bootstrap on the identical OOF records for model differences.
unique_groups=np.unique(groups)
idx_by_group={g:np.flatnonzero(groups==g) for g in unique_groups}
rng=np.random.default_rng(SEED)
diffs={f'{a}-{b}':[] for a,b in [('C_plus_reporting','C_chemistry'),('C_plus_reporting_plus_process','C_plus_reporting'),('C_plus_reporting_plus_process','C_chemistry')]}
for _ in range(N_BOOT):
    draw=rng.choice(unique_groups,size=len(unique_groups),replace=True)
    ix=np.concatenate([idx_by_group[g] for g in draw])
    if len(np.unique(y[ix]))<2: continue
    for key in diffs:
        a,b=key.split('-')
        diffs[key].append(average_precision_score(y[ix],preds[a][ix])-average_precision_score(y[ix],preds[b][ix]))
boot={k:{'mean':float(np.mean(v)),'median':float(np.median(v)),'ci95_percentile':[float(np.quantile(v,.025)),float(np.quantile(v,.975))],'positive_fraction':float(np.mean(np.array(v)>0))} for k,v in diffs.items()}

summary={
 'schema':'v238_zeosyn_grouped_doi_process_ablation',
 'source':'ZeoSyn public synthesis dataset, official GitHub repository; dataset table filtered to DOI-present, normed=yes rows',
 'input_path':str(INP.relative_to(ROOT)),
 'input_sha256':hashlib.sha256(INP.read_bytes()).hexdigest(),
 'n_spreadsheet_rows':int(len(raw)),
 'n_doi_normed_rows':int(len(raw.loc[raw['doi'].notna() & raw['normed'].astype(str).str.lower().eq('yes')])),
 'endpoint':'positive = standardized Code1 IZA framework present; negative = product1 explicitly amorphous or dense; other/missing outcomes excluded',
 'n_analysis':int(len(df)),'n_positive':int(y.sum()),'n_negative':int((1-y).sum()),'prevalence':float(y.mean()),'n_doi':int(len(unique_groups)),
 'features':{'composition_numeric':composition_num,'composition_categorical':composition_cat,'process_numeric':process_num,'reporting_control':'count of nonmissing process_numeric fields'},
 'model':'L2 logistic regression C=1.0; fixed feature blocks; no tuning; 5-fold GroupKFold by DOI; same folds for all models',
 'split_audit':{'all_doi_disjoint':True,'folds':fold_rows},
 'metrics_oof':metrics,
 'paired_doi_cluster_bootstrap':{'draws':N_BOOT,'seed':SEED,'resampling_unit':'DOI; paired identical out-of-fold rows', 'average_precision_differences':boot},
 'interpretation_boundary':'Source-adapted, within-ZeoSyn framework-formation task. This is not a PATH-Mat frozen zero-shot evaluation, does not test impurity prediction, and ZeoSyn does not encode verified reagent-addition chronology. It can support cross-domain usefulness of route conditions under DOI-held-out evaluation only if the process increment survives the reporting-count control.',
}
(OUT/'V238_ZEOSYN_DOI_GROUPED_PROCESS_AUDIT.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
pd.DataFrame(fold_rows).to_csv(OUT/'V238_ZEOSYN_DOI_GROUPED_FOLD_METRICS.csv',index=False)
pd.DataFrame({'doi':groups,'y':y,**{k:v for k,v in preds.items()}}).to_csv(OUT/'V238_ZEOSYN_DOI_GROUPED_OOF_PREDICTIONS.csv',index=False)
print(json.dumps({'n':summary['n_analysis'],'positive':summary['n_positive'],'negative':summary['n_negative'],'prevalence':summary['prevalence'],'n_doi':summary['n_doi'],'metrics':metrics,'paired_doi_cluster_bootstrap':boot},indent=2))
