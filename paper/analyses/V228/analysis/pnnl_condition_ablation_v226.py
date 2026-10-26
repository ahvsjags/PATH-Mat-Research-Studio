from pathlib import Path
import json, hashlib, sys
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, accuracy_score, confusion_matrix

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'PATH-Mat_Evidence_Strengthening_V226'/'analysis'/'ironoxide_pnnl_validation_23.csv'
OUT=ROOT/'PATH-Mat_Evidence_Strengthening_V226'/'results'
OUT.mkdir(parents=True,exist_ok=True)


def make_model(num_cols,cat_cols):
    num=Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())])
    cat=Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))])
    pre=ColumnTransformer([('numeric',num,num_cols),('categorical',cat,cat_cols)],remainder='drop')
    return Pipeline([('features',pre),('logistic',LogisticRegression(C=1.0,solver='liblinear',max_iter=500))])

def pred_cv(df,num_cols,cat_cols,groups=None):
    y=df.observed_nonhematite.to_numpy(int); X=df[num_cols+cat_cols].copy(); scores=np.zeros(len(y)); fold=np.full(len(y),-1,int)
    splitter=LeaveOneOut() if groups is None else LeaveOneGroupOut()
    splits=splitter.split(X,y) if groups is None else splitter.split(X,y,groups=groups)
    for k,(tr,te) in enumerate(splits):
        model=make_model(num_cols,cat_cols); model.fit(X.iloc[tr],y[tr]); scores[te]=model.predict_proba(X.iloc[te])[:,1]; fold[te]=k
    return scores,fold

def met(y,s):
    return {'AP':float(average_precision_score(y,s)),'AUROC':float(roc_auc_score(y,s)),'Brier':float(brier_score_loss(y,s)),
            'accuracy_at_fixed_0.5':float(accuracy_score(y,s>=.5)),'n_correct_at_fixed_0.5':int(np.sum((s>=.5)==y)),
            'score_sd':float(np.std(s,ddof=1)),'unique_scores':int(np.unique(np.round(s,12)).size)}

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    d=pd.read_csv(DATA); d['additive']=d.additive.fillna('none').astype(str)
    base=d[d.additive.eq('none')].copy().reset_index(drop=True)
    y=base.observed_nonhematite.to_numpy(int)
    # Label-free grouping by FeCl3 concentration tertile; all preprocessing stays inside folds.
    groups=pd.qcut(base.fecl3_mM,3,labels=False,duplicates='drop').to_numpy()
    group_sizes={str(int(g)):int(np.sum(groups==g)) for g in np.unique(groups)}
    positive_by_group={str(int(g)):int(base.loc[groups==g,'observed_nonhematite'].sum()) for g in np.unique(groups)}
    ablations={
      'temperature_only':['temp_C'], 'pH_only':['pH'], 'FeCl3_concentration_only':['fecl3_mM'],
      'temperature_plus_pH':['temp_C','pH'], 'temperature_plus_concentration':['temp_C','fecl3_mM'],
      'pH_plus_concentration':['pH','fecl3_mM'], 'all_three_measured_conditions':['fecl3_mM','pH','temp_C']}
    result={'dataset':'Liu et al. 2023 Tables 1-2; printed experiment rows 1-17 and 19-24; row 18 is absent from printed tables',
      'source_data_sha256':sha(DATA),'n_printed_rows':len(d),'n_same_composition_additive_free':len(base),
      'endpoint':'source-table XRD phase label; 1=non-hematite phase vs target hematite, 0=hematite',
      'same_composition_control':{'target':'Fe2O3','precursors':['FeCl3','NaOH'],'additive':'none','duration_h_all_16':bool(base.duration_h.nunique()==1 and base.duration_h.iloc[0]==16),
        'same_reported_action_template':'solution mixing -> pH adjustment -> hydrothermal heating; same action count and order for all 17 rows',
        'n_positive':int(y.sum()),'prevalence':float(y.mean()),'concentration_band_sizes':group_sizes,'positive_by_band':positive_by_group,
        'model':'fixed LogisticRegression(C=1, solver=liblinear); numeric features imputed/standardized within each training fold; no tuning or threshold selection',
        'LOO_and_leave_one_concentration_band_out':{},'feature_ablation_posthoc_descriptive':{}},
      'full_23_row_source_native_comparison':{},
      'interpretation':'This estimates source-native within-panel predictability for a physical XRD-assigned outcome; it is not a frozen PATH-Mat score, not a cross-study test, and not a causal estimate.'}
    same_scores={}; same_metrics={}
    for name,cols in ablations.items():
      sloo,_=pred_cv(base,cols,[]); sloco,fold=pred_cv(base,cols,[],groups)
      same_metrics[name]={'features':cols,'LOO':met(y,sloo),'leave_one_concentration_band_out':met(y,sloco),
        'held_out_band_detail':{str(int(g)):{'n':int(np.sum(groups==g)),'positive':int(y[groups==g].sum()),'correct_at_0.5':int(np.sum((sloco[groups==g]>=.5)==y[groups==g])),'AP':float(average_precision_score(y[groups==g],sloco[groups==g])),'AUROC':float(roc_auc_score(y[groups==g],sloco[groups==g]))} for g in np.unique(groups)}}
      same_scores[name]=(sloo,sloco)
    result['same_composition_control']['LOO_and_leave_one_concentration_band_out']=same_metrics['all_three_measured_conditions']
    result['same_composition_control']['feature_ablation_posthoc_descriptive']={k:{'LOO':v['LOO'],'leave_one_concentration_band_out':v['leave_one_concentration_band_out']} for k,v in same_metrics.items()}
    out=base.copy(); out['FeCl3_concentration_band']=groups
    for name,(a,b) in same_scores.items(): out[name+'_LOO_score']=a; out[name+'_heldout_concentration_band_score']=b
    out.to_csv(OUT/'V226_PNNL_SAME_COMPOSITION_CONDITION_ABLATIONS.csv',index=False)
    # Whole 23-row source-native model comparison; no CV-based model selection.
    Xcols=['fecl3_mM','pH','temp_C','duration_h','additive_mM']
    y23=d.observed_nonhematite.to_numpy(int)
    s0,_=pred_cv(d,[],['additive']); s1,_=pred_cv(d,Xcols,['additive'])
    result['full_23_row_source_native_comparison']={'n':len(d),'positive':int(y23.sum()),'prevalence':float(y23.mean()),
      'composition_identity_only':met(y23,s0),'composition_plus_measured_conditions':met(y23,s1),
      'note':'Source-native cross-validation; target and precursor identities are constant and composition baseline is therefore an additive-identity baseline.'}
    df=d.copy(); df['composition_identity_only_LOO_score']=s0; df['composition_plus_conditions_LOO_score']=s1
    df.to_csv(OUT/'V226_PNNL_23ROW_SOURCE_NATIVE_PREDICTIONS.csv',index=False)
    (OUT/'V226_PNNL_CONDITION_ABLATION_AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'same_comp':result['same_composition_control']['LOO_and_leave_one_concentration_band_out'],'ablations':result['same_composition_control']['feature_ablation_posthoc_descriptive'],'full23':result['full_23_row_source_native_comparison']},indent=2))
if __name__=='__main__': main()

