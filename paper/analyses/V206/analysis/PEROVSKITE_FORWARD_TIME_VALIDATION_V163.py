from __future__ import annotations
import importlib.util, json, hashlib, csv, time
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor
HERE=Path(__file__).resolve().parent
SOURCE_SCRIPT=HERE/'PEROVSKITE_MEASURED_CHRONOLOGY_V162.py'
DATA=HERE/'perovskite_in_situ_data.h5'
OUT_JSON=HERE/'PEROVSKITE_FORWARD_TIME_RESULTS_V163.json'
OUT_CSV=HERE/'PEROVSKITE_FORWARD_TIME_DATE_SUMMARY_V163.csv'
SEED=20261007
NBOOT=20000

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def main():
 t=time.time()
 spec=importlib.util.spec_from_file_location('chrono_v162',SOURCE_SCRIPT)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 num,obj=m.load_hdf(DATA); c=m.build_cohort(num,obj,'full')
 y,dates,Xbag,Xchron=c['y'],c['dates'],c['X_bag'],c['X_order']
 unique=np.unique(dates)
 # Frozen, audit-friendly expanding-window design. Earliest eight dates train first;
 # later dates tested in four non-overlapping forward blocks.
 bounds=[(8,10),(10,12),(12,14),(14,len(unique))]
 pred_bag=np.full(len(y),np.nan);pred_chron=np.full(len(y),np.nan)
 folds=[]
 for fold,(train_end,test_end) in enumerate(bounds,1):
  train_dates=unique[:train_end];test_dates=unique[train_end:test_end]
  train=np.isin(dates,train_dates);test=np.isin(dates,test_dates)
  if train.sum()==0 or test.sum()==0:raise RuntimeError('Empty fold')
  for X,pred,label in [(Xbag,pred_bag,'sorted'),(Xchron,pred_chron,'chronological')]:
   model=ExtraTreesRegressor(n_estimators=500,min_samples_leaf=3,max_features=1.0,random_state=20261006,n_jobs=1)
   model.fit(X[train],y[train]);pred[test]=model.predict(X[test])
  folds.append({'fold':fold,'train_dates':[int(x) for x in train_dates],'test_dates':[int(x) for x in test_dates],'n_train':int(train.sum()),'n_test':int(test.sum())})
 test=np.isfinite(pred_bag)&np.isfinite(pred_chron)
 if not test.any():raise RuntimeError('No predictions')
 # aggregate per-date errors; source rows/IDs are not exported
 date_rows=[]
 for d in unique:
  ix=(dates==d)&test
  if ix.any():
   date_rows.append({'date':int(d),'n':int(ix.sum()),'mae_sorted':float(mean_absolute_error(y[ix],pred_bag[ix])),'mae_chronological':float(mean_absolute_error(y[ix],pred_chron[ix])),'delta_mae_sorted_minus_chronological':float(mean_absolute_error(y[ix],pred_bag[ix])-mean_absolute_error(y[ix],pred_chron[ix]))})
 ix=np.flatnonzero(test)
 delta=float(mean_absolute_error(y[ix],pred_bag[ix])-mean_absolute_error(y[ix],pred_chron[ix]))
 # Nonparametric cluster bootstrap over the non-overlapping held-out acquisition dates.
 rng=np.random.default_rng(SEED);ids=np.arange(len(date_rows));w=np.array([r['n'] for r in date_rows],float);dd=np.array([r['delta_mae_sorted_minus_chronological'] for r in date_rows]);boot=np.empty(NBOOT)
 for b in range(NBOOT):
  pick=rng.choice(ids,size=len(ids),replace=True);boot[b]=np.sum(w[pick]*dd[pick])/np.sum(w[pick])
 pos=sum(v>0 for v in dd);neg=sum(v<0 for v in dd);nsg=pos+neg
 from math import comb
 p_one=sum(comb(nsg,k) for k in range(pos,nsg+1))/(2**nsg) if nsg else 1
 p_two=min(1.0,2*min(p_one,sum(comb(nsg,k) for k in range(0,neg+1))/(2**nsg))) if nsg else 1
 out={'dataset_doi':'10.5281/zenodo.14609789','source_article_doi':'10.1039/D4EE03445G','data_sha256':sha(DATA),'source_script_sha256':sha(SOURCE_SCRIPT),'cohort':'finite PCE_backward, ratio, molarity, evacuation duration, and valid four-channel traces; exactly as primary V162 chronology cohort','design':'post-hoc expanding-window future-date validation; four non-overlapping future test blocks; earliest eight dates used as initial training history','fit_dates': [int(x) for x in unique], 'folds':folds,'n_rows_test':int(test.sum()),'n_test_dates':len(date_rows),'model':'ExtraTreesRegressor(n_estimators=500,min_samples_leaf=3,max_features=1.0,random_state=20261006,n_jobs=1)','matched_control':'same 32-point/channel interpolated signal multiset sorted within each channel; shared covariates unchanged','mae_sorted':float(mean_absolute_error(y[ix],pred_bag[ix])),'mae_chronological':float(mean_absolute_error(y[ix],pred_chron[ix])),'delta_mae_sorted_minus_chronological':delta,'rmse_sorted':float(mean_squared_error(y[ix],pred_bag[ix])**.5),'rmse_chronological':float(mean_squared_error(y[ix],pred_chron[ix])**.5),'r2_sorted':float(r2_score(y[ix],pred_bag[ix])),'r2_chronological':float(r2_score(y[ix],pred_chron[ix])),'date_cluster_bootstrap_resamples':NBOOT,'date_cluster_bootstrap_seed':SEED,'date_cluster_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist(),'dates_favoring_chronology':pos,'dates_favoring_sorted':neg,'two_sided_sign_test_p':p_two,'date_rows':date_rows,'runtime_seconds':time.time()-t}
 OUT_JSON.write_text(json.dumps(out,indent=2,ensure_ascii=False,default=lambda x: x.item() if isinstance(x,np.generic) else str(x))+'\n',encoding='utf-8')
 with OUT_CSV.open('w',newline='',encoding='utf-8-sig') as f:
  wri=csv.DictWriter(f,fieldnames=list(date_rows[0]));wri.writeheader();wri.writerows(date_rows)
 print(json.dumps(out,indent=2,ensure_ascii=False,default=lambda x: x.item() if isinstance(x,np.generic) else str(x)))
if __name__=='__main__':main()

