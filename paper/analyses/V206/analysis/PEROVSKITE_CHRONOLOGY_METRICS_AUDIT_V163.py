from __future__ import annotations
import csv,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
HERE=Path(__file__).resolve().parent
DATA=HERE/'perovskite_in_situ_data.h5'
LODO=HERE/'PEROVSKITE_OOF_full_finite_PCE_V162.csv'
LODO_RESULT=HERE/'PEROVSKITE_MEASURED_CHRONOLOGY_RESULTS_V162.json'
FWD=HERE/'PEROVSKITE_FORWARD_TIME_RESULTS_V163.json'
OUT=HERE/'PEROVSKITE_CHRONOLOGY_METRICS_AUDIT_V163.json'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def close(a,b,tol=1e-10):
 if abs(float(a)-float(b))>tol: raise AssertionError(f'{a} != {b}')

def date_boot(rows, reps, seed):
 rng=np.random.default_rng(seed)
 w=np.asarray([r['n'] for r in rows],dtype=float)
 d=np.asarray([r['delta'] for r in rows],dtype=float)
 ids=np.arange(len(rows)); out=np.empty(reps)
 for i in range(reps):
  ix=rng.choice(ids,size=len(ids),replace=True)
  out[i]=np.sum(w[ix]*d[ix])/np.sum(w[ix])
 return float(out.mean()),np.quantile(out,[.025,.975]).tolist()

def main():
 with LODO.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
 y=np.asarray([float(r['PCE_backward']) for r in rows]); ps=np.asarray([float(r['pred_sorted_control']) for r in rows]); pc=np.asarray([float(r['pred_chronological']) for r in rows]); dates=np.asarray([int(r['date']) for r in rows])
 m=json.loads(LODO_RESULT.read_text(encoding='utf-8'))['models']['full_finite_PCE']['ExtraTrees_primary']
 assert len(rows)==5857 and len(np.unique(dates))==17
 close(mean_absolute_error(y,ps),m['mae_sorted']);close(mean_absolute_error(y,pc),m['mae_chronological'])
 # Reproduce original cluster bootstrap directly from saved device-level OOF predictions.
 rng=np.random.default_rng(20261006); uniq=np.unique(dates); mem={d:np.flatnonzero(dates==d) for d in uniq}; boot=np.empty(10000)
 for b in range(10000):
  sample=rng.choice(uniq,size=len(uniq),replace=True); ix=np.concatenate([mem[d] for d in sample])
  boot[b]=mean_absolute_error(y[ix],ps[ix])-mean_absolute_error(y[ix],pc[ix])
 expected=m['date_cluster_bootstrap_ci95']
 assert np.allclose(np.quantile(boot,[.025,.975]),expected,rtol=0,atol=1e-12)
 # Verify the saved source-protocol ExtraTrees sensitivity as well.
 src=HERE/'PEROVSKITE_OOF_source_protocol_sensitivity_V162.csv'
 with src.open(encoding='utf-8-sig',newline='') as f: sr=list(csv.DictReader(f))
 sy=np.asarray([float(r['PCE_backward']) for r in sr]); sa=np.asarray([float(r['pred_sorted_control']) for r in sr]); sc=np.asarray([float(r['pred_chronological']) for r in sr]); sd=np.asarray([int(r['date']) for r in sr])
 sm=json.loads(LODO_RESULT.read_text(encoding='utf-8'))['models']['source_protocol_sensitivity']['ExtraTrees_primary']
 assert len(sr)==2215 and len(np.unique(sd))==6
 close(mean_absolute_error(sy,sa),sm['mae_sorted']);close(mean_absolute_error(sy,sc),sm['mae_chronological'])
 srng=np.random.default_rng(20261006); su=np.unique(sd); smem={d:np.flatnonzero(sd==d) for d in su}; sboot=np.empty(10000)
 for i in range(10000):
  six=np.concatenate([smem[d] for d in srng.choice(su,size=len(su),replace=True)])
  sboot[i]=mean_absolute_error(sy[six],sa[six])-mean_absolute_error(sy[six],sc[six])
 assert np.allclose(np.quantile(sboot,[.025,.975]),sm['date_cluster_bootstrap_ci95'],rtol=0,atol=1e-12)
 # Independent checks on forward-only aggregate output and saved fold chronology.
 fw=json.loads(FWD.read_text(encoding='utf-8')); fr=fw['date_rows']
 n=sum(int(r['n']) for r in fr)
 ma=float(sum(r['n']*r['mae_sorted'] for r in fr)/n)
 mc=float(sum(r['n']*r['mae_chronological'] for r in fr)/n)
 close(n,fw['n_rows_test']);close(ma,fw['mae_sorted'],1e-8);close(mc,fw['mae_chronological'],1e-8)
 close(ma-mc,fw['delta_mae_sorted_minus_chronological'],1e-8)
 rows_boot=[{'n':r['n'],'delta':r['delta_mae_sorted_minus_chronological']} for r in fr]
 mean,ci=date_boot(rows_boot,fw['date_cluster_bootstrap_resamples'],fw['date_cluster_bootstrap_seed'])
 assert np.allclose(ci,fw['date_cluster_bootstrap_ci95'],rtol=0,atol=1e-10)
 last_train=None
 for fold in fw['folds']:
  tr=fold['train_dates']; te=fold['test_dates']
  if not tr or not te or max(tr)>=min(te):raise AssertionError('Forward fold is not strictly forward')
  if last_train is not None and max(tr)<=last_train: raise AssertionError('Training window did not expand')
  last_train=max(tr)
 assert len(fr)==9 and n==3292
 # Experiment IDs must remain nested within an acquisition date for the LODO grouping claim.
 spec=importlib.util.spec_from_file_location('chrono_v162',HERE/'PEROVSKITE_MEASURED_CHRONOLOGY_V162.py')
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 num,obj=mod.load_hdf(DATA); cohort=mod.build_cohort(num,obj,'full')
 expids=np.asarray(obj['expID'])[cohort['row_ids']]
 exp_date={}
 for eid,d in zip(expids,cohort['dates']): exp_date.setdefault(str(eid),set()).add(int(d))
 n_cross=sum(len(v)>1 for v in exp_date.values())
 assert n_cross==0
 out={'audit':'Saved-output verification; no model refit in this audit.','source_data_sha256':sha(DATA),'source_protocol_sha256':sha(HERE/'PEROVSKITE_MEASURED_CHRONOLOGY_PROTOCOL_V162.md'),'lodo_oof_csv_sha256':sha(LODO),'lodo_rows':len(rows),'lodo_dates':len(uniq),'lodo_mae_sorted_recomputed':mean_absolute_error(y,ps),'lodo_mae_chronological_recomputed':mean_absolute_error(y,pc),'lodo_delta_recomputed':mean_absolute_error(y,ps)-mean_absolute_error(y,pc),'lodo_bootstrap_ci_recomputed':np.quantile(boot,[.025,.975]).tolist(),'lodo_ci_matches_saved':True,'source_protocol_oof_rows':len(sr),'source_protocol_oof_dates':int(len(np.unique(sd))),'source_protocol_extra_trees_delta_recomputed':float(mean_absolute_error(sy,sa)-mean_absolute_error(sy,sc)),'source_protocol_bootstrap_ci_recomputed':np.quantile(sboot,[.025,.975]).tolist(),'source_protocol_ci_matches_saved':True,'forward_json_sha256':sha(FWD),'forward_test_rows_recomputed':n,'forward_mae_sorted_recomputed':ma,'forward_mae_chronological_recomputed':mc,'forward_delta_recomputed':ma-mc,'forward_bootstrap_ci_recomputed':ci,'forward_ci_matches_saved':True,'forward_folds_strictly_chronological':True,'expIDs_crossing_acquisition_dates':n_cross,'full_model_rerun_performed':False}
 OUT.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=='__main__':main()

