from pathlib import Path
import csv, hashlib, json, math, os, sys
import numpy as np

HERE=Path(__file__).resolve().parent
RESULT=HERE/'PEROVSKITE_MEASURED_CHRONOLOGY_RESULTS_V162.json'
LODO=HERE/'PEROVSKITE_OOF_full_finite_PCE_V162.csv'
SRC=HERE/'PEROVSKITE_OOF_source_protocol_sensitivity_V162.csv'
FWD=HERE/'PEROVSKITE_FORWARD_TIME_RESULTS_V163.json'
PROTOCOL=HERE/'PEROVSKITE_MEASURED_CHRONOLOGY_PROTOCOL_V162.md'
MODEL_SCRIPT=HERE/'PEROVSKITE_MEASURED_CHRONOLOGY_V162.py'
SOURCE_H5=Path(sys.argv[1] if len(sys.argv)>1 else os.environ.get('PATHMAT_PEROVSKITE_DATA', HERE/'perovskite_in_situ_data.h5'))
OUT=HERE/'V204_SAVED_OUTPUT_AUDIT.json'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):
            h.update(b)
    return h.hexdigest()

def mean_absolute_error(y,p):
    return float(np.mean(np.abs(np.asarray(y)-np.asarray(p))))

def row_data(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    y=np.array([float(r['PCE_backward']) for r in rows])
    ps=np.array([float(r['pred_sorted_control']) for r in rows])
    pc=np.array([float(r['pred_chronological']) for r in rows])
    dates=np.array([int(r['date']) for r in rows])
    return rows,y,ps,pc,dates

def row_boot(y,ps,pc,dates,reps,seed):
    rng=np.random.default_rng(seed)
    uniq=np.unique(dates)
    members={d:np.flatnonzero(dates==d) for d in uniq}
    vals=np.empty(reps)
    for i in range(reps):
        sampled=rng.choice(uniq,size=len(uniq),replace=True)
        ix=np.concatenate([members[d] for d in sampled])
        vals[i]=mean_absolute_error(y[ix],ps[ix])-mean_absolute_error(y[ix],pc[ix])
    return np.quantile(vals,[0.025,0.975]).tolist()

def summary_boot(date_rows,reps,seed):
    rng=np.random.default_rng(seed)
    n=np.asarray([int(r['n']) for r in date_rows],dtype=float)
    d=np.asarray([float(r['delta_mae_sorted_minus_chronological']) for r in date_rows])
    ix0=np.arange(len(n))
    vals=np.empty(reps)
    for i in range(reps):
        ix=rng.choice(ix0,size=len(ix0),replace=True)
        vals[i]=np.sum(n[ix]*d[ix])/np.sum(n[ix])
    return np.quantile(vals,[0.025,0.975]).tolist()

def check_close(a,b,tol=1e-10):
    if abs(float(a)-float(b))>tol:
        raise AssertionError(f'{a} != {b}')

def main():
    res=json.loads(RESULT.read_text(encoding='utf-8'))
    audit={}
    for path, cohort, model, reps, seed in [
        (LODO,'full_finite_PCE','ExtraTrees_primary',10000,20261006),
        (SRC,'source_protocol_sensitivity','ExtraTrees_primary',10000,20261006),
    ]:
        rows,y,ps,pc,dates=row_data(path)
        m=res['models'][cohort][model]
        check_close(mean_absolute_error(y,ps),m['mae_sorted'])
        check_close(mean_absolute_error(y,pc),m['mae_chronological'])
        ci=row_boot(y,ps,pc,dates,reps,seed)
        if not np.allclose(ci,m['date_cluster_bootstrap_ci95'],rtol=0,atol=1e-12):
            raise AssertionError(f'bootstrap mismatch: {path.name} {ci}')
        audit[path.name]={'n':len(rows),'dates':len(np.unique(dates)),'mae_sorted':float(mean_absolute_error(y,ps)),'mae_chronological':float(mean_absolute_error(y,pc)),'delta':float(mean_absolute_error(y,ps)-mean_absolute_error(y,pc)),'bootstrap_ci95':ci,'matches_saved':True,'sha256':sha(path)}
    # Recreate cluster bootstrap for all four model/cohort summaries from their per-date losses.
    rng_specs=[
        ('full_finite_PCE','ExtraTrees_primary',10000,20261006),
        ('full_finite_PCE','Ridge_sensitivity',10000,20261006),
        ('source_protocol_sensitivity','ExtraTrees_primary',10000,20261006),
        ('source_protocol_sensitivity','Ridge_sensitivity',10000,20261006),
    ]
    for c,mname,reps,seed in rng_specs:
        m=res['models'][c][mname]
        drows=m['date_specific']
        rng=np.random.default_rng(seed)
        n=np.asarray([int(r['n_rows']) for r in drows],dtype=float)
        d=np.asarray([float(r['delta_mae_sorted_minus_chronological']) for r in drows])
        idx0=np.arange(len(drows)); vals=np.empty(reps)
        for i in range(reps):
            ix=rng.choice(idx0,size=len(idx0),replace=True)
            vals[i]=np.sum(n[ix]*d[ix])/np.sum(n[ix])
        ci=np.quantile(vals,[.025,.975]).tolist()
        if not np.allclose(ci,m['date_cluster_bootstrap_ci95'],rtol=0,atol=1e-10):
            raise AssertionError(f'per-date bootstrap mismatch {c}/{mname}: {ci} vs {m["date_cluster_bootstrap_ci95"]}')
        audit[f'{c}/{mname}']={'ci95_recomputed_from_saved_date_summaries':ci,'matches_saved':True,'delta':m['delta_mae_sorted_minus_chronological'],'dates_favoring_chronology':m['date_sign_test']['dates_favoring_chronology'],'dates_favoring_sorted':m['date_sign_test']['dates_favoring_sorted']}
    fw=json.loads(FWD.read_text(encoding='utf-8'))
    fr=fw['date_rows']
    n=sum(int(r['n']) for r in fr)
    ms=sum(int(r['n'])*float(r['mae_sorted']) for r in fr)/n
    mc=sum(int(r['n'])*float(r['mae_chronological']) for r in fr)/n
    check_close(ms,fw['mae_sorted']);check_close(mc,fw['mae_chronological'])
    ci=summary_boot(fr,fw['date_cluster_bootstrap_resamples'],fw['date_cluster_bootstrap_seed'])
    if not np.allclose(ci,fw['date_cluster_bootstrap_ci95'],rtol=0,atol=1e-10):
        raise AssertionError('future-date bootstrap mismatch')
    for f in fw['folds']:
        if max(f['train_dates'])>=min(f['test_dates']):
            raise AssertionError('Future-date split is not strictly chronological')
    audit['future_date']={'n':n,'dates':len(fr),'mae_sorted':ms,'mae_chronological':mc,'delta':ms-mc,'ci95':ci,'strict_forward_folds':True,'matches_saved':True}
    check_hashes={'source_data':sha(SOURCE_H5),'protocol':sha(PROTOCOL),'model_script':sha(MODEL_SCRIPT)}
    expected={'source_data':res['data_sha256'],'protocol':res['protocol_sha256'],'model_script':res['analysis_script_sha256']}
    if check_hashes!=expected:
        raise AssertionError(f'provenance hash mismatch: {check_hashes} != {expected}')
    audit['provenance']={'computed_sha256':check_hashes,'matches_saved':True,'full_model_refit_performed':False}
    out={'audit':'Independent saved-output reproduction; no models refit.','results':audit}
    OUT.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
