"""Post-inspection source-only fitting; no external-label adaptation.

An explicit model lock is written before opening the external archive. All three
predeclared models and all six AP contrasts are retained, irrespective of sign.
No source operation list is interpreted as a verified global chronology.
"""
from __future__ import annotations
import collections, csv, gzip, hashlib, json, pathlib, sys, time, warnings, zipfile
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
warnings.filterwarnings('ignore', message='No Pauling electronegativity')
import os
import v152_portable_helpers as comp
clean=comp
DSU=comp.DSU
CODE=HERE
HERE=pathlib.Path(os.environ.get('PATHMAT_ANALYSIS_OUT',str(CODE/'analysis_outputs'))).resolve()
HERE.mkdir(parents=True,exist_ok=True)

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def norm(s): return ''.join(str(s or '').split()).lower()

def common_type(s):
    s=str(s).lower()
    if s in ('heating','heat'): return 'heat'
    if s in ('mixing','grinding','mix'): return 'mix'
    if s in ('shaping','pressing','deposition','shape'): return 'shape'
    return 'other'

def process(acts):
    count=collections.Counter(common_type(a['name']) for a in acts)
    temp=[x for a in acts for x in a['temps']]
    duration=[x for a in acts for x in a['times']]
    return {'n':len(acts),'proportions':{k:count[k]/max(1,len(acts)) for k in ('heat','mix','shape','other')},
            'values':[max(temp) if temp else np.nan, np.mean(temp) if temp else np.nan,
                      max(duration) if duration else np.nan,np.mean(duration) if duration else np.nan],
            'present':[int(bool(temp)),int(bool(duration))]}

def sol_number(obj,kind):
    if not isinstance(obj,dict): return []
    val=obj.get('max_value') if obj.get('max_value') is not None else obj.get('min_value')
    try: val=float(val)
    except (ValueError,TypeError): return []
    # The same strict unit/range code is used for both sources.
    return clean.numeric([{'values':[val],'units':obj.get('unit')}],kind)

def make_features(sr1,r,p,variant,med):
    f=comp.extended(sr1,r)
    if variant=='composition':return f
    f['direct_action_log1p_count']=float(np.log1p(p['n']))
    for k,v in p['proportions'].items():f['common_action_proportion='+k]=v
    if variant=='full':
        for j,k in enumerate(('maximum_temperature','mean_temperature','maximum_duration','mean_duration')):
            v=p['values'][j] if np.isfinite(p['values'][j]) else med[j]
            f[k]=float(v/(1500 if j<2 else 24))
        f['temperature_present']=p['present'][0];f['duration_present']=p['present'][1]
    return f

def metrics(y,raw,p,t):
    pred=p>=t; pos=y==1;neg=~pos
    ece=0.
    for a,b in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        take=(p>=a)&((p<b) if b<1 else (p<=b))
        if take.any():ece+=take.mean()*abs(p[take].mean()-y[take].mean())
    return {'n':len(y),'positive':int(y.sum()),'prevalence':float(y.mean()),
      'raw_auprc':float(average_precision_score(y,raw)), 'raw_auroc':float(roc_auc_score(y,raw)),
      'calibrated_auprc':float(average_precision_score(y,p)), 'calibrated_auroc':float(roc_auc_score(y,p)),
      'brier':float(brier_score_loss(y,p)),'ece_10_equal_width_bins':float(ece),
      'threshold_source_only':float(t),'classified_positive':int(pred.sum()),
      'TP':int((pos&pred).sum()),'FP':int((neg&pred).sum()),'FN':int((pos&~pred).sum()),'TN':int((neg&~pred).sum()),
      'distinct_raw_scores':len(np.unique(raw)),'distinct_calibrated_scores':len(np.unique(p)),
      'raw_score_sd':float(np.std(raw)),'calibrated_score_sd':float(np.std(p)),
      'raw_score_quantiles':np.quantile(raw,[0,.25,.5,.75,1]).tolist(),
      'calibrated_score_quantiles':np.quantile(p,[0,.25,.5,.75,1]).tolist()}

class WeightedAP:
    def __init__(self,y,p):
        self.idx=np.argsort(-p,kind='stable');self.y=y[self.idx]
        ps=p[self.idx];self.end=np.r_[np.flatnonzero(ps[1:]!=ps[:-1]),len(ps)-1]
    def __call__(self,w):
        w=w[self.idx];tp=np.cumsum(w*self.y)[self.end];tot=np.cumsum(w)[self.end]
        if not tp[-1]: return np.nan
        return float(np.sum(np.diff(np.r_[0,tp]) * np.divide(tp,tot,out=np.zeros_like(tp),where=tot>0))/tp[-1])

def boot(y,pred,groups):
    unique,inv=np.unique(groups,return_inverse=True);g=len(unique)
    funcs={k:WeightedAP(y,p) for k,p in pred.items()}
    # Validate weighted tie handling against sklearn, including zero weights.
    rng=np.random.default_rng(1522026)
    w=rng.integers(0,4,len(y)).astype(float)
    for k,f in funcs.items(): assert abs(f(w)-average_precision_score(y,pred[k],sample_weight=w))<1e-12
    pairs=[('actions','composition'),('full','composition'),('full','actions')]
    vals=[];zero=np.ones(len(y))
    for _ in range(3000):
        gw=np.bincount(rng.integers(0,g,g),minlength=g);w=gw[inv].astype(float)
        if np.dot(w,y)==0 or np.dot(w,1-y)==0: continue
        ap={k:f(w) for k,f in funcs.items()};vals.append([ap[a]-ap[b] for a,b in pairs])
    vals=np.asarray(vals)
    return [{'contrast':a+' minus '+b,'delta_raw_auprc':funcs[a](zero)-funcs[b](zero),
      'ci95':np.quantile(vals[:,i],[.025,.975]).tolist(),
      'ci99_1667_bonferroni_six':np.quantile(vals[:,i],[.05/12,1-.05/12]).tolist(),
      'valid_draws':len(vals),'components':g} for i,(a,b) in enumerate(pairs)]

def main():
    start=time.time();_,sr1,rows,ih,ph,_,_=comp.old.load_locked()
    inds=comp.old.split_indices(rows,None);y=np.asarray([sr1.label(r) for r in rows])
    raw_path=comp.required_input('PATHMAT_LEE_RAW_GZ');assert sha(raw_path)==clean.EXPECTED_RAW
    raw=json.load(gzip.open(raw_path,'rt',encoding='utf-8'))
    byhash={hashlib.sha256(json.dumps(r,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest():r for r in raw}
    assert len(byhash)==len(raw)
    pro=[];audit=collections.Counter();examples=[];source_formula=set();source_doi=set()
    for r in rows:
        rr=byhash[r['source']['content_hash']];assert rr['DOI']==r['source']['source_doi']
        aa=rr.get('conditions_forDOI') or [];assert len(aa)==len(r.get('actions') or [])
        direct=[a for a in aa if a.get('act_type')!='Starting' and not a.get('ref_act')]
        ids=[a.get('act_id') for a in direct]
        audit['records']+=1;audit['direct_non_starting_acts']+=len(direct)
        audit['starting_entries']+=sum(a.get('act_type')=='Starting' for a in aa)
        audit['reference_entries']+=sum(bool(a.get('ref_act')) for a in aa)
        audit['empty_direct_list_records']+=not bool(direct)
        allids=all(isinstance(i,int) for i in ids)
        audit['all_integer_local_ids_records']+=allids
        if allids:
            dup=len(set(ids))<len(ids);unsorted=ids!=sorted(ids)
            audit['repeated_local_id_records']+=dup;audit['list_local_id_discordant_records']+=unsorted
            if dup and len(examples)<3:examples.append({'source_content_hash':r['source']['content_hash'],'doi':rr['DOI'],'local_id_type_pairs':[[a['act_id'],a['act_type']] for a in direct]})
        pro.append(process([{'name':a.get('act_type'),'temps':clean.numeric(a.get('temp_values'),'temperature'),
                             'times':clean.numeric(a.get('time_values'),'duration')} for a in direct]))
        source_formula.add(norm(r['target_formula']));source_doi.add(norm(rr['DOI']))
    source_units=dict(clean.unit_stats);clean.unit_stats.clear()
    audit_result={'exact_cohort_n':len(rows),'canonical_hash_matches':len(rows),'counts':dict(audit),
       'global_sentence_or_timestamp_key_in_action_export':False,
       'observed_action_keys':sorted({k for rr in raw[:100] for a in rr.get('conditions_forDOI') or [] for k in a}),
       'conclusion':'Local token/action IDs and stored list order do not establish a global physical synthesis chronology. No global sort or order-success claim is made.',
       'examples':examples,'source_sha256':ih,'raw_sha256':sha(raw_path)}
    (HERE/'V152_ORDER_RECOVERABILITY_AUDIT.json').write_text(json.dumps(audit_result,indent=2),encoding='utf-8')
    del raw,byhash
    X=np.asarray([p['values'] for p in pro]);med=np.nanmedian(X[inds['D_dev']],axis=0);assert np.isfinite(med).all()
    models={};lock={'protocol_sha256':sha(CODE/'V152_PRE_EXECUTION_PROTOCOL.md'),'source_sha256':ih,
       'partition_sha256':ph,'script_sha256':sha(pathlib.Path(__file__)),'imputation_medians':med.tolist(),
       'fit_scope':'Lee D_dev only; calibration D_cal; threshold D_select; external archive not yet opened',
       'source_unit_diagnostics':source_units,'models':{}}
    for name in ('composition','actions','full'):
        def fs(ix):return [make_features(sr1,rows[i],pro[i],name,med) for i in ix]
        vec=DictVectorizer(sparse=True);xd=vec.fit_transform(fs(inds['D_dev']))
        xc=vec.transform(fs(inds['D_cal']));xs=vec.transform(fs(inds['D_select']))
        est=LogisticRegression(C=1,max_iter=600,solver='liblinear',class_weight='balanced',random_state=31415).fit(xd,y[inds['D_dev']])
        assert est.n_iter_.max()<600, 'Model did not converge; do not proceed to external evaluation'
        iso=IsotonicRegression(out_of_bounds='clip').fit(est.predict_proba(xc)[:,1],y[inds['D_cal']])
        threshold=sr1.choose_threshold(y[inds['D_select']],iso.predict(est.predict_proba(xs)[:,1]))
        lock['models'][name]={'feature_names':vec.feature_names_,'coefficient':est.coef_.tolist(),'intercept':est.intercept_.tolist(),
          'calibration_x':iso.X_thresholds_.tolist(),'calibration_y':iso.y_thresholds_.tolist(),'threshold':float(threshold),
          'n_iter':est.n_iter_.tolist()}
        models[name]=(vec,est,iso,threshold)
        print('fitted',name,len(vec.feature_names_),flush=True)
    lock_path=HERE/'V152_SOURCE_ONLY_MODEL_LOCK.json';lock_path.write_text(json.dumps(lock,indent=2),encoding='utf-8')
    lock_hash=sha(lock_path);print('Source-only model lock saved',lock_hash,flush=True)
    # External labels and input objects are read only AFTER the fitted lock.
    arc=comp.required_input('PATHMAT_SOLGEL_ZIP')
    assert sha(arc)=='65da667fcecedf73942a4b8b898e14a1e6479c7371be3fa50162ac9e3f0d5d03'
    erows=[];ep=[];ey=[];counts=collections.Counter()
    with zipfile.ZipFile(arc) as z:
        with z.open('sol_gel_dataset.jsonl') as f:
            for line in f:
                rr=json.loads(line);counts['all_archive_records']+=1
                pr=rr.get('protocol') or {};target=pr.get('target') or {};pp=rr.get('phase_purity') or {}
                lab=str(pp.get('classification') or '').strip().lower();formula=str(target.get('material_formula') or '').strip()
                doi=str(pr.get('doi') or '').strip()
                if lab not in ('pure','impure') or not formula or not doi: counts['ineligible']+=1;continue
                if norm(formula) in source_formula:counts['formula_overlap_excluded']+=1;continue
                if norm(doi) in source_doi:counts['doi_overlap_excluded_after_formula']+=1;continue
                erows.append({'target_formula':formula,'precursor_formulas':[str(p['material_formula']) for p in pr.get('metal_precursors') or [] if p.get('material_formula')],
                               'doi':norm(doi),'analysis_id':counts['all_archive_records']-1})
                ep.append(process([{'name':a.get('operation_type'),'temps':sol_number(a.get('temperature'),'temperature'),
                                    'times':sol_number(a.get('time'),'duration')} for a in rr.get('operations') or []]))
                ey.append(int(lab=='impure'))
    assert len(erows)>5000;ey=np.asarray(ey);external_units=dict(clean.unit_stats)
    ds=DSU()
    for r in erows:
        a='D:'+r['doi'];b='F:'+norm(r['target_formula']);ds.add(a);ds.add(b);ds.union(a,b)
    eg=np.asarray([ds.find('D:'+r['doi']) for r in erows])
    assert not {norm(r['target_formula']) for r in erows}&source_formula
    assert not {r['doi'] for r in erows}&source_doi
    data={};preds={};result={'status':'post-inspection exploratory source-only zero-shot transfer; not confirmatory replication',
      'model_lock_sha256':lock_hash,'source_lock_saved_before_external_label_read':True,'prior_external_archive_inspection':True,
      'order_audit':audit_result,'external_selection_counts':dict(counts),'external_sha256':sha(arc),'external_unit_diagnostics':external_units,
      'formula_normalization':'whitespace stripped and lowercase; near-equivalent chemistry not excluded','cohorts':{}}
    for cohort,rs,ps,yy,gg in [('primary',[rows[i] for i in inds['D_test']],[pro[i] for i in inds['D_test']],y[inds['D_test']],np.asarray([rows[i]['group_id'] for i in inds['D_test']])),
                              ('external_solgel',erows,ep,ey,eg)]:
        cohort_metrics={};rawpred={};coverage={}
        for name,(vec,est,iso,t) in models.items():
            xx=vec.transform([make_features(sr1,r,p,name,med) for r,p in zip(rs,ps)])
            raw=est.predict_proba(xx)[:,1];cal=iso.predict(raw);rawpred[name]=raw
            cohort_metrics[name]=metrics(yy,raw,cal,t)
            coverage[name]={'zero_feature_rows':int(np.sum(np.diff(xx.indptr)==0)),
                            'mean_active_features':float(np.mean(np.diff(xx.indptr)))}
            preds[cohort+'_'+name+'_raw']=raw;preds[cohort+'_'+name+'_cal']=cal
            print(cohort,name,json.dumps(cohort_metrics[name]),flush=True)
        result['cohorts'][cohort]={'n':len(yy),'components':len(np.unique(gg)),'metrics':cohort_metrics,
           'coverage':coverage,'temperature_present_rows':int(sum(p['present'][0] for p in ps)),
           'duration_present_rows':int(sum(p['present'][1] for p in ps)),
           'contrasts':boot(yy,rawpred,gg)}
        preds[cohort+'_y']=yy;preds[cohort+'_group']=gg
        preds[cohort+'_id']=np.asarray([str(r.get('route_id') or r.get('analysis_id')) for r in rs])
    result['elapsed_seconds']=round(time.time()-start,2)
    assert sha(lock_path)==lock_hash, 'Fitted lock changed after external outcomes were read'
    (HERE/'V152_HARMONIZED_ZERO_SHOT_RESULTS.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.savez_compressed(HERE/'V152_ZERO_SHOT_PREDICTIONS_INTERNAL.npz',**preds)
    with (HERE/'V152_HARMONIZED_METRICS.csv').open('w',encoding='utf-8',newline='') as f:
        ww=csv.DictWriter(f,fieldnames=['cohort','model']+list(result['cohorts']['primary']['metrics']['composition']))
        ww.writeheader()
        for cc,v in result['cohorts'].items():
            for k,m in v['metrics'].items():ww.writerow(dict(cohort=cc,model=k,**m))
    print(json.dumps({c:v['contrasts'] for c,v in result['cohorts'].items()},indent=2),flush=True)

if __name__=='__main__':main()
