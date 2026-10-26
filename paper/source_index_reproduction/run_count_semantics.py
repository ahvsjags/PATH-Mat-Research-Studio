"""All fixed response branches, both score scales, same exact primary cohort."""
from __future__ import annotations
import collections,gzip,hashlib,json,pathlib,re,sys,time,warnings
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score
H=pathlib.Path(__file__).resolve().parent;R=H.parent
import os
import v153_helpers as comp
import v153_helpers as clean
def common_type(s):
    s=str(s).lower()
    if s in ('heating','heat'): return 'heat'
    if s in ('mixing','grinding','mix'): return 'mix'
    if s in ('shaping','pressing','deposition','shape'): return 'shape'
    return 'other'


class WeightedAP:
    def __init__(self,y,p):
        self.idx=np.argsort(-p,kind='stable');self.y=y[self.idx]
        ps=p[self.idx];self.end=np.r_[np.flatnonzero(ps[1:]!=ps[:-1]),len(ps)-1]
    def __call__(self,w):
        w=w[self.idx];tp=np.cumsum(w*self.y)[self.end];tot=np.cumsum(w)[self.end]
        if not tp[-1]: return np.nan
        return float(np.sum(np.diff(np.r_[0,tp]) * np.divide(tp,tot,out=np.zeros_like(tp),where=tot>0))/tp[-1])


warnings.filterwarnings('ignore',message='No Pauling electronegativity')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 t=time.time();_,sr1,rows,ih,ph,_,_=comp.old.load_locked();ind=comp.old.split_indices(rows,None)
 y=np.array([sr1.label(r) for r in rows],int);te=ind['D_test'];group=np.array([rows[i]['group_id'] for i in te])
 rawpath=comp.required_input('PATHMAT_LEE_RAW_GZ');assert digest(rawpath)==clean.EXPECTED_RAW
 raw=json.load(gzip.open(rawpath,'rt',encoding='utf-8'))
 by={hashlib.sha256(json.dumps(r,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest():r for r in raw}
 nums=[];cnt=[];words=[];native=[];common=[]
 for r in rows:
  rr=by[r['source']['content_hash']];aa=rr.get('conditions_forDOI') or []
  assert rr['DOI']==r['source']['source_doi'] and len(aa)==len(r.get('actions') or [])
  start=[a for a in aa if a.get('act_type')=='Starting'];ref=[a for a in aa if a.get('act_type')!='Starting' and a.get('ref_act')]
  direct=[a for a in aa if a.get('act_type')!='Starting' and not a.get('ref_act')]
  assert len(aa)==len(start)+len(ref)+len(direct)
  cnt.append([len(aa),len(start),len(ref),len(direct)])
  tt=[x for a in aa for x in clean.numeric(a.get('temp_values'),'temperature')];dd=[x for a in aa for x in clean.numeric(a.get('time_values'),'duration')]
  nums.append([max(tt) if tt else np.nan,np.mean(tt) if tt else np.nan,max(dd) if dd else np.nan,np.mean(dd) if dd else np.nan])
  words.append(sum(len(re.findall(r'\w+',str(a.get('subject') or '')+' '+str(a.get('act_token') or ''))) for a in aa))
  native.append({k:v/max(1,len(aa)) for k,v in collections.Counter(str(a.get('act_type') or 'Unknown') for a in aa).items()})
  cc=collections.Counter(common_type(a.get('act_type')) for a in direct);common.append({k:cc[k]/max(1,len(direct)) for k in ('heat','mix','shape','other')})
 del raw,by
 cnt=np.array(cnt,int);values=np.array(nums,float);pres=np.isfinite(values[:,[0,2]]).astype(float)
 med=np.nanmedian(values[ind['D_dev']],axis=0);assert np.isfinite(med).all()
 values=np.where(np.isfinite(values),values,med);values[:,:2]/=1500;values[:,2:]/=24
 audit={'source_n':len(rows),'direct_count_definition':'Non-Starting and false/empty ref_act','category_order':['total','Starting','non_Starting_reference','direct'],
 'mutually_exclusive_counts_sum_verified':True,'counts':{},'records_with_non_starting_reference':int((cnt[:,2]>0).sum()),'correlations_pearson':np.corrcoef(cnt.T).tolist()}
 for part,ix in [('all',np.arange(len(rows))),('test',te)]:
  audit['counts'][part]={k:{'sum':int(cnt[ix,j].sum()),'zero_rows':int((cnt[ix,j]==0).sum()),'quantiles_0_25_50_75_95_100':np.quantile(cnt[ix,j],[0,.25,.5,.75,.95,1]).tolist()} for j,k in enumerate(audit['category_order'])}
 (H/'V153_COUNT_SEMANTICS_AUDIT.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
 names=['composition','total_count','direct_count','excluded_count','direct_taxonomy','word_presence','native_conditions']
 def feat(i,name):
  f=comp.extended(sr1,rows[i])
  if name=='composition':return f
  if name in ('total_count','word_presence','native_conditions'):f['action_count']=float(cnt[i,0])
  if name in ('direct_count','direct_taxonomy'):f['direct_count']=float(cnt[i,3])
  if name=='excluded_count':f['excluded_count']=float(cnt[i,1]+cnt[i,2])
  if name=='direct_taxonomy':f.update({'common_action_proportion='+k:v for k,v in common[i].items()})
  if name in ('word_presence','native_conditions'):
   f['operation_subject_token_words']=float(words[i]);f['source_temperature_present']=pres[i,0];f['source_duration_present']=pres[i,1]
  if name=='native_conditions':
   f.update({'native_action_proportion='+k:v for k,v in native[i].items()})
   f.update({k:float(values[i,j]) for j,k in enumerate(('max_temperature','mean_temperature','max_duration','mean_duration'))})
  return f
 metrics={};pred={};fits=H/'count_semantics_models_internal';fits.mkdir(exist_ok=True)
 import pickle
 for name in names:
  vec=DictVectorizer(sparse=True);xd=vec.fit_transform([feat(i,name) for i in ind['D_dev']]);xc=vec.transform([feat(i,name) for i in ind['D_cal']]);xt=vec.transform([feat(i,name) for i in te])
  for family in (('logistic',) if os.environ.get('PATHMAT_REPRO_LOGISTIC_ONLY')=='1' else ('logistic','extra_trees')):
   est=LogisticRegression(max_iter=600,solver='liblinear',class_weight='balanced',random_state=31415) if family=='logistic' else ExtraTreesClassifier(n_estimators=200,min_samples_leaf=10,max_features=.8,class_weight='balanced',n_jobs=8,random_state=31415)
   est.fit(xd,y[ind['D_dev']]);iso=IsotonicRegression(out_of_bounds='clip').fit(est.predict_proba(xc)[:,1],y[ind['D_cal']])
   raw=est.predict_proba(xt)[:,1];cal=iso.predict(raw)
   key=family+'__'+name;metrics[key]={'features':xd.shape[1]}
   for scale,p in [('raw',raw),('calibrated',cal)]:
    pred[key+'__'+scale]=p;metrics[key][scale]={'ap':float(average_precision_score(y[te],p)),'auroc':float(roc_auc_score(y[te],p)),'distinct_scores':len(np.unique(p)),'score_sd':float(np.std(p))}
   with (fits/(key+'.pkl')).open('wb') as fp:pickle.dump({'vectorizer':vec,'model':est,'calibrator':iso},fp)
   print(key,json.dumps(metrics[key]),flush=True)
   del est,iso
  del xd,xc,xt,vec
 audit['source_reconstruction_verified_by_input_digest']=True
 pairs=[('total_count','composition'),('direct_count','composition'),('excluded_count','composition'),('direct_count','total_count'),('direct_taxonomy','direct_count'),('native_conditions','word_presence')]
 specs=[(fam,a,b,scale) for fam in (('logistic',) if os.environ.get('PATHMAT_REPRO_LOGISTIC_ONLY')=='1' else ('logistic','extra_trees')) for scale in ('raw','calibrated') for a,b in pairs]
 funcs={k:WeightedAP(y[te],p) for k,p in pred.items()};ug,inv=np.unique(group,return_inverse=True);rng=np.random.default_rng(1532031);draws=[]
 for z in range(10000):
  w=np.bincount(rng.integers(0,len(ug),len(ug)),minlength=len(ug))[inv].astype(float)
  if np.dot(w,y[te])==0 or np.dot(w,1-y[te])==0:continue
  ap={k:f(w) for k,f in funcs.items()};draws.append([ap[fam+'__'+a+'__'+s]-ap[fam+'__'+b+'__'+s] for fam,a,b,s in specs])
 draws=np.array(draws);contrasts=[]
 for j,(fam,a,b,s) in enumerate(specs):
  contrasts.append({'classifier':fam,'scale':s,'contrast':a+' minus '+b,'delta_ap':metrics[fam+'__'+a][s]['ap']-metrics[fam+'__'+b][s]['ap'],
   'ci95':np.quantile(draws[:,j],[.025,.975]).tolist(),'bonferroni24':np.quantile(draws[:,j],[.05/48,1-.05/48]).tolist()})
 out={'status':'post-review exploratory matched fixed fits','source_sha256':ih,'partition_sha256':ph,'protocol_sha256':digest(H/'V153_REVIEW_RESPONSE_ANALYSIS_PROTOCOL.md'),
 'metrics':metrics,'contrasts':contrasts,'test_n':len(te),'test_positive':int(y[te].sum()),'test_components':len(ug),'draws':len(draws),'seed':1532031,'local_family_size':24,
 'conditions_imputation_development_medians':med.tolist(),'count_audit':audit,'elapsed_seconds':round(time.time()-t,2)}
 np.savez_compressed(H/'V153_COUNT_SEMANTICS_PREDICTIONS_INTERNAL.npz',route_id=np.array([rows[i]['route_id'] for i in te]),group_id=group,y=y[te],**pred)
 (H/'V153_COUNT_SEMANTICS_RESULTS.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps({'contrasts':contrasts,'seconds':out['elapsed_seconds']}),flush=True)
if __name__=='__main__':main()
