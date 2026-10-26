"""Recompute published held-out metrics without loading a legacy pickle."""
import argparse, csv, gzip, hashlib, json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score

ap=argparse.ArgumentParser();ap.add_argument('--predictions',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
args=ap.parse_args()
expected_sha='5ef2d8f93514ecd2e92de0aec0ed4248de5f03798c741902eeadb722a4832966'
assert hashlib.sha256(args.predictions.read_bytes()).hexdigest()==expected_sha,'Frozen prediction snapshot mismatch'
with gzip.open(args.predictions,'rt',encoding='utf-8') as f:rows=list(csv.DictReader(f))
y=np.asarray([int(r['label']) for r in rows]);assert len(rows)==5880 and int(y.sum())==1288
assert len(set(r['group_id'] for r in rows))==2769
names=['B0_composition','B1_process_bag','B2_ordered_process','B3_guarded_process'];thresholds=[.28,.26,.27,.28]
expected_ap=[.2669631952989753,.31243648913351163,.3183302812789369,.32024769244170415]
out={}
for name,t,reference in zip(names,thresholds,expected_ap):
    p=np.asarray([float(r[name]) for r in rows]);assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
    auprc=float(average_precision_score(y,p));assert abs(auprc-reference)<1e-12
    bins=np.minimum((p*12).astype(int),11);ece=sum(np.mean(bins==j)*abs(p[bins==j].mean()-y[bins==j].mean()) for j in range(12) if np.any(bins==j))
    out[name]={'auprc':auprc,'auroc':float(roc_auc_score(y,p)),'macro_f1':float(f1_score(y,p>=t,average='macro')),'brier':float(brier_score_loss(y,p)),'ece12':float(ece),'threshold':t}
payload={'n':len(rows),'positive':int(y.sum()),'components':2769,'fixed_prediction_metrics':out,
         'scope':'Recomputes frozen-score metrics; does not claim same-version legacy model reconstruction'}
args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(payload,indent=2),encoding='utf-8')
print(json.dumps(payload,indent=2))
