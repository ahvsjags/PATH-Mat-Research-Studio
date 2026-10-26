#!/usr/bin/env python3
"""Post-publication, frozen V136 challenge on the PNNL iron-oxide validation rows.

The paper's Tables 1-2 list 23 rows (Exp. 1-17 and 19-24); the article text
states that 24 experiments were run. This script scores only the 23 rows
actually printed in those tables. It never fits or calibrates a model.
"""
from __future__ import annotations

import csv, hashlib, json, math, pickle, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).with_name("ironoxide_pnnl_validation_23.csv")
MODEL_DIR = ROOT / "PATH-Mat_Evidence_Strengthening_V173" / "external_sources" / "ASTRAL" / "frozen_v136_models"
OUT = ROOT / "PATH-Mat_Evidence_Strengthening_V226" / "results"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_HASHES = {
    "B0_composition": "4a1ccd2ae6f6a51cf743a41da5b79ff7cec6f8688dbc47e001ab4dbfdd59d19e",
    "B1_process_bag": "364c85b24bc0c4d6d3a5651969bfc6bf3e0ad59e4af4a1fc905e5e3a9ae0ad3e",
    "B2_ordered_process": "f401a42cdfe20e71d3a879d961312934b13edcd82848071a7d2817ad36b1e0f6",
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def record_from_row(row: pd.Series) -> dict:
    precursors = ["FeCl3", "NaOH"]
    if row.additive == "SDS":
        precursors.append("C12H25NaO4S")
    elif row.additive == "sodium citrate":
        precursors.append("Na3C6H5O7")
    # Procedure read from the source paper: dissolve/stir, adjust pH, then
    # transfer to autoclave and hold. pH/concentration are recorded in the CSV
    # but are outside V136's feature contract and are therefore not injected.
    actions = [
        {"name": "solution mixing"},
        {"name": "pH adjustment"},
        {"name": "hydrothermal heating", "temperature_C": float(row.temp_C),
         "duration_h": float(row.duration_h)},
    ]
    return {"target_formula": "Fe2O3", "precursor_formulas": precursors, "actions": actions}

def manual_score(obj: dict, records: list[dict], features: dict):
    names = list(obj["feature_names"]); index = {n:i for i,n in enumerate(names)}
    fn = features[obj["feature_function"]]
    X = np.zeros((len(records), len(names)), dtype=float)
    present, missing = set(), set()
    for ri, rec in enumerate(records):
        for k,v in fn(rec).items():
            if k in index:
                X[ri,index[k]] += float(v); present.add(k)
            else: missing.add(k)
    logits = np.clip(X @ obj["model"].coef_[0] + float(obj["model"].intercept_[0]), -700, 700)
    raw = 1/(1+np.exp(-logits))
    cal = np.interp(raw, obj["calibrator"].X_thresholds_, obj["calibrator"].y_thresholds_)
    return raw, cal, {"model_feature_count":len(names), "raw_feature_keys_seen":len(present|missing),
        "matched_keys":len(present), "key_coverage":len(present)/max(1,len(present|missing)),
        "unmatched_raw_feature_keys":sorted(missing), "unique_feature_rows":int(np.unique(X,axis=0).shape[0])}

def safe_auc(y, s):
    return None if len(np.unique(y)) < 2 else float(roc_auc_score(y,s))

def row_bootstrap(y, s, seed=22617, draws=20000):
    # Descriptive sample-level interval only: it cannot represent between-lab
    # or between-material uncertainty because this is one source/system.
    rng=np.random.default_rng(seed); vals_ap=[]; vals_auc=[]
    n=len(y)
    for _ in range(draws):
        ix=rng.integers(0,n,n); yy=y[ix]
        if len(np.unique(yy))<2: continue
        vals_ap.append(float(average_precision_score(yy,s[ix])))
        vals_auc.append(float(roc_auc_score(yy,s[ix])))
    def ci(a): return [float(np.quantile(a,.025)),float(np.quantile(a,.975))] if a else [None,None]
    return {"bootstrap_unit":"individual printed experimental row; descriptive only",
            "draws_requested":draws,"draws_valid":len(vals_ap),"seed":seed,
            "AP_CI95":ci(vals_ap),"AUROC_CI95":ci(vals_auc)}

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    sys.path.insert(0, str(MODEL_DIR))
    from PATH_FEATURES import FEATURES
    data=pd.read_csv(DATA)
    assert len(data)==23 and data.experiment.tolist()==list(range(1,18))+list(range(19,25))
    assert set(data.observed_nonhematite.unique())=={0,1}
    records=[record_from_row(r) for r in data.itertuples(index=False)]
    scored=data.copy(); model_summaries={}; model_paths={}
    y=data.observed_nonhematite.to_numpy(dtype=int)
    for name, expected in MODEL_HASHES.items():
        p=MODEL_DIR/f"{name}.pkl"; digest=sha256(p); assert digest==expected,(name,digest)
        model_paths[name]=digest
        with p.open("rb") as f: obj=pickle.load(f)
        raw,cal,support=manual_score(obj,records,FEATURES)
        scored[name+"_raw_score"]=raw; scored[name+"_calibrated_score"]=cal
        score=cal
        model_summaries[name]={"n":len(y),"positive_nonhematite":int(y.sum()),"prevalence":float(y.mean()),
            "average_precision":float(average_precision_score(y,score)),"prevalence_baseline_AP":float(y.mean()),
            "AUROC":safe_auc(y,score),"Brier_vs_physical_binary_endpoint":float(brier_score_loss(y,score)),
            "unique_calibrated_scores":int(np.unique(score).size),"score_sd":float(np.std(score,ddof=1)),
            "score_min":float(score.min()),"score_max":float(score.max()),"feature_support":support}
        model_summaries[name]["row_bootstrap_descriptive"] = row_bootstrap(y,score)
    scored.to_csv(OUT/"V226_PNNL_IRONOXIDE_FROZEN_PREDICTIONS.csv",index=False)
    brier_prevalence=float(np.mean((y-y.mean())**2))
    result={"schema":"pathmat_v226_pnnl_ironoxide_frozen_external_physical_endpoint",
        "analysis_status":"post_publication_exploratory_frozen_model_challenge; outcome table was public before scoring",
        "source":"Liu et al. (2023), arXiv:2303.11244, Tables 1-2; source model's explicitly described experimental validation subset",
        "table_reconciliation":{"article_text_claims_24_experiments":True,"rows_printed_in_Tables_1_2":23,
            "printed_experiments":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22,23,24],
            "missing_printed_experiment_number":18,"analysis_includes_only_printed_rows":True},
        "endpoint":{"positive": "observed non-hematite phase in XRD table, relative to prespecified Fe2O3/hematite target",
            "negative":"reported Hem phase","n":len(data),"positive_n":int(y.sum()),"prevalence":float(y.mean()),"prevalence_Brier":brier_prevalence},
        "feature_mapping":{"target_formula":"Fe2O3 for every route (hematite target)",
            "precursor_formulas":"FeCl3 and NaOH; plus SDS=C12H25NaO4S or sodium citrate=Na3C6H5O7 where reported",
            "ordered_actions":"solution mixing -> pH adjustment -> hydrothermal heating at table-reported temperature/duration",
            "available_but_unencoded_fields":["pH","FeCl3 concentration","additive concentration","solution volume","autoclave metadata"],
            "post-reaction wash/centrifugation excluded from synthesis-input actions":True},
        "freeze":{"models":"original V136 B0/B1/B2; no fit, recalibration, threshold adjustment, or feature tuning",
            "model_sha256":model_paths,"prediction_data_sha256":sha256(DATA),"software_sklearn":__import__('sklearn').__version__},
        "results":model_summaries,
        "claim_boundary":{"source_independent_physical_challenge":"evaluated without refitting on a separately reported laboratory experiment table; same chemistry family and one lab/source",
            "not_blind":True,"not_formula_novelity_test":True,"not_order_test":"all routes share the same reported action chronology; no order contrast",
            "not_causal_policy_test":True,"not_universal_generalization":True}}
    (OUT/"V226_PNNL_IRONOXIDE_FROZEN_AUDIT.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"models":{k:{a:v for a,v in s.items() if a in ["average_precision","prevalence_baseline_AP","AUROC","Brier_vs_physical_binary_endpoint","unique_calibrated_scores","score_sd"]} for k,s in model_summaries.items()},
        "endpoint":result["endpoint"],"support":{k:v["feature_support"] for k,v in model_summaries.items()}},indent=2))

if __name__=="__main__": main()
