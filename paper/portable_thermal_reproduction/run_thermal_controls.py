"""Reproducible correction of the profile discretization comparator; see protocol."""
from __future__ import annotations
import csv, hashlib, json, os, pathlib, platform, re, sys, time
from collections import Counter
import numpy as np
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedGroupKFold

ROOT = pathlib.Path(__file__).resolve().parents[1]
CODEDIR = pathlib.Path(__file__).resolve().parent
OUT = pathlib.Path(os.environ.get('PATHMAT_ANALYSIS_OUT', str(CODEDIR / 'outputs')))
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / 'PATH-Mat_Evidence_Strengthening_V150'))
import sequence_controls_core as old
v = old.v
v.LEDGER = pathlib.Path(os.environ['PATHMAT_PUBLIC_LEDGER'])
if v.sha(v.LEDGER) != 'f8b8aabb5b8718ad912075b9aa10373a5e81659b496348f14c926e432eb42dac':
    raise RuntimeError('The source snapshot differs from the reported ledger')
SEED = 14829
N_BOOT = 10000

original_phase_metrics = v.phase_metrics

def phase_formula(name):
    prefix = str(name).split('_')[0]
    formula = re.sub(r'\(\d{3,}\)$', '', prefix)
    return formula, v.reduced(formula)

def corrected_phase_metrics(s, case):
    pfs = case.get('phase_weights') or {}
    if not pfs: return None
    values = np.asarray(list(pfs.values()), float)
    if not np.isfinite(values).all() or np.any(values < 0) or values.sum() <= 0:
        raise ValueError('invalid active refinement weights')
    precursor_formulas = {v.reduced(p.get('formula_clean') or p.get('formula')) for p in s.get('precursors') or []}
    precursor_formulas.discard(None)
    if not precursor_formulas: raise ValueError('unparsed precursor endpoint')
    residual = unknown = 0.; matched = []
    for name, w in pfs.items():
        _, f = phase_formula(name)
        if f is None: unknown += float(w)
        elif f in precursor_formulas: residual += float(w); matched.append(name)
    return residual/values.sum(), values.sum(), unknown/values.sum(), matched

v.phase_metrics = corrected_phase_metrics

def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def audit_source(rows):
    samples = json.loads(v.LEDGER.read_text(encoding='utf-8'))['samples']
    audit = Counter(); weights = []; raw_counts = []; discrepancy = []; mapping = {}; changed = []; fingerprints = Counter()
    for s in samples:
        case = v.active_case(s)
        if case is None: continue
        ver = case.get('verification') or {}
        if ver.get('is_accepted') is not True or ver.get('human_quality_score') not in (1, 2): continue
        pfs = case.get('phase_weights') or {}
        vals = np.asarray(list(pfs.values()), float)
        audit['accepted_refinements'] += 1
        audit['negative_weight_cases'] += int(np.any(vals < 0))
        audit['nonfinite_weight_cases'] += int(not np.isfinite(vals).all())
        weights.append(float(vals.sum()))
        old_metric = original_phase_metrics(s, case); new_metric = corrected_phase_metrics(s, case)
        audit['old_positive_unparsed_weight_cases'] += int(old_metric[2] > 0)
        audit['corrected_positive_unparsed_weight_cases'] += int(new_metric[2] > 0)
        if abs(new_metric[0]-old_metric[0]) > 1e-12:
            changed.append({'sample_id': s['sample_id'], 'old_residual_fraction': old_metric[0], 'corrected_residual_fraction': new_metric[0]})
        for name, weight in pfs.items():
            f, red = phase_formula(name)
            mapping[name] = {'phase_identifier':name, 'parsed_formula':f, 'reduced_formula':red,
                             'terminal_catalogue_suffix_removed':f != name.split('_')[0]}
        prof = v.get_profile(s)
        if prof:
            ordered, legacy, _, _ = prof
            discrepancy.append(float(np.max(np.abs(np.sort(ordered) - legacy))))
            raw_counts.append(len((s.get('synthesis') or {})['temperature_profile']['temperature_celsius']))
            pr=s['synthesis']['temperature_profile']
            fingerprint=hashlib.sha256(json.dumps([pr['time_minutes'],pr['temperature_celsius']],separators=(',',':')).encode()).hexdigest()
            fingerprints[fingerprint] += 1
    available = [r for r in rows if np.isfinite(r['ordered']).all()]
    for r in available:
        assert np.array_equal(np.sort(r['ordered']), r['exact_sorted'])
    (OUT/'V151_PHASE_IDENTIFIER_MAPPING.json').write_text(json.dumps(list(mapping.values()),indent=2),encoding='utf-8')
    with (OUT/'V151_ENDPOINT_CORRECTION_DIFFERENCES.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['sample_id','old_residual_fraction','corrected_residual_fraction']);w.writeheader();w.writerows(changed)
    return {
        **dict(audit), 'phase_weight_sum_range': [min(weights), max(weights)],
        'phase_weight_sum_quantiles': np.quantile(weights, [0, .25, .5, .75, 1]).tolist(),
        'profile_raw_measurement_count_range': [min(raw_counts), max(raw_counts)],
        'legacy_vs_exact_sorted_nonidentical_profiles': int(np.sum(np.asarray(discrepancy) > 1e-10)),
        'legacy_vs_exact_sorted_max_abs_celsius_quantiles': np.quantile(discrepancy, [0, .25, .5, .75, .95, 1]).tolist(),
        'corrected_exact_multiset_assertions_passed': len(available),
        'endpoint_changed_records':len(changed), 'maximum_absolute_endpoint_correction':max(abs(x['corrected_residual_fraction']-x['old_residual_fraction']) for x in changed),
        'distinct_full_profile_fingerprints':len(fingerprints), 'records_with_shared_profile':sum(n for n in fingerprints.values() if n>1),
        'largest_profile_fingerprint_group':max(fingerprints.values()),
    }

def folds_for(rows, scheme):
    groups = np.asarray([r['group'] for r in rows], object)
    y = np.asarray([r['y'] for r in rows], float)
    if scheme == 'formula_cv':
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
        folds = [{'name': f'formula_fold_{i+1}', 'train': tr, 'test': te}
                 for i, (tr, te) in enumerate(cv.split(np.zeros(len(y)), (y > 0).astype(int), groups))]
    elif scheme == 'precursor_lopo':
        folds = [{'name': z['precursor'], 'train': z['train'], 'test': z['test']} for z in old.make_folds(rows)]
    else:
        parent={i:i for i in range(len(rows))}
        def find(i):
            while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
            return i
        seen={}
        for i,r in enumerate(rows):
            for feature in [('target',r['group']),('profile',r['profile_hash'])]:
                if feature in seen:parent[find(i)]=find(seen[feature])
                else:seen[feature]=i
        splitgroups=np.asarray([str(find(i)) for i in range(len(rows))])
        for r,group in zip(rows,splitgroups):r['strict_group']=group
        cv=StratifiedGroupKFold(5,shuffle=True,random_state=SEED)
        folds=[{'name':f'profile_formula_fold_{i+1}','train':tr,'test':te} for i,(tr,te) in enumerate(cv.split(np.zeros(len(y)),(y>0).astype(int),splitgroups))]
    counts = np.zeros(len(rows), int)
    for f in folds:
        tr, te = f['train'], f['test']; counts[te] += 1
        assert not set(groups[tr]) & set(groups[te])
        assert not set(tr) & set(te)
        if scheme == 'precursor_lopo':
            assert all(f['name'] not in rows[i]['precs'] for i in tr)
            assert all(f['name'] in rows[i]['precs'] for i in te)
        if scheme=='profile_formula_cv':
            assert not {rows[i]['profile_hash'] for i in tr} & {rows[i]['profile_hash'] for i in te}
    assert np.all(counts == (2 if scheme == 'precursor_lopo' else 1))
    return folds, counts

def conditional_bootstrap(y, groups, control, ordered):
    ids, inv = np.unique(groups, return_inverse=True)
    diff = np.abs(y - control) - np.abs(y - ordered)
    sums = np.bincount(inv, weights=diff); counts = np.bincount(inv)
    rng = np.random.default_rng(15129)
    draws = np.empty(N_BOOT)
    for start in range(0, N_BOOT, 100):
        ix = rng.integers(0, len(ids), (min(100, N_BOOT-start), len(ids)))
        draws[start:start+len(ix)] = sums[ix].sum(axis=1) / counts[ix].sum(axis=1)
    return {
        'mae_gain_exact_sorted_minus_chronological': float(diff.mean()),
        'ci95_conditional_formula_bootstrap': np.quantile(draws, [.025, .975]).tolist(),
        'ci99_5_bonferroni_ten_contrasts_conditional': np.quantile(draws, [.0025, .9975]).tolist(),
        'draws': N_BOOT, 'seed': 15129,
        'median_record_absolute_error_gain': float(np.median(diff)),
        'fraction_records_error_reduced': float(np.mean(diff > 0)),
        'record_error_gain_quantiles': np.quantile(diff, [.05, .25, .5, .75, .95]).tolist(),
    }

def main():
    start = time.time()
    rows = old.extract_rows()
    raw_by_id = {str(s['sample_id']): s for s in json.loads(v.LEDGER.read_text(encoding='utf-8'))['samples']}
    # Re-extract eligible identifiers explicitly and verify row alignment, not iteration guesses.
    ids = []
    for s in raw_by_id.values():
        case = v.active_case(s)
        if case is None: continue
        ver = case.get('verification') or {}
        precs = {v.reduced(p.get('formula_clean') or p.get('formula')) for p in s.get('precursors') or []}; precs.discard(None)
        if ver.get('is_accepted') is True and ver.get('human_quality_score') in (1, 2) and v.phase_metrics(s, case) is not None and s.get('target_compound') and len(precs) >= 2:
            ids.append(str(s['sample_id']))
    assert len(ids) == len(rows) == 1005
    assert len(set(ids)) == len(ids)
    for r, sid in zip(rows, ids):
        s = raw_by_id[sid]
        assert r['group'] == v.reduced(s['target_compound'])
        assert r['y'] == v.phase_metrics(s, v.active_case(s))[0]
        r['id'] = sid
        profile=(s.get('synthesis') or {}).get('temperature_profile') or {}
        r['profile_hash']=hashlib.sha256(json.dumps([profile.get('time_minutes'),profile.get('temperature_celsius')],separators=(',',':')).encode()).hexdigest()
        r['exact_sorted'] = np.sort(r['ordered'])
        # Original base layout: comp236 + inputchem12 + condition2 + summary12.
        r['clean_base'] = np.r_[r['base'][:-12], r['base'][-12]]
    audit = audit_source(rows)
    payload = {'schema': 'pathmat_v151_exact_profile_controls',
               'source_sha256': sha(v.LEDGER), 'protocol_sha256': sha(CODEDIR / 'V151_PREANALYSIS_PROTOCOL.md'),
               'script_sha256': sha(__file__), 'addendum_sha256':sha(CODEDIR/'V151_ENDPOINT_AND_PROFILE_DEPENDENCE_ADDENDUM.md'), 'source_doi': '10.5281/zenodo.21285546',
               'status': 'post-inspection exploratory correction, target-source supervised fitting; not zero-shot transfer',
               'audit': audit, 'results': {},
               'uncertainty': 'fixed-prediction formula-group bootstrap except strict profile/formula component bootstrap; excludes refit/source uncertainty; overlapping LOPO fits',
               'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__}}
    table = []
    for dataset in ('legacy_base_all', 'order_blind_base_complete'):
        rr = rows if dataset == 'legacy_base_all' else [r for r in rows if np.isfinite(r['ordered']).all()]
        base = np.vstack([r['base' if dataset == 'legacy_base_all' else 'clean_base'] for r in rr])
        y = np.asarray([r['y'] for r in rr], float); g = np.asarray([r['group'] for r in rr], str)
        variants = {'exact_sorted': np.c_[base, np.vstack([r['exact_sorted'] for r in rr])],
                    'chronological': np.c_[base, np.vstack([r['ordered'] for r in rr])]}
        schemes=('formula_cv', 'precursor_lopo') if dataset=='legacy_base_all' else ('formula_cv','precursor_lopo','profile_formula_cv')
        for scheme in schemes:
            folds, counts = folds_for(rr, scheme)
            splitgroups=np.asarray([r['strict_group'] for r in rr],str) if scheme=='profile_formula_cv' else g
            key = dataset + '__' + scheme
            (OUT / f'V151_FOLDS_{key}.json').write_text(json.dumps({
                'ids': [r['id'] for r in rr], 'target_formula_groups': g.tolist(),'bootstrap_groups':splitgroups.tolist(),'profile_hashes':[r['profile_hash'] for r in rr],
                'folds': [{'name': f['name'], 'train': f['train'].tolist(), 'test': f['test'].tolist()} for f in folds],
                'formula_overlap': 0, 'held_precursor_overlap': 0 if scheme == 'precursor_lopo' else None}, indent=2), encoding='utf-8')
            shared=sum(sum(r['profile_hash'] in {rr[i]['profile_hash'] for i in f['train']} for r in [rr[i] for i in f['test']] if np.isfinite(r['ordered']).all()) for f in folds)
            payload['results'][key] = {'n': len(rr), 'target_formula_groups': len(set(g)), 'bootstrap_groups':len(set(splitgroups)), 'folds': len(folds), 'complete_profile_row_fold_predictions_with_train_profile_overlap':shared,'models': {}}
            baseline = np.zeros(len(rr))
            for f in folds: baseline[f['test']] += y[f['train']].mean()
            baseline /= counts
            for model in ('ridge', 'extra_trees'):
                mats=dict(variants)
                if dataset=='legacy_base_all' and scheme=='formula_cv':
                    mats.update({'composition':base[:,:236],'composition_input_chemistry':base[:,:248],'condition_summary':base})
                sums = {z: np.zeros(len(rr)) for z in mats}; fold_results = []
                for fi, f in enumerate(folds):
                    fp = {}
                    for z, X in mats.items():
                        est = old.estimator(model); est.fit(X[f['train']], y[f['train']])
                        pp = np.clip(est.predict(X[f['test']]), 0, 1); sums[z][f['test']] += pp; fp[z] = pp
                    fold_results.append({'fold': f['name'], 'n_train': len(f['train']), 'n_test': len(f['test']),
                                         'mae_gain': float(mean_absolute_error(y[f['test']], fp['exact_sorted']) - mean_absolute_error(y[f['test']], fp['chronological']))})
                    if (fi+1) % 10 == 0 or fi+1 == len(folds): print(f'{key} {model} {fi+1}/{len(folds)}', flush=True)
                pred = {z: p/counts for z,p in sums.items()}
                result = {z: {'mae': float(mean_absolute_error(y,p)), 'rmse': float(mean_squared_error(y,p)**.5)} for z,p in pred.items()}
                result['training_fold_mean_baseline_mae'] = float(mean_absolute_error(y,baseline))
                result['contrast'] = conditional_bootstrap(y, splitgroups, pred['exact_sorted'], pred['chronological'])
                if 'composition' in pred:
                    result['additional_process_contrasts']={name:conditional_bootstrap(y,g,pred[a],pred[b]) for name,a,b in [('thermal_vs_input_chemistry','composition_input_chemistry','condition_summary'),('input_chemistry_vs_composition','composition','composition_input_chemistry'),('process_vs_composition','composition','condition_summary')]}
                result['relative_mae_reduction_percent'] = 100 * result['contrast']['mae_gain_exact_sorted_minus_chronological'] / result['exact_sorted']['mae']
                result['fold_results'] = fold_results
                payload['results'][key]['models'][model] = result
                np.savez_compressed(OUT / f'V151_PREDICTIONS_{key}_{model}.npz', sample_id=np.asarray([r['id'] for r in rr]), target_formula=g, bootstrap_group=splitgroups, y=y, training_fold_mean=baseline, **pred)
                table.append({'dataset': dataset, 'split': scheme, 'model': model, 'n':len(rr),
                              'exact_sorted_mae': result['exact_sorted']['mae'], 'chronological_mae': result['chronological']['mae'],
                              'mean_baseline_mae':result['training_fold_mean_baseline_mae'], **result['contrast'],
                              'relative_mae_reduction_percent':result['relative_mae_reduction_percent']})
                payload['runtime']['elapsed_seconds'] = round(time.time()-start,2)
                (OUT/'V151_EXACT_PROFILE_RESULTS.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    with (OUT/'V151_EXACT_PROFILE_COMPARISONS.csv').open('w', encoding='utf-8', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    print(json.dumps({'status':'complete', 'audit':audit,'results':table}, indent=2),flush=True)

if __name__ == '__main__': main()
