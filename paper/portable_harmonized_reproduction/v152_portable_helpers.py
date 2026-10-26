from __future__ import annotations
import collections,csv,gzip,hashlib,json,pathlib,re,sys,time,functools,os
from collections import Counter,defaultdict
from types import SimpleNamespace
import numpy as np
from pymatgen.core import Composition,Element
from sklearn.metrics import f1_score
ELEMENT=re.compile(r"[A-Z][a-z]?")
EXPECTED_RAW='59fb0608e594721a4055fb6a623f180651f2d7aef737a5d4e13db63595b039c0'
EXPECTED_INPUT='4cc3210fd0d782e26cfc75ecfb84dd1afb12c4d34c545174704b21f90a3b30e8'
EXPECTED_PART='5e5fc8c40ed530803dfdbff4881837556bbc0b242c787b7e780ce6fe4f17a7d7'
unit_stats=collections.Counter()
def required_input(key):
    value=os.environ.get(key)
    if not value: raise RuntimeError('Supply an authorized input path using '+key)
    p=pathlib.Path(value).resolve()
    if not p.is_file():raise RuntimeError('Input does not exist: '+key)
    return p
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def base(record):
 o={};target=str(record.get("target_formula") or "");te=sorted(set(ELEMENT.findall(target)));pre=record.get("precursor_formulas") or []
 for x in te:o["target_element="+x]=1.0
 o["target_element_count"]=float(len(te));o["precursor_count"]=float(len(pre))
 for p in pre:
  for x in sorted(set(ELEMENT.findall(str(p)))):o["precursor_element="+x]=o.get("precursor_element="+x,0)+1
 return o

def label(r):return int(bool(r.get("competing_phases")) or r.get("reported_outcome")=="reported_impurity")

def choose_threshold(y,p):
 grid=np.arange(.05,.951,.01);return float(max(grid,key=lambda t:f1_score(y,(p>=t).astype(int),average="macro",labels=[0,1],zero_division=0)))

class DSU:
    def __init__(self):
        self.p: dict[str, str] = {}

    def add(self, x: str) -> None:
        self.p.setdefault(x, x)

    def find(self, x: str) -> str:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)

def assignments(rows: list[dict]) -> tuple[dict[str, dict], str, dict]:
    dsu = DSU()
    for r in rows:
        source = r.get("source") if isinstance(r.get("source"), dict) else {}
        doi = str(source.get("source_doi") or "unknown")
        target = str(r.get("target_formula") or "unknown")
        dsu.add("D:" + doi); dsu.add("T:" + target); dsu.union("D:" + doi, "T:" + target)
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        source = r.get("source") if isinstance(r.get("source"), dict) else {}
        groups[dsu.find("D:" + str(source.get("source_doi") or "unknown"))].append(r)
    out: dict[str, dict] = {}
    part_counts: Counter[str] = Counter()
    component_sizes = []
    doi_parts: dict[str, set[str]] = defaultdict(set)
    formula_parts: dict[str, set[str]] = defaultdict(set)
    for rs in groups.values():
        nodes = []
        for r in rs:
            source = r.get("source") if isinstance(r.get("source"), dict) else {}
            nodes.extend(("D:" + str(source.get("source_doi") or "unknown"), "T:" + str(r.get("target_formula") or "unknown")))
        canonical = min(nodes)
        b = int(hashlib.sha256(("PATH-Mat-SR1-source-native-v1|" + canonical).encode()).hexdigest()[:12], 16) % 100
        part = "D_dev" if b < 60 else "D_cal" if b < 75 else "D_select" if b < 85 else "D_test"
        gid = hashlib.sha256(canonical.encode()).hexdigest()
        component_sizes.append(len(rs)); part_counts[part] += len(rs)
        for r in rs:
            route_id = str(r.get("route_id") or "")
            if not route_id or route_id in out:
                raise RuntimeError("Missing or duplicate route_id while rebuilding the primary partition")
            source = r.get("source") if isinstance(r.get("source"), dict) else {}
            doi = str(source.get("source_doi") or "unknown")
            target = str(r.get("target_formula") or "unknown")
            out[route_id] = {"route_id": route_id, "group_id": gid, "partition": part, "doi": doi, "target_formula": target}
            doi_parts[doi].add(part); formula_parts[target].add(part)
    lines = [json.dumps({k: v for k, v in out[rid].items() if k in {"route_id", "group_id", "partition"}}, sort_keys=True) + "\n" for rid in sorted(out)]
    digest = hashlib.sha256("".join(lines).encode()).hexdigest()
    if any(len(p) > 1 for p in doi_parts.values()) or any(len(p) > 1 for p in formula_parts.values()):
        raise RuntimeError("DOI or target formula crosses a reconstructed partition")
    summary = {
        "partition_counts": dict(sorted(part_counts.items())),
        "component_count": len(component_sizes),
        "singleton_components": sum(n == 1 for n in component_sizes),
        "largest_component_rows": max(component_sizes),
        "doi_cross_partition_count": sum(len(p) > 1 for p in doi_parts.values()),
        "target_formula_cross_partition_count": sum(len(p) > 1 for p in formula_parts.values()),
    }
    return out, digest, summary

def numeric(items,kind):
    out=[]
    for d in items or []:
        if not isinstance(d,dict): continue
        unit=str(d.get('units') or '').strip().lower().replace(' ','')
        unit_stats[kind+'::'+unit]+=1
        vals=d.get('values') or [d.get('max') if d.get('max') is not None else d.get('min')]
        for val in vals:
            try:x=float(val)
            except (ValueError,TypeError):continue
            if not np.isfinite(x):continue
            if kind=='temperature':
                if unit in ['°c','c','℃','celsius']:pass
                elif unit in ['k','°k','kelvin']:x-=273.15
                else:unit_stats['temperature_unsupported_numeric_values']+=1;continue
                if x < -273.15 or x > 10000:unit_stats['temperature_invalid_range_values']+=1;continue
            else:
                if unit in ['h','hr','hrs','hour','hours']:pass
                elif unit in ['min','mins','minute','minutes']:x/=60
                elif unit in ['s','sec','secs','second','seconds']:x/=3600
                elif unit in ['d','day','days']:x*=24
                else:unit_stats['duration_unsupported_numeric_values']+=1;continue
                if x<0 or x>100000:unit_stats['duration_invalid_range_values']+=1;continue
            out.append(x)
    return out

PROPS = np.array([[z/118, float(Element.from_Z(z).atomic_mass)/300,
                   float(Element.from_Z(z).X)/4 if Element.from_Z(z).X is not None and np.isfinite(Element.from_Z(z).X) else np.nan] for z in range(1,119)])

@functools.lru_cache(maxsize=None)
def atomic_fraction(formula):
    out = np.zeros(118)
    try:
        c=Composition(formula)
        if c.num_atoms <= 0: raise ValueError('nonpositive atom count')
        for e,n in c.get_el_amt_dict().items(): out[Element(e).Z-1]=n/c.num_atoms
        if not np.isfinite(out).all() or np.any(out<0): raise ValueError('invalid fractions')
        return out, False
    except Exception: return out, True

def extended(sr1, r):
    feat=sr1.base(r)
    target,tbad=atomic_fraction(str(r.get('target_formula') or ''))
    pp=[atomic_fraction(str(p)) for p in r.get('precursor_formulas') or []]
    prec=np.mean([p[0] for p in pp],axis=0) if pp else np.zeros(118)
    feat['target_formula_parse_failed']=float(tbad)
    feat['precursor_formula_parse_failed_fraction']=float(np.mean([p[1] for p in pp])) if pp else 1.
    for name,a in [('target',target),('precursor_mean',prec)]:
        for z,x in enumerate(a):
            if x: feat[f'{name}_atomic_fraction_Z{z+1}']=float(x)
        for j,prop in enumerate(['atomic_number','atomic_mass','electronegativity']):
            ok=np.isfinite(PROPS[:,j]); total=a[ok].sum()
            mean=float(np.sum(a[ok]*PROPS[ok,j])/total) if total>0 else 0.
            var=float(np.sum(a[ok]*(PROPS[ok,j]-mean)**2)/total) if total>0 else 0.
            feat[f'{name}_{prop}_mean']=mean;feat[f'{name}_{prop}_sd']=var**.5
            feat[f'{name}_{prop}_missing_fraction']=float(a[~ok].sum())
    return feat

def load_locked():
    p=required_input('PATHMAT_PRIMARY_JSONL');ih=sha(p)
    assert ih==EXPECTED_INPUT, 'Not the exact conflict-filtered primary input'
    rows=[]
    with p.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():continue
            r=json.loads(line);s=r.get('source') or {}
            if s.get('source_id')=='lee2025_impurities' and r.get('route_id') and s.get('source_doi') and r.get('target_formula'): rows.append(r)
    assert len(rows)==70636
    parts,ph,summary=assignments(rows)
    assert ph==EXPECTED_PART
    assert summary['partition_counts']=={'D_cal':5901,'D_dev':55082,'D_select':3773,'D_test':5880}
    for r in rows:r['group_id'],r['partition']=parts[r['route_id']]['group_id'],parts[r['route_id']]['partition']
    return None,sys.modules[__name__],rows,ih,ph,None,summary
def split_indices(rows,unused):
    return {p:np.asarray([i for i,r in enumerate(rows) if r['partition']==p],dtype=int) for p in ('D_dev','D_cal','D_select','D_test')}
old=SimpleNamespace(load_locked=load_locked,split_indices=split_indices)
