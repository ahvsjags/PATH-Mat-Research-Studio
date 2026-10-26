import csv,gzip,hashlib,json,pathlib
def canonical(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'))
def formula(x):return ''.join(str(x or '').split()) or None
def extract(raw,route_id,content_hash):
 tg=raw.get('target') or []
 if isinstance(tg,list):tg=tg[0] if tg else {}
 target=formula(tg.get('material_formula') if isinstance(tg,dict) else tg) or formula(raw.get('target_formula'))
 pp=raw.get('precursors') or raw.get('precursor_formulas') or [];pp=pp if isinstance(pp,list) else [pp]
 prec=sorted(set(p for p in [formula(a.get('material_formula') or a.get('formula')) if isinstance(a,dict) else formula(a) for a in pp] if p))
 acts=[{'name':'Mixing' if a.get('act_type')=='Mixing' else 'Unknown','temperature_C':None,'duration_h':None} for a in raw.get('conditions_forDOI') or []]
 label=int(bool(raw.get('impurity_reaction')) or bool(raw.get('impurity_phase')))
 return {'route_id':route_id,'target_formula':target,'precursor_formulas':prec,'actions':acts,'reported_outcome':'reported_impurity' if label else 'reported_synthesis','source':{'source_id':'lee2025_impurities','source_doi':raw.get('DOI') or raw.get('doi'),'content_hash':content_hash}}
def relevant(r):
 return {'route_id':r['route_id'],'target_formula':r.get('target_formula'),'precursor_formulas':r.get('precursor_formulas') or [],
 'source_doi':r['source']['source_doi'],'label':int(bool(r.get('competing_phases')) or r.get('reported_outcome')=='reported_impurity'),
 'actions':[{'name':a.get('name'),'temperature_C':a.get('temperature_C'),'duration_h':a.get('duration_h')} for a in r.get('actions') or []]}

def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--lee-raw',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);a=ap.parse_args()
 checks=json.loads((pathlib.Path(__file__).parent/'expected_checks.json').read_text(encoding='utf-8'))
 assert hashlib.sha256(a.lee_raw.read_bytes()).hexdigest()==checks['lee_raw_sha256']
 raw=json.load(gzip.open(a.lee_raw,'rt',encoding='utf-8'));selectors=list(csv.DictReader((pathlib.Path(__file__).parent/'primary_source_selector.csv').open(encoding='utf-8')))
 assert len(raw)==80806 and len(selectors)==70636
 assert hashlib.sha256((pathlib.Path(__file__).parent/'primary_source_selector.csv').read_bytes()).hexdigest()==checks['selector_sha256']
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open('w',encoding='utf-8',newline='\n') as fp:
  for s in selectors:
   z=raw[int(s['raw_row_index_zero_based'])];h=hashlib.sha256(canonical(z).encode()).hexdigest();assert h==s['source_content_sha256']
   fp.write(canonical(extract(z,s['route_id'],h))+'\n')
 assert hashlib.sha256(a.out.read_bytes()).hexdigest()==checks['semantic_projection_sha256']
 print(json.dumps({'status':'PASS public source + explicit selection reconstructs exact model-relevant projection','rows':len(selectors),'projection_sha256':checks['semantic_projection_sha256']}))
if __name__=='__main__':main()
