"""Recompute descriptive published order-by-context contrasts.

Inputs are published summary means, not route-level observations. No batch-level
uncertainty is inferred where source SDs or replicate identity are unavailable.
AgNP rows are concentration-confounded and remain sensitivity evidence only.
"""
from __future__ import annotations
import csv,hashlib,json
from collections import defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent
INPUT=HERE/'AUNP_ORDER_CONTEXT_SOURCE_VALUES_V206.csv'
OUTPUT=HERE/'AUNP_ORDER_CONTEXT_AUDIT_V206.json'
def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()
rows=list(csv.DictReader(INPUT.open(encoding='utf-8-sig',newline='')))
groups=defaultdict(dict)
for row in rows:
    key=(row['study'],row['system'],row['endpoint'],row['context_value'])
    for label in (row['order_a'],row['order_b']):
        assert label not in groups[key],f'duplicate order {label}: {key}'
    groups[key][row['order_a']]=row
    groups[key][row['order_b']]=row
contrasts=[]
for (study,system,endpoint,context),cells in sorted(groups.items()):
    first=next(iter(cells.values())); second=cells[first['order_b']]
    effect=round(float(first['mean_a_nm'])-float(first['mean_b_nm']),4)
    contrasts.append({'study':study,'system':system,'endpoint':endpoint,'context':context,
      'contrast':f"{first['order_a']} - {first['order_b']}",'effect_nm':effect,
      'final_composition_matched':first['final_composition_matched'].lower()=='true',
      'independent_repeat_n':first['independent_repeat_n'],
      'replicate_note':first['replicate_note'],'source_doi':first['source_doi']})
by_context=defaultdict(dict)
for item in contrasts:
    by_context[(item['study'],item['system'],item['endpoint'])][item['context']]=item['effect_nm']
interactions=[]
for (study,system,endpoint),effects in sorted(by_context.items()):
    if study=='Salloum2025':
        low,high=effects['5:1'],effects['17:1']; label='17:1 minus 5:1 order contrast'
    else:
        low,high=effects['slow'],effects['fast']; label='fast minus slow feed/mixing order contrast'
    interactions.append({'study':study,'system':system,'endpoint':endpoint,
      'interaction_definition':label,'slow_or_low_context_order_effect_nm':round(low,4),
      'fast_or_high_context_order_effect_nm':round(high,4),
      'difference_in_differences_nm':round(high-low,4),
      'matched_final_composition':all(x['final_composition_matched'] for x in contrasts
        if x['study']==study and x['system']==system and x['endpoint']==endpoint),
      'inference':'descriptive arithmetic contrast of published means; no inferential interval computed'})
expected={
 ('Salloum2025','AuNP','TEM particle diameter','5:1'):1.0,
 ('Salloum2025','AuNP','TEM particle diameter','17:1'):-8.0,
 ('Besenhard2018','AuNP','TEM particle diameter','slow'):1.7,
 ('Besenhard2018','AuNP','TEM particle diameter','fast'):-3.1,
 ('Besenhard2018','AgNP','TEM particle diameter','slow'):1.6,
 ('Besenhard2018','AgNP','TEM particle diameter','fast'):-4.8,
 ('Besenhard2018','AuNP','UV-Vis surface-plasmon peak wavelength','slow'):0.8,
 ('Besenhard2018','AuNP','UV-Vis surface-plasmon peak wavelength','fast'):-0.6,
}
for x in contrasts:
    assert x['effect_nm']==expected[(x['study'],x['system'],x['endpoint'],x['context'])],x
expected_interactions={
 ('Salloum2025','AuNP','TEM particle diameter'):-9.0,
 ('Besenhard2018','AuNP','TEM particle diameter'):-4.8,
 ('Besenhard2018','AgNP','TEM particle diameter'):-6.4,
 ('Besenhard2018','AuNP','UV-Vis surface-plasmon peak wavelength'):-1.4,
}
for x in interactions:
    assert x['difference_in_differences_nm']==expected_interactions[(x['study'],x['system'],x['endpoint'])],x
result={'version':'V206','purpose':'recompute descriptive order-by-context point contrasts from published nanoparticle summary means',
 'input':INPUT.name,'input_sha256':sha256(INPUT),'sources':[
 {'doi':'10.1039/D5NR02727F','type':'peer-reviewed article','scope':'AuNP TEM size by citrate:gold ratio'},
 {'doi':'10.1039/C8CE01014E','type':'peer-reviewed article and SI','scope':'AuNP TEM size and UV-Vis peak wavelength by feed/mixing regime; AgNP sensitivity'}],
 'contrasts':contrasts,'interactions':interactions,
 'interpretation':{'two_independent_Au_studies_show_context_dependent_TEM_size_reversal':True,
  'Besenhard_AuNP_UVVis_endpoint_shows_same_within_study_sign_reversal':True,
  'Besenhard_AuNP_UVVis_replicate_species_attribution_ambiguous_due_to_SI_caption_parenthetical':True,
  'Besenhard_AuNP_UVVis_numeric_SD_values_not_recovered':True,
  'no_batch_level_CI_or_p_value_calculated':True,
  'AgNP_final_concentrations_mismatched':True,'AgNP_isolated_order_effect_claim':False,
  'does_not_evaluate_PATH_Mat_or_establish_PATH_Mat_policy':True}}
OUTPUT.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({'output':str(OUTPUT),'sha256':sha256(OUTPUT),'contrasts':contrasts,'interactions':interactions},indent=2,ensure_ascii=False))
