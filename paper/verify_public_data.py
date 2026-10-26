from pathlib import Path
import csv,json,gzip

ROOT=Path(__file__).resolve().parent
def main():
    p=ROOT/'data/fixed_budget/S124_formula_level_source_data.csv'
    with p.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==432
    b=sum(int(x['baseline_unique_accepted_basins']) for x in rows)
    c=sum(int(x['candidate_unique_accepted_basins']) for x in rows)
    delta=[int(x['delta_candidate_minus_baseline']) for x in rows]
    assert (b,c,sum(delta))==(1572,1589,17)
    assert (sum(x>0 for x in delta),sum(x==0 for x in delta),sum(x<0 for x in delta))==(17,415,0)
    summary=json.loads((ROOT/'data/figure7/Figure7_summary.json').read_text())
    assert (summary['cod_distinct'],summary['all_three_energy_screen'],summary['joint_structure_chgnet_screen'])==(43,28,24)
    assert len(list((ROOT/'data/figure7/structure_cifs').glob('*.cif')))==6
    print(json.dumps({'formulae':len(rows),'baseline_basins':b,'policy_basins':c,'net_gain':sum(delta),'candidate_summary_count':summary['cod_distinct']},indent=2))
if __name__=='__main__':main()
