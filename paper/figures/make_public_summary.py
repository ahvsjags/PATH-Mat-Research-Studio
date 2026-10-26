"""Plot only authorized aggregate candidate counts and fixed-budget search results."""
from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
def main():
    summary=json.loads((ROOT/'data/figure7/Figure7_summary.json').read_text())
    with (ROOT/'data/fixed_budget/S124_formula_level_source_data.csv').open(encoding='utf-8-sig')as f:rows=list(csv.DictReader(f))
    assert len(rows)==432
    b=sum(int(x['baseline_unique_accepted_basins'])for x in rows)
    c=sum(int(x['candidate_unique_accepted_basins'])for x in rows)
    assert (b,c)==(1572,1589)
    fig,axs=plt.subplots(1,3,figsize=(10,3),layout='constrained')
    labels=['Proposed','Geometry','COD-distinct','Three-potential','Joint screen']
    values=[summary[k]for k in ['design_proposals','geometry_retained','cod_distinct','all_three_energy_screen','joint_structure_chgnet_screen']]
    axs[0].barh(labels[::-1],values[::-1],color='#188d89');axs[0].set_xlabel('Candidate count')
    axs[1].bar(['Static FIRE','Failure-aware'],[b,c],color=['#71838a','#188d89']);axs[1].set_ylabel('Unique accepted basins')
    m=summary['mean_delta']
    for y,key in enumerate(['pair_ci','two_way_ci']):
        lo,hi=summary[key];axs[2].errorbar(m,y,xerr=[[m-lo],[hi-m]],fmt='o',color='#188d89',capsize=3)
    axs[2].axvline(0,color='#71838a',linestyle='--');axs[2].set_yticks([0,1],['Pair clusters','A-site × pair clusters']);axs[2].set_xlabel('Additional basins per formula')
    for ax in axs:
        ax.spines[['top','right']].set_visible(False)
    out=ROOT/'generated_figures';out.mkdir(exist_ok=True)
    for ext in ['pdf','svg','png']:fig.savefig(out/f'public_summary.{ext}',dpi=300)
if __name__=='__main__':main()
