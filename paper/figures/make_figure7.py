"""Figure 7: verified source data, actual CIF projections and integrated modules.

The two numerical cohorts are displayed separately. Existing main figures are
not read or modified. All plotting and structural projections use Python.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, Rectangle
from pymatgen.core import Structure


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
DATA = ROOT / "figures"
OUT = ROOT / "generated_figures"
SOURCE = ROOT / "data" / "fixed_budget"
TEAL, INK, GOLD, RUST = "#188d89", "#183c4c", "#b9872e", "#bc6451"
GREY, LINE = "#71838a", "#cbd6d9"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare() -> tuple[list[dict], list[dict], dict]:
    inputs = DATA / "inputs"
    paths = {
        "design": inputs / "design_summary.json",
        "structure_screen": inputs / "candidate_chgnet_screen.csv",
        "common_geometry": inputs / "common_geometry_energies.csv",
        "tensornet": inputs / "tensornet_energies.csv",
        "confirmation": SOURCE / "failure_aware_relaxation_analysis.json",
        "formula_data": SOURCE / "S124_formula_level_source_data.csv",
    }
    missing = [p.name for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit("Full-panel rendering requires the non-public candidate energy inputs: " + ", ".join(missing) + ". Use make_public_summary.py for the released summaries.")
    design = json.loads(paths["design"].read_text(encoding="utf-8-sig"))
    common = load_csv(paths["common_geometry"])
    tensors = {x["candidate_id"]: x for x in load_csv(paths["tensornet"])}
    screens = {x["candidate_id"]: x for x in load_csv(paths["structure_screen"])}
    assert len(common) == len(tensors) == len(screens) == 43
    assert {x["candidate_id"] for x in common} == set(tensors) == set(screens)
    candidates = []
    for row in common:
        tensor, screen = tensors[row["candidate_id"]], screens[row["candidate_id"]]
        assert row["candidate_relaxed_cif_sha256"] == tensor["candidate_relaxed_cif_sha256"]
        assert row["reference_relaxed_cif_sha256"] == tensor["reference_relaxed_cif_sha256"]
        vals = [float(row["chgnet_singlepoint_delta_meV_atom"]),
                float(row["m3gnet_singlepoint_delta_meV_atom"]),
                float(tensor["tensornet_singlepoint_delta_meV_atom"])]
        assert np.isfinite(vals).all()
        candidates.append({"candidate_id": row["candidate_id"], "formula": row["formula"],
            "chgnet_delta_meV_atom": vals[0], "m3gnet_delta_meV_atom": vals[1],
            "tensornet_delta_meV_atom": vals[2],
            "all_three_le_50_meV_atom": all(v <= 50 for v in vals),
            "joint_structure_chgnet_screen": screen["screen_pass"] == "True"})
    candidates.sort(key=lambda x: (max(x["chgnet_delta_meV_atom"], x["m3gnet_delta_meV_atom"],
                                         x["tensornet_delta_meV_atom"]), x["candidate_id"]))
    formulae = load_csv(paths["formula_data"])
    analysis = json.loads(paths["confirmation"].read_text(encoding="utf-8-sig"))
    assert len(formulae) == 432
    baseline = sum(int(x["baseline_unique_accepted_basins"]) for x in formulae)
    candidate = sum(int(x["candidate_unique_accepted_basins"]) for x in formulae)
    repair = [x for x in formulae if x["candidate_fourth_call_method"] == "LBFGS_force_only_continuation"]
    fallback = [x for x in formulae if x["candidate_fourth_call_method"] == "FIRE_catalog_fallback_class3"]
    assert (baseline, candidate, len(repair), len(fallback)) == (1572, 1589, 89, 343)
    assert sum(int(x["delta_candidate_minus_baseline"]) for x in repair) == 17
    assert all(int(x["delta_candidate_minus_baseline"]) == 0 for x in fallback)
    assert sum(x["all_three_le_50_meV_atom"] for x in candidates) == 28
    assert sum(x["joint_structure_chgnet_screen"] for x in candidates) == 24
    counts = design["counts"]
    assert (counts["composition_pairs_enumerated"], counts["geometry_screened_candidates"]) == (72,45)
    summary = {"design_proposals": 72, "geometry_retained": 45, "cod_distinct": 43,
        "all_three_energy_screen": 28, "joint_structure_chgnet_screen": 24,
        "search_compositions": 432, "baseline_basins": baseline, "candidate_basins": candidate,
        "continuation_count": 89, "fallback_count": 343, "gains":17,"ties":415,"losses":0,
        "mean_delta": analysis["mean_delta_candidate_minus_baseline_per_formula"],
        "pair_ci": analysis["pair_cluster_bootstrap_95_ci_primary"],
        "two_way_ci": analysis["two_way_a_site_x_pair_cluster_bootstrap_95_ci_primary"],
        "source_hashes": {k: {"file": str(p), "sha256":sha(p)} for k,p in paths.items()}}
    DATA.mkdir(exist_ok=True)
    write_csv(DATA / "Figure7_candidate_energies.csv", candidates)
    write_csv(SOURCE / "Figure7_candidate_energies.csv", candidates)
    (DATA / "Figure7_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    (SOURCE / "Figure7_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    return candidates, formulae, summary


def panel(fig, rect, letter, title):
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    fig.text(rect[0]-.024, rect[1]+rect[3]+.012, letter, weight="bold", size=10.0, color=INK)
    fig.text(rect[0], rect[1]+rect[3]+.012, title, weight="bold", size=8.0, color=INK)
    return ax


def box(ax, x,y,w,h,text, *, fc="#eef4f5", color=INK, fs=7.1):
    ax.add_patch(Rectangle((x,y),w,h,facecolor=fc,edgecolor=LINE,lw=.65))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs,color=color,linespacing=1.22)


def arrow(ax,x1,y1,x2,y2,*,color=GREY):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=6.5,
                                color=color,lw=.75,shrinkA=0,shrinkB=0))


def draw_workflow(ax):
    ax.set_xlim(0,1);ax.set_ylim(0,1)
    labels=["Designer\nCIF + routes","Verifier\nTool checks","Skeptic\nSpecific issue",
            "Verifier\nConfirm issue","Repairer\nNew artifact","Recheck\nSaved output"]
    xs=np.linspace(0,.86,6)
    for i,(x,label) in enumerate(zip(xs,labels)):
        box(ax,float(x),.36,.14,.40,label,fc="#e9f3f1" if i in (0,4,5) else "#f0f3f5",fs=7.0)
        if i<5:arrow(ax,float(x)+.14,.56,float(xs[i+1]),.56)
    ax.text(.25,.97,"Controller · bounded rounds, repairs and relaxation assignments",ha="center",va="center",size=7.0,color=INK)
    ax.plot([.02,.80],[.86,.86],color=GOLD,lw=1.05)
    box(ax,.30,.015,.60,.20,"Verified memory · store / retrieve; evaluation read-only",fc="#f7f3e8",fs=6.6)
    arrow(ax,.78,.36,.78,.225,color=GOLD)
    # Retrieval is an exposed API, not evidence of a learned policy update.


def draw_screen(ax,s):
    ax.set_xlim(0,1);ax.set_ylim(0,1)
    box(ax,.05,.78,.90,.17,"72 proposed compositions",fs=7.4)
    box(ax,.05,.49,.90,.17,"45 geometry-retained CIFs",fs=7.4)
    box(ax,.05,.20,.90,.17,"43 COD-distinct candidates",fc="#e8f2f0",fs=7.4)
    arrow(ax,.5,.78,.5,.66);arrow(ax,.5,.49,.5,.37)
    ax.text(.02,-.005,"28 / 43",size=8.2,weight="bold",color=TEAL,ha="left",va="top")
    ax.text(.02,-.092,"Three-potential\nenergy screen",size=6.8,ha="left",va="top",color=INK)
    ax.text(.60,-.005,"24 / 43",size=8.2,weight="bold",color=GOLD,ha="left",va="top")
    ax.text(.60,-.092,"Joint structural /\nCHGNet screen",size=6.8,ha="left",va="top",color=INK)


def project_structure(ax,path):
    structure=Structure.from_file(path,primitive=False) * [2,2,2]
    coords=np.asarray(structure.cart_coords)
    corner=np.array([[i,j,k] for i in (0,1) for j in (0,1) for k in (0,1)]) @ structure.lattice.matrix
    center=corner.mean(axis=0)
    a,b=np.deg2rad(33),np.deg2rad(24)
    rz=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    rx=np.array([[1,0,0],[0,np.cos(b),-np.sin(b)],[0,np.sin(b),np.cos(b)]])
    rotation=rx@rz
    xyz=(coords-center)@rotation.T;cell=(corner-center)@rotation.T
    roles={"O":"O"}
    formula=structure.composition.get_el_amt_dict()
    a_site=next(x for x in formula if x in {"Ba","Sr","Ca"})
    b_prime=next(x for x in formula if x in {"Nb","Ta"})
    b_site=next(x for x in formula if x not in {a_site,b_prime,"O"})
    colors={a_site:"#7996a5",b_site:TEAL,b_prime:GOLD,"O":RUST}
    sizes={a_site:18,b_site:13,b_prime:13,"O":7}
    symbols=[s.specie.symbol for s in structure]
    for i in range(len(structure)):
        if symbols[i] not in {b_site,b_prime}:continue
        for j in range(len(structure)):
            if symbols[j]=="O" and 1.2<float(np.linalg.norm(coords[i]-coords[j]))<2.7:
                ax.plot([xyz[i,0],xyz[j,0]],[xyz[i,1],xyz[j,1]],color="#a7b2b4",lw=.36,zorder=1)
    for i in range(8):
        for j in range(i+1,8):
            if sum(abs(np.array([(i>>2)&1,(i>>1)&1,i&1])-np.array([(j>>2)&1,(j>>1)&1,j&1])))==1:
                ax.plot([cell[i,0],cell[j,0]],[cell[i,1],cell[j,1]],color="#526870",lw=.45,zorder=2)
    for k,i in enumerate(np.argsort(xyz[:,2])):
        symbol=symbols[int(i)]
        ax.scatter(xyz[i,0],xyz[i,1],s=sizes[symbol],c=colors[symbol],edgecolors="white",linewidths=.20,zorder=3+k*.01)
    span=np.ptp(cell[:,:2],axis=0).max()*1.12
    ax.set_xlim(-span/2,span/2);ax.set_ylim(-span/2,span/2)
    ax.set_aspect("equal");ax.set_axis_off()


def main():
    rows,formulae,s=prepare()
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["DejaVu Sans","Arial"],
        "font.size":7.0,"axes.labelsize":7.0,"xtick.labelsize":7.0,"ytick.labelsize":7.0,
        "svg.fonttype":"none","pdf.fonttype":42,"ps.fonttype":42,
        "axes.linewidth":.7,"axes.spines.top":False,"axes.spines.right":False})
    fig=plt.figure(figsize=(7.20,8.65),facecolor="white")
    ax=panel(fig,[.07,.845,.88,.105],"a","Integrated candidate-design agent workflow")
    draw_workflow(ax)
    fig.text(.046,.804,"CANDIDATE DESIGN",size=7.5,weight="bold",color=TEAL)
    fig.text(.31,.804,"72 proposed compositions · structure and energy screening",size=7,color=GREY)
    fig.add_artist(plt.Line2D([.046,.95],[.793,.793],transform=fig.transFigure,color=LINE,lw=.65))
    ax=panel(fig,[.07,.565,.245,.178],"b","Candidate screening")
    draw_screen(ax,s)
    c_rect=[.36,.53,.59,.213]
    panel(fig,c_rect,"c","Computed candidate structures")
    files=sorted((ROOT/'data'/'figure7'/'structure_cifs').glob('*.cif'))
    assert len(files)==6
    names=["Sr2NbFeO6","Sr2NbCrO6","Sr2ScTaO6","Sr2TaCoO6","Ba2ScNbO6","Ba2TaCoO6"]
    structure_sources=[]
    subscript=str.maketrans('0123456789','₀₁₂₃₄₅₆₇₈₉')
    for i,name in enumerate(names):
        path=next(p for p in files if ('_'+name+'_') in p.name)
        col,row=i%3,i//3
        rect=[.36+col*.198,.545+(1-row)*.112,.182,.087]
        ax=fig.add_axes(rect);project_structure(ax,path)
        fig.text(rect[0]+rect[2]/2,rect[1]-.008,name.translate(subscript),ha="center",va="top",size=7.0,color=INK)
        structure_sources.append({'formula':name,'file':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'display_supercell':[2,2,2],'source_sites':len(Structure.from_file(path,primitive=False))})
    for i,(label,color) in enumerate([('A site','#7996a5'),('B site',TEAL),('B′ site',GOLD),('O',RUST)]):
        x=.40+i*.13
        fig.add_artist(plt.Line2D([x],[.508],transform=fig.transFigure,marker='o',markersize=3.5,
                                 color=color,linestyle='none'))
        fig.text(x+.012,.508,label,size=6.6,va='center',color=GREY)
    (DATA/'Figure7_structure_sources.json').write_text(json.dumps(structure_sources,indent=2)+'\n',encoding='utf-8')
    # All 43 candidate/reference pairs are displayed, with no outcome-based omission.
    fig.text(.046,.468,"d",weight='bold',size=10,color=INK)
    fig.text(.07,.468,"Relative-energy agreement on the same geometries",weight='bold',size=8,color=INK)
    fig.text(.95,.468,"28 / 43 pass all three screens",ha='right',size=7,color=TEAL)
    energy=np.array([[x[k] for x in rows] for k in ['chgnet_delta_meV_atom','m3gnet_delta_meV_atom','tensornet_delta_meV_atom']])
    ax=fig.add_axes([.205,.351,.70,.091])
    cmap=LinearSegmentedColormap.from_list('relative_energy',['#476f83','#f1f2ee','#b8614e'])
    im=ax.imshow(energy,aspect='auto',interpolation='nearest',cmap=cmap,norm=TwoSlopeNorm(vmin=-100,vcenter=0,vmax=175))
    ax.set_yticks([0,1,2],['CHGNet','M3GNet','TensorNet'])
    ax.set_xticks([0,9,19,29,42],['1','10','20','30','43'])
    ax.tick_params(axis='both',length=0,pad=4)
    ax.set_xlabel('Candidate rank (sorted by maximum relative energy)',labelpad=4)
    for spine in ax.spines.values():spine.set_visible(False)
    gate=fig.add_axes([.205,.447,.70,.008])
    gate.imshow([[int(x['all_three_le_50_meV_atom']) for x in rows]],aspect='auto',interpolation='nearest',
        cmap=LinearSegmentedColormap.from_list('gate',['#e4e9e9',TEAL]),vmin=0,vmax=1)
    gate.set_axis_off()
    cax=fig.add_axes([.923,.351,.012,.091]);cb=fig.colorbar(im,cax=cax,ticks=[-100,0,50,175])
    cb.ax.tick_params(labelsize=6.1,length=2,pad=2);cb.outline.set_linewidth(.5)
    fig.text(.925,.333,'meV/atom',size=6.6,ha='left',va='top',color=GREY)
    fig.text(.046,.291,"LOCKED SEARCH CONFIRMATION",size=7.5,weight='bold',color=TEAL)
    fig.text(.47,.291,"432 compositions · separate cohort",size=7,color=GREY)
    fig.add_artist(plt.Line2D([.046,.95],[.280,.280],transform=fig.transFigure,color=LINE,lw=.65))
    ax=panel(fig,[.07,.070,.425,.172],"e","Failure-aware fourth-slot allocation")
    ax.set_xlim(0,1);ax.set_ylim(0,1)
    box(ax,.02,.75,.96,.19,'Shared FIRE starts · classes 0–2',fs=7.0)
    box(ax,.02,.33,.45,.26,'89 verified triggers\nLBFGS continuation',fc='#e9f3f1',fs=7.0)
    box(ax,.53,.33,.45,.26,'343 no triggers\nClass-3 FIRE',fs=7.0)
    arrow(ax,.27,.75,.24,.59);arrow(ax,.73,.75,.76,.59)
    ax.text(.245,.18,'89 accepted continuations\n17 basin gains; 72 ties',ha='center',va='center',size=6.7,color=TEAL)
    ax.text(.755,.18,'343 ties',ha='center',va='center',size=7.0,color=INK)
    ax.text(.5,-.035,'Four relaxation assignments per formula in each arm',ha='center',va='center',size=6.7,color=GREY)
    fig.text(.565,.254,'f',size=10,weight='bold',color=INK)
    fig.text(.59,.254,'Paired basin-count improvement',size=8,weight='bold',color=INK)
    fig.text(.59,.228,'1,572 → 1,589 unique accepted basins',size=7.3,color=TEAL)
    ax=fig.add_axes([.73,.105,.215,.100])
    point=s['mean_delta']
    for y,ci in zip([1,0],[s['pair_ci'],s['two_way_ci']]):
        ax.plot(ci,[y,y],color=TEAL,lw=1.9,solid_capstyle='round')
        ax.scatter([point],[y],s=23,color=TEAL,edgecolor='white',lw=.4,zorder=3)
    ax.axvline(0,color=GREY,lw=.75,ls=(0,(3,2)))
    ax.set_yticks([1,0],['B/B′ pair\nclusters','A-site × pair\nclusters'])
    ax.tick_params(axis='y',length=0,pad=5,labelsize=6.7)
    ax.set_ylim(-.6,1.6);ax.set_xlim(-.006,.10)
    ax.set_xticks([0,.04,.08]);ax.tick_params(axis='x',length=2.5,pad=3)
    ax.set_xlabel('Additional basins / formula',labelpad=4,size=6.8)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x',color='#e1e7e8',lw=.5)
    fig.text(.59,.047,'17 gains · 415 ties · 0 losses',size=7.2,color=INK)
    fig.text(.59,.027,'Points: mean; intervals: 95% cluster bootstrap',size=6.4,color=GREY)
    OUT.mkdir(parents=True,exist_ok=True)
    stem=OUT/'Figure7_Agent_Design_Recovery_V284'
    fig.savefig(stem.with_suffix('.pdf'))
    fig.savefig(stem.with_suffix('.svg'))
    fig.savefig(stem.with_suffix('.png'),dpi=600)
    fig.savefig(stem.with_suffix('.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
    report={'figure':'Figure 7','dimensions_mm':[182.88,219.71],
        'source_script_sha256':sha(Path(__file__)),'data_summary':s,
        'structure_sources':structure_sources,
        'exports':[{'file':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size}
                   for p in [stem.with_suffix(x) for x in ['.pdf','.svg','.png','.tiff']]],
        'integrity_notes':['All 43 candidate energy rows displayed.','Six rendered structures are actual stored CIFs reused from the supporting candidate illustrations.',
                           'Candidate-design cohort and 432-formula search grid remain separate.',
                           'No new scientific evaluation or independently replicated experiment is implied.']}
    (DATA/'Figure7_provenance.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'FIGURE7_RENDERED','candidate_rows':len(rows),'search_rows':len(formulae),'exports':report['exports']},indent=2))


if __name__=='__main__':
    main()



