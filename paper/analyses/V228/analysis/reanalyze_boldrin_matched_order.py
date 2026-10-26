from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)
MG = [0.1, 0.5, 1, 2, 3, 4, 5, 6]
RU = [0.05, 0.1, 0.2]
# Duplicate XRD-well Scherrer particle sizes (nm), nested Mg -> Ru -> well.
# Source: Boldrin et al. SI Table S3.
PRE2 = [
 [[12.79,11.66],[10.51,12.34],[11.07,9.94]], [[10.03,8.90],[11.77,9.80],[11.56,10.81]],
 [[9.81,11.89],[11.28,10.31],[8.94,8.77]], [[10.39,9.73],[12.63,11.96],[11.53,10.89]],
 [[12.60,10.38],[10.08,10.96],[8.58,9.64]], [[9.63,10.02],[11.48,12.15],[9.40,9.19]],
 [[12.27,10.60],[9.35,8.86],[10.99,10.83]], [[8.75,10.82],[9.21,9.68],[8.03,10.00]]]
PRE3 = [
 [[14.92,14.13],[11.71,13.61],[12.46,13.15]], [[14.41,12.50],[13.90,12.10],[11.48,11.80]],
 [[11.97,12.07],[12.41,13.39],[12.82,14.36]], [[12.83,12.02],[13.63,12.73],[13.55,12.19]],
 [[12.99,14.38],[13.92,12.15],[12.60,12.41]], [[12.59,10.36],[10.84,12.08],[11.22,10.09]],
 [[10.76,12.62],[12.05,11.90],[13.21,11.67]], [[11.57,13.11],[11.76,13.20],[11.51,10.66]]]
# Source: Boldrin et al. SI Table S4.
POST2 = [
 [[15.92,14.09],[12.17,9.41],[11.94,9.36]], [[12.49,11.97],[8.80,8.67],[9.78,10.75]],
 [[10.44,12.51],[11.37,9.96],[12.85,12.85]], [[10.69,9.75],[10.51,12.20],[12.50,12.48]],
 [[11.09,12.51],[10.41,10.85],[11.05,10.97]], [[10.98,11.00],[11.31,12.29],[11.64,12.18]],
 [[13.77,12.92],[13.26,10.02],[12.83,12.54]], [[11.80,11.29],[11.06,10.85],[10.26,8.07]]]
POST3 = [
 [[14.35,16.21],[16.04,15.59],[15.97,14.37]], [[15.22,15.22],[16.34,16.03],[12.92,14.87]],
 [[16.87,16.02],[13.04,10.71],[15.68,16.19]], [[14.11,15.99],[18.24,15.96],[14.73,15.74]],
 [[15.20,15.18],[13.90,12.51],[16.51,14.85]], [[14.37,15.98],[11.83,13.72],[15.57,14.57]],
 [[12.41,14.94],[14.44,14.06],[12.74,14.37]], [[13.12,12.71],[15.59,15.89],[14.78,14.55]]]

def avg(x):
    a=np.asarray(x,float)
    assert a.shape==(8,3,2), a.shape
    return a.mean(axis=2)

def summarize(d,rng):
    z=d.ravel()
    lo,hi=stats.t.interval(.95,len(z)-1,loc=float(z.mean()),scale=float(stats.sem(z)))
    mg_means=d.mean(axis=1)
    mg_lo,mg_hi=stats.t.interval(.95,len(mg_means)-1,loc=float(mg_means.mean()),scale=float(stats.sem(mg_means)))
    # Mg-cluster leave-one-out and bootstrap; all three Ru strata remain together.
    loo=np.array([np.delete(mg_means,i).mean() for i in range(len(mg_means))])
    idx=rng.integers(0,8,size=(100000,8))
    boot=d[idx,:].mean(axis=(1,2))
    return {
      "mean_difference_nm":float(z.mean()), "median_difference_nm":float(np.median(z)),
      "paired_cell_t_95ci_nm":[float(lo),float(hi)],
      "mg_cluster_t_95ci_nm":[float(mg_lo),float(mg_hi)],
      "mg_cluster_bootstrap_95ci_nm":[float(q) for q in np.quantile(boot,[.025,.975])],
      "mg_clusters_lower_in_ooa2":int((mg_means<0).sum()),"mg_clusters_total":len(mg_means),
      "leave_one_mg_cluster_out_mean_range_nm":[float(loo.min()),float(loo.max())],
      "cells_lower_in_ooa2":int((z<0).sum()),"cells_total":len(z),
      "direction_counts_are_descriptive":True}

def main():
    pre2,pre3,post2,post3=map(avg,[PRE2,PRE3,POST2,POST3])
    dpre=pre2-pre3; dpost=post2-post3
    growth2=post2-pre2; growth3=post3-pre3; dgrowth=growth2-growth3
    rng=np.random.default_rng(20261016)
    rows=[]
    for i,mg in enumerate(MG):
      for j,ru in enumerate(RU):
        rows.append({
          "Mg_wt_pct":mg,"Ru_wt_pct":ru,
          "OoA2_pre_well1_nm":PRE2[i][j][0],"OoA2_pre_well2_nm":PRE2[i][j][1],"OoA2_pre_mean_nm":pre2[i,j],
          "OoA3_pre_well1_nm":PRE3[i][j][0],"OoA3_pre_well2_nm":PRE3[i][j][1],"OoA3_pre_mean_nm":pre3[i,j],"pre_OoA2_minus_OoA3_nm":dpre[i,j],
          "OoA2_post_well1_nm":POST2[i][j][0],"OoA2_post_well2_nm":POST2[i][j][1],"OoA2_post_mean_nm":post2[i,j],
          "OoA3_post_well1_nm":POST3[i][j][0],"OoA3_post_well2_nm":POST3[i][j][1],"OoA3_post_mean_nm":post3[i,j],"post_OoA2_minus_OoA3_nm":dpost[i,j],
          "OoA2_growth_nm":growth2[i,j],"OoA3_growth_nm":growth3[i,j],"growth_OoA2_minus_OoA3_nm":dgrowth[i,j]})
    out_csv=OUT/"V228_BOLDRIN_OOA2_VS_OOA3_MATCHED_CELLS.csv"
    with out_csv.open("w",newline="",encoding="utf-8") as f:
      w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    audit={
      "schema":"pathmat_v228_boldrin_matched_addition_order_reanalysis",
      "source":{"article":"Boldrin et al., Proxy-based accelerated discovery of Fischer–Tropsch catalysts, Chemical Science (2015), DOI 10.1039/C4SC02116A",
        "article_url":"https://pubs.rsc.org/en/content/articlelanding/2015/sc/c4sc02116a",
        "supporting_information_url":"https://www.rsc.org/suppdata/sc/c4/c4sc02116a/c4sc02116a1.pdf",
        "data_tables":["Supporting Information Table S3 (pre-ageing XRD)","Supporting Information Table S4 (post-ageing XRD)"]},
      "contrast":{"OoA2":"Mg -> 120 °C drying -> Co -> 120 °C drying -> Ru; common final 120 °C drying and 250 °C calcination",
        "OoA3":"Co -> 120 °C drying -> Ru -> 120 °C drying -> Mg; common final 120 °C drying and 250 °C calcination",
        "matched_factors":"20 wt% Co; same Mg and Ru loading in every pair; neither arm has 550 °C inter-step calcination",
        "factorial_cells":24,"mg_clusters":8,"ru_levels_per_mg":3,"unique_preparations_per_protocol":24,"technical_xrd_wells_per_preparation":2},
      "estimand":"Mean difference in XRD-derived Scherrer Co particle size across the tested 8 Mg × 3 Ru grid, OoA2 minus OoA3; duplicate wells averaged within each unique preparation.",
      "results":{"pre_age_particle_size":summarize(dpre,rng),"post_100h_age_particle_size":summarize(dpost,rng),
        "ageing_growth_post_minus_pre":summarize(dgrowth,rng),
        "grand_means_nm":{"OoA2_pre":float(pre2.mean()),"OoA3_pre":float(pre3.mean()),
          "OoA2_post":float(post2.mean()),"OoA3_post":float(post3.mean()),
          "OoA2_growth":float(growth2.mean()),"OoA3_growth":float(growth3.mean()),
          "pre_size_reduction_percent_vs_ooa3":float((pre3.mean()-pre2.mean())/pre3.mean()*100),
          "post_size_reduction_percent_vs_ooa3":float((post3.mean()-post2.mean())/post3.mean()*100),
          "growth_reduction_percent_vs_ooa3":float((growth3.mean()-growth2.mean())/growth3.mean()*100)}},
      "bootstrap":{"method":"percentile cluster bootstrap over 8 Mg-load clusters, retaining all 3 Ru strata","draws":100000,"seed":20261016},
      "limitations":["Retrospective, post hoc reanalysis of published SI data.",
        "Duplicate XRD wells are technical measurements, averaged before analysis; they are not independent synthesis replicates.",
        "Intervals describe variation across the tested factorial grid, not a population of catalyst systems.",
        "The source does not document randomized assignment or independent batch replication for each cell.",
        "The contrast isolates Mg-before-Co/Ru versus Co/Ru-before-Mg within the shared 120 °C drying schedule, but does not establish a universal order rule.",
        "This does not evaluate PATH-Mat predictions or a PATH-Mat-generated intervention."]}
    (OUT/"V228_BOLDRIN_MATCHED_ORDER_AUDIT.json").write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(audit["results"],indent=2))
    print("Wrote",out_csv)
if __name__=="__main__": main()

