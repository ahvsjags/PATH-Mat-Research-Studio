from __future__ import annotations
import csv, json
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]
OUT = MODULE / "runs" / "published_dft_stability_20261017_v1"


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def test_published_dft_screen_reconciles_candidate_set_and_scope() -> None:
    summary = json.loads((OUT / "pathmat_candidate_published_dft_stability.json").read_text(encoding="utf-8"))
    rows = read_csv(OUT / "pathmat_candidate_published_dft_stability.csv")
    assert len(rows) == summary["n_cod_distinct_generated_candidates"] == 43
    assert len({r["formula"] for r in rows}) == 43
    assert summary["n_found_in_source_DFT_near_hull_table"] == 43
    assert summary["n_exactly_on_source_hull"] == 29
    assert summary["source_relaxed_structure_class_counts"] == {"perovskite": 39, "non-perovskite": 4}
    assert summary["source_hull_distance_eV_atom"]["max"] == 0.0277
    assert summary["n_generated_cif_matches_to_any_same_formula_source_POSCAR"] == 0
    assert all(int(r["source_poscar_count_same_formula"]) > 0 for r in rows)
    assert all(r["generated_cif_matches_source_poscar"] == "False" for r in rows)


def test_dft_target_gate_preserves_route_traceability_and_abstentions() -> None:
    summary = json.loads((OUT / "pathmat_candidate_published_dft_stability.json").read_text(encoding="utf-8"))
    shortlist = read_csv(OUT / "pathmat_dft_target_filtered_candidates.csv")
    routes = read_csv(OUT / "pathmat_dft_target_filtered_route_hypotheses.csv")
    assert len(shortlist) == summary["n_dft_target_screen_candidates"] == 39
    assert len(routes) == summary["n_dft_target_screen_route_hypotheses"] == 117
    assert summary["n_dft_target_screen_route_model_abstentions"] == 117
    assert {r["source_relaxed_structure_class"] for r in shortlist} == {"perovskite"}
    assert all(float(r["source_decomposition_energy_above_hull_eV_atom"]) <= 0.05 for r in shortlist)
    assert all(r["model_abstain"].lower() == "true" for r in routes)
    assert all("unseen_source" in r["abstention_reasons"] for r in routes)
    best = [r for r in routes if str(r["within_candidate_rank"]) == "1"]
    assert len(best) == 39
    assert all(r["schedule_id"] == "lower_thermal_budget" for r in best)
