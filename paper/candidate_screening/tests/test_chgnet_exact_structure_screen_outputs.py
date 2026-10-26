import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


MODULE = Path(__file__).resolve().parents[1]
RUN = MODULE / "runs" / "chgnet_exact_structure_screen_20261017_v1"


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def test_chgnet_exact_structure_screen_is_complete_and_lineage_checked():
    summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    rows = read_csv(RUN / "candidate_chgnet_screen.csv")
    routes = read_csv(RUN / "pathmat_routes_for_chgnet_pass_candidates.csv")

    assert len(rows) == summary["n_candidates"] == 43
    assert len({r["candidate_id"] for r in rows}) == 43
    assert summary["n_candidate_force_converged"] == 43
    assert summary["n_reference_force_converged"] == 43
    assert sum(r["screen_pass"] == "True" for r in rows) == summary["target_gate"]["n_pass"] == 24
    assert summary["n_candidate_relaxations_collapsed_to_source_structure"] == 0
    assert summary["n_relaxed_candidates_matching_exact_formula_cod_structure"] == 0
    assert summary["n_exact_formula_cod_references_compared"] == 6

    for row in rows:
        assert row["candidate_force_converged"] == "True"
        assert row["source_force_converged"] == "True"
        relaxed = RUN / row["relaxed_candidate_cif"]
        assert relaxed.is_file()
        assert digest(relaxed) == row["relaxed_candidate_cif_sha256"]

    assert len(routes) == 72
    counts = Counter(r["candidate_id"] for r in routes)
    passed_ids = {r["candidate_id"] for r in rows if r["screen_pass"] == "True"}
    assert set(counts) == passed_ids
    assert set(counts.values()) == {3}
    assert all(r["model_abstain"] == "True" for r in routes)
