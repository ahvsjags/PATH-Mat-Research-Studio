from __future__ import annotations

"""Render a compact, fully vector supplementary figure after independent audit."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


HERE = Path(__file__).resolve()
PROJECT = HERE.parents[1]
RUN = PROJECT / "runs" / "v280_failure_repair_fresh_b4b4_v1" / "scored"
ANALYSIS = RUN / "analysis_v280_confirmation.json"
AUDIT = RUN / "independent_audit_v280_v4.json"
OUT = PROJECT / "reports" / "figures"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if not ANALYSIS.is_file() or not AUDIT.is_file():
        raise FileNotFoundError("figure requires completed results and an independent audit")
    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if not str(audit.get("status", "")).startswith("INDEPENDENT_V280_AUDIT_PASS"):
        raise RuntimeError("independent recorded-primary audit did not pass; figure will not be released")
    if not audit.get("dual_primary_interval_criterion_met"):
        raise RuntimeError("independent audit did not reproduce the locked primary criterion")
    if audit.get("analysis_json_sha256") != sha(ANALYSIS):
        raise RuntimeError("audited analysis JSON changed")
    rows = audit["per_formula"]
    a_sites = sorted({r["a_site"] for r in rows})
    pairs = sorted({r["pair_cluster"] for r in rows})
    matrix = np.zeros((len(pairs), len(a_sites)), dtype=float)
    pi, ai = {p: i for i, p in enumerate(pairs)}, {a: i for i, a in enumerate(a_sites)}
    for r in rows:
        matrix[pi[r["pair_cluster"]], ai[r["a_site"]]] = r["delta"]
    if matrix.shape != (72, 6):
        raise RuntimeError(f"unexpected grid shape {matrix.shape}")

    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig = plt.figure(figsize=(7.25, 5.35), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.78, 1.22], width_ratios=[1.1, 0.9])
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    # Panel a: frozen decision rule, intentionally schematic rather than data art.
    ax0.set_axis_off()
    ax0.set_xlim(0, 1); ax0.set_ylim(0, 1)
    ax0.text(0.0, 0.96, "a", fontsize=11, fontweight="bold", va="top")
    ax0.text(0.035, 0.96, "Failure-aware candidate selection under an equal four-call budget", fontsize=9, fontweight="bold", va="top")

    def box(x, y, w, h, text, fc, ec="#315767", fontsize=8):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                               facecolor=fc, edgecolor=ec, linewidth=0.85)
        ax0.add_patch(patch)
        ax0.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize, wrap=True)

    def arrow(x1, y1, x2, y2, color="#647984"):
        ax0.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
                                      linewidth=0.8, color=color))

    box(0.02, 0.50, 0.18, 0.27, "Common starts\nclasses 0–2\nFIRE", "#e7f3f1", fontsize=7.5)
    box(0.27, 0.50, 0.20, 0.27, "Gate\nV0 passes; serialized\nforce > 0.050001 eV/Å", "#f5f2e9", fontsize=7.2)
    box(0.56, 0.63, 0.18, 0.20, "Trigger met\nLBFGS continuation\nfrom lowest-index class", "#dcefe9", fontsize=7.0)
    box(0.56, 0.33, 0.18, 0.20, "No trigger\nclass 3 FIRE\nfallback", "#edf0f2", fontsize=7.2)
    box(0.81, 0.50, 0.17, 0.27, "Four slots per arm\nV0 + force gate\nunique-basin matcher", "#edf3f7", fontsize=7.0)
    arrow(0.20, 0.63, 0.27, 0.63); arrow(0.47, 0.65, 0.56, 0.72)
    arrow(0.47, 0.59, 0.56, 0.43); arrow(0.74, 0.72, 0.81, 0.65); arrow(0.74, 0.43, 0.81, 0.58)

    # Panel b: 72 × 6 fully paired cation grid.
    ax1.text(-0.10, 1.04, "b", transform=ax1.transAxes, fontsize=11, fontweight="bold", va="top")
    ax1.set_title("Paired change in unique accepted basins", loc="left", pad=6, fontweight="bold")
    cmap = ListedColormap(["#c76552", "#e9eceb", "#15968f"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    ax1.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax1.set_xticks(range(len(a_sites)), a_sites)
    ax1.set_xlabel("A-site cation")
    ax1.set_ylabel("B/B′ pair cluster (72)")
    ax1.set_yticks([0, 17, 35, 53, 71], ["1", "18", "36", "54", "72"])
    ax1.tick_params(length=0)
    ax1.set_xticks(np.arange(-.5, len(a_sites), 1), minor=True)
    ax1.set_yticks(np.arange(-.5, len(pairs), 1), minor=True)
    ax1.grid(which="minor", color="white", linewidth=0.28)
    ax1.tick_params(which="minor", bottom=False, left=False)
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(color="#15968f", label="+1"), Patch(color="#e9eceb", label="0"),
                        Patch(color="#c76552", label="−1")], title="candidate − baseline",
               loc="upper left", bbox_to_anchor=(0.0, -0.18), ncol=3, frameon=False, fontsize=7, title_fontsize=7)

    # Panel c: exact frozen bootstrap decision.
    ax2.text(-0.12, 1.04, "c", transform=ax2.transAxes, fontsize=11, fontweight="bold", va="top")
    ax2.set_title("Paired basin-count difference", loc="left", pad=6, fontweight="bold")
    labels = ["formula i.i.d.\n(secondary)", "B/B′ pair\n(primary)", "A site × pair\n(primary)"]
    cis = [audit["formula_iid_bootstrap_95_ci_secondary"],
           audit["pair_cluster_bootstrap_95_ci_primary"],
           audit["two_way_a_site_x_pair_cluster_bootstrap_95_ci_primary"]]
    point = float(audit["mean_delta_candidate_minus_baseline_per_formula"])
    ys = np.arange(len(labels))[::-1]
    colors = ["#8b9da5", "#147f7a", "#147f7a"]
    for y, ci, color in zip(ys, cis, colors):
        ax2.plot(ci, [y, y], color=color, linewidth=2.0, solid_capstyle="round")
        ax2.scatter([point], [y], color=color, s=22, zorder=3, edgecolor="white", linewidth=0.4)
    ax2.axvline(0, color="#66777e", linestyle=(0, (3, 2)), linewidth=0.8)
    ax2.set_yticks(ys, labels)
    ax2.set_xlabel("change in accepted basins per formula")
    ax2.grid(axis="x", color="#e1e7e8", linewidth=0.6)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(axis="y", length=0)
    lo = min(-0.15, min(x[0] for x in cis) - 0.03)
    hi = max(0.20, max(x[1] for x in cis) + 0.03)
    ax2.set_xlim(lo, hi)
    ax2.text(0.02, -0.30,
             f"mean = {point:+.3f}; trigger {audit['repair_summary']['triggered_formula_count']}/432; "
             f"LBFGS accepted {audit['repair_summary']['repair_calls_accepted']}",
             transform=ax2.transAxes, fontsize=7, color="#425158")

    fig.suptitle("Failure-aware local relaxation: locked confirmation", fontsize=10, fontweight="bold", x=0.02, ha="left")
    pdf = OUT / "V280_failure_aware_relaxation_confirmation.pdf"
    svg = OUT / "V280_failure_aware_relaxation_confirmation.svg"
    png = OUT / "V280_failure_aware_relaxation_confirmation.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"status": "FIGURE_EXPORTED_AFTER_AUDIT", "pdf_sha256": sha(pdf),
                      "svg_sha256": sha(svg), "png_sha256": sha(png)}, indent=2))


if __name__ == "__main__":
    main()



