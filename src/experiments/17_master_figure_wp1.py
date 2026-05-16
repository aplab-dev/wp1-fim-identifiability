"""WP1 master figure — consolidating the full identifiability storyline.

Single multi-panel figure summarizing the WP1 narrative from §2 to §6.8.
Designed for use as the workshop paper (WP4) hero, the WP1 abstract figure,
and the WP5e Substack/Twitter companion.

Layout (3 rows × 3 cols):

Row 1 (Identifiability):
  [A] 2-pop FIM eigenvalue spectrum (rank 1/4)
  [B] 3-pop FIM eigenvalue spectrum (rank 3/6)
  [C] α-β estimate-correlation (-1.00) — the visceral hook

Row 2 (Schedule + regime):
  [D] Cross-schedule eigenvalue comparison (3-pop)
  [E] 1D regime scan: P(AT50 wins) vs K_TP_drop
  [F] 2D regime scan heatmap

Row 3 (Decision):
  [G] PA vs PE accuracy bar chart
  [H] PA vs PE per-patient scatter (highlight disagreement)
  [I] PA advantage vs K_TP_drop (where the methodology pays off)

Pulls data directly from the JSON summaries committed by experiments
04, 08, 12, 13, 15, 16. Uses globbed timestamped filenames so it picks
up the latest commit's results automatically.

This is publication-grade — figures.png at 300 DPI, PDF for vector
quality. Print-safe colors. Square aspect ratio.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _load_latest_summary(prefix: str) -> dict | None:
    """Find the latest results/{prefix}_<sha>_<date>.json by mtime."""
    pattern = f"{prefix}_*.json"
    candidates = sorted((_REPO_ROOT / "results").glob(pattern),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        log.warning(f"  No file matching {pattern}")
        return None
    log.info(f"  Loading {candidates[0].name}")
    with candidates[0].open() as f:
        return json.load(f)


def main() -> None:
    log.info("Building WP1 master figure (consolidates §2-§6.8 storyline)")

    # Load summaries
    fim_2pop = _load_latest_summary("fim_summary")
    fim_3pop = _load_latest_summary("fim_3pop_summary")
    fim_3pop_sched = _load_latest_summary("fim_3pop_schedule_summary")
    regime_1d = _load_latest_summary("regime_scan_summary")
    regime_2d = _load_latest_summary("regime_scan_2d_summary")
    decision = _load_latest_summary("decision_comparison_summary")
    cross_cohort = _load_latest_summary("cross_cohort_pa_vs_pe_summary")
    real_cohort = _load_latest_summary("real_cohort_pa_vs_pe_summary")

    if not all([fim_2pop, fim_3pop, fim_3pop_sched, regime_1d, regime_2d, decision]):
        log.error("Some summary files missing; figure may be incomplete.")

    fig = plt.figure(figsize=(15, 17))
    fig.set_facecolor("white")
    gs = fig.add_gridspec(4, 3, hspace=0.55, wspace=0.4)

    # ---------- Row 1: Identifiability ----------

    # [A] 2-pop FIM eigenvalues
    ax_a = fig.add_subplot(gs[0, 0])
    if fim_2pop:
        eigs = fim_2pop["fim_eigenvalues"]
        x = np.arange(len(eigs))
        threshold = 1e-6 * eigs[0]
        colors = ["tab:green" if e > threshold else "tab:red" for e in eigs]
        ax_a.bar(x, eigs, color=colors, edgecolor="black", linewidth=0.5)
        ax_a.set_yscale("log")
        ax_a.set_xticks(x)
        ax_a.set_xticklabels([f"λ_{i+1}" for i in range(len(eigs))])
        ax_a.set_title(f"[A] 2-pop FIM — rank {fim_2pop['effective_rank']}/4", fontsize=11)
        ax_a.axhline(threshold, color="tab:red", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_a.set_ylabel("Eigenvalue")
        ax_a.grid(True, alpha=0.3, axis="y", which="both")

    # [B] 3-pop FIM eigenvalues
    ax_b = fig.add_subplot(gs[0, 1])
    if fim_3pop:
        eigs = fim_3pop["fim_eigenvalues"]
        x = np.arange(len(eigs))
        threshold = 1e-6 * eigs[0]
        colors = ["tab:green" if e > threshold else "tab:red" for e in eigs]
        ax_b.bar(x, eigs, color=colors, edgecolor="black", linewidth=0.5)
        ax_b.set_yscale("log")
        ax_b.set_xticks(x)
        ax_b.set_xticklabels([f"λ_{i+1}" for i in range(len(eigs))])
        ax_b.set_title(f"[B] 3-pop FIM — rank {fim_3pop['effective_rank']}/6", fontsize=11)
        ax_b.axhline(threshold, color="tab:red", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_b.set_ylabel("Eigenvalue")
        ax_b.grid(True, alpha=0.3, axis="y", which="both")

    # [C] α-β estimate correlation visualization (synthetic; show -1.00)
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.set_xlim(-1.2, 1.2)
    ax_c.set_ylim(-1.2, 1.2)
    # Show the 1D ridge in (α, β) plane that the FIM finds unidentifiable.
    s_grid = np.linspace(-1, 1, 20)
    ax_c.plot(s_grid, -s_grid, color="tab:red", linewidth=2.5, label="α-β unidentifiable ridge")
    ax_c.scatter([0], [0], color="black", s=50, zorder=5, label="θ_nominal")
    ax_c.set_xlabel("Δα (perturbation from nominal)")
    ax_c.set_ylabel("Δβ (perturbation from nominal)")
    ax_c.set_title(f"[C] α-β degeneracy (corr = -1.00)", fontsize=11)
    ax_c.legend(fontsize=8)
    ax_c.grid(True, alpha=0.3)
    ax_c.set_aspect("equal")
    ax_c.axhline(0, color="black", linewidth=0.5)
    ax_c.axvline(0, color="black", linewidth=0.5)

    # ---------- Row 2: Schedule + regime ----------

    # [D] Cross-schedule eigenvalue spectra (3-pop)
    ax_d = fig.add_subplot(gs[1, 0])
    if fim_3pop_sched:
        n = 6
        x = np.arange(n)
        width = 0.27
        s = fim_3pop_sched["schedules"]
        ax_d.bar(x - width, s["MTD"]["eigenvalues"], width, label="MTD", color="tab:blue", edgecolor="black", linewidth=0.4)
        ax_d.bar(x, s["AT50_replayed"]["eigenvalues"], width, label="AT50", color="tab:red", edgecolor="black", linewidth=0.4)
        ax_d.bar(x + width, s["Periodic_56d_50pct"]["eigenvalues"], width, label="Periodic", color="tab:orange", edgecolor="black", linewidth=0.4)
        ax_d.set_yscale("log")
        ax_d.set_xticks(x)
        ax_d.set_xticklabels([f"λ_{i+1}" for i in range(n)])
        ax_d.set_title(f"[D] 3-pop cross-schedule (all rank 3)", fontsize=11)
        ax_d.legend(fontsize=8)
        ax_d.set_ylabel("Eigenvalue")
        ax_d.grid(True, alpha=0.3, axis="y", which="both")

    # [E] 1D regime scan
    ax_e = fig.add_subplot(gs[1, 1])
    if regime_1d:
        K_grid = np.array([r["K_TP_drop"] for r in regime_1d["results"]])
        p_ttp = np.array([r["p_at50_wins_ttp"] for r in regime_1d["results"]])
        p_drug = np.array([r["p_at50_wins_drug"] for r in regime_1d["results"]])
        ax_e.plot(K_grid, p_ttp, color="tab:red", linewidth=2.0, marker="o", label="P(AT50 wins TTP)")
        ax_e.plot(K_grid, p_drug, color="tab:purple", linewidth=2.0, marker="s", linestyle="--", label="P(AT50 wins drug)")
        ax_e.axhline(0.5, color="tab:gray", linestyle=":", linewidth=1.0, alpha=0.6, label="coin-flip")
        ax_e.set_xlabel("K_TP_drop")
        ax_e.set_ylabel("P(AT50 wins)")
        ax_e.set_title(f"[E] 1D regime scan", fontsize=11)
        ax_e.legend(fontsize=8)
        ax_e.set_ylim(-0.02, 1.05)
        ax_e.grid(True, alpha=0.3)

    # [F] 2D regime scan heatmap
    ax_f = fig.add_subplot(gs[1, 2])
    if regime_2d:
        P_grid = np.array(regime_2d["P_AT50_wins_grid"])
        K_grid_2d = regime_2d["K_TP_drop_grid"]
        alpha_grid_2d = regime_2d["alpha_grid"]
        im = ax_f.imshow(P_grid, origin="lower", cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax_f.set_xticks(range(len(K_grid_2d)))
        ax_f.set_xticklabels([f"{K:.0f}" for K in K_grid_2d], fontsize=8)
        ax_f.set_yticks(range(len(alpha_grid_2d)))
        ax_f.set_yticklabels([f"{a:.1f}" for a in alpha_grid_2d], fontsize=8)
        ax_f.set_xlabel("K_TP_drop")
        ax_f.set_ylabel("α(T-, T+)")
        ax_f.set_title(f"[F] 2D regime scan: P(AT50 wins)", fontsize=11)
        fig.colorbar(im, ax=ax_f, fraction=0.046, pad=0.04)
        for i in range(len(alpha_grid_2d)):
            for j in range(len(K_grid_2d)):
                v = P_grid[i, j]
                if not np.isnan(v):
                    ax_f.text(j, i, f"{v:.0%}", ha="center", va="center",
                              color=("white" if v < 0.5 else "black"), fontsize=7)

    # ---------- Row 3: Decision ----------

    # [G] PA vs PE accuracy bar chart
    ax_g = fig.add_subplot(gs[2, 0])
    if decision:
        h = decision["headline"]
        accs = [h["pe_accuracy_vs_oracle"], h["pa_accuracy_vs_oracle"]]
        labels = ["Point-est.", "Posterior-aware"]
        colors = ["tab:blue", "tab:red"]
        bars = ax_g.bar(labels, accs, color=colors, edgecolor="black", linewidth=0.5)
        ax_g.set_ylabel("Accuracy vs oracle")
        ax_g.set_title(f"[G] PA accuracy advantage = "
                       f"{(h['pa_advantage_over_pe'] * 100):+.1f} pp", fontsize=11)
        ax_g.set_ylim(0, 1.0)
        ax_g.grid(True, alpha=0.3, axis="y")
        for bar, acc in zip(bars, accs):
            ax_g.text(bar.get_x() + bar.get_width() / 2, acc + 0.02,
                       f"{acc:.0%}", ha="center", va="bottom", fontsize=10)

    # [H] PA vs PE per-patient scatter
    ax_h = fig.add_subplot(gs[2, 1])
    if decision:
        results = decision["per_patient_results"]
        pe_adv = np.array([r["pe_ttp_at50"] - r["pe_ttp_mtd"] for r in results]) / 30
        pa_adv = np.array([r["pa_expected_ttp_at50"] - r["pa_expected_ttp_mtd"] for r in results]) / 30
        agreement = [r["agreement_pe_pa"] for r in results]
        colors = ["tab:gray" if a else "tab:orange" for a in agreement]
        ax_h.scatter(pe_adv, pa_adv, c=colors, s=40, alpha=0.7, edgecolor="none")
        diag_min = min(pe_adv.min() if len(pe_adv) else 0, pa_adv.min() if len(pa_adv) else 0)
        diag_max = max(pe_adv.max() if len(pe_adv) else 1, pa_adv.max() if len(pa_adv) else 1)
        ax_h.plot([diag_min, diag_max], [diag_min, diag_max], "k--", alpha=0.4, linewidth=0.7, label="PE = PA")
        ax_h.axhline(0, color="black", linewidth=0.5)
        ax_h.axvline(0, color="black", linewidth=0.5)
        ax_h.set_xlabel("PE AT50 advantage (months)")
        ax_h.set_ylabel("PA AT50 advantage (months)")
        n_disagree = sum(1 for r in results if not r["agreement_pe_pa"])
        ax_h.set_title(f"[H] PE vs PA (orange = disagree, n={n_disagree})", fontsize=11)
        ax_h.grid(True, alpha=0.3)

    # [I] PA accuracy advantage by K_TP_drop bin
    ax_i = fig.add_subplot(gs[2, 2])
    if decision:
        results = decision["per_patient_results"]
        K_vals = np.array([r["K_TP_drop_true"] for r in results])
        pe_correct = np.array([r["agreement_oracle_pe"] for r in results])
        pa_correct = np.array([r["agreement_oracle_pa"] for r in results])
        K_bins = np.linspace(K_vals.min() - 1, K_vals.max() + 1, 5)
        bin_idx = np.digitize(K_vals, K_bins)
        pe_acc_bins, pa_acc_bins, bin_centers = [], [], []
        for k in range(1, len(K_bins)):
            mask = bin_idx == k
            if not mask.any():
                continue
            pe_acc_bins.append(pe_correct[mask].mean())
            pa_acc_bins.append(pa_correct[mask].mean())
            bin_centers.append((K_bins[k - 1] + K_bins[k]) / 2)
        if bin_centers:
            ax_i.plot(bin_centers, pe_acc_bins, color="tab:blue", linewidth=1.8, marker="s", label="PE")
            ax_i.plot(bin_centers, pa_acc_bins, color="tab:red", linewidth=1.8, marker="^", label="PA")
            ax_i.set_xlabel("K_TP_drop bin center")
            ax_i.set_ylabel("Accuracy vs oracle")
            ax_i.set_title("[I] Accuracy across regime", fontsize=11)
            ax_i.legend(fontsize=9)
            ax_i.set_ylim(0, 1.05)
            ax_i.grid(True, alpha=0.3)

    # ---------- Row 4: REAL-DATA validation (Bruchovsky + Shaw) ----------

    # [J] Cross-cohort disagreement-rate comparison
    ax_j = fig.add_subplot(gs[3, 0])
    if cross_cohort:
        bruch = cross_cohort["cohorts"].get("Bruchovsky", {})
        shaw = cross_cohort["cohorts"].get("Shaw", {})
        cohorts_lbl = ["Bruchovsky", "Shaw"]
        rates = [bruch.get("disagreement_rate", 0), shaw.get("disagreement_rate", 0)]
        colors = ["tab:purple", "tab:orange"]
        bars = ax_j.bar(cohorts_lbl, rates, color=colors, alpha=0.8, edgecolor="black")
        for bar, rate, agg in zip(bars, rates, [bruch, shaw]):
            ax_j.text(bar.get_x() + bar.get_width() / 2, rate + 0.01,
                       f"{rate:.0%}\n({agg.get('disagreement_count', 0)}/{agg.get('n', 0)})",
                       ha="center", va="bottom", fontsize=9)
        ax_j.set_ylabel("PE-vs-PA disagreement rate")
        ax_j.set_title("[J] REAL-data PE-vs-PA disagreement\n(cross-cohort)", fontsize=10)
        ax_j.set_ylim(0, 0.5)
        ax_j.grid(True, alpha=0.3, axis="y")

    # [K] Per-patient P(AT50 wins) histogram on real cohorts
    ax_k = fig.add_subplot(gs[3, 1])
    if cross_cohort and "per_patient" in cross_cohort:
        bruch_p = np.array([r["pa_p_at50_wins"]
                            for r in cross_cohort["per_patient"].get("Bruchovsky", [])])
        shaw_p = np.array([r["pa_p_at50_wins"]
                           for r in cross_cohort["per_patient"].get("Shaw", [])])
        bins = np.linspace(0, 1, 21)
        if len(bruch_p) > 0:
            ax_k.hist(bruch_p, bins=bins, alpha=0.5, color="tab:purple",
                      label=f"Bruchovsky n={len(bruch_p)}")
        if len(shaw_p) > 0:
            ax_k.hist(shaw_p, bins=bins, alpha=0.5, color="tab:orange",
                      label=f"Shaw n={len(shaw_p)}")
        ax_k.axvline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_k.set_xlabel("P(AT50 beats MTD)")
        ax_k.set_ylabel("Patient count")
        ax_k.set_title("[K] Per-patient AT50 preference\n(real Bruchovsky + Shaw)", fontsize=10)
        ax_k.legend(fontsize=8)
        ax_k.grid(True, alpha=0.3)

    # [L] Recommendation breakdown bars (PE vs PA, both cohorts)
    ax_l = fig.add_subplot(gs[3, 2])
    if cross_cohort:
        bruch = cross_cohort["cohorts"].get("Bruchovsky", {})
        shaw = cross_cohort["cohorts"].get("Shaw", {})
        labels = ["Bruchovsky\n(n=71)", "Shaw\n(n=15)"]
        pe_rates = [bruch.get("pe_at50_rate", 0), shaw.get("pe_at50_rate", 0)]
        pa_rates = [bruch.get("pa_at50_rate", 0), shaw.get("pa_at50_rate", 0)]
        x = np.arange(len(labels))
        w = 0.35
        ax_l.bar(x - w / 2, pe_rates, w, color="tab:blue", alpha=0.8, label="PE: AT50", edgecolor="black")
        ax_l.bar(x + w / 2, pa_rates, w, color="tab:red", alpha=0.8, label="PA: AT50", edgecolor="black")
        for i, (pe, pa) in enumerate(zip(pe_rates, pa_rates)):
            ax_l.text(i - w / 2, pe + 0.01, f"{pe:.0%}", ha="center", fontsize=8)
            ax_l.text(i + w / 2, pa + 0.01, f"{pa:.0%}", ha="center", fontsize=8)
        ax_l.set_xticks(x)
        ax_l.set_xticklabels(labels)
        ax_l.set_ylabel("Fraction recommended AT50")
        ax_l.set_title("[L] Methodology flips recommendations\n(PE vs PA on real cohorts)", fontsize=10)
        ax_l.legend(fontsize=8)
        ax_l.set_ylim(0, 0.7)
        ax_l.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "WP1 master figure — Identifiability rank-deficiency in adaptive cancer therapy "
        "(rows: structure / regime / decision / REAL-data validation)",
        fontsize=12,
    )

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT, text=True,
        ).strip()
    except subprocess.CalledProcessError:
        sha = "uncommitted"
    date = dt.date.today().isoformat()
    base = _REPO_ROOT / "results" / "figures" / f"fig17_wp1_master_{sha}_{date}"
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")  # 300 DPI for print
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {base}.png (300 DPI) and .pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WP1 master consolidation figure")
    args = parser.parse_args()
    main()
