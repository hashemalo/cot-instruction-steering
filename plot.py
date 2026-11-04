"""Figures from the saved metrics (CPU).

fig_dev_sweep.png   the three vectors on the 48-question dev set: strict compliance by layer and strength; hollow
                    markers = the setting collapsed thinking (> 2 of 48 traces under 32 tokens) or made the model
                    announce the rule (meta-mention rate > 10 pts above the unsteered reference)
fig_test.png        held-out test set: strict compliance and answer accuracy with 95% Wilson CIs
Usage: python plot.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
VEC = {"A": "A: Stolfo vector\n(instruction present − absent, last prompt token)",
       "B": "B: in-CoT vector\n(instruction present − absent, mean over thinking tokens)",
       "C": "C: lowercase-style vector\n(same thinking text lowercased − original)"}
LAYER_COL = {8: "#9ecae1", 14: "#4292c6", 20: "#2171b5", 26: "#08306b"}
TEST = [("T_noinstr_none", "no instruction, no steering"), ("T_instr_none", "instruction only"),
        ("T_instr_B", "instruction + B (in-CoT vector), L26 r=0.4"), ("T_instr_C", "instruction + C (lowercase-style), L26 r=0.4"),
        ("T_instr_rand", "instruction + random vector, same norm"), ("T_noinstr_primary", "no instruction + B")]


def dev_sweep():
    d = pd.read_csv(RES / "dev_summary.csv", index_col=0)
    ref = d.loc["dev_instr_none"]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.3), sharey=True)
    for ax, v in zip(axs, "ABC"):
        for L, col in LAYER_COL.items():
            rows = d[(d.vec == v) & (d.layer == L)].sort_values("r")
            bad = (rows.short32 > 2) | (rows.meta_rate > ref.meta_rate + 0.10)
            ax.plot(rows.r, rows.strict, color=col, lw=2, label=f"layer {L}")
            ax.scatter(rows.r[~bad], rows.strict[~bad], color=col, s=40, zorder=3)
            ax.scatter(rows.r[bad], rows.strict[bad], facecolors="white", edgecolors=col, s=46, lw=1.6, zorder=3)
        ax.axhline(ref.strict, color="#999", ls=":", lw=1, label=f"instruction only ({ref.strict:.0%})")
        ax.set(title=VEC[v], xlabel="strength r (fraction of the residual norm)", xscale="log", xticks=[0.1, 0.2, 0.4, 0.8],
               xticklabels=["0.1", "0.2", "0.4", "0.8"], ylim=(-0.02, 1.02))
        ax.title.set_fontsize(9.5)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=0.25)
    axs[0].set_ylabel("strict lowercase compliance (CoTControl grader)")
    axs[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Dev sweep (48 questions): filled = usable setting, hollow = thinking collapsed or the rule is announced instead of followed", fontsize=10)
    fig.tight_layout()
    fig.savefig(RES / "fig_dev_sweep.png", dpi=150)


def test():
    d = pd.read_csv(RES / "test_summary.csv", index_col=0)
    rows = [(k, n) for k, n in TEST if k in d.index]
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    y = list(range(len(rows)))[::-1]
    for ax, m, lab in zip(axs, ("strict", "acc"), ("strict lowercase compliance", "answer accuracy (forced, unsteered)")):
        for yi, (k, n) in zip(y, rows):
            r = d.loc[k]
            ax.errorbar(r[m], yi, xerr=[[r[m] - r[f"{m}_lo"]], [r[f"{m}_hi"] - r[m]]], fmt="o", color="#2171b5", capsize=3)
            ax.text(1.09, yi, f"{r[m]:.0%}  (n={int(r.n)})", va="center", ha="left", fontsize=8, color="#444")
        ax.axvline(d.loc["T_instr_none", m], color="#c0392b", ls="--", lw=1, label="instruction only")
        ax.set(xlim=(0, 1.32), xlabel=lab, xticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(axis="x", alpha=0.25)
    axs[0].set_yticks(y, [n for _, n in rows], fontsize=9)
    axs[0].legend(fontsize=8, loc="upper center")
    fig.suptitle("Held-out test questions, 95% Wilson CIs. Vector A had no usable dev setting and was not run on the test set.", fontsize=10)
    fig.tight_layout()
    fig.savefig(RES / "fig_test.png", dpi=150)


if __name__ == "__main__":
    dev_sweep()
    test()
    print("wrote results/fig_dev_sweep.png, results/fig_test.png")
