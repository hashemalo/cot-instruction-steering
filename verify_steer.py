"""Independent checks + figures for the steering test run (reads only runs/test/*.jsonl).

  1. strict compliance re-derived from the saved token ids with the CoTControl repo grader (fresh decode)
  2. length-controlled compliance: (a) traces with >= 512 thinking tokens only, (b) first-256-token prefix of every
     trace with >= 256 tokens -- steering that merely shortens thinking cannot win these
  3. accuracy vs instruction-only: paired exact McNemar test
  4. topicality: fraction of the question's content words (len >= 5) that appear in the thinking (catches
     on-rule-but-off-task text)
  5. hack checks: letter-free / near-empty traces, non-ASCII letters, meta-mentions
  6. figures: compliance & accuracy by condition (95% CI), wrong-case fraction by thinking window (decay)
  7. results/samples.md: 2 random traces per condition (seeded), first 700 chars

Usage: python verify_steer.py
"""
from __future__ import annotations

import json
import math
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from common import build_prompt  # noqa: E402
from grading import grade_lowercase_thinking  # noqa: E402

RUNS = ROOT / "runs" / "test"
RES = ROOT / "results"
META = re.compile(r"lower ?case|upper ?case|capital|all caps|no caps|small letters", re.I)
NAMES = {
    "T_noinstr_none": "no instruction, no steering",
    "T_instr_none": "instruction only",
    "T_instr_A": "instr + A (Stolfo prompt-end vector)",
    "T_instr_B": "instr + B (in-CoT instruction vector)",
    "T_instr_C": "instr + C (lowercase-style vector)",
    "T_instr_rand": "instr + random vector (same norm)",
    "T_noinstr_primary": "NO instr + B",
    "T_instr_primary_promptonly": "instr + B, prompt positions only",
    "T_instr_primary_think0-64": "instr + B, thinking tokens 0-63 only",
    "T_instr_primary_think0-512": "instr + B, thinking tokens 0-511 only",
    "T_instr_primary_onset256": "instr + B, from thinking token 256 on",
    "T_instr_primary_seed1": "instr + B (replication, seed 1)",
    "T_instr_B_r0.2": "instr + B at half strength (r=0.2; added post hoc)",
}


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mcnemar_exact(a, b):
    """a, b: dict qid->bool. Two-sided exact binomial p on discordant pairs."""
    q = sorted(set(a) & set(b))
    n01 = sum((not a[i]) and b[i] for i in q); n10 = sum(a[i] and (not b[i]) for i in q)
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return min(1.0, p), n01, n10


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    data = {f.stem: [json.loads(l) for l in open(f)] for f in sorted(RUNS.glob("*.jsonl"))}
    order = [k for k in NAMES if k in data]
    ref = {r["qid"]: r for r in data["T_instr_none"]}
    qtext = {}
    rows, samples = [], ["# Random steered traces (2 per condition, seed 0, first 700 chars)\n"]
    for name in order:
        R = data[name]
        texts = [tok.decode(r["think_ids"]) for r in R]
        strict = np.array([grade_lowercase_thinking(t) for t in texts])
        assert (strict == np.array([r["strict"] for r in R])).all(), f"{name}: strict mismatch vs saved"
        n_think = np.array([len(r["think_ids"]) for r in R])
        long_ = n_think >= 512
        pre = [(grade_lowercase_thinking(tok.decode(r["think_ids"][:256]))) for r in R if len(r["think_ids"]) >= 256]
        topical = []
        for r, t in zip(R, texts):
            if r["qid"] not in qtext:
                s = build_prompt(r, "baseline")["sample"]
                qtext[r["qid"]] = {w for w in re.findall(r"[a-z]{5,}", (s["question"]).lower())}
            qw = qtext[r["qid"]]
            topical.append(len(qw & set(re.findall(r"[a-z]{5,}", t.lower()))) / len(qw) if qw else np.nan)
        letters = [sum(c.isalpha() for c in t) for t in texts]
        acc = {r["qid"]: r["correct"] for r in R}
        p, n01, n10 = mcnemar_exact({q: v["correct"] for q, v in ref.items()}, acc)
        k = int(strict.sum()); n = len(R)
        rows.append({
            "condition": name, "description": NAMES[name], "n": n,
            "strict": k / n, "strict_lo": wilson(k, n)[0], "strict_hi": wilson(k, n)[1],
            "strict_len>=512": strict[long_].mean() if long_.any() else np.nan, "n_len>=512": int(long_.sum()),
            "strict_first256": np.mean(pre) if pre else np.nan, "n_first256": len(pre),
            "acc": np.mean(list(acc.values())),
            "acc_instr_only_same_q": np.mean([ref[q]["correct"] for q in acc]),
            "strict_instr_only_same_q": np.mean([ref[q]["strict"] for q in acc]),
            "distinct3_instr_only_same_q": float(np.nanmean([ref[q]["distinct3"] if ref[q]["distinct3"] is not None else np.nan for q in acc])),
            "acc_lo": wilson(sum(acc.values()), n)[0],
            "acc_hi": wilson(sum(acc.values()), n)[1], "mcnemar_p_vs_instr_only": p,
            "acc_gained": n01, "acc_lost": n10,
            "mean_think_tokens": n_think.mean(), "median_think_tokens": float(np.median(n_think)),
            "truncated_frac": np.mean([r["truncated"] for r in R]),
            "near_empty(<50 letters)": int(sum(l < 50 for l in letters)),
            "nonascii_letters": float(np.nanmean([r["nonascii_alpha_frac"] if r["nonascii_alpha_frac"] is not None else np.nan for r in R])),
            "meta_rate": np.mean([bool(META.search(t)) for t in texts]),
            "topicality": float(np.nanmean(topical)),
            "distinct3": float(np.nanmean([r["distinct3"] if r["distinct3"] is not None else np.nan for r in R])),
            **{c: float(np.nanmean([r[c] if r.get(c) is not None else np.nan for r in R]))
               for c in R[0] if c.startswith("wrong_frac_")},
        })
        rng = random.Random(0)
        samples.append(f"\n## {name}: {NAMES[name]}\n")
        for i in rng.sample(range(n), 2):
            samples.append(f"**{R[i]['qid']}** · {len(R[i]['think_ids'])} tokens · strict={int(strict[i])} · "
                           f"correct={R[i]['correct']}\n```text\n{texts[i][:700]}\n```\n")
    df = pd.DataFrame(rows).set_index("condition")
    df.to_csv(RES / "test_verified.csv")
    (RES / "samples.md").write_text("\n".join(samples))
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
    print(df.drop(columns=["description"]).round(3).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    y = np.arange(len(df))[::-1]
    for ax, (m, lo, hi, lab) in zip(axes, [("strict", "strict_lo", "strict_hi", "strict CoT compliance (repo grader)"),
                                           ("acc", "acc_lo", "acc_hi", "answer accuracy (forced, unsteered)")]):
        ax.errorbar(df[m], y, xerr=[df[m] - df[lo], df[hi] - df[m]], fmt="o", color="#4C72B0", capsize=3)
        ax.set_yticks(y, [NAMES[c] for c in df.index], fontsize=8)
        ax.set_xlabel(lab); ax.set_xlim(0, 1); ax.grid(axis="x", alpha=0.3)
        ax.axvline(df.loc["T_instr_none", m], color="#C44E52", ls="--", lw=1, label="instruction only")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].set_yticks(y, [""] * len(y))
    fig.suptitle("Qwen3-8B lowercase_thinking: steering the thinking (n=200 held-out questions, 95% Wilson CI)", fontsize=10)
    fig.tight_layout(); fig.savefig(RES / "fig_test_compliance_accuracy.png", dpi=140); plt.close(fig)

    wcols = [c for c in df.columns if c.startswith("wrong_frac_")]
    mids = [np.mean([int(x) for x in c.split("_")[2:4]]) for c in wcols]
    plotted = [n for n in ["T_instr_none", "T_instr_B", "T_instr_primary_promptonly", "T_instr_primary_think0-64",
                           "T_instr_primary_think0-512", "T_instr_primary_onset256"] if n in data]
    common = set.intersection(*[{r["qid"] for r in data[n]} for n in plotted])  # same questions for every curve
    win_means = {n: [float(np.nanmean([r[c] if r.get(c) is not None else np.nan for r in data[n] if r["qid"] in common]))
                     for c in wcols] for n in plotted}
    pd.DataFrame(win_means, index=wcols).T.to_csv(RES / "test_decay_windows.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, st in [("T_instr_none", ("#C44E52", "-")), ("T_instr_primary", None), ("T_instr_B", ("#4C72B0", "-")),
                     ("T_instr_primary_promptonly", ("#8172B2", ":")), ("T_instr_primary_think0-64", ("#DD8452", "--")),
                     ("T_instr_primary_think0-512", ("#55A868", "--")), ("T_instr_primary_onset256", ("#937860", "-."))]:
        if st is None or name not in win_means:
            continue
        ax.plot(mids, win_means[name], marker="o", color=st[0], ls=st[1], label=NAMES[name])
    for x in [64, 256, 512]:
        ax.axvline(x, color="#DDDDDD", lw=0.8)
    ax.set_xscale("log"); ax.set_xlabel("thinking-token window (midpoint, log)")
    ax.set_ylabel("fraction of letters in the wrong case"); ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"Where steering acts and how long it lasts (vector B; same {len(common)} questions per curve)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(RES / "fig_test_decay.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
