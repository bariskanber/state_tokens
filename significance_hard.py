"""Paired per-seed significance analysis for the hard-world experiment.

C (state-gated) vs D (ungated) and D vs B (post-hoc) on the key OOD metrics,
paired by training seed (same data/init luck per pair). Reports per-seed
differences, sign consistency, paired t-test, and Wilcoxon signed-rank.
"""
import json
import os
from collections import defaultdict

from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "hard_world_results.jsonl")

# metric, eval set, direction (+1 = higher is better, -1 = lower is better)
METRICS = [
    ("temptation", "tempted_harm_rate", -1, "tempted harm"),
    ("temptation", "utility", +1, "temptation utility@k=2"),
    ("temptation", "agr_caring", +1, "temptation agr. caring expert"),
    ("micro", "harm_rate", -1, "micro harm"),
    ("micro", "utility", +1, "micro utility@k=2"),
    ("allstake", "harm_rate", -1, "allstake harm"),
    ("heldout", "utility", +1, "heldout utility@k=2"),
    ("heldout", "agr_caring", +1, "heldout agr. caring expert"),
]


def load():
    by_cond = defaultdict(dict)   # cond -> seed -> sets
    with open(RESULTS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_cond[r["cond"]][r["seed"]] = r["sets"]
    return by_cond


def paired_test(by_cond, cond_a, cond_b):
    seeds = sorted(set(by_cond[cond_a]) & set(by_cond[cond_b]))
    print(f"\n=== {cond_a} vs {cond_b}  (n={len(seeds)} paired seeds) ===")
    print(f"{'metric':32s} {'meanA':>8s} {'meanB':>8s} {'diff':>8s} "
          f"{'sign(A>B)':>10s} {'t p':>8s} {'wilcoxon p':>10s}")
    for setname, key, sign, label in METRICS:
        a = [by_cond[cond_a][s][setname][key] * sign for s in seeds]
        b = [by_cond[cond_b][s][setname][key] * sign for s in seeds]
        d = [x - y for x, y in zip(a, b)]
        n_better = sum(1 for x in d if x > 0)
        n_worse = sum(1 for x in d if x < 0)
        t, tp = stats.ttest_rel(a, b)
        try:
            wp = stats.wilcoxon(a, b).pvalue if (n_better and n_worse) else 0.0
        except ValueError:
            wp = float("nan")
        print(f"{label:32s} {sum(a)/len(a)*1:8.3f} {sum(b)/len(b):8.3f} "
              f"{sum(d)/len(d):+8.3f} {n_better:4d}/{len(d):<2d}    "
              f"{tp:8.3f} {wp:10.3f}")


if __name__ == "__main__":
    by_cond = load()
    conds = sorted(by_cond)
    print(f"conditions: {conds}")
    for c in conds:
        print(f"  {c}: seeds {sorted(by_cond[c])}")
    if "C_care_state" in by_cond and "D_caring" in by_cond:
        paired_test(by_cond, "C_care_state", "D_caring")
    if "D_caring" in by_cond and "B_dpo" in by_cond:
        paired_test(by_cond, "D_caring", "B_dpo")
