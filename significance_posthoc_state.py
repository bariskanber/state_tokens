"""Paired significance tests for the post-hoc + CARE retrofit conditions.

Key comparisons (paired by training seed, two-sided t + Wilcoxon):
  1. Bmix_state vs B_state readout (ID): does dense state BCE rescue the
     readout under sparse behavioral pairs?  (mechanism isolation)
  2. Ball_state vs B_all coherence (heldout agr): does retrofitting the state
     channel onto the dense pipeline change behavior?  (retrofit is free?)
  3. Ball_state vs B_state coherence: dense vs sparse pairs with the channel.
  4. Ball_state readout vs C readout (ID): retrofit vs from-scratch auditability.

Usage: python significance_posthoc_state.py
"""
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path, conds):
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("cond") in conds:
                    rows[(r["cond"], r["seed"])] = r
    return rows


def paired(a_rows, b_rows, path, name):
    seeds = sorted(set(s for (c, s) in a_rows) & set(s for (c, s) in b_rows))
    a = np.array([path(a_rows[(c, s)]) for c, s in a_rows if s in seeds])
    # rebuild in seed order for correct pairing
    ca = {s: path(v) for (c, s), v in a_rows.items()}
    cb = {s: path(v) for (c, s), v in b_rows.items()}
    a = np.array([ca[s] for s in seeds])
    b = np.array([cb[s] for s in seeds])
    d = a - b
    t, tp = stats.ttest_rel(a, b)
    try:
        wp = stats.wilcoxon(a, b).pvalue
    except ValueError:
        wp = float("nan")
    print(f"{name:55s} n={len(seeds)}  {d.mean():+.3f}  "
          f"t p={tp:.2e}  Wilcoxon p={wp:.2e}  sign {int((d > 0).sum())}/{len(seeds)}")


def main():
    ph = load(os.path.join(HERE, "posthoc_state_results.jsonl"),
              {"A_state", "B_state", "Bmix_state", "Ball_state"})
    ball = load(os.path.join(HERE, "hard_world_ball_results.jsonl"), {"B_all"})
    hw = load(os.path.join(HERE, "hard_world_results.jsonl"),
              {"A_selfish", "B_dpo", "D_caring", "C_care_state"})

    def cond(rows, name):
        return {(c, s): v for (c, s), v in rows.items() if c == name}

    # pair key helper: make both sides keyed by (label, seed)
    def relabel(rows, old, new):
        return {(new, s): v for (c, s), v in rows.items() if c == old}

    ro_id = lambda r: r["readout"]["heldout"]
    ro_mi = lambda r: r["readout"]["micro"]
    agr_h = lambda r: r["sets"]["heldout"]["agr_caring"]
    agr_t = lambda r: r["sets"]["temptation"]["agr_caring"]
    th = lambda r: r["sets"]["temptation"]["tempted_harm_rate"]

    print("== mechanism: state-BCE coverage rescues the readout (sparse pairs) ==")
    paired(cond(ph, "Bmix_state"), cond(ph, "B_state"), ro_id,
           "readout ID: Bmix_state - B_state")
    paired(cond(ph, "Bmix_state"), cond(ph, "B_state"), ro_mi,
           "readout micro: Bmix_state - B_state")

    print("\n== retrofit is free? (dense pipeline, with vs without channel) ==")
    paired(relabel(ph, "Ball_state", "X"), relabel(ball, "B_all", "Y"), agr_h,
           "heldout agr: Ball_state - B_all")
    paired(relabel(ph, "Ball_state", "X"), relabel(ball, "B_all", "Y"), th,
           "tempted harm: Ball_state - B_all")

    print("\n== sparse vs dense pairs WITH the channel ==")
    paired(cond(ph, "Ball_state"), cond(ph, "B_state"), agr_h,
           "heldout agr: Ball_state - B_state")

    print("\n== retrofit vs from-scratch C (auditability + coherence) ==")
    paired(relabel(ph, "Ball_state", "X"), relabel(hw, "C_care_state", "Y"), ro_id,
           "readout ID: Ball_state - C")
    paired(relabel(ph, "Ball_state", "X"), relabel(hw, "C_care_state", "Y"), agr_h,
           "heldout agr: Ball_state - C")


if __name__ == "__main__":
    main()
