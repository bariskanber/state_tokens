"""Summarize the post-hoc + CARE retrofit conditions vs the existing ones.

Table: condition x key metrics (hard world), mean +/- std over seeds.
Reference conditions come from hard_world_results.jsonl (A/B/D/C) and
hard_world_ball_results.jsonl (B_all); retrofit conditions from
posthoc_state_results.jsonl (A_state/B_state/Ball_state).

Usage: python summarize_posthoc_state.py [--seeds 42 43 44]
"""
import argparse
import json
import os
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path, conds):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("cond") in conds:
                    rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    args = ap.parse_args()

    ref_conds = {"A_selfish": "A", "B_dpo": "B", "D_caring": "D",
                 "C_care_state": "C", "B_all": "B_all"}
    rows = (load(os.path.join(HERE, "hard_world_results.jsonl"), set(ref_conds))
            + load(os.path.join(HERE, "hard_world_ball_results.jsonl"), {"B_all"})
            + load(os.path.join(HERE, "posthoc_state_results.jsonl"),
                   {"A_state", "B_state", "Bmix_state", "Ball_state"}))
    if args.seeds:
        rows = [r for r in rows if r["seed"] in set(args.seeds)]

    by = defaultdict(list)
    for r in rows:
        name = ref_conds.get(r["cond"], r["cond"])
        by[name].append(r)

    def m(rs, path):
        vals = []
        for r in rs:
            v = r
            for k in path:
                v = v[k]
            vals.append(v)
        return st.mean(vals) * 100, (st.stdev(vals) * 100 if len(vals) > 1 else 0.0)

    def fmt(rs, path):
        if not rs:
            return "      ---"
        mean, sd = m(rs, path)
        return f"{mean:6.1f}±{sd:4.1f}"

    hdr = (f"{'cond':11s} {'n':>2} {'agrHeld':>11} {'agrTemp':>11} "
           f"{'tHarm%':>11} {'utilTemp':>11} {'k*':>5} {'roID%':>6} "
           f"{'roMicro%':>8} {'clamp0/1%':>10}")
    print(hdr)
    print("-" * len(hdr))
    order = ["A", "B", "B_all", "D", "C", "A_state", "B_state", "Bmix_state", "Ball_state"]
    for name in order:
        rs = by.get(name, [])
        if not rs:
            continue
        ks = [r.get("kstar") for r in rs if r.get("kstar") is not None]
        kstar_s = f"{st.mean(ks):5.2f}" if ks else "  ---"
        ro_id = fmt(rs, ["readout", "heldout"]) if rs[0].get("readout") else "   ---"
        ro_mi = fmt(rs, ["readout", "micro"]) if rs[0].get("readout") else "     ---"
        cf = ""
        if rs[0].get("clamp_flip"):
            c0 = st.mean([r["clamp_flip"]["clamp0"] for r in rs]) * 100
            c1 = st.mean([r["clamp_flip"]["clamp1"] for r in rs]) * 100
            cf = f"{c0:4.1f}/{c1:4.1f}"
        print(f"{name:11s} {len(rs):>2} {fmt(rs, ['sets','heldout','agr_caring']):>11} "
              f"{fmt(rs, ['sets','temptation','agr_caring']):>11} "
              f"{fmt(rs, ['sets','temptation','tempted_harm_rate']):>11} "
              f"{fmt(rs, ['sets','temptation','utility']):>11} "
              f"{kstar_s} {ro_id:>6} {ro_mi:>8} {cf:>10}")


if __name__ == "__main__":
    main()
