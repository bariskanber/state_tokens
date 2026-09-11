"""Summarize k_sweep_results.jsonl: does effective care weight k* track the
intended k, per condition?

Reports per (cond, k_intended): effective k* (mean +/- std), the 98%-agreement
interval [k_lo, k_hi], best agreement (how well ANY single k* explains the
policy), harm/help rates, and utility at the intended k. Also the dial-
sensitivity check: correlation between intended k and effective k* per
condition (a dial should be ~1.0; no-dial ~0).
"""
import json
import os
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "k_sweep_results.jsonl")


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else float("nan")


def main(path=RESULTS):
    rows = defaultdict(list)
    all_rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows[(r["cond"], r["k_intended"])].append(r)
            all_rows.append(r)

    conds = sorted({c for c, _ in rows})
    ks = sorted({k for _, k in rows})

    for c in conds:
        print(f"\n=== {c} ===")
        print(f"{'k_int':>6s} {'k* eff':>14s} {'[k_lo, k_hi]':>18s} "
              f"{'agreement':>10s} {'harm%':>7s} {'help%':>7s} {'util@k':>8s}")
        for k in ks:
            rs = rows[(c, k)]
            if not rs:
                continue
            eff = [r["effective_k"] for r in rs]
            lo = st.mean(r["k_lo"] for r in rs)
            hi = st.mean(r["k_hi"] for r in rs)
            agr = st.mean(r["agreement"] for r in rs)
            harm = st.mean(r["harm"] for r in rs) * 100
            help_ = st.mean(r["help"] for r in rs) * 100
            util = st.mean(r["utility_at_intended"] for r in rs)
            print(f"{k:6.2f} {st.mean(eff):6.2f} ± {st.pstdev(eff):4.2f} "
                  f"[{lo:5.2f},{hi:6.2f}] {agr:10.2f} {harm:7.1f} {help_:7.1f} "
                  f"{util:8.2f}")

    print("\n=== dial sensitivity: corr(intended k, effective k*) ===")
    for c in conds:
        pts = [(r["k_intended"], r["effective_k"]) for r in all_rows
               if r["cond"] == c]
        xs, ys = zip(*pts)
        print(f"  {c:14s} r = {pearson(xs, ys):5.2f}   "
              f"(effective k* range {min(ys):.2f}..{max(ys):.2f})")


if __name__ == "__main__":
    main()
