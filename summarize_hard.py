"""Summarize hard_world_results.jsonl (nonlinear-world care-state experiment).

Same table layout as summarize_care.py, plus the learnability diagnostics
(agr_caring / agr_selfish per eval set) and C's stake-readout accuracy.
"""
import json
import os
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "hard_world_results.jsonl")

SETS = ["heldout", "temptation", "micro", "allstake"]
CONDITIONS = ["selfish_expert", "caring_expert", "A_selfish", "B_dpo",
              "D_caring", "C_care_state"]


def main(path=RESULTS):
    rows = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            for sname, m in r["sets"].items():
                for kk, v in m.items():
                    if isinstance(v, (int, float)):
                        rows[r["cond"]][sname][kk].append(v)
            if r.get("readout"):
                for sname, acc in r["readout"].items():
                    rows[r["cond"]][sname]["readout_acc"].append(acc)

    def ms(vals, pct=True):
        if not vals:
            return "   ---"
        m = st.mean(vals) * (100 if pct else 1)
        s = st.pstdev(vals) * (100 if pct else 1)
        return f"{m:6.1f} ± {s:4.1f}"

    for sname in SETS:
        print(f"\n=== {sname} ===")
        print(f"{'condition':16s} {'harm%':>12s} {'temptedHarm%':>13s} "
              f"{'help%':>12s} {'utility':>12s} {'agrCar%':>12s} {'agrSelf%':>12s} {'readout%':>12s}")
        for c in CONDITIONS:
            if c not in rows or sname not in rows[c]:
                continue
            d = rows[c][sname]
            print(f"{c:16s} {ms(d['harm_rate']):>12s} {ms(d['tempted_harm_rate']):>13s} "
                  f"{ms(d['help_rate']):>12s} {ms(d['utility'], pct=False):>12s} "
                  f"{ms(d['agr_caring']):>12s} {ms(d['agr_selfish']):>12s} "
                  f"{ms(d['readout_acc']) if d.get('readout_acc') else '       ---':>12s}")

    print("\n=== headline (OOD): tempted-harm / utility on temptation, harm / utility on micro ===")
    for c in CONDITIONS:
        if c not in rows:
            continue
        th = st.mean(rows[c]["temptation"]["tempted_harm_rate"]) * 100
        tu = st.mean(rows[c]["temptation"]["utility"])
        mh = st.mean(rows[c]["micro"]["harm_rate"]) * 100
        mu = st.mean(rows[c]["micro"]["utility"])
        print(f"  {c:16s} tempt: harm {th:5.1f} util {tu:5.2f} | micro: harm {mh:5.1f} util {mu:5.2f}")


if __name__ == "__main__":
    main()
