"""Summarize care_state_results.jsonl: mean +/- std per condition and eval set.

Headline comparison (the experiment's reason to exist):
  OOD tempted-harm-rate of B_dpo (post-hoc preference tuning) vs
  D_caring (from-scratch value cloning) vs C_care_state (value cloning +
  ground-truth-supervised CARE state gating behavior), with A_selfish as
  the dangerous floor and the two experts as analytic references.
"""
import json
import os
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "care_state_results.jsonl")

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
                for k, v in m.items():
                    if isinstance(v, (int, float)):
                        rows[r["cond"]][sname][k].append(v)
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
              f"{'help%':>12s} {'utility':>12s} {'readout%':>12s}")
        for c in CONDITIONS:
            if c not in rows or sname not in rows[c]:
                continue
            d = rows[c][sname]
            print(f"{c:16s} {ms(d['harm_rate']):>12s} {ms(d['tempted_harm_rate']):>13s} "
                  f"{ms(d['help_rate']):>12s} {ms(d['utility'], pct=False):>12s} "
                  f"{ms(d['readout_acc']) if d.get('readout_acc') else '       ---':>12s}")

    # headline
    print("\n=== headline: OOD temptation, tempted-harm-rate (%) ===")
    vals = {c: st.mean(rows[c]["temptation"]["tempted_harm_rate"]) * 100
            for c in CONDITIONS if c in rows}
    for c, v in vals.items():
        print(f"  {c:16s} {v:6.1f}")
    if all(c in vals for c in ("B_dpo", "C_care_state", "D_caring")):
        print(f"\n  C (state-gated) - B (post-hoc DPO): "
              f"{vals['C_care_state'] - vals['B_dpo']:+.1f} pp")
        print(f"  D (clone, no state) - B:            "
              f"{vals['D_caring'] - vals['B_dpo']:+.1f} pp")
        print(f"  C - D:                              "
              f"{vals['C_care_state'] - vals['D_caring']:+.1f} pp")


if __name__ == "__main__":
    main()
