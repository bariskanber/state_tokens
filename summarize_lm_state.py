"""Summarize the LM bridge experiment (state tokens in DistilBERT).

Dial  : k* tracking per fed k (heldout), harm steering (temptation),
        per-seed Pearson r(fed k, k*).
Care  : readout accuracy, token-swap flip rate (the literal clamp test),
        coherence (agr_caring), forced-token behavior.
Plain : coherence/k* reference (no-token D analog).

Usage: python summarize_lm_state.py
"""
import json
import math
import os
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def main():
    rows = [json.loads(l) for l in open(os.path.join(HERE, "lm_state_results.jsonl"))]
    seeds = sorted(set(r["seed"] for r in rows))
    print(f"rows {len(rows)}  seeds {seeds}")

    # ---- dial ----
    dial = [r for r in rows if r["model"] == "dial"]
    if dial:
        dseeds = sorted(set(r["seed"] for r in dial))
        print(f"\n== DIAL-LM ({len(dseeds)} seeds) ==")
        byk = defaultdict(list)
        for r in dial:
            if r["set"] == "heldout":
                byk[r["fed_k"]].append(r["kstar"])
        print("fed k -> k* (heldout, mean over seeds):")
        for k in sorted(byk):
            v = byk[k]
            print(f"  {k:>4g} -> {st.mean(v):5.2f}  (seeds: "
                  + " ".join(f"{x:.2f}" for x in sorted(v)) + ")")
        rs = []
        for s in dseeds:
            g = sorted([r for r in dial if r["seed"] == s and r["set"] == "heldout"],
                       key=lambda r: r["fed_k"])
            if len(g) > 2:
                rs.append(pearson([r["fed_k"] for r in g], [r["kstar"] for r in g]))
        if rs:
            print(f"per-seed r(fed k, k*): mean {st.mean(rs):.3f}"
                  + (f" ± {st.stdev(rs):.3f}" if len(rs) > 1 else "")
                  + f"  range {min(rs):.3f}-{max(rs):.3f}")
        harm = defaultdict(list)
        for r in dial:
            if r["set"] == "temptation":
                harm[r["fed_k"]].append(r["tempted_harm_rate"] * 100)
        print("tempted-harm steering (mean over seeds):")
        print("  " + "  ".join(f"{k:g}:{st.mean(v):.0f}%" for k, v in sorted(harm.items())))
        agrk2 = [r["agr_caring"] for r in dial if r["set"] == "heldout" and r["fed_k"] == 2.0]
        if agrk2:
            print(f"agr_caring @ fed k=2 (heldout): {st.mean(agrk2)*100:.1f}%")

    # ---- care ----
    care = [r for r in rows if r["model"] == "care"]
    if care:
        cseeds = sorted(set(r["seed"] for r in care))
        print(f"\n== CARE-LM ({len(cseeds)} seeds) ==")
        pred = [r for r in care if r["mode"] == "pred"]
        for name in ("heldout", "temptation"):
            g = [r for r in pred if r["set"] == name]
            if g:
                ro_c = st.mean([r["readout_acc"] for r in g]) * 100
                ro_x = (st.mean([r["readout_acc_neutral"] for r in g]) * 100
                        if "readout_acc_neutral" in g[0] else float("nan"))
                ro_n = (st.mean([r["readout_acc_nocare"] for r in g]) * 100
                        if "readout_acc_nocare" in g[0] else float("nan"))
                print(f"[{name}] agr {st.mean([r['agr_caring'] for r in g])*100:.1f}%  "
                      f"k* {st.mean([r['kstar'] for r in g]):.2f}  "
                      f"readout C/neutral/N: {ro_c:.0f}/{ro_x:.0f}/{ro_n:.0f}%  "
                      f"token-swap flips {st.mean([r['flip_rate'] for r in g])*100:.1f}%")
        fc = [r for r in care if r["mode"] == "force_care"]
        fn = [r for r in care if r["mode"] == "force_nocare"]
        for name in ("heldout", "temptation"):
            a = [r for r in fc if r["set"] == name]
            b = [r for r in fn if r["set"] == name]
            if a and b:
                print(f"[{name}] forced [CARE]: agr {st.mean([r['agr_caring'] for r in a])*100:.1f}% "
                      f"k* {st.mean([r['kstar'] for r in a]):.2f} | "
                      f"forced [NOCARE]: agr {st.mean([r['agr_caring'] for r in b])*100:.1f}% "
                      f"k* {st.mean([r['kstar'] for r in b]):.2f}")

    # ---- plain ----
    plain = [r for r in rows if r["model"] == "plain"]
    if plain:
        print(f"\n== PLAIN-LM ({len(set(r['seed'] for r in plain))} seeds) ==")
        for name in ("heldout", "temptation"):
            g = [r for r in plain if r["set"] == name]
            if g:
                print(f"[{name}] agr {st.mean([r['agr_caring'] for r in g])*100:.1f}%  "
                      f"k* {st.mean([r['kstar'] for r in g]):.2f}  "
                      f"tHarm {st.mean([r['tempted_harm_rate'] for r in g])*100:.1f}%")


if __name__ == "__main__":
    main()
