"""Generate all paper figures from the final result jsonls (F1-F6).

Outputs (state_tokens/figs/):
  F1_dial_linear.png     k* vs intended k, linear world (B vs D/C)
  F1_dial_hard.png       same, hard world
  F2_safe_but_broken.png hard world: coherence (agr) vs harm, all conditions
  F3_runtime_dial.png    k* vs fed k + tempted-harm vs fed k (one model)
  F4_goodhart.png        PILLAR: harm vs pressure steps (5 subjects)
                        + dial steering spread panel
  F5_data_efficiency.png agr/utility vs n_train
  F6_noise.png           agr vs pair-noise eps (B vs D_noisy vs clean)

Run: python make_figures.py
"""
import json
import os
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

C = {"A": "#888888", "B": "#d62728", "D": "#1f77b4", "C": "#2ca02c",
     "Dial": "#9467bd", "expert": "#000000", "B_all": "#ff9896",
     "B_card": "#8c564b"}


def load(path):
    with open(os.path.join(HERE, path), encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def f1_dial(jsonl, out, title, extra_dense=False):
    rows = defaultdict(lambda: defaultdict(list))
    for r in load(jsonl):
        rows[r["cond"]][r["k_intended"]].append(r)
    if extra_dense and os.path.exists(os.path.join(HERE, "b_enriched_results.jsonl")):
        for r in load("b_enriched_results.jsonl"):
            rows[r["cond"]][r["k_intended"]].append(r)
    series = [("B_dpo", C["B"], "o", "post-hoc preference tuning (B, sparse pairs)"),
              ("D_caring", C["D"], "s", "value cloning (D)"),
              ("C_care_state", C["C"], "^", "value cloning + CARE state (C)")]
    if extra_dense:
        series += [("B_all", C["B_all"], "v", "dense preference tuning (B$_{all}$)"),
                   ("B_card", C["B_card"], "x", "dense + margins (B$_{card}$)")]
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    for cond, color, marker, label in series:
        if cond not in rows:
            continue
        ks = sorted(rows[cond])
        eff = [st.mean(r["effective_k"] for r in rows[cond][k]) for k in ks]
        lo = [st.mean(r["k_lo"] for r in rows[cond][k]) for k in ks]
        hi = [st.mean(r["k_hi"] for r in rows[cond][k]) for k in ks]
        yerr = [[max(0, e - l) for e, l in zip(eff, lo)],
                [max(0, h - e) for e, h in zip(eff, hi)]]
        ax.errorbar(ks, eff, yerr=yerr, color=color, marker=marker, ms=6,
                    capsize=3, lw=1.6, label=label)
    ax.plot([0, 8.2], [0, 8.2], "k--", lw=1, alpha=0.6, label="intended k (identity)")
    ax.axhline(6.05, color="gray", ls=":", lw=1)
    ax.text(0.3, 6.3, "identification ceiling (~6)", fontsize=8, color="gray")
    ax.set_xlabel("intended care weight k")
    ax.set_ylabel("effective care weight k*")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0, 8.2); ax.set_ylim(0, 8.2)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, out), dpi=200); plt.close(fig)


def f2_quadrant():
    by = defaultdict(lambda: defaultdict(list))
    for r in load("hard_world_results.jsonl"):
        by[r["cond"]][r["seed"]] = r["sets"]
    if os.path.exists(os.path.join(HERE, "hard_world_ball_results.jsonl")):
        for r in load("hard_world_ball_results.jsonl"):
            by[r["cond"]][r["seed"]] = r["sets"]
    conds = ["A_selfish", "B_dpo", "B_all", "D_caring", "C_care_state", "caring_expert"]
    labels = {"A_selfish": "A selfish", "B_dpo": "B post-hoc (sparse)",
              "B_all": "B dense pairs", "D_caring": "D explicit",
              "C_care_state": "C state-gated", "caring_expert": "caring expert"}
    colors = {"A_selfish": C["A"], "B_dpo": C["B"], "B_all": C["B_all"],
              "D_caring": C["D"], "C_care_state": C["C"], "caring_expert": C["expert"]}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, setname in zip(axes, ["heldout", "temptation"]):
        for c in conds:
            if c not in by:
                continue
            xs = [st.mean([by[c][s][setname]["harm_rate"] for s in by[c]]) * 100]
            ys = [st.mean([by[c][s][setname]["agr_caring"] for s in by[c]]) * 100]
            xe = [st.pstdev([by[c][s][setname]["harm_rate"] for s in by[c]]) * 100]
            ye = [st.pstdev([by[c][s][setname]["agr_caring"] for s in by[c]]) * 100]
            ax.errorbar(xs, ys, xerr=xe, yerr=ye, fmt="o", ms=9, capsize=3,
                        color=colors[c], label=labels[c])
            ax.annotate(labels[c], (xs[0], ys[0]), textcoords="offset points",
                        xytext=(8, -3), fontsize=8)
        ax.set_xlabel("harm rate (%)")
        ax.set_ylabel("value coherence: agreement w/ caring expert (%)")
        ax.set_title(f"safe-but-broken: {setname} (10 seeds)", fontsize=10)
        ax.set_xlim(-3, 60); ax.set_ylim(10, 105)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "F2_safe_but_broken.png"), dpi=200); plt.close(fig)


def f3_runtime_dial():
    rows = defaultdict(list)
    for r in load("dial_input_results.jsonl"):
        rows[r["k_input"]].append(r)
    ks = sorted(rows)
    eff = [st.mean(r["effective_k"] for r in rows[k]) for k in ks]
    eff_e = [st.pstdev(r["effective_k"] for r in rows[k]) for k in ks]
    th = [st.mean(r["tempted_harm"] for r in rows[k]) * 100 for k in ks]
    th_e = [st.pstdev(r["tempted_harm"] for r in rows[k]) * 100 for k in ks]
    train_ks = [0.0, 0.5, 1.0, 2.0, 4.0]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    ax = axes[0]
    ax.errorbar(ks, eff, yerr=eff_e, marker="o", ms=6, capsize=3, color=C["Dial"], lw=1.6)
    ax.plot([0, 8.2], [0, 8.2], "k--", lw=1, alpha=0.6)
    for tk in train_ks:
        ax.axvline(tk, color="gray", ls=":", lw=0.8)
    ax.annotate("training ks", (4.1, 0.4), fontsize=8, color="gray")
    ax.set_xlabel("dial setting fed at inference (k)")
    ax.set_ylabel("effective k*")
    ax.set_title("one model tracks the runtime dial\n(interpolates at k=3, ceiling at 8)", fontsize=10)
    ax = axes[1]
    ax.errorbar(ks, th, yerr=th_e, marker="s", ms=6, capsize=3, color=C["B"], lw=1.6)
    ax.set_xlabel("dial setting fed at inference (k)")
    ax.set_ylabel("tempted-harm rate (%)")
    ax.set_title("runtime value steering\n(no retraining)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "F3_runtime_dial.png"), dpi=200); plt.close(fig)


def f4_goodhart():
    rows = defaultdict(lambda: defaultdict(list))
    for r in load("goodhart_results.jsonl"):
        rows[r["model"]][r["steps"]].append(r)
    steps = sorted(next(iter(rows.values())))
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2))
    if os.path.exists(os.path.join(HERE, "goodhart_ball_results.jsonl")):
        for r in load("goodhart_ball_results.jsonl"):
            rows[r["model"]][r["steps"]].append(r)
    ax = axes[0]
    order = [("A_selfish", "A selfish (ceiling)", C["A"]), ("B_dpo", "B post-hoc (sparse)", C["B"]),
             ("B_all", "B dense pairs", C["B_all"]), ("D_caring", "D explicit", C["D"]),
             ("C_care_state", "C state-gated", C["C"]), ("Dial", "Dial @ k=2", C["Dial"])]
    for m, label, color in order:
        ys = [st.mean(r["tempt_harm"] for r in rows[m][s]) * 100 for s in steps]
        es = [st.pstdev(r["tempt_harm"] for r in rows[m][s]) * 100 for s in steps]
        ax.errorbar(steps, ys, yerr=es, marker="o", ms=4, capsize=2, lw=1.6,
                    color=color, label=label)
    ax.set_xlabel("optimization pressure (REINFORCE steps on self-reward)")
    ax.set_ylabel("tempted-harm rate (%)")
    ax.set_title("value survival under pressure (10 seeds)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax = axes[1]
    spread = [st.mean(r["dial_harm_k0"] - r["dial_harm_k8"] for r in rows["Dial"][s]) * 100
              for s in steps]
    spread_e = [st.pstdev((r["dial_harm_k0"] - r["dial_harm_k8"]) for r in rows["Dial"][s]) * 100
                for s in steps]
    ax.errorbar(steps, spread, yerr=spread_e, marker="D", ms=5, capsize=3,
                color=C["Dial"], lw=1.6)
    ax.set_xlabel("optimization pressure (steps)")
    ax.set_ylabel("dial steering spread (harm@k0 − harm@k8, pp)")
    ax.set_title("the dial keeps a recovery lever\nafter degradation", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "F4_goodhart.png"), dpi=200); plt.close(fig)


def f5_data_efficiency():
    rows = defaultdict(lambda: defaultdict(list))
    for r in load("data_efficiency_results.jsonl"):
        rows[r["cond"]][r["n_train"]].append(r)
    if os.path.exists(os.path.join(HERE, "b_all_dataeff_results.jsonl")):
        for r in load("b_all_dataeff_results.jsonl"):
            rows[r["cond"]][r["n_train"]].append(r)
    ns = sorted(next(iter(rows.values())))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for ax, key, ylabel, title in (
            (axes[0], "agr_caring", "value coherence: agr. caring expert (%)", "coherence vs data"),
            (axes[1], "utility", "utility on temptation @k=2", "value delivery vs data")):
        scale = 100 if key == "agr_caring" else 1
        series = [("B_dpo", "B post-hoc (sparse)", C["B"], "o"),
                  ("B_all", "B dense pairs", C["B_all"], "v"),
                  ("D_caring", "D explicit", C["D"], "s"),
                  ("C_care_state", "C state-gated", C["C"], "^")]
        for cond, label, color, marker in series:
            if cond not in rows:
                continue
            ys = [st.mean(r["sets"]["heldout"][key] for r in rows[cond][n]) * scale for n in ns]
            es = [st.pstdev(r["sets"]["heldout"][key] for r in rows[cond][n]) * scale for n in ns]
            ax.errorbar(ns, ys, yerr=es, marker=marker, ms=5, capsize=3, color=color,
                        lw=1.6, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("training games (log)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "F5_data_efficiency.png"), dpi=200); plt.close(fig)


def f6_noise():
    rows = defaultdict(lambda: defaultdict(list))
    for r in load("pair_noise_results.jsonl"):
        rows[r["cond"]][r["eps"]].append(r)
    eps = sorted(rows["B_dpo_noisy"])
    all_clean = [r for rs in rows["D_caring_clean"].values() for r in rs]
    clean_agr = st.mean(r["sets"]["heldout"]["agr_caring"] for r in all_clean) * 100
    clean_u = st.mean(r["sets"]["heldout"]["utility"] for r in all_clean)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for ax, key, scale, clean, ylabel in (
            (axes[0], "agr_caring", 100, clean_agr, "value coherence (%)"),
            (axes[1], "utility", 1, clean_u, "utility @k=2 (heldout)")):
        for cond, label, color, marker in (
                ("B_dpo_noisy", "B post-hoc (noisy pairs)", C["B"], "o"),
                ("D_caring_noisy", "D explicit (noisy labels)", C["D"], "s")):
            ys = [st.mean(r["sets"]["heldout"][key] for r in rows[cond][e]) * scale for e in eps]
            es = [st.pstdev(r["sets"]["heldout"][key] for r in rows[cond][e]) * scale for e in eps]
            ax.errorbar(eps, ys, yerr=es, marker=marker, ms=5, capsize=3, color=color,
                        lw=1.6, label=label)
        ax.axhline(clean, color=C["C"], ls="--", lw=1.4, label="D clean (by construction)")
        ax.set_xlabel("supervision noise (label flip rate)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    axes[0].set_title("B is noise-nonmonotone (unanchored);\nD degrades gracefully", fontsize=10)
    axes[1].set_title("same, utility view", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "F6_noise.png"), dpi=200); plt.close(fig)


if __name__ == "__main__":
    f1_dial("k_sweep_results.jsonl", "F1_dial_linear.png",
            "The value dial (linear world)")
    f1_dial("k_sweep_hard_results.jsonl", "F1_dial_hard.png",
            "The value dial survives nonlinearity (hard world)", extra_dense=True)
    f2_quadrant()
    f3_runtime_dial()
    f4_goodhart()
    f5_data_efficiency()
    f6_noise()
    for f in sorted(os.listdir(FIGS)):
        print("wrote", os.path.join("figs", f))
