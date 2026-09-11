"""P0 control (review): does richer supervision close the preference-tuning gap?

The review's confound: D (explicit-value cloning) sees the exact argmax of
self + k*other on EVERY game; B (incumbent) sees only ordinal preference pairs
on the ~20% of games where the caring and selfish experts disagree. The
calibration gap (r=.99 vs .64) might be a supervision-RICHNESS artifact.

Conditions (hard world, dial/k-sweep protocol, 3 seeds, 6 k levels):
  B_sparse - incumbent (from k_sweep_hard_results.jsonl; ordinal pairs,
             disagreement games only)  [reference, not re-run]
  B_all    - dense ordinal pairs: for EVERY game, preferred = argmax
             (self + k*other), dispreferred = argmin(...). CPO-anchored.
  B_card   - cardinal margins: same dense pairs, but the DPO margin weighted
             by the utility gap (u_pref - u_disp)/mean_gap, so the loss knows
             HOW MUCH better the preferred action is.

If B_all/B_card track the dial like D (r ~ .99), the claim narrows to
"sparse ordinal supervision is unanchored". If they still fail, the
constructed-semantics claim survives a richness-matched control.

Rows -> b_enriched_results.jsonl (k_sweep schema).
"""
import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from care_state_experiment import Policy, set_seed, train_clone
from hard_world_experiment import gen_games
from k_sweep import eval_row

HERE = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def choose(model, g):
    x = torch.tensor(g["features"].reshape(len(g["self"]), -1), dtype=torch.float32)
    logits, _ = model(x)
    return logits.argmax(dim=-1).numpy()


def train_b(model, ref, g, k, epochs, mode, beta=0.5, alpha=1.0, lr=1e-2):
    """CPO-anchored preference training. mode: 'all' (dense ordinal pairs) or
    'card' (dense pairs with cardinal margin weights)."""
    x_all = torch.tensor(g["features"].reshape(len(g["self"]), -1), dtype=torch.float32)
    u = g["self"] + k * g["other"]
    pref = u.argmax(axis=1)
    disp = u.argmin(axis=1)                       # dense: every game yields a pair
    gap = u[np.arange(len(u)), pref] - u[np.arange(len(u)), disp]
    w = torch.tensor(gap / max(gap.mean(), 1e-6), dtype=torch.float32)
    pref = torch.tensor(pref, dtype=torch.long)
    disp = torch.tensor(disp, dtype=torch.long)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(x_all)
    idx = torch.arange(n)
    for _ in range(epochs):
        opt.zero_grad()
        logits, _ = model(x_all)
        with torch.no_grad():
            ref_logits, _ = ref(x_all)
        lp = F.log_softmax(logits, dim=-1)
        lpr = F.log_softmax(ref_logits, dim=-1)
        margin = beta * ((lp[idx, pref] - lpr[idx, pref])
                         - (lp[idx, disp] - lpr[idx, disp]))
        if mode == "card":
            margin = margin * w
        loss = (-F.logsigmoid(margin).mean()
                + alpha * F.cross_entropy(logits, pref))
        loss.backward()
        opt.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=float, nargs="+",
                    default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=str,
                    default=os.path.join(HERE, "b_enriched_results.jsonl"))
    args = ap.parse_args()

    ks = [0.5, 2.0] if args.quick else args.ks
    seeds = [42] if args.quick else args.seeds
    epochs = 800 if args.quick else args.epochs
    n_train = 600 if args.quick else args.n_train
    n_eval = 300 if args.quick else args.n_eval
    in_dim = 3 * 8

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["cond"], r["k_intended"], r["seed"]))

    for seed in seeds:
        set_seed(seed)
        rng = np.random.RandomState(seed)
        train = gen_games(n_train, rng, "train")
        sets = [gen_games(n_eval, np.random.RandomState(seed + 1000 + i), name)
                for i, name in enumerate(["heldout", "allstake"])]

        mA = Policy(hidden=128, in_dim=in_dim)
        train_clone(mA, train, "selfish", epochs)
        ref = copy.deepcopy(mA)
        for p in ref.parameters():
            p.requires_grad_(False)

        for k in ks:
            for cond, mode in (("B_all", "all"), ("B_card", "card")):
                if (cond, k, seed) in done:
                    continue
                mB = copy.deepcopy(mA)
                train_b(mB, ref, train, k, epochs, mode)
                row = {"cond": cond, "k_intended": k, "seed": seed,
                       **eval_row(sets, [choose(mB, g) for g in sets], k)}
                with open(args.out, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"seed {seed} k={k} {cond}: k*={row['effective_k']:.2f} "
                      f"(agr {row['agreement']:.2f})", flush=True)

    print("ALL DONE. Results in", args.out)


if __name__ == "__main__":
    main()
