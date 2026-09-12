"""Sparse-supervision coverage schemes (review P6).

The incumbent B samples its ~20% preference pairs from the games where the
caring and selfish experts disagree. The observed miscalibration could be a
property of SPARSITY ITSELF or of WHERE those examples sit. This script
holds the objective and the pair construction FIXED (CPO-anchored, pairs =
(argmax u, argmin u) on covered games) and varies only the 20% coverage
scheme:

  rand  - a uniformly random 20% of training games
  dis   - the disagreement games (caring != selfish expert; ~20% of games;
          same LOCATION as incumbent B, different pair construction)
  gap   - the 20% of games with the LARGEST utility gap u(argmax)-u(argmin)
          (a principled informative subset: where the value matters most)

Protocol identical to b_enriched_sweep.py (hard world, 6 intended k levels,
3 seeds, eval_row) so results are directly comparable with B (r=0.50,
caring-vs-selfish pairs on disagreement games) and B_all (r=0.963, every
game, same pair construction as here).

Question (review): does the calibration failure arise from sparsity itself,
or from the location of the sparse examples?

Rows -> b_coverage_results.jsonl (schema of b_enriched_results.jsonl).

Usage: python b_coverage_schemes.py [--quick] [--schemes rand dis gap]
"""
import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from care_state_experiment import Policy, set_seed, train_clone, expert_choice
from hard_world_experiment import gen_games
from k_sweep import eval_row

HERE = os.path.dirname(os.path.abspath(__file__))


def coverage_mask(g, k, scheme, rng):
    n = len(g["self"])
    u = g["self"] + k * g["other"]
    if scheme == "rand":
        m = np.zeros(n, dtype=bool)
        m[rng.choice(n, size=max(1, int(0.2 * n)), replace=False)] = True
    elif scheme == "dis":
        m = expert_choice(g, "caring", k) != expert_choice(g, "selfish")
    elif scheme == "gap":
        gap = u.max(axis=1) - u.min(axis=1)
        thr = np.quantile(gap, 0.8)
        m = gap >= thr
    else:
        raise ValueError(scheme)
    return m


def train_b_cov(model, ref, g, k, epochs, scheme, beta=0.5, alpha=1.0, lr=1e-2,
                rng=None):
    """CPO-anchored preference tuning on pairs (argmax u, argmin u), coverage
    restricted to the scheme's 20% subset. Identical loss to train_b mode
    'all'; only the covered games differ."""
    x_all = torch.tensor(g["features"].reshape(len(g["self"]), -1),
                         dtype=torch.float32)
    m = coverage_mask(g, k, scheme, rng if rng is not None else
                      np.random.RandomState(0))
    u = g["self"] + k * g["other"]
    pref = u.argmax(axis=1)
    disp = u.argmin(axis=1)
    x = x_all[m]
    pref = torch.tensor(pref[m], dtype=torch.long)
    disp = torch.tensor(disp[m], dtype=torch.long)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(x)
    idx = torch.arange(n)
    for _ in range(epochs):
        opt.zero_grad()
        logits, _ = model(x)
        with torch.no_grad():
            ref_logits, _ = ref(x)
        lp = F.log_softmax(logits, dim=-1)
        lpr = F.log_softmax(ref_logits, dim=-1)
        margin = beta * ((lp[idx, pref] - lpr[idx, pref])
                         - (lp[idx, disp] - lpr[idx, disp]))
        loss = (-F.logsigmoid(margin).mean()
                + alpha * F.cross_entropy(logits, pref))
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def choose(model, g):
    x = torch.tensor(g["features"].reshape(len(g["self"]), -1), dtype=torch.float32)
    logits, _ = model(x)
    return logits.argmax(dim=-1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=float, nargs="+",
                    default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--schemes", nargs="+", default=["rand", "dis", "gap"])
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=str,
                    default=os.path.join(HERE, "b_coverage_results.jsonl"))
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
        cov_rng = np.random.RandomState(seed + 77)   # rand-scheme draw

        mA = Policy(hidden=128, in_dim=in_dim)
        train_clone(mA, train, "selfish", epochs)
        ref = copy.deepcopy(mA)
        for p in ref.parameters():
            p.requires_grad_(False)

        for k in ks:
            for scheme in args.schemes:
                cond = f"B_{scheme}20"
                if (cond, k, seed) in done:
                    continue
                mB = copy.deepcopy(mA)
                train_b_cov(mB, ref, train, k, epochs, scheme, rng=cov_rng)
                row = {"cond": cond, "k_intended": k, "seed": seed,
                       **eval_row(sets, [choose(mB, g) for g in sets], k)}
                with open(args.out, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"seed {seed} k={k} {cond}: k*={row['effective_k']:.2f} "
                      f"(agr {row['agreement']:.2f})", flush=True)

    print("ALL DONE. Results in", args.out, flush=True)


if __name__ == "__main__":
    main()
