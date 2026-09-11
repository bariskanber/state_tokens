"""Supervision-noise robustness: what happens when the supervision is noisy?

Realistic asymmetry: preference pairs come from (noisy) human judgment, while
constructible supervision (the specified k, the ground-truth stake) is exact
by construction. We sweep pair-label noise eps in {0, .05, .1, .2, .3} for
B (pref/disp swapped on a random eps fraction) and, as a control, apply the
SAME label corruption to D's cloning targets (eps of targets replaced by the
selfish expert's choice) -- D_noisy. Clean D (noiseless by construction) is
the reference. Hard world, k=2, 3 seeds.

Rows -> pair_noise_results.jsonl: {cond, eps, seed, sets}
"""
import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from care_state_experiment import (K_EVAL, Policy, expert_choice, metrics,
                                   set_seed, train_clone)
from hard_world_experiment import gen_games

HERE = os.path.dirname(os.path.abspath(__file__))


def train_dpo_noisy(model, ref, g, k, epochs, eps, rng, beta=0.5, alpha=1.0, lr=1e-2):
    """CPO-anchored mini-DPO with eps fraction of preference pairs flipped."""
    x_all = torch.tensor(g["features"].reshape(len(g["self"]), -1), dtype=torch.float32)
    pref = expert_choice(g, "caring", k).copy()
    disp = expert_choice(g, "selfish").copy()
    mask = pref != disp
    x, pref, disp = x_all[mask], pref[mask], disp[mask]
    flip = rng.rand(len(x)) < eps
    pref[flip], disp[flip] = disp[flip], pref[flip]
    pref = torch.tensor(pref, dtype=torch.long)
    disp = torch.tensor(disp, dtype=torch.long)
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


def train_clone_noisy(model, g, epochs, eps, rng, k=K_EVAL):
    """Caring cloning with eps of targets corrupted to the selfish choice."""
    x = torch.tensor(g["features"].reshape(len(g["self"]), -1), dtype=torch.float32)
    y = expert_choice(g, "caring", k).copy()
    y_self = expert_choice(g, "selfish")
    flip = rng.rand(len(y)) < eps
    y[flip] = y_self[flip]
    y = torch.tensor(y, dtype=torch.long)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        logits, _ = model(x)
        F.cross_entropy(logits, y).backward()
        opt.step()
    return model


@torch.no_grad()
def eval_sets(model, sets):
    out = {}
    for name, g in sets.items():
        x = torch.tensor(g["features"].reshape(len(g["self"]), -1), dtype=torch.float32)
        chosen = model(x)[0].argmax(dim=-1).numpy()
        m = metrics(g, chosen)
        m["agr_caring"] = float((chosen == expert_choice(g, "caring", K_EVAL)).mean())
        out[name] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps_list", type=float, nargs="+",
                    default=[0.0, 0.05, 0.1, 0.2, 0.3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=str,
                    default=os.path.join(HERE, "pair_noise_results.jsonl"))
    args = ap.parse_args()

    eps_list = [0.0, 0.1] if args.quick else args.eps_list
    seeds = [42] if args.quick else args.seeds
    epochs = 800 if args.quick else args.epochs
    n_train = 600 if args.quick else args.n_train
    n_eval = 300 if args.quick else args.n_eval
    in_dim = 3 * 8

    for seed in seeds:
        set_seed(seed)
        rng = np.random.RandomState(seed)
        noise_rng = np.random.RandomState(seed + 777)
        train = gen_games(n_train, rng, "train")
        sets = {name: gen_games(n_eval, np.random.RandomState(seed + 1000 + i), name)
                for i, name in enumerate(["heldout", "temptation"])}

        # shared base + frozen reference for B
        mA = Policy(hidden=128, in_dim=in_dim)
        train_clone(mA, train, "selfish", epochs)
        ref = copy.deepcopy(mA)
        for p in ref.parameters():
            p.requires_grad_(False)

        # clean references (noiseless supervision, by construction)
        mD = Policy(hidden=128, in_dim=in_dim)
        train_clone(mD, train, "caring", epochs, k=K_EVAL)
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps({"cond": "D_caring_clean", "eps": 0.0, "seed": seed,
                                "sets": eval_sets(mD, sets)}) + "\n")

        for eps in eps_list:
            rows = []
            mB = copy.deepcopy(mA)
            train_dpo_noisy(mB, ref, train, K_EVAL, epochs, eps, noise_rng)
            rows.append({"cond": "B_dpo_noisy", "eps": eps, "seed": seed,
                         "sets": eval_sets(mB, sets)})
            mDn = Policy(hidden=128, in_dim=in_dim)
            train_clone_noisy(mDn, train, epochs, eps, noise_rng)
            rows.append({"cond": "D_caring_noisy", "eps": eps, "seed": seed,
                         "sets": eval_sets(mDn, sets)})

            with open(args.out, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            h = rows[0]["sets"]["heldout"]
            print(f"seed {seed} eps={eps}: B agr {h['agr_caring']:.2f} util {h['utility']:.2f} | "
                  f"Dnoisy agr {rows[1]['sets']['heldout']['agr_caring']:.2f} "
                  f"util {rows[1]['sets']['heldout']['utility']:.2f}", flush=True)

    print("ALL DONE. Results in", args.out, flush=True)


if __name__ == "__main__":
    main()
