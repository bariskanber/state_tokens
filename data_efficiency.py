"""Data-efficiency: how does each condition behave as training data shrinks?

Sweeps n_train in {100, 300, 1000, 3000} (n=4000 covered by the main hard-world
run) for A/B/D/C in the hard world at k=2, 3 seeds. For B we also record the
number of preference pairs available (games where the caring and selfish
experts disagree) -- in reality pairs are the expensive supervision.

Rows -> data_efficiency_results.jsonl:
  {cond, n_train, seed, pairs, sets: {heldout: {...}, temptation: {...}}}
"""
import argparse
import copy
import json
import os

import numpy as np
import torch

from care_state_experiment import (K_EVAL, Policy, expert_choice, metrics,
                                   set_seed, train_clone, train_dpo)
from hard_world_experiment import gen_games

HERE = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def eval_sets(model, sets):
    out = {}
    for name, g in sets.items():
        x = torch.tensor(g["features"].reshape(len(g["self"]), -1), dtype=torch.float32)
        logits, _ = model(x)
        chosen = logits.argmax(dim=-1).numpy()
        m = metrics(g, chosen)
        m["agr_caring"] = float((chosen == expert_choice(g, "caring", K_EVAL)).mean())
        out[name] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 300, 1000, 3000])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--dpo_epochs", type=int, default=4000)
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=str,
                    default=os.path.join(HERE, "data_efficiency_results.jsonl"))
    args = ap.parse_args()

    sizes = [100, 300] if args.quick else args.sizes
    seeds = [42] if args.quick else args.seeds
    epochs = 800 if args.quick else args.epochs
    dpo_epochs = 1200 if args.quick else args.dpo_epochs
    n_eval = 300 if args.quick else args.n_eval
    in_dim = 3 * 8

    for seed in seeds:
        set_seed(seed)
        for n_train in sizes:
            rng = np.random.RandomState(seed)
            train = gen_games(n_train, rng, "train")
            sets = {name: gen_games(n_eval, np.random.RandomState(seed + 1000 + i), name)
                    for i, name in enumerate(["heldout", "temptation"])}

            pairs = int((expert_choice(train, "caring", K_EVAL)
                         != expert_choice(train, "selfish")).sum())

            rows = []
            mA = Policy(hidden=128, in_dim=in_dim)
            train_clone(mA, train, "selfish", epochs)
            rows.append({"cond": "A_selfish", "n_train": n_train, "seed": seed,
                         "pairs": pairs, "sets": eval_sets(mA, sets)})

            ref = copy.deepcopy(mA)
            for p in ref.parameters():
                p.requires_grad_(False)
            mB = copy.deepcopy(mA)
            train_dpo(mB, ref, train, K_EVAL, dpo_epochs)
            rows.append({"cond": "B_dpo", "n_train": n_train, "seed": seed,
                         "pairs": pairs, "sets": eval_sets(mB, sets)})

            mD = Policy(hidden=128, in_dim=in_dim)
            train_clone(mD, train, "caring", epochs, k=K_EVAL)
            rows.append({"cond": "D_caring", "n_train": n_train, "seed": seed,
                         "pairs": pairs, "sets": eval_sets(mD, sets)})

            mC = Policy(use_state=True, hidden=128, in_dim=in_dim)
            train_clone(mC, train, "caring", epochs, k=K_EVAL, use_state=True)
            rows.append({"cond": "C_care_state", "n_train": n_train, "seed": seed,
                         "pairs": pairs, "sets": eval_sets(mC, sets)})

            with open(args.out, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            print(f"seed {seed} n={n_train} (pairs {pairs}) done", flush=True)

    print("ALL DONE. Results in", args.out, flush=True)


if __name__ == "__main__":
    main()
