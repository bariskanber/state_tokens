"""k-sweep in the HARD (nonlinear) world: does the value dial survive
nonlinearity? Same protocol as k_sweep.py but with hard-world games
(features = frozen MLP(payoffs) + distractors).

Rows -> k_sweep_hard_results.jsonl (same schema as k_sweep_results.jsonl).
"""
import argparse
import copy
import json
import os

import numpy as np
import torch

from care_state_experiment import Policy, set_seed, train_clone, train_dpo
from hard_world_experiment import gen_games
from k_sweep import K_GRID, eval_row

HERE = os.path.dirname(os.path.abspath(__file__))


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
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--dpo_epochs", type=int, default=4000)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=str,
                    default=os.path.join(HERE, "k_sweep_hard_results.jsonl"))
    args = ap.parse_args()

    ks = [0.5, 2.0] if args.quick else args.ks
    seeds = [42] if args.quick else args.seeds
    epochs = 800 if args.quick else args.epochs
    dpo_epochs = 1200 if args.quick else args.dpo_epochs
    n_train = 600 if args.quick else args.n_train
    n_eval = 300 if args.quick else args.n_eval
    in_dim = 3 * 8

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
            rows = []
            mB = copy.deepcopy(mA)
            train_dpo(mB, ref, train, k, dpo_epochs)
            rows.append({"cond": "B_dpo", "k_intended": k, "seed": seed,
                         **eval_row(sets, [choose(mB, g) for g in sets], k)})
            mD = Policy(hidden=128, in_dim=in_dim)
            train_clone(mD, train, "caring", epochs, k=k)
            rows.append({"cond": "D_caring", "k_intended": k, "seed": seed,
                         **eval_row(sets, [choose(mD, g) for g in sets], k)})
            mC = Policy(use_state=True, hidden=128, in_dim=in_dim)
            train_clone(mC, train, "caring", epochs, k=k, use_state=True)
            rows.append({"cond": "C_care_state", "k_intended": k, "seed": seed,
                         **eval_row(sets, [choose(mC, g) for g in sets], k)})

            with open(args.out, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            print(f"seed {seed} k={k}: " + " | ".join(
                f"{r['cond']}: k*={r['effective_k']:.2f} (agr {r['agreement']:.2f})"
                for r in rows), flush=True)

    print("ALL DONE. Results in", args.out, flush=True)


if __name__ == "__main__":
    main()
