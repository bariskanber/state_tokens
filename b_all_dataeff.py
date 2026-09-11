"""Richness control for the data-efficiency sweep (review follow-up).

The original sweep (data_efficiency.py) showed sparse-B flat at ~30-40%
coherence across a 40x data range while D/C rose 72%->96%. The P0 control
(b_enriched_sweep.py) showed dense supervision closes the calibration and
coherence gaps at full data. An obvious follow-up: is dense-B's advantage
also present as data shrinks, or does it need the full corpus?

This script sweeps n_train in {100, 300, 1000, 3000} for B_all (dense
ordinal pairs, identical protocol to b_enriched_sweep.py) in the hard world
at k=2, 3 seeds, with the SAME eval protocol as data_efficiency.py so rows
are directly comparable.

Rows -> b_all_dataeff_results.jsonl:
  {cond, n_train, seed, pairs, sets: {heldout: {...}, temptation: {...}}}
"""
import argparse
import copy
import json
import os

import numpy as np
import torch

from care_state_experiment import K_EVAL, Policy, expert_choice, set_seed, train_clone
from hard_world_experiment import gen_games
from data_efficiency import eval_sets
from b_enriched_sweep import train_b

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 300, 1000, 3000])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=str,
                    default=os.path.join(HERE, "b_all_dataeff_results.jsonl"))
    args = ap.parse_args()

    sizes = [100, 300] if args.quick else args.sizes
    seeds = [42] if args.quick else args.seeds
    epochs = 800 if args.quick else args.epochs
    n_eval = 300 if args.quick else args.n_eval
    in_dim = 3 * 8

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["cond"], r["n_train"], r["seed"]))

    with open(args.out, "a") as f:
        for seed in seeds:
            set_seed(seed)
            for n_train in sizes:
                if ("B_all", n_train, seed) in done:
                    continue
                rng = np.random.RandomState(seed)
                train = gen_games(n_train, rng, "train")
                sets = {name: gen_games(n_eval, np.random.RandomState(seed + 1000 + i), name)
                        for i, name in enumerate(["heldout", "temptation"])}

                pairs_dense = len(train)
                mA = Policy(hidden=128, in_dim=in_dim)
                train_clone(mA, train, "selfish", epochs)
                ref = copy.deepcopy(mA)
                for p in ref.parameters():
                    p.requires_grad_(False)
                mB = copy.deepcopy(mA)
                train_b(mB, ref, train, K_EVAL, epochs, "all")
                row = {"cond": "B_all", "n_train": n_train, "seed": seed,
                       "pairs": pairs_dense, "sets": eval_sets(mB, sets)}
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"seed {seed} n={n_train}: B_all heldout agr "
                      f"{row['sets']['heldout']['agr_caring']:.3f}", flush=True)

    print("ALL DONE")


if __name__ == "__main__":
    main()
