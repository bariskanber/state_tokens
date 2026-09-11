"""Follow-through to the P0 control: does dense supervision fix the OTHER B
pathologies (hard-world coherence, pressure survival), or only calibration?

Trains ONE new condition in the hard world:
  B_all - anchored preference tuning on DENSE ordinal pairs (every game),
          identical protocol to b_enriched_sweep.py's B_all.

Experiments (matching existing protocols exactly):
  1. Hard-world eval at 10 seeds (42-51)  -> hard_world_ball_results.jsonl
     (compare vs hard_world_results.jsonl rows for A/B/D/C at 10 seeds)
  2. Pressure (REINFORCE self-reward, 800 steps, 3 seeds 42-44)
     -> goodhart_ball_results.jsonl
     (compare vs goodhart_results.jsonl)

Usage: python b_all_full.py [--quick]
"""
import argparse
import copy
import json
import os

import numpy as np
import torch

from care_state_experiment import (K_EVAL, Policy, expert_choice, metrics,
                                   set_seed, train_clone)
from hard_world_experiment import gen_games, evaluate
from goodhart_pressure import pressure_step, evaluate_checkpoint, kstar, CHECKPOINTS
from b_enriched_sweep import train_b, choose

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--pressure_seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    quick = args.quick
    seeds = [42] if quick else args.seeds
    pseeds = [42] if quick else args.pressure_seeds
    epochs = 800 if quick else args.epochs
    n_train = 600 if quick else args.n_train
    n_eval = 300 if quick else args.n_eval
    in_dim = 3 * 8

    out_hw = os.path.join(HERE, "hard_world_ball_results.jsonl")
    out_pr = os.path.join(HERE, "goodhart_ball_results.jsonl")

    # ---- 1. hard-world eval ----
    done = set()
    if os.path.exists(out_hw):
        with open(out_hw) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add(r["seed"])
    with open(out_hw, "a") as f:
        for seed in seeds:
            if seed in done:
                continue
            set_seed(seed)
            rng = np.random.RandomState(seed)
            train = gen_games(n_train, rng, "train")
            sets = {name: gen_games(n_eval, np.random.RandomState(seed + 1000 + i), name)
                    for i, name in enumerate(["heldout", "temptation", "micro", "allstake"])}
            mA = Policy(hidden=128, in_dim=in_dim)
            train_clone(mA, train, "selfish", epochs)
            ref = copy.deepcopy(mA)
            for p in ref.parameters():
                p.requires_grad_(False)
            mB = copy.deepcopy(mA)
            train_b(mB, ref, train, K_EVAL, epochs, "all")
            ev, _ = evaluate(mB, sets)
            f.write(json.dumps({"cond": "B_all", "seed": seed, "sets": ev,
                                "readout": None}) + "\n")
            f.flush()
            print(f"hw seed {seed}: B_all heldout agr {ev['heldout']['agr_caring']:.3f} "
                  f"temptHarm {ev['temptation']['tempted_harm_rate']*100:.0f}%", flush=True)

    # ---- 2. pressure ----
    done = set()
    if os.path.exists(out_pr):
        with open(out_pr) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add(r["seed"])
    with open(out_pr, "a") as f:
        for seed in pseeds:
            if seed in done:
                continue
            set_seed(seed)
            rng = np.random.RandomState(seed)
            train = gen_games(n_train, rng, "train")
            pressure_games = gen_games(2000 if not quick else 400,
                                       np.random.RandomState(seed + 66), "train")
            heldout = gen_games(n_eval, np.random.RandomState(seed + 1000), "heldout")
            tempt = gen_games(n_eval, np.random.RandomState(seed + 1001), "temptation")
            allstake = gen_games(n_eval, np.random.RandomState(seed + 1003), "allstake")
            sets_flat = [heldout, allstake]

            mA = Policy(hidden=128, in_dim=in_dim)
            train_clone(mA, train, "selfish", epochs)
            ref = copy.deepcopy(mA)
            for p in ref.parameters():
                p.requires_grad_(False)
            mB = copy.deepcopy(mA)
            train_b(mB, ref, train, K_EVAL, epochs, "all")

            opt = torch.optim.SGD(mB.parameters(), lr=5e-3)
            step = 0
            for target in [s for s in CHECKPOINTS if s <= 800]:
                while step < target:
                    pressure_step(mB, pressure_games, opt)
                    step += 1
                row = evaluate_checkpoint(mB, sets_flat, tempt, heldout, allstake,
                                          "B_all", seed, target)
                f.write(json.dumps(row) + "\n")
                f.flush()
            print(f"pressure seed {seed} done", flush=True)

    print("ALL DONE")


if __name__ == "__main__":
    main()
