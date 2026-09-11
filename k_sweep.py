"""k-sweep: does the value live in a settable dial, or is it whatever
preference tuning happens to produce?

For each intended care weight k in {0.25, 0.5, 1, 2, 4, 8} we train
  B : post-hoc preference tuning (CPO-anchored mini-DPO) with (caring_k >
      selfish) pairs -- the incumbent approach; k only changes WHICH games
      yield preference pairs, never the ordinal direction
  D : from-scratch cloning of the caring_k expert (no state head)
  C : same + CARE state head (teacher-forced stake conditioning)

and estimate each model's EFFECTIVE care weight k*: the k* whose expert
(argmax self + k* * other) best agrees with the model's actual choices on
held-out games (grid k* in [0, 20]). If the value is a dial, k* tracks k
(D, C). If preference tuning has no dial, k* is (near-)constant in k (B).

Rows appended to k_sweep_results.jsonl:
  cond, k_intended, seed, effective_k, agreement, k_lo, k_hi,
  harm, help, utility_at_intended
(k_lo/k_hi = range of k* within 98% of best agreement.)
"""
import argparse
import copy
import json
import os

import numpy as np
import torch

from care_state_experiment import (Policy, gen_games, set_seed, train_clone,
                                   train_dpo)

HERE = os.path.dirname(os.path.abspath(__file__))
K_GRID = np.round(np.arange(0.0, 20.0001, 0.05), 4)


@torch.no_grad()
def choose(model, g):
    x = torch.tensor(g["features"].reshape(len(g["self"]), -1), dtype=torch.float32)
    logits, _ = model(x)
    return logits.argmax(dim=-1).numpy()


def eval_row(g_sets, choices_sets, k_int):
    cs_l, co_l, harm_m, help_m = [], [], [], []
    n_total = sum(len(c) for c in choices_sets)
    agree = np.zeros_like(K_GRID)
    for g, ch in zip(g_sets, choices_sets):
        ar = np.arange(len(ch))
        cs_l.append(g["self"][ar, ch])
        co_l.append(g["other"][ar, ch])
        harm_m.append((g["other"] < 0).any(1) & (g["other"] >= 0).any(1))
        help_m.append((g["other"] > 0).any(1))
        util = g["self"][:, :, None] + K_GRID[None, None, :] * g["other"][:, :, None]
        agree += (util.argmax(axis=1) == ch[:, None]).sum(axis=0)
    agree /= n_total
    best = int(agree.argmax())
    inside = np.where(agree >= 0.98 * agree[best])[0]
    cs, co = np.concatenate(cs_l), np.concatenate(co_l)
    harm_m, help_m = np.concatenate(harm_m), np.concatenate(help_m)
    return {
        "effective_k": float(K_GRID[best]),
        "agreement": float(agree[best]),
        "k_lo": float(K_GRID[inside[0]]),
        "k_hi": float(K_GRID[inside[-1]]),
        "harm": float((co[harm_m] < 0).mean()) if harm_m.any() else float("nan"),
        "help": float((co[help_m] > 0).mean()) if help_m.any() else float("nan"),
        "utility_at_intended": float((cs + k_int * co).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=float, nargs="+",
                    default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=2500)
    ap.add_argument("--dpo_epochs", type=int, default=4000)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=str,
                    default=os.path.join(HERE, "k_sweep_results.jsonl"))
    args = ap.parse_args()

    ks = [0.5, 2.0] if args.quick else args.ks
    seeds = [42] if args.quick else args.seeds
    epochs = 600 if args.quick else args.epochs
    dpo_epochs = 1200 if args.quick else args.dpo_epochs
    n_train = 600 if args.quick else args.n_train
    n_eval = 300 if args.quick else args.n_eval

    for seed in seeds:
        set_seed(seed)
        rng = np.random.RandomState(seed)
        train = gen_games(n_train, rng, "train")
        sets = [gen_games(n_eval, np.random.RandomState(seed + 1000 + i), name)
                for i, name in enumerate(["heldout", "allstake"])]

        # B's base policy (selfish) and frozen reference: shared across k
        mA = Policy()
        train_clone(mA, train, "selfish", epochs)
        ref = copy.deepcopy(mA)
        for p in ref.parameters():
            p.requires_grad_(False)

        for k in ks:
            rows = []
            # B: preference pairs generated with intended k
            mB = copy.deepcopy(mA)
            train_dpo(mB, ref, train, k, dpo_epochs)
            ch = [choose(mB, g) for g in sets]
            rows.append({"cond": "B_dpo", "k_intended": k, "seed": seed,
                         **eval_row(sets, ch, k)})
            # D: caring_k cloning, no state head
            mD = Policy()
            train_clone(mD, train, "caring", epochs, k=k)
            ch = [choose(mD, g) for g in sets]
            rows.append({"cond": "D_caring", "k_intended": k, "seed": seed,
                         **eval_row(sets, ch, k)})
            # C: caring_k cloning + state head
            mC = Policy(use_state=True)
            train_clone(mC, train, "caring", epochs, k=k, use_state=True)
            ch = [choose(mC, g) for g in sets]
            rows.append({"cond": "C_care_state", "k_intended": k, "seed": seed,
                         **eval_row(sets, ch, k)})

            with open(args.out, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            print(f"seed {seed} k={k}: " + " | ".join(
                f"{r['cond']}: k*={r['effective_k']:.2f} (agr {r['agreement']:.2f})"
                for r in rows), flush=True)

    print("ALL DONE. Results in", args.out, flush=True)


if __name__ == "__main__":
    main()
