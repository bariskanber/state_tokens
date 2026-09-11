"""Runtime value dial: ONE model, k as a conditioning INPUT.

Motivation: the static binary-stake conditioning (C in the main experiments)
turns out NOT to be causally used at inference (clamping it flips <0.5% of
choices -- features make it redundant; see intervention_results.jsonl). The
C>D edge is therefore a training-time representation effect.

Here we instead train a SINGLE policy on MIXED care levels: each training game
gets a per-game k sampled from {0, 0.5, 1, 2, 4}; the target is the caring-k
expert's choice; the normalized k is fed as a conditioning input (reusing the
state-head input channel). The k input is then the only thing distinguishing
WHICH expert to follow for identical features, so the policy must read it.

Tests (hard world, 3 seeds -> dial_input_results.jsonl):
  1. dial tracking : effective k* (agreement grid) vs fed k, including
                     interpolation (k=3, unseen) and extrapolation (6, 8)
  2. one-model vs many-models: compare against k_sweep_hard's per-k models
  3. runtime steering: feeding different k changes behavior in the predicted
     direction on OOD sets (temptation) -- the causal test the binary state
                     failed, done with a semantically meaningful dial.

Rows: {seed, k_input, effective_k, agreement, k_lo, k_hi, harm, help,
       utility_at_fed} for heldout+allstake; plus a temptation block per
       k_input with tempted_harm_rate.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from care_state_experiment import K_EVAL, Policy, expert_choice, metrics, set_seed
from hard_world_experiment import gen_games
from k_sweep import K_GRID, eval_row

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_KS = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
K_NORM = 4.0  # kc = k / K_NORM -> [0, 1] over the training range


def dial_targets(g, ks):
    """Per-game expert choice under that game's k."""
    ar = np.arange(len(ks))
    util = g["self"] + ks[:, None] * g["other"]
    return util.argmax(axis=1), ar


class DialPolicy(nn.Module):
    """Early-fusion dial policy: the normalized k is part of the INPUT, so the
    encoder can form the products (k x action-features) needed to express
    argmax(self + k*other). (Late fusion -- appending k at the output layer --
    provably cannot: a linear head only adds a game-independent bias, and the
    quick smoke of that variant ignored the dial entirely.)"""

    def __init__(self, hidden=128, feat_dim=3 * 8, n_actions=3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feat_dim + 1, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))

    def forward(self, x, kc):
        logits = self.net(torch.cat([x, kc.unsqueeze(-1)], dim=-1))
        return logits, None


def train_dial(model, g, ks, epochs, lr=1e-2):
    x = torch.tensor(g["features"].reshape(len(ks), -1), dtype=torch.float32)
    y, _ = dial_targets(g, ks)
    y = torch.tensor(y, dtype=torch.long)
    kc = torch.tensor(ks / K_NORM, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        logits, _ = model(x, kc)
        F.cross_entropy(logits, y).backward()
        opt.step()
    return model


@torch.no_grad()
def choose_at_k(model, g, k):
    x = torch.tensor(g["features"].reshape(len(g["self"]), -1), dtype=torch.float32)
    kc = torch.full((len(g["self"]),), k / K_NORM, dtype=torch.float32)
    logits, _ = model(x, kc)
    return logits.argmax(dim=-1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--n_train", type=int, default=6000,
                    help="larger: mixed-k needs coverage of each k level")
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--eval_ks", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=str,
                    default=os.path.join(HERE, "dial_input_results.jsonl"))
    args = ap.parse_args()

    seeds = [42] if args.quick else args.seeds
    epochs = 800 if args.quick else args.epochs
    n_train = 1200 if args.quick else args.n_train
    n_eval = 300 if args.quick else args.n_eval
    in_dim = 3 * 8

    for seed in seeds:
        set_seed(seed)
        rng = np.random.RandomState(seed)
        train = gen_games(n_train, rng, "train")
        ks = rng.choice(TRAIN_KS, size=n_train)
        sets = [gen_games(n_eval, np.random.RandomState(seed + 1000 + i), name)
                for i, name in enumerate(["heldout", "allstake"])]
        tempt = gen_games(n_eval, np.random.RandomState(seed + 1004), "temptation")

        model = DialPolicy(hidden=128, feat_dim=in_dim)
        train_dial(model, train, ks, epochs)

        with open(args.out, "a", encoding="utf-8") as f:
            for k in args.eval_ks:
                ch = [choose_at_k(model, g, k) for g in sets]
                row = {"seed": seed, "k_input": k,
                       **eval_row(sets, ch, k)}
                # runtime steering on temptation
                ct = choose_at_k(model, tempt, k)
                mt = metrics(tempt, ct)
                row["tempted_harm"] = mt["tempted_harm_rate"]
                row["tempt_utility_at_k2"] = mt["utility"]
                f.write(json.dumps(row) + "\n")
                print(f"seed {seed} k={k}: k*={row['effective_k']:.2f} "
                      f"(agr {row['agreement']:.2f}) temptHarm {mt['tempted_harm_rate']*100:.0f}%",
                      flush=True)

    print("ALL DONE. Results in", args.out, flush=True)


if __name__ == "__main__":
    main()
