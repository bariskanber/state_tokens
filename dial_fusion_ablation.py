"""Architectural ablation around early fusion (review P8).

The paper's lesson (3) argues theoretically that the value dial must form
game- and action-specific interactions (bilinear k*o terms), which a
late-fused scalar (a game-independent additive offset) cannot express. This
script supplies the empirical ablation: the SAME mixed-k dial protocol
(hard world, train ks {0,.5,1,2,4}, eval at 8 fed ks incl. unseen 3,6,8)
across five fusion architectures:

  early     - k concatenated to the input features (the paper's DialPolicy)
  late      - k concatenated to the output layer's input (known negative)
  mid       - k concatenated to the SECOND hidden layer's input
  film      - k -> (gamma, beta); h' = h * (1 + gamma) + beta after layer 1
              (FiLM-style multiplicative modulation of the hidden state)
  bilinear  - input = [x, k_tilde * x]  (explicit multiplicative interaction
              at the input; the network need not form k*x products itself)

Question: which injection sites yield causal dial control (k* tracking,
interpolation, steering)? Hypothesis: any site that permits game-dependent
multiplicative interaction (early, film, bilinear — and mid, partially)
works; purely additive output-layer fusion fails.

Rows -> dial_fusion_results.jsonl:
  {variant, seed, k_input, effective_k, agreement, tempted_harm, ...}

Usage: python dial_fusion_ablation.py [--quick] [--variants early late ...]
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from care_state_experiment import set_seed, metrics
from hard_world_experiment import gen_games
from dial_input_experiment import TRAIN_KS, K_NORM
from k_sweep import eval_row

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_KS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]


class FusionPolicy(nn.Module):
    """2x128 ReLU MLP with the dial injected at one of five sites."""

    def __init__(self, variant, hidden=128, feat_dim=24, n_actions=3):
        super().__init__()
        self.variant = variant
        h = hidden
        self.l1 = nn.Linear(feat_dim, h)
        self.l2 = nn.Linear(h, h)
        if variant == "early":
            self.l1 = nn.Linear(feat_dim + 1, h)
        elif variant == "mid":
            self.l2 = nn.Linear(h + 1, h)
        elif variant == "late":
            self.out = nn.Linear(h + 1, n_actions)
        elif variant == "film":
            self.film = nn.Linear(1, 2 * h)
        elif variant == "bilinear":
            self.l1 = nn.Linear(2 * feat_dim, h)
        else:
            raise ValueError(variant)
        if not hasattr(self, "out"):
            self.out = nn.Linear(h, n_actions)

    def forward(self, x, kc):
        k = kc.unsqueeze(-1)                      # (B,1)
        if self.variant == "early":
            h = F.relu(self.l1(torch.cat([x, k], dim=-1)))
            h = F.relu(self.l2(h))
        elif self.variant == "bilinear":
            h = F.relu(self.l1(torch.cat([x, k * x], dim=-1)))
            h = F.relu(self.l2(h))
        elif self.variant == "mid":
            h = F.relu(self.l1(x))
            h = F.relu(self.l2(torch.cat([h, k], dim=-1)))
        elif self.variant == "late":
            h = F.relu(self.l1(x))
            h = F.relu(self.l2(h))
            return self.out(torch.cat([h, k], dim=-1)), None
        elif self.variant == "film":
            g, b = self.film(k).chunk(2, dim=-1)
            h = F.relu(self.l1(x))
            h = h * (1 + g) + b                   # multiplicative modulation
            h = F.relu(self.l2(h))
        return self.out(h), None


def train_fusion(model, g, ks, epochs, lr=1e-2):
    x = torch.tensor(g["features"].reshape(len(ks), -1), dtype=torch.float32)
    util = g["self"] + ks[:, None] * g["other"]
    y = torch.tensor(util.argmax(axis=1), dtype=torch.long)
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
    ap.add_argument("--variants", nargs="+",
                    default=["early", "late", "mid", "film", "bilinear"])
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--n_train", type=int, default=6000)
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=str,
                    default=os.path.join(HERE, "dial_fusion_results.jsonl"))
    args = ap.parse_args()

    seeds = [42] if args.quick else args.seeds
    epochs = 800 if args.quick else args.epochs
    n_train = 1200 if args.quick else args.n_train
    n_eval = 300 if args.quick else args.n_eval
    in_dim = 3 * 8

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["variant"], r["seed"]))

    for seed in seeds:
        set_seed(seed)
        rng = np.random.RandomState(seed)
        train = gen_games(n_train, rng, "train")
        ks = rng.choice(TRAIN_KS, size=n_train)
        sets = [gen_games(n_eval, np.random.RandomState(seed + 1000 + i), name)
                for i, name in enumerate(["heldout", "allstake"])]
        tempt = gen_games(n_eval, np.random.RandomState(seed + 1004), "temptation")

        for variant in args.variants:
            if (variant, seed) in done:
                continue
            model = FusionPolicy(variant, hidden=128, feat_dim=in_dim)
            train_fusion(model, train, ks, epochs)
            with open(args.out, "a") as f:
                for k in EVAL_KS:
                    ch = [choose_at_k(model, g, k) for g in sets]
                    row = {"variant": variant, "seed": seed, "k_input": k,
                           **eval_row(sets, ch, k)}
                    ct = choose_at_k(model, tempt, k)
                    mt = metrics(tempt, ct)
                    row["tempted_harm"] = mt["tempted_harm_rate"]
                    f.write(json.dumps(row) + "\n")
                f.flush()
            # per-variant summary print
            rows = [json.loads(l) for l in open(args.out)
                    if f'"variant": "{variant}"' in l and f'"seed": {seed}' in l]
            track = " ".join(f"{r['k_input']:g}->{r['effective_k']:.2f}"
                             for r in sorted(rows, key=lambda r: r["k_input"]))
            harms = [r["tempted_harm"] * 100 for r in sorted(rows, key=lambda r: r["k_input"])]
            spread = max(harms) - min(harms)
            print(f"seed {seed} {variant:9s} | k* track: {track} | harm spread "
                  f"{spread:.0f}pp", flush=True)

    print("ALL DONE. Results in", args.out, flush=True)


if __name__ == "__main__":
    main()
