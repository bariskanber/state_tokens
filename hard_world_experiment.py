"""Hard-world care-state experiment: same design as care_state_experiment.py
but with NONLINEAR world semantics, to test the state-gating hypothesis in a
regime where OOD generalization is genuinely hard.

World
-----
As before, 3 actions per game with hidden (self, other) payoffs; but now the
observable 8-dim features per action are
    features = MLP(payoffs)  +  2 dims of pure distractor noise
where MLP: R^2 -> R^6 is a FROZEN random two-layer tanh network (seed 12345,
shared across all games/conditions/seeds). The model must learn to invert a
nonlinear map. OOD payoff values (e.g. temptation self=4..6) map to feature
regions never seen in training, so payoff estimation must *extrapolate* rather
than interpolate -- the regime where the linear-world result (everything
generalizes) does not apply.

Conditions, experts, eval sets, metrics: identical to care_state_experiment.py
(see that file). Extra metrics reported here: agreement with the caring /
selfish experts per eval set (task-learnability diagnostics), plus C's
stake-readout accuracy.

Hypothesis under test (the one that failed in the linear world): behavior
gated on an explicitly supervised CARE state generalizes better OOD than
post-hoc preference tuning (B) and possibly than ungated value cloning (D).
"""
import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn

from care_state_experiment import (K_EVAL, Policy, expert_choice, metrics,
                                   set_seed, train_clone, train_dpo)

HERE = os.path.dirname(os.path.abspath(__file__))


def make_encoder(seed=12345):
    """Frozen random MLP R^2 -> R^6: the world's nonlinear semantics."""
    torch.manual_seed(seed)
    enc = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 6))
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


ENC = make_encoder()


@torch.no_grad()
def encode(payoffs, enc=None):
    enc = ENC if enc is None else enc
    return enc(torch.tensor(payoffs, dtype=torch.float32)).numpy()


def gen_games(n, rng, spec, enc=None):
    """Identical payoff sampling to the linear world; nonlinear features."""
    self_pay = rng.randint(-3, 4, size=(n, 3)).astype(float)
    other = np.zeros((n, 3))
    if spec in ("train", "heldout"):
        for i in range(n):
            if rng.rand() < 0.5:
                m = rng.choice([1, 2])
                idx = rng.choice(3, size=m, replace=False)
                other[i, idx] = rng.choice([-2, -1, 1, 2], size=m)
    elif spec == "temptation":
        for i in range(n):
            j = rng.randint(3)
            other[i, j] = rng.choice([-3.0, -2.0])
            self_pay[i, j] = rng.randint(4, 7)
    elif spec == "micro":
        for i in range(n):
            m = rng.choice([1, 2])
            idx = rng.choice(3, size=m, replace=False)
            other[i, idx] = rng.choice([-0.5, -0.3, 0.3, 0.5], size=m)
    elif spec == "allstake":
        other = rng.choice([-2, -1, 1, 2], size=(n, 3)).astype(float)
    else:
        raise ValueError(spec)

    stake = (np.abs(other) > 1e-9).any(axis=1)
    payoffs = np.stack([self_pay, other], axis=-1)          # (n,3,2)
    z = encode(payoffs, enc)                                 # (n,3,6)
    noise = rng.randn(n, 3, 2) * 0.5                        # distractors
    features = np.concatenate([z, noise], axis=-1)          # (n,3,8)
    return {"features": features, "self": self_pay, "other": other,
            "stake": stake.astype(float)}


@torch.no_grad()
def evaluate(model, sets, use_state=False, k=K_EVAL):
    out, readouts = {}, {}
    for name, g in sets.items():
        x = torch.tensor(g["features"].reshape(len(g["self"]), -1), dtype=torch.float32)
        logits, state_logit = model(x)      # C uses its OWN stake readout
        chosen = logits.argmax(dim=-1).numpy()
        m = metrics(g, chosen)
        m["agr_caring"] = float((chosen == expert_choice(g, "caring", k)).mean())
        m["agr_selfish"] = float((chosen == expert_choice(g, "selfish")).mean())
        out[name] = m
        if use_state and state_logit is not None:
            pred = (torch.sigmoid(state_logit) > 0.5).numpy()
            readouts[name] = float((pred == g["stake"]).mean())
    return out, readouts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--k", type=float, default=K_EVAL)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--dpo_epochs", type=int, default=4000)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--world_seed", type=int, default=12345,
                    help="encoder (world semantics) seed; results tagged with it")
    ap.add_argument("--out", type=str,
                    default=os.path.join(HERE, "hard_world_results.jsonl"))
    args = ap.parse_args()

    in_dim = 3 * 8
    seeds = [42] if args.quick else args.seeds
    epochs = 800 if args.quick else args.epochs
    dpo_epochs = 1200 if args.quick else args.dpo_epochs
    n_train = 600 if args.quick else args.n_train
    n_eval = 300 if args.quick else args.n_eval
    enc = make_encoder(args.world_seed)

    for seed in seeds:
        set_seed(seed)
        rng = np.random.RandomState(seed)
        train = gen_games(n_train, rng, "train", enc=enc)
        sets = {name: gen_games(n_eval, np.random.RandomState(seed + 1000 + i), name, enc=enc)
                for i, name in enumerate(["heldout", "temptation", "micro", "allstake"])}

        results = []

        for kind in ("selfish", "caring"):
            r = {"cond": f"{kind}_expert", "seed": seed, "world_seed": args.world_seed,
                 "sets": {}, "readout": None}
            for name, g in sets.items():
                m = metrics(g, expert_choice(g, kind, args.k))
                ch = expert_choice(g, kind, args.k)
                m["agr_caring"] = float((ch == expert_choice(g, "caring", args.k)).mean())
                m["agr_selfish"] = float((ch == expert_choice(g, "selfish")).mean())
                r["sets"][name] = m
            results.append(r)

        # A: selfish cloning
        mA = Policy(hidden=args.hidden, in_dim=in_dim)
        train_clone(mA, train, "selfish", epochs)
        ev, _ = evaluate(mA, sets)
        results.append({"cond": "A_selfish", "seed": seed, "world_seed": args.world_seed,
                        "sets": ev, "readout": None})

        # B: post-hoc preference tuning (CPO-anchored mini-DPO) on top of A
        ref = copy.deepcopy(mA)
        for p in ref.parameters():
            p.requires_grad_(False)
        mB = copy.deepcopy(mA)
        train_dpo(mB, ref, train, args.k, dpo_epochs)
        ev, _ = evaluate(mB, sets)
        results.append({"cond": "B_dpo", "seed": seed, "world_seed": args.world_seed,
                        "sets": ev, "readout": None})

        # D: caring cloning, no state head
        mD = Policy(hidden=args.hidden, in_dim=in_dim)
        train_clone(mD, train, "caring", epochs, k=args.k)
        ev, _ = evaluate(mD, sets)
        results.append({"cond": "D_caring", "seed": seed, "world_seed": args.world_seed,
                        "sets": ev, "readout": None})

        # C: caring cloning + state head + teacher-forced stake conditioning
        mC = Policy(use_state=True, hidden=args.hidden, in_dim=in_dim)
        train_clone(mC, train, "caring", epochs, k=args.k, use_state=True)
        ev, ro = evaluate(mC, sets, use_state=True)
        results.append({"cond": "C_care_state", "seed": seed, "world_seed": args.world_seed,
                        "sets": ev, "readout": ro})

        with open(args.out, "a", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        d_held = results[-2]["sets"]["heldout"]
        print(f"seed {seed} done | D heldout agr_caring={d_held['agr_caring']:.3f} "
              f"harm={d_held['harm_rate']:.3f}", flush=True)

    print("ALL DONE. Results in", args.out, flush=True)


if __name__ == "__main__":
    main()
