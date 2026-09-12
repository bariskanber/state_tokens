"""Combined post-hoc + CARE-token conditions (the retrofit test).

Question
--------
C (state-gated) trains the CARE channel JOINTLY from scratch. The incumbent
pipeline is post-hoc: capability training first (A), preference tuning after
(B). Can the CARE state token be installed on the post-hoc pipeline instead --
i.e., does the token still buy auditability (and change behavior) when the
policy was already trained selfish-first?

Conditions (hard world, k=2, protocols identical to hard_world_experiment.py):
  A_state    - Policy(use_state=True): selfish cloning + stake BCE, action
               head conditioned on the teacher-forced TRUE stake. Capability
               training with a reserved semantic channel (the selfish expert
               ignores stakes; the readout is supervised from the start).
  B_state    - A_state + CPO-anchored mini-DPO on SPARSE pairs (games where
               the caring and selfish experts disagree, ~20%) + stake BCE on
               those pair games only, teacher-forced stake; predicted stake at
               inference. NOTE: sparse pairs are all staked games (experts can
               only disagree where other != 0), so the state BCE sees only
               positive examples — the readout is expected to collapse.
  Bmix_state - A_state + SPARSE behavioral pairs (like B_state) but the stake
               BCE is computed on EVERY training game (mixed coverage: sparse
               behavior, dense state). Isolates whether B_state's readout
               collapse is a state-supervision-coverage effect.
  Ball_state - A_state + the same objective on DENSE pairs (every game;
               identical pair construction to b_enriched_sweep.train_b).

Compare against A/B/D/C (hard_world_results.jsonl) and B_all
(hard_world_ball_results.jsonl).

Extra measurements per condition:
  - stake-readout accuracy on every eval set (auditability, ID + OOD)
  - causal clamp test (stake forced 0/1): fraction of choices that flip vs
    the model's own readout -- steering wheel or mirror?
  - effective value k* on the heldout set (grid 0..20)

Rows -> posthoc_state_results.jsonl (schema compatible with
hard_world_results.jsonl: {cond, seed, world_seed, sets, readout, kstar,
clamp_flip}).

Usage: python posthoc_state_experiment.py [--quick]
"""
import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from care_state_experiment import (K_EVAL, Policy, expert_choice, set_seed,
                                   train_clone)
from hard_world_experiment import gen_games, evaluate, make_encoder
from goodhart_pressure import kstar

HERE = os.path.dirname(os.path.abspath(__file__))


def train_dpo_state(model, ref, g, k, epochs, mode, beta=0.5, alpha=1.0,
                    lr=1e-2, lambda_state=1.0):
    """CPO-anchored preference tuning WITH the CARE channel: teacher-forced
    true stake into the conditioning, BCE on the state readout.
    mode: 'sparse' (expert-disagreement games, like B), 'sparse_mix' (same
    behavioral pairs but stake BCE on EVERY game — mixed coverage), or
    'all' (dense pairs on every game, like B_all / train_b)."""
    x_all = torch.tensor(g["features"].reshape(len(g["self"]), -1),
                         dtype=torch.float32)
    if mode in ("sparse", "sparse_mix"):
        pref = expert_choice(g, "caring", k)
        disp = expert_choice(g, "selfish")
        mask = pref != disp
    elif mode == "all":
        u = g["self"] + k * g["other"]
        pref = u.argmax(axis=1)
        disp = u.argmin(axis=1)
        mask = np.ones(len(u), dtype=bool)
    else:
        raise ValueError(mode)
    x = x_all[mask]
    pref = torch.tensor(pref[mask], dtype=torch.long)
    disp = torch.tensor(disp[mask], dtype=torch.long)
    s_true = torch.tensor(g["stake"][mask], dtype=torch.float32)
    # mixed coverage: state BCE over every game (both stake values)
    s_all = torch.tensor(g["stake"], dtype=torch.float32) \
        if mode == "sparse_mix" else None

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(x)
    idx = torch.arange(n)
    for _ in range(epochs):
        opt.zero_grad()
        logits, state_logit = model(x, cond_stake=s_true)
        with torch.no_grad():
            ref_logits, _ = ref(x, cond_stake=s_true)
        lp = F.log_softmax(logits, dim=-1)
        lpr = F.log_softmax(ref_logits, dim=-1)
        margin = beta * ((lp[idx, pref] - lpr[idx, pref])
                         - (lp[idx, disp] - lpr[idx, disp]))
        loss = -F.logsigmoid(margin).mean() + alpha * F.cross_entropy(logits, pref)
        if s_all is not None:
            _, sl_all = model(x_all, cond_stake=s_all)
            loss = loss + lambda_state * F.binary_cross_entropy_with_logits(
                sl_all, s_all)
        else:
            loss = loss + lambda_state * F.binary_cross_entropy_with_logits(
                state_logit, s_true)
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def clamp_flip_rate(model, sets):
    """Fraction of choices that change when the stake input is clamped to
    0 or 1 (vs the model's own readout), pooled over all eval sets."""
    flips0, flips1, total = 0, 0, 0
    for g in sets.values():
        x = torch.tensor(g["features"].reshape(len(g["self"]), -1),
                         dtype=torch.float32)
        n = len(g["self"])
        base, _ = model(x)
        base = base.argmax(dim=-1)
        for mode in ("clamp0", "clamp1"):
            s = torch.full((n,), 0.0 if mode == "clamp0" else 1.0)
            logits, _ = model(x, cond_stake=s)
            ch = logits.argmax(dim=-1)
            d = float((ch != base).sum())
            if mode == "clamp0":
                flips0 += d
            else:
                flips1 += d
        total += n
    return {"clamp0": flips0 / total, "clamp1": flips1 / total}


@torch.no_grad()
def heldout_kstar(model, g):
    x = torch.tensor(g["features"].reshape(len(g["self"]), -1),
                     dtype=torch.float32)
    logits, _ = model(x)
    ch = logits.argmax(dim=-1).numpy()
    return kstar([g], [ch])


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
    ap.add_argument("--world_seed", type=int, default=12345)
    ap.add_argument("--out", type=str,
                    default=os.path.join(HERE, "posthoc_state_results.jsonl"))
    args = ap.parse_args()

    in_dim = 3 * 8
    seeds = [42] if args.quick else args.seeds
    epochs = 800 if args.quick else args.epochs
    dpo_epochs = 1200 if args.quick else args.dpo_epochs
    n_train = 600 if args.quick else args.n_train
    n_eval = 300 if args.quick else args.n_eval
    enc = make_encoder(args.world_seed)

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["cond"], r["seed"]))

    with open(args.out, "a", encoding="utf-8") as f:
        for seed in seeds:
            conds = ("A_state", "B_state", "Bmix_state", "Ball_state")
            if all((c, seed) in done for c in conds):
                continue
            set_seed(seed)
            rng = np.random.RandomState(seed)
            train = gen_games(n_train, rng, "train", enc=enc)
            sets = {name: gen_games(n_eval, np.random.RandomState(seed + 1000 + i),
                                    name, enc=enc)
                    for i, name in enumerate(["heldout", "temptation", "micro",
                                              "allstake"])}

            def emit(cond, model):
                ev, ro = evaluate(model, sets, use_state=True)
                row = {"cond": cond, "seed": seed, "world_seed": args.world_seed,
                       "sets": ev, "readout": ro,
                       "kstar": heldout_kstar(model, sets["heldout"]),
                       "clamp_flip": clamp_flip_rate(model, sets)}
                f.write(json.dumps(row) + "\n")
                f.flush()
                h = ev["heldout"]
                print(f"seed {seed} {cond:11s} | heldout agr {h['agr_caring']:.3f} "
                      f"harm {h['harm_rate']*100:.0f}% | k* {row['kstar']:.2f} | "
                      f"readout {ro['heldout']*100:.0f}/{ro['micro']*100:.0f} "
                      f"(ID/micro) | clamp flips "
                      f"{row['clamp_flip']['clamp0']*100:.1f}/"
                      f"{row['clamp_flip']['clamp1']*100:.1f}%", flush=True)
                return ev

            # A_state: capability training with the reserved CARE channel,
            # trained ONCE per seed; both post-hoc variants build on this
            # same base model (deterministic: first training after set_seed).
            mA = Policy(use_state=True, hidden=args.hidden, in_dim=in_dim)
            train_clone(mA, train, "selfish", epochs, use_state=True)
            if ("A_state", seed) not in done:
                emit("A_state", mA)

            # shared frozen reference for both post-hoc variants
            ref = copy.deepcopy(mA)
            for p in ref.parameters():
                p.requires_grad_(False)

            # B_state: sparse post-hoc preference tuning + CARE channel
            # (state BCE only on the pair games — all of them staked)
            if ("B_state", seed) not in done:
                mB = copy.deepcopy(mA)
                train_dpo_state(mB, ref, train, args.k, dpo_epochs, "sparse")
                emit("B_state", mB)

            # Bmix_state: sparse behavioral pairs, dense state BCE (mechanism
            # isolation: does the readout collapse follow state-supervision
            # coverage, independent of the behavioral pathology?)
            if ("Bmix_state", seed) not in done:
                mB = copy.deepcopy(mA)
                train_dpo_state(mB, ref, train, args.k, dpo_epochs, "sparse_mix")
                emit("Bmix_state", mB)

            # Ball_state: dense post-hoc preference tuning + CARE channel
            if ("Ball_state", seed) not in done:
                mB = copy.deepcopy(mA)
                train_dpo_state(mB, ref, train, args.k, dpo_epochs, "all")
                emit("Ball_state", mB)

    print("ALL DONE. Results in", args.out, flush=True)


if __name__ == "__main__":
    main()
