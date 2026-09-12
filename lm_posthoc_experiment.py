"""Post-hoc preference tuning in the LM bridge (the B conditions).

The bridge (lm_state_tokens_experiment.py) compared token-based control
(dial-LM, care-LM) against a plain fixed-value LM (the D analog) -- but NOT
against the paper's incumbent, post-hoc preference tuning. This script adds:

  B_dpo  - selfish-trained LM + CPO-anchored mini-DPO on SPARSE pairs
           (games where the caring and selfish experts disagree, ~20%;
           identical pair construction to the toy's B)
  B_all  - same base, same objective, DENSE pairs (every game; the toy's
           B_all coverage control)

Same linear-world generator as the bridge (ground truth survives: k*,
expert agreement, harm). If sparse-B is unanchored/overshooting here too,
the coverage law replicates in a real LM; if the pretrained substrate
rescues it, sparse-pair pathology is specific to from-scratch models --
either result is informative.

Rows -> lm_state_results.jsonl (schema matches plain rows):
  {"model": "B_dpo"|"B_all", "seed", "set", "kstar", "agr_caring", **metrics}

Usage: python lm_posthoc_experiment.py [--quick] [--seeds 42 43 44]
       [--conds sparse dense]
"""
import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from care_state_experiment import (K_EVAL, expert_choice, gen_games, metrics,
                                   set_seed)
from lm_state_tokens_experiment import (action_logits, build_model,
                                        build_tokenizer, encode, kstar_of,
                                        render_game, train_lm)

HERE = os.path.dirname(os.path.abspath(__file__))


def build_pairs(g, k, mode):
    if mode == "sparse":
        pref = expert_choice(g, "caring", k)
        disp = expert_choice(g, "selfish")
        mask = pref != disp
    elif mode == "dense":
        u = g["self"] + k * g["other"]
        pref = u.argmax(axis=1)
        disp = u.argmin(axis=1)
        mask = np.ones(len(u), dtype=bool)
    else:
        raise ValueError(mode)
    return pref[mask], disp[mask], mask


def train_dpo_lm(model, ref, ids, am, pref, disp, device, epochs=3, bs=8,
                 beta=0.5, alpha=1.0, lr=5e-5):
    """CPO-anchored preference tuning (matches the toy's B loss:
    -logsigmoid(beta * margin) + alpha * CE on the preferred action)."""
    fresh = [p for n, p in model.named_parameters()
             if any(s in n for s in ("action_head", "stake_head", "k_proj"))]
    other = [p for n, p in model.named_parameters()
             if not any(s in n for s in ("action_head", "stake_head", "k_proj"))]
    opt = torch.optim.AdamW(
        [{"params": other, "lr": lr}, {"params": fresh, "lr": max(lr * 20, 1e-3)}],
        weight_decay=0.01)
    pref = torch.tensor(pref, dtype=torch.long, device=device)
    disp = torch.tensor(disp, dtype=torch.long, device=device)
    n = len(pref)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            logits, _ = model(ids[idx], am[idx])
            with torch.no_grad():
                ref_logits, _ = ref(ids[idx], am[idx])
            lp = F.log_softmax(logits, dim=-1)
            lpr = F.log_softmax(ref_logits, dim=-1)
            ar = torch.arange(len(idx), device=device)
            margin = beta * ((lp[ar, pref[idx]] - lpr[ar, pref[idx]])
                             - (lp[ar, disp[idx]] - lpr[ar, disp[idx]]))
            loss = (-F.logsigmoid(margin).mean()
                    + alpha * F.cross_entropy(logits, pref[idx]))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--conds", nargs="+", default=["sparse", "dense"],
                    choices=["sparse", "dense"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--n_train", type=int, default=6000)
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=str,
                    default=os.path.join(HERE, "lm_state_results.jsonl"))
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.epochs, args.n_train, args.n_eval = [42], 1, 400, 150

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    tok = build_tokenizer()

    with open(args.out, "a", encoding="utf-8") as f:
        for seed in args.seeds:
            set_seed(seed)
            rng = np.random.RandomState(seed)
            train = gen_games(args.n_train, rng, "train")
            sets = {name: gen_games(args.n_eval,
                                    np.random.RandomState(seed + 1000 + i), name)
                    for i, name in enumerate(["heldout", "temptation"])}
            texts = [render_game(train, i) for i in range(len(train["self"]))]
            ids, am = encode(tok, texts, device)
            y_self = torch.tensor(expert_choice(train, "selfish"), dtype=torch.long,
                                  device=device)

            # shared selfish base (capability training), frozen reference
            base = build_model(tok, device)
            train_lm(base, ids, am, y_self, device, epochs=args.epochs)
            ref = copy.deepcopy(base)
            for p in ref.parameters():
                p.requires_grad_(False)

            for mode in args.conds:
                name = "B_dpo" if mode == "sparse" else "B_all"
                pref, disp, mask = build_pairs(train, K_EVAL, mode)
                mB = copy.deepcopy(base)
                train_dpo_lm(mB, ref, ids[mask], am[mask], pref, disp, device,
                             epochs=args.epochs)
                for sname, g in sets.items():
                    tx = [render_game(g, i) for i in range(len(g["self"]))]
                    gi, ga = encode(tok, tx, device)
                    logits, _ = action_logits(mB, gi, ga)
                    ch = logits.argmax(dim=-1).numpy()
                    m = metrics(g, ch)
                    m["agr_caring"] = float(
                        (ch == expert_choice(g, "caring", K_EVAL)).mean())
                    row = {"model": name, "seed": seed, "set": sname,
                           "kstar": kstar_of(g, ch), "n_pairs": int(mask.sum()), **m}
                    f.write(json.dumps(row) + "\n")
                f.flush()
                h = [json.loads(l) for l in open(args.out)]
                hr = [r for r in h if r["model"] == name and r["seed"] == seed]
                for r in hr:
                    print(f"seed {seed} {name:5s} [{r['set']}] k* {r['kstar']:.2f} "
                          f"agr {r['agr_caring']:.3f} "
                          f"tHarm {r['tempted_harm_rate']*100:.0f}%", flush=True)
                del mB
                torch.cuda.empty_cache()
            del base, ref
            torch.cuda.empty_cache()

    print("ALL DONE. Results in", args.out, flush=True)


if __name__ == "__main__":
    main()
