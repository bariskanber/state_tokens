"""Causal intervention test: is the CARE state a steering wheel or a mirror?

Retrains C (caring clone + state head, hard world) per seed, then evaluates
behavior with the stake input to the action head:
  pred   - the model's own readout (normal inference)
  clamp0 - stake forced to 0 (no care active)  -> should drift selfish-like
  clamp1 - stake forced to 1 (care always on)  -> should drift overcaring-like

If clamping moves behavior in the predicted directions, the state token is a
CAUSAL gate on the value -- an inference-time control interface, not merely a
correlated readout. Rows -> intervention_results.jsonl:
  {seed, set, mode, ...metrics, agr_caring}
"""
import argparse
import json
import os

import numpy as np
import torch

from care_state_experiment import K_EVAL, Policy, expert_choice, metrics, set_seed, train_clone
from hard_world_experiment import gen_games

HERE = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def eval_clamped(model, g, mode):
    x = torch.tensor(g["features"].reshape(len(g["self"]), -1), dtype=torch.float32)
    n = len(g["self"])
    if mode == "pred":
        logits, _ = model(x)
    else:
        s = torch.full((n,), 0.0 if mode == "clamp0" else 1.0)
        logits, _ = model(x, cond_stake=s)
    chosen = logits.argmax(dim=-1).numpy()
    m = metrics(g, chosen)
    m["agr_caring"] = float((chosen == expert_choice(g, "caring", K_EVAL)).mean())
    m["agr_selfish"] = float((chosen == expert_choice(g, "selfish")).mean())
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=str,
                    default=os.path.join(HERE, "intervention_results.jsonl"))
    args = ap.parse_args()

    seeds = [42] if args.quick else args.seeds
    epochs = 800 if args.quick else args.epochs
    n_train = 600 if args.quick else args.n_train
    n_eval = 300 if args.quick else args.n_eval

    for seed in seeds:
        set_seed(seed)
        rng = np.random.RandomState(seed)
        train = gen_games(n_train, rng, "train")
        sets = {name: gen_games(n_eval, np.random.RandomState(seed + 1000 + i), name)
                for i, name in enumerate(["heldout", "temptation", "micro", "allstake"])}

        mC = Policy(use_state=True, hidden=128, in_dim=3 * 8)
        train_clone(mC, train, "caring", epochs, k=K_EVAL, use_state=True)

        with open(args.out, "a", encoding="utf-8") as f:
            for name, g in sets.items():
                for mode in ("pred", "clamp0", "clamp1"):
                    m = eval_clamped(mC, g, mode)
                    f.write(json.dumps({"seed": seed, "set": name, "mode": mode, **m}) + "\n")
        h = {m: eval_clamped(mC, sets["heldout"], m) for m in ("pred", "clamp0", "clamp1")}
        print(f"seed {seed} | heldout harm: clamp0 {h['clamp0']['harm_rate']*100:.1f} / "
              f"pred {h['pred']['harm_rate']*100:.1f} / clamp1 {h['clamp1']['harm_rate']*100:.1f} | "
              f"agrSelf clamp0 {h['clamp0']['agr_selfish']*100:.0f}%", flush=True)

    print("ALL DONE. Results in", args.out, flush=True)


if __name__ == "__main__":
    main()
