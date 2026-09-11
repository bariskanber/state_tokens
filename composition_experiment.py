"""Composition: CARE + UNKNOWABLE state tokens in ONE model (bridge to ABSTAIN).

Question
--------
Do constructible state tokens COMPOSE? We train a single policy that must
simultaneously externalize two states:
  UNKNOWABLE (the ABSTAIN analogue): the game's features are corrupted and
      payoff information is destroyed -> the correct action is ABSTAIN (a 4th
      action, like the QA null span), regardless of how attractive the
      (unreadable) payoffs are.
  CARE + value dial: on VALID games, behavior follows argmax(self + k*other)
      with k fed as an early-fused input (the runtime dial).

Ground truth for both states is by construction (we corrupt the features
ourselves; we know the payoffs when generating).

Training
--------
Mixed-k valid games (k in {0,.5,1,2,4}) -> caring-k expert target;
invalid games (features + Gaussian noise, sigma=3) -> ABSTAIN target.
Joint loss: CE(4 actions) + lambda*BCE(valid) + lambda*BCE(stake | valid only).

Composition / transfer tests (the point of the experiment)
----------------------------------------------------------
  T1 selectivity : valid-head vs stake-head confusion matrix (each readout
                   predicts its own state).
  T2 cross-family transfer of UNKNOWABLE (the ABSTAIN-paper result, in
                   miniature): TRAIN on mixed corruptions {gauss, perm, zero};
                   test on UNSEEN families {dropout, scale, dimshuffle}.
                   (Diagnostic first run with single-corruption training
                   showed NO transfer -- from-scratch MLPs learn the trained
                   family's statistics only; mixed training is the fair test.)
  T3 composition under conflict: INVALID games built on temptation payoffs
                   (self 4..6 available) -> must abstain anyway; and valid
                   temptation -> caring behavior. The states must not leak.
  T4 dial-abstain independence: abstain rate on invalid games vs fed k
                   (should be ~flat: UNKNOWABLE masks CARE).

Rows -> composition_results.jsonl:
  {seed, set, k, abstain_rate, agr_caring, harm_rate, mean_self, mean_other,
   utility, n, valid_readout_acc, stake_readout_acc}
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from care_state_experiment import K_EVAL, expert_choice, metrics, set_seed
from hard_world_experiment import gen_games
from k_sweep import K_GRID

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_KS = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
K_NORM = 4.0
ABSTAIN = 3  # 4th action index
SIGMA = 3.0  # training corruption magnitude


class CompPolicy(nn.Module):
    """One encoder (features + dial early-fused), three heads: 4 actions,
    stake readout, valid readout."""

    def __init__(self, hidden=128, feat_dim=3 * 8, n_actions=4):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(feat_dim + 1, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.action_head = nn.Linear(hidden, n_actions)
        self.stake_head = nn.Linear(hidden, 1)
        self.valid_head = nn.Linear(hidden, 1)

    def forward(self, x, kc):
        h = self.enc(torch.cat([x, kc.unsqueeze(-1)], dim=-1))
        return (self.action_head(h),
                self.stake_head(h).squeeze(-1),
                self.valid_head(h).squeeze(-1))


TRAIN_CORRUPTIONS = ("gauss", "perm", "zero")
UNSEEN_CORRUPTIONS = ("dropout", "scale", "dimshuffle")


def corrupt_features(features, rng, kind):
    f = features.copy()
    n = len(f)
    if kind == "gauss":            # seen in training
        return f + rng.randn(*f.shape) * SIGMA
    if kind == "perm":             # seen: permute the 3 action feature vectors
        for i in range(n):
            f[i] = f[i][rng.permutation(3)]
        return f
    if kind == "zero":             # seen: destroy all information
        return np.zeros_like(f)
    if kind == "dropout":           # UNSEEN: zero a random 50% of feature dims
        mask = rng.rand(*f.shape) < 0.5
        return f * mask
    if kind == "scale":             # UNSEEN: inflate feature magnitude 4x
        return f * 4.0
    if kind == "dimshuffle":        # UNSEEN: permute feature dims within actions
        for i in range(n):
            f[i] = f[i][:, rng.permutation(f.shape[2])]
        return f
    raise ValueError(kind)


def make_set(base_games, rng, kind):
    """Invalid variant of a base game set (features corrupted, labels unknown)."""
    return {"features": corrupt_features(base_games["features"], rng, kind),
            "self": base_games["self"], "other": base_games["other"],
            "stake": base_games["stake"], "valid": np.zeros(len(base_games["self"]))}


def dial_choice(g, k):
    util = g["self"] + k * g["other"]
    return util.argmax(axis=1)


def train(model, valid_games, ks, invalid_games, epochs, lr=1e-2, lam=1.0):
    xv = torch.tensor(valid_games["features"].reshape(len(ks), -1), dtype=torch.float32)
    xi = torch.tensor(invalid_games["features"].reshape(len(invalid_games["self"]), -1),
                      dtype=torch.float32)
    yv = torch.tensor(np.argmax(valid_games["self"] + ks[:, None] * valid_games["other"], axis=1),
                      dtype=torch.long)
    yi = torch.full((len(xi),), ABSTAIN, dtype=torch.long)
    sv = torch.tensor(valid_games["stake"], dtype=torch.float32)
    vv = torch.ones(len(xv))
    vi = torch.zeros(len(xi))
    x = torch.cat([xv, xi])
    y = torch.cat([yv, yi])
    kcv = torch.tensor(ks / K_NORM, dtype=torch.float32)
    kci = torch.full((len(xi),), np.mean(TRAIN_KS) / K_NORM, dtype=torch.float32)
    kc = torch.cat([kcv, kci])
    s = torch.cat([sv, torch.zeros(len(xi))])
    v = torch.cat([vv, vi])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        logits, s_logit, v_logit = model(x, kc)
        loss = F.cross_entropy(logits, y)
        loss = loss + lam * F.binary_cross_entropy_with_logits(v_logit, v)
        stake_mask = v > 0.5
        loss = loss + lam * F.binary_cross_entropy_with_logits(
            s_logit[stake_mask], s[stake_mask])
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def evaluate(model, g, k):
    x = torch.tensor(g["features"].reshape(len(g["self"]), -1), dtype=torch.float32)
    kc = torch.full((len(g["self"]),), k / K_NORM, dtype=torch.float32)
    logits, s_logit, v_logit = model(x, kc)
    chosen = logits.argmax(dim=-1).numpy()
    abst = float((chosen == ABSTAIN).mean())
    valid = g.get("valid", np.ones(len(chosen)))
    answered = chosen != ABSTAIN
    m = metrics(g, np.where(answered, chosen, 0))  # self/other for answered games
    agr = float((chosen[valid > .5] ==
                 np.argmax(g["self"] + k * g["other"], axis=1)[valid > .5]).mean()) \
        if (valid > .5).any() else float("nan")
    # restrict metrics to actually-answered games
    if answered.any():
        g_ans = {"features": None, "self": g["self"][answered],
                 "other": g["other"][answered], "stake": g["stake"][answered]}
        mm = metrics(g_ans, chosen[answered])
    else:
        mm = {"harm_rate": float("nan"), "help_rate": float("nan"),
              "utility": float("nan")}
    v_pred = (torch.sigmoid(v_logit) > 0.5).numpy()
    s_pred = (torch.sigmoid(s_logit) > 0.5).numpy()
    return {"abstain_rate": abst, "agr_caring": agr,
            "harm_rate": mm["harm_rate"], "utility": mm["utility"],
            "valid_readout_acc": float((v_pred == (valid > .5)).mean()),
            "stake_readout_acc": float((s_pred[valid > .5] == (g["stake"][valid > .5] > .5))
                                       .mean()) if (valid > .5).any() else float("nan"),
            "n": len(chosen)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--n_train", type=int, default=6000)
    ap.add_argument("--n_invalid", type=int, default=2000)
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--eval_ks", type=float, nargs="+", default=[0.0, 2.0, 4.0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=str,
                    default=os.path.join(HERE, "composition_results.jsonl"))
    args = ap.parse_args()

    seeds = [42] if args.quick else args.seeds
    epochs = 800 if args.quick else args.epochs
    n_train = 1200 if args.quick else args.n_train
    n_invalid = 400 if args.quick else args.n_invalid
    n_eval = 300 if args.quick else args.n_eval

    for seed in seeds:
        set_seed(seed)
        rng = np.random.RandomState(seed)
        valid_train = gen_games(n_train, rng, "train")
        ks = rng.choice(TRAIN_KS, size=n_train)
        # mixed-corruption training: equal thirds, each kind on fresh base games
        parts = []
        for ci, kind in enumerate(TRAIN_CORRUPTIONS):
            base = gen_games(n_invalid // len(TRAIN_CORRUPTIONS),
                             np.random.RandomState(seed + 55 + ci), "train")
            parts.append(make_set(base, np.random.RandomState(seed + 56 + ci), kind))
        invalid_train = {k: np.concatenate([p[k] for p in parts])
                         for k in ("features", "self", "other", "stake", "valid")}

        model = CompPolicy()
        train(model, valid_train, ks, invalid_train, epochs)

        # eval sets
        base = {"heldout": gen_games(n_eval, np.random.RandomState(seed + 1000), "heldout"),
                "temptation": gen_games(n_eval, np.random.RandomState(seed + 1001), "temptation"),
                "micro": gen_games(n_eval, np.random.RandomState(seed + 1002), "micro")}
        sets = {}
        for name, g in base.items():
            g = dict(g)
            g["valid"] = np.ones(len(g["self"]))
            sets[f"valid_{name}"] = g
            for ci, kind in enumerate(TRAIN_CORRUPTIONS + UNSEEN_CORRUPTIONS):
                sets[f"invalid_{kind}_{name}"] = make_set(
                    g, np.random.RandomState(seed + 2000 + ci * 37), kind)

        with open(args.out, "a", encoding="utf-8") as f:
            for name, g in sets.items():
                for k in args.eval_ks:
                    r = evaluate(model, g, k)
                    f.write(json.dumps({"seed": seed, "set": name, "k": k, **r}) + "\n")
        vh = evaluate(model, sets["valid_heldout"], 2.0)
        ip = evaluate(model, sets["invalid_dropout_heldout"], 2.0)
        iz = evaluate(model, sets["invalid_scale_temptation"], 2.0)
        print(f"seed {seed} | valid heldout: abst {vh['abstain_rate']*100:.1f}% agr {vh['agr_caring']*100:.0f}% | "
              f"invalid dropout (unseen): abst {ip['abstain_rate']*100:.1f}% | "
              f"invalid scale tempt (unseen): abst {iz['abstain_rate']*100:.1f}%", flush=True)

    print("ALL DONE. Results in", args.out, flush=True)


if __name__ == "__main__":
    main()
