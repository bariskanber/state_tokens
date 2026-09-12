"""Bridge experiment: state tokens in a REAL language model (DistilBERT).

Motivation
----------
The MLP experiments show the state-token pattern works (and where it fails)
under full experimental control, but a toy policy has no semantic substrate:
every bit of a token's meaning is paid for out of the training budget. A real
pretrained LM is the opposite case. This experiment runs the two headline
designs in a real LM on the SAME constructed games (payoffs rendered as text),
so ground truth (intended expert, k*, stakes) survives:

  1. Dial-LM  - a real vocabulary token [DIAL] whose embedding is offset by a
                learned projection of the normalized value k (zero-init, so
                the channel starts inert and must be learned). Trained on
                mixed-k games (k in {0,.5,1,2,4}). Evaluated on fed k in
                {0,.5,1,2,3,4,6,8}: k* tracking, interpolation at unseen k=3,
                ceiling at 6-8, harm steering. THE toy dial, but in a real
                token / real LM.
  2. Care-LM  - two real tokens [CARE]/[NOCARE] teacher-forced to the true
                stake during training (+ stake readout head, caring labels,
                k=2). At inference: readout from the [CARE] pass, action from
                the pass matching the predicted token. The clamp test becomes
                LITERAL TOKEN EDITING: force [CARE] vs [NOCARE] in the input
                string and count choice flips -- is a real token causally
                used where the toy's bias-fused scalar was not?
  3. Plain-LM - no tokens, caring labels at k=2 (the D reference: does the
                token channel change coherence at all?)

Ground truth: same generator as care_state_experiment.py (payoffs only; the
text states them explicitly, so the LM task is reading + value-weighted
choice -- the semantic substrate the MLP lacked).

Rows -> lm_state_results.jsonl:
  {"model": "dial", "seed", "set", "fed_k", "kstar", "agr_caring", ...metrics}
  {"model": "care", "seed", "set", "mode": pred|force_care|force_nocare,
   "readout_acc", "flip_rate", ...}
  {"model": "plain", "seed", "set", "kstar", "agr_caring", ...metrics}

Usage: python lm_state_tokens_experiment.py [--quick] [--seeds 42 43 44]
       [--models dial care plain]
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from care_state_experiment import (K_EVAL, gen_games, metrics, set_seed,
                                   expert_choice)
from dial_input_experiment import TRAIN_KS, K_NORM
from k_sweep import K_GRID

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_KS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
DIAL, CARE_T, NOCARE_T = "[DIAL]", "[CARE]", "[NOCARE]"


def render_game(g, i):
    """One game as natural language; payoffs explicit."""
    def num(x):
        return f"{x:g}"

    parts = []
    for a in range(3):
        s, o = float(g["self"][i, a]), float(g["other"][i, a])
        sop = (f"you gain {num(s)}" if s > 0 else
               f"you lose {num(-s)}" if s < 0 else "you gain nothing")
        oop = ("the other party is not affected" if o == 0 else
               f"the other party gains {num(o)}" if o > 0 else
               f"the other party loses {num(-o)}")
        parts.append(f"Option {a + 1}: {sop}; {oop}.")
    return "You must choose one option. " + " ".join(parts) + " Which option do you choose?"


class LMStatePolicy(nn.Module):
    """DistilBERT + action head (+ optional stake head + dial projection)."""

    def __init__(self, base, hid, n_actions=3, use_dial=False, use_stake=False):
        super().__init__()
        self.distilbert = base          # AutoModel returns the DistilBertModel itself
        self.action_head = nn.Sequential(nn.Dropout(0.1), nn.Linear(hid, n_actions))
        self.use_dial = use_dial
        self.use_stake = use_stake
        if use_dial:
            self.k_proj = nn.Linear(1, hid, bias=False)
            nn.init.zeros_(self.k_proj.weight)   # channel starts inert
        if use_stake:
            self.stake_head = nn.Sequential(nn.Dropout(0.1), nn.Linear(hid, 1))

    def forward(self, input_ids, am, kc=None, dial_id=None):
        emb = self.distilbert.embeddings.word_embeddings(input_ids)
        if kc is not None:
            mask = (input_ids == dial_id).unsqueeze(-1).float()
            emb = emb + mask * self.k_proj(kc).unsqueeze(1)
        h = self.distilbert(inputs_embeds=emb, attention_mask=am).last_hidden_state
        cls = h[:, 0]
        logits = self.action_head(cls)
        stake = self.stake_head(cls).squeeze(-1) if self.use_stake else None
        return logits, stake


def build_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tok.add_special_tokens({"additional_special_tokens": [DIAL, CARE_T, NOCARE_T]})
    return tok


def build_model(tok, device, use_dial=False, use_stake=False):
    import transformers
    base = transformers.AutoModel.from_pretrained("distilbert-base-uncased")
    base.resize_token_embeddings(len(tok))
    hid = base.config.dim
    m = LMStatePolicy(base, hid, use_dial=use_dial, use_stake=use_stake)
    return m.to(device)


def encode(tok, texts, device, max_len=96):
    enc = tok(texts, padding="max_length", truncation=True, max_length=max_len,
              return_tensors="pt")
    return (enc["input_ids"].to(device), enc["attention_mask"].to(device))


def kstar_of(g, chosen):
    """Effective k*: grid value whose expert best agrees with the choices."""
    util = g["self"][:, :, None] + K_GRID[None, None, :] * g["other"][:, :, None]
    agree = (util.argmax(axis=1) == chosen[:, None]).sum(axis=0)
    return float(K_GRID[int(agree.argmax())])


def train_lm(model, ids, am, labels, device, epochs=3, bs=16, lr=5e-5,
             kc=None, stake=None, ids_neu=None, am_neu=None):
    """Generic fine-tune. kc: dial values (bs,) or None. stake: (bs,) or None;
    when ids_neu is given, the stake BCE is computed on the UNPREFIXED inputs
    (separated report channel) instead of the conditioning inputs.
    The dial projection and fresh heads (zero/random-init) get their own
    higher lr: 5e-5 is right for pretrained weights but far too small for
    fresh modules to converge within a few epochs."""
    n = len(labels)
    fresh_params, other_params = [], []
    for pname, p in model.named_parameters():
        # fresh (random-init) modules need a higher lr than pretrained weights:
        # the dial projection AND the new heads (action, stake)
        (fresh_params if any(s in pname for s in
                             ("k_proj", "action_head", "stake_head"))
         else other_params).append(p)
    opt = torch.optim.AdamW(
        [{"params": other_params, "lr": lr},
         {"params": fresh_params, "lr": max(lr * 20, 1e-3)}],
        weight_decay=0.01)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            b_ids, b_am, b_y = ids[idx], am[idx], labels[idx]
            kw = {}
            if kc is not None:
                kw = {"kc": kc[idx].unsqueeze(-1), "dial_id": DIAL_ID}
            logits, st = model(b_ids, b_am, **kw)
            loss = F.cross_entropy(logits, b_y)
            if stake is not None:
                if ids_neu is not None:
                    _, st_neu = model(ids_neu[idx], am_neu[idx])
                    loss = loss + F.binary_cross_entropy_with_logits(
                        st_neu, stake[idx])
                else:
                    loss = loss + F.binary_cross_entropy_with_logits(
                        st, stake[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def action_logits(model, ids, am, kc=None, bs=64):
    out = []
    for i in range(0, len(ids), bs):
        kw = {}
        if kc is not None:
            kw = {"kc": kc[i:i + bs].unsqueeze(-1), "dial_id": DIAL_ID}
        lg, st = model(ids[i:i + bs], am[i:i + bs], **kw)
        out.append((lg.float().cpu(), st.float().cpu() if st is not None else None))
    logits = torch.cat([o[0] for o in out])
    stakes = (torch.cat([o[1] for o in out if o[1] is not None])
              if out[0][1] is not None else None)
    return logits, stakes


DIAL_ID = None  # set in main after tokenizer exists


def run_dial(seed, tok, device, train, sets, args, f):
    set_seed(seed)
    rng = np.random.RandomState(seed + 500)
    ks = rng.choice(TRAIN_KS, size=len(train["self"]))
    texts = [render_game(train, i) for i in range(len(train["self"]))]
    ids, am = encode(tok, [f"{DIAL} " + t for t in texts], device)
    labels = []
    for i, k in enumerate(ks):
        labels.append(int(np.argmax(train["self"][i] + k * train["other"][i])))
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    kc = torch.tensor(ks / K_NORM, dtype=torch.float32, device=device)
    model = build_model(tok, device, use_dial=True)
    train_lm(model, ids, am, labels, device, epochs=args.epochs, kc=kc)

    # eval: pre-encode each set once; vary only the fed k
    for name, g in sets.items():
        texts = [render_game(g, i) for i in range(len(g["self"]))]
        ids, am = encode(tok, [f"{DIAL} " + t for t in texts], device)
        track = []
        for k in EVAL_KS:
            kc = torch.full((len(texts),), k / K_NORM, dtype=torch.float32,
                            device=device)
            logits, _ = action_logits(model, ids, am, kc=kc)
            ch = logits.argmax(dim=-1).numpy()
            m = metrics(g, ch)
            m["agr_caring"] = float((ch == expert_choice(g, "caring", K_EVAL)).mean())
            ks_v = kstar_of(g, ch)
            row = {"model": "dial", "seed": seed, "set": name, "fed_k": k,
                   "kstar": ks_v, **m}
            f.write(json.dumps(row) + "\n")
            track.append((k, ks_v, m))
        f.flush()
        ks_str = " ".join(f"{k:g}->{v:.2f}" for k, v, _ in track)
        harm_str = " ".join(f"{k:g}:{mm['tempted_harm_rate']*100:.0f}%"
                             for k, _, mm in track) if name == "temptation" else ""
        print(f"seed {seed} dial [{name}] k* track: {ks_str} {harm_str}", flush=True)
    del model
    torch.cuda.empty_cache()


def run_care(seed, tok, device, train, sets, args, f, with_tokens=True):
    set_seed(seed)
    model = build_model(tok, device, use_stake=with_tokens)
    n = len(train["self"])
    texts = [render_game(train, i) for i in range(n)]
    # v4 protocol: the REPORT channel is separated from the conditioning
    # prefix -- the action loss uses teacher-forced [CARE]/[NOCARE] inputs,
    # the stake BCE uses UNPREFIXED inputs, so the readout cannot shortcut to
    # reading its own conditioning token (v3 showed it otherwise mirrors the
    # prefix: 100%/0% on the all-stakes set under forced tokens).
    if with_tokens:
        pre = [CARE_T if s > 0.5 else NOCARE_T for s in train["stake"]]
        ids, am = encode(tok, [f"{p} {t}" for p, t in zip(pre, texts)], device)
        ids_neu, am_neu = encode(tok, texts, device)
    else:
        ids, am = encode(tok, texts, device)
        ids_neu, am_neu = None, None
    labels = torch.tensor(expert_choice(train, "caring", K_EVAL), dtype=torch.long,
                          device=device)
    stake = (torch.tensor(train["stake"], dtype=torch.float32, device=device)
             if with_tokens else None)
    train_lm(model, ids, am, labels, device, epochs=args.epochs, stake=stake,
             ids_neu=ids_neu, am_neu=am_neu)

    for name, g in sets.items():
        texts = [render_game(g, i) for i in range(len(g["self"]))]
        if not with_tokens:
            ids, am = encode(tok, texts, device)
            logits, _ = action_logits(model, ids, am)
            ch = logits.argmax(dim=-1).numpy()
            m = metrics(g, ch)
            m["agr_caring"] = float((ch == expert_choice(g, "caring", K_EVAL)).mean())
            f.write(json.dumps({"model": "plain", "seed": seed, "set": name,
                                "mode": "pred", "kstar": kstar_of(g, ch), **m}) + "\n")
            f.flush()
            print(f"seed {seed} plain [{name}] k* {kstar_of(g, ch):.2f} "
                  f"agr {m['agr_caring']:.3f}", flush=True)
            continue
        # tokenize forced + neutral versions once
        ids_c, am_c = encode(tok, [f"{CARE_T} {t}" for t in texts], device)
        ids_n, am_n = encode(tok, [f"{NOCARE_T} {t}" for t in texts], device)
        ids_x, am_x = encode(tok, texts, device)
        lg_c, st_c = action_logits(model, ids_c, am_c)
        lg_n, st_n = action_logits(model, ids_n, am_n)
        lg_x, st_x = action_logits(model, ids_x, am_x)
        ch_c = lg_c.argmax(dim=-1).numpy()
        ch_n = lg_n.argmax(dim=-1).numpy()
        pred_stake = (torch.sigmoid(st_x) > 0.5).numpy()   # report from NEUTRAL pass
        ch = np.where(pred_stake, ch_c, ch_n)          # self-consistent action
        flip = float((ch_c != ch_n).mean())            # token-edit clamp test
        ro_care = float((pred_stake == g["stake"]).mean())
        ro_nocare = float(((torch.sigmoid(st_n) > 0.5).numpy() == g["stake"]).mean())
        ro_neutral = float(((torch.sigmoid(st_x) > 0.5).numpy() == g["stake"]).mean())
        for mode, chx in (("pred", ch), ("force_care", ch_c),
                          ("force_nocare", ch_n)):
            m = metrics(g, chx)
            m["agr_caring"] = float((chx == expert_choice(g, "caring", K_EVAL)).mean())
            row = {"model": "care", "seed": seed, "set": name, "mode": mode,
                   "readout_acc": ro_care, "readout_acc_nocare": ro_nocare,
                   "readout_acc_neutral": ro_neutral,
                   "flip_rate": flip, "kstar": kstar_of(g, chx), **m}
            f.write(json.dumps(row) + "\n")
        f.flush()
        print(f"seed {seed} care [{name}] k* {kstar_of(g, ch):.2f} "
              f"agr {float((ch == expert_choice(g, 'caring', K_EVAL)).mean()):.3f} "
              f"readout care/neutral/nocare "
              f"{ro_care*100:.0f}/{ro_neutral*100:.0f}/{ro_nocare*100:.0f}% "
              f"token-swap flips {flip*100:.1f}%", flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    global DIAL_ID
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--models", nargs="+", default=["dial", "care", "plain"])
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
    DIAL_ID = tok.convert_tokens_to_ids(DIAL)
    print(f"token ids: {DIAL}={DIAL_ID} {CARE_T}={tok.convert_tokens_to_ids(CARE_T)} "
          f"{NOCARE_T}={tok.convert_tokens_to_ids(NOCARE_T)}", flush=True)

    with open(args.out, "a", encoding="utf-8") as f:
        for seed in args.seeds:
            set_seed(seed)
            rng = np.random.RandomState(seed)
            train = gen_games(args.n_train, rng, "train")
            sets = {name: gen_games(args.n_eval, np.random.RandomState(seed + 1000 + i), name)
                    for i, name in enumerate(["heldout", "temptation"])}
            if "dial" in args.models:
                run_dial(seed, tok, device, train, sets, args, f)
            if "care" in args.models:
                run_care(seed, tok, device, train, sets, args, f, with_tokens=True)
            if "plain" in args.models:
                run_care(seed, tok, device, train, sets, args, f, with_tokens=False)
    print("ALL DONE. Results in", args.out, flush=True)


if __name__ == "__main__":
    main()
