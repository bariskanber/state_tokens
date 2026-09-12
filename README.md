# State Tokens: Care as an Internal State (small-scale experiment)

**Framing note (2026-09-11):** tokens need not represent pieces of text. A
token can externalize an *internal state* of the model — epistemic (ABSTAIN:
"this input is unknowable"), evaluative (CARE: "another's welfare is at
stake"), parametric (a value dial k), or operational (validity/UNKNOWABLE).
This directory is an exploration of that generalization, building on the
ABSTAIN paper's QA results: state tokens as a small, auditable protocol for
injecting and reading out non-textual semantics in learned systems.

Exploratory follow-up to the ABSTAIN paper. Philosophical motivation: intelligence
without affective valuation can be dangerous (cf. Damasio's somatic marker
hypothesis; care ethics). Engineering question: instead of teaching values
*post-hoc* (preference tuning), can a value-relevant internal state — a
welfare-stake (**CARE**) state with ground truth *by construction*, exactly like
ABSTAIN's corruption-induced unknowability — be trained jointly with behavior and
used to gate it?

## Design (`care_state_experiment.py`)

- 3-action decision games. Actions are 4-dim raw features; a fixed secret linear
  projection maps them to (self payoff, other's welfare delta) — the model must
  learn the world's semantics.
- Ground-truth stake = any action with nonzero welfare delta (constructible label).
- Experts: **selfish** = argmax self; **caring** = argmax self + k·other (k = 2,
  the explicit "care dial").
- Conditions: **A** selfish cloning (dangerous baseline) · **B** post-hoc
  preference tuning on top of A (mini-DPO + SFT anchor, CPO-style) ·
  **D** caring cloning from scratch (no state head) · **C** caring cloning +
  stake-state head, action head conditioned on TRUE stake in training
  (teacher-forced) and PREDICTED stake at inference.
- OOD sets: **temptation** (harm pays 4–6, outside train range), **micro**
  (fractional stakes ±0.3–0.5; train had integers ≥1), **allstake** (every action
  carries stakes; train had 1–2 of 3).
- 3 seeds; results in `care_state_results.jsonl`; `summarize_care.py` for tables.

### Methodological note: plain DPO fails in this toy
Plain mini-DPO satisfied its margin objective while never choosing the preferred
action — it demoted the dispreferred action and let the *third* action win
(likelihood displacement, cf. Razin et al. 2024). β=2 saturated instantly
(pref-win 0.0). B therefore uses a CPO-style SFT anchor (β=0.5, α=1), which
reaches pref-win 1.0 — a fair, strong incumbent rather than a strawman.

## Results (3 seeds, mean ± std)

Tempted-harm-rate on OOD temptation: A 100% · caring expert 27.9% · **B 0.0%** ·
D 27.9% · C 27.9%.  Harm on OOD micro: A 39.1% · expert 27.4% · **B 16.3%** ·
D 25.7% · C 26.0%.  Caring utility (intended k=2) on micro: expert/D/C 1.8 ·
**B 0.7**.  C's stake-readout accuracy: 100% (heldout, temptation, allstake),
**25.9% (micro)**.

## k-sweep: the value-dial experiment (`k_sweep.py`)

**Question:** is the care level a *settable dial*, or whatever preference tuning
happens to produce? Train B/D/C at six intended care weights k ∈ {0.25, 0.5, 1,
2, 4, 8}; estimate each model's **effective k\*** = the k whose expert
(argmax self + k·other) best agrees with its actual choices (grid 0–20;
3 seeds; `k_sweep_results.jsonl`, `summarize_k_sweep.py`, figure `k_sweep.png`).

**Results** (agreement = how well the *best* single k\* explains the policy):

| intended k | 0.25 | 0.5 | 1 | 2 | 4 | 8 | corr(k, k\*) |
|---|---|---|---|---|---|---|---|
| B_dpo k\* | 0.38 | 1.75 | 5.05 | 6.03 | 6.05 | 6.05 | 0.64 |
| D_caring k\* | 0.25 | 0.50 | 1.00 | 2.00 | 4.00 | 6.05† | **0.99** |
| C_care_state k\* | 0.30 | 0.50 | 1.00 | 2.00 | 4.00 | 6.05† | **0.99** |

† k=8 saturates at the game's *identification ceiling*: with integer payoffs
(max self-difference 6, min other-difference 1), no expert above k≈6 is
distinguishable — 6.05 is the measurement's resolution limit, not a policy
failure (D/C agreement stays 1.00; they behave exactly like a k≥6 expert).

- **D/C: the dial works.** Exact tracking across a 16× range, zero variance
  across seeds, agreement 0.99–1.00 — the policy *is* the specified tradeoff.
- **B: no dial.** Overshoots at every level (0.5→1.75, 1→5.05), then pins at
  maximal care for any k ≥ 2: two modes ("weak" / "maximal"), no control of the
  transition. Agreement ≤ 0.78 means no single tradeoff explains B's policy at
  all — it is off the k-manifold. Cost: at intended k=1, B realizes utility
  1.13 vs D's 2.00 (burns 44% of intended utility by overcaring).
- Caveat (honest): B's effective k is in principle tunable via hyperparameters
  (β, anchor weight, steps) — but only by measuring outcomes and re-tuning per
  target, indirectly and fragilely. D/C expose a semantically meaningful,
  directly settable constant; that is the claim demonstrated here.

## Hard-world experiment (`hard_world_experiment.py`)

Same design, but the world's semantics are **nonlinear**: features = frozen
random MLP(payoffs) + 2 distractor dims (8-dim features per action). Payoffs
are still sampled identically; the model must learn to invert a nonlinear map,
and OOD payoffs (temptation self=4..6, micro stakes) land in feature regions
never seen in training. Learnable in-distribution (D agreement with caring
expert 96.4–97.4% heldout), genuinely harder OOD.

**Results** (3 seeds; `hard_world_results.jsonl`, `summarize_hard.py`):

| OOD set | metric | A | B | D | C | expert |
|---|---|---|---|---|---|---|
| temptation | tempted harm % | 89.8 | **2.4** | 36.7 | 35.1 | 27.9 |
| temptation | utility@k=2 | 0.35 | 1.03 | 1.49 | **1.51** | 1.56 |
| temptation | agr. caring % | 38.1 | 61.6 | 89.2 | **91.7** | 100 |
| micro | harm % | 39.1 | **23.7** | 27.7 | 27.1 | 27.4 |
| micro | utility@k=2 | 1.69 | 0.29 | 1.74 | **1.75** | 1.77 |
| heldout | utility@k=2 | 1.7 | 0.8 | 2.1 | 2.1 | 2.1 |
| heldout | agr. caring % | 80.4 | 38.0 | 96.9 | **98.0** | 100 |

Findings:

1. **Post-hoc tuning degrades into incoherence in the hard world.** In the
   linear world B overshot coherently; here its behavior matches *no* expert
   (agreement with the caring expert 38% in-distribution!) while destroying
   utility everywhere (heldout 0.8 vs intended 2.1; micro 0.29 vs 1.77). B
   remains the most harm-averse, but partly through degraded competence: it
   is "safe but broken" — overcaring (effective k≈6 from the sweep) makes it
   sacrifice 3 self for +1 other even when that is net-negative at the
   intended k=2, and under nonlinear semantics it can no longer even do that
   coherently.
2. **Explicit-value training stays coherent OOD.** D/C keep 89–95% agreement
   with the intended k=2 expert on all OOD sets and deliver 94–99% of its
   utility.
3. **A small, consistent OOD edge for state-gating appears (C > D)** on every
   OOD set and metric (temptation harm 35.1 vs 36.7; agr 91.7 vs 89.2; micro
   harm 27.1 vs 27.7; allstake harm 2.5 vs 2.9) — exactly zero in the linear
   world. At 3 seeds the gap (~1–2 pp, std ~1) is suggestive, not significant;
   more seeds would be needed to claim it.
4. **C's readout is imperfect OOD here** (84.2% temptation, 30.7% micro vs
   99.7% heldout at 10 seeds): the "silent detection failure" risk is real in
   the hard world, though behavior absorbed it.

### 10-seed confirmation (seeds 42–51) + paired significance (`significance_hard.py`)

The 3-seed results replicate and the suggestive edges become significant
(paired by training seed; t-test and Wilcoxon signed-rank):

**C (state-gated) > D (ungated), 7 of 8 metrics significant:**

| metric | C | D | sign C>D | t p | Wilcoxon p |
|---|---|---|---|---|---|
| temptation utility@k=2 | 1.513 | 1.485 | **10/10** | .001 | .000 |
| tempted harm % | 35.0 | 37.9 | 9/10 | .000 | .004 |
| temptation agr. caring % | 92.2 | 88.7 | 9/10 | .000 | .004 |
| micro harm % | 28.7 | 29.4 | 9/10 | .001 | .004 |
| micro utility@k=2 | 1.770 | 1.763 | 9/10 | .000 | .006 |
| heldout utility@k=2 | 2.130 | 2.119 | **10/10** | .000 | .000 |
| heldout agr. caring % | 97.9 | 96.6 | **10/10** | .000 | .000 |
| allstake harm % | 3.0 | 3.3 | 6/10 | .059 | .051 |

Effects are small (~1–3 pp, +0.01–0.03 utility) but strikingly consistent
(9–10/10 seeds per metric). Interpretation: the explicitly supervised stake
state gives the policy an extra, noise-filtered input that acts as a mild
regularizer on *where* the caring policy applies — paying off most under
covariate shift (temptation), least on near-i.i.d. sets (allstake).

**D vs B (explicit value vs post-hoc), 10 seeds — "safe but broken"
confirmed:** B is more harm-avoidant (tempted harm 2.4 vs 37.9%, 10/10 for B)
but catastrophically incoherent: agreement with the intended expert 36.4% vs
96.6% (10/10 for D, p<.001), utility at intended k destroyed everywhere
(heldout 0.74 vs 2.12; micro 0.25 vs 1.76; all 10/10, p<.001). Evaluating
alignment only by harm-avoidance would rank B first; evaluating by *delivering
the specified values* ranks it last.

## Comprehensive-paper experiment suite (2026-09-10/11)

### Causal interventions on the binary stake state (`intervention_test.py`)
**Honest negative.** At full training, clamping the stake input (0 or 1) flips
<0.5% of choices on every eval set (3 seeds): the features make the binary
state redundant at inference. The C>D edge is therefore a *training-time
representation effect*, not an inference-time gate. This motivated the
runtime-dial design below — and connects to the interpretability literature's
mirror-vs-steering distinction. (Lesson: if you want a causal state, make the
information *only* available through the state channel.)

### World-seed robustness (`hard_world_w23456/34567.jsonl`)
Two additional frozen-encoder worlds, extended to 10 seeds each (42–51).
Replicates in all three worlds: B always lowest tempted-harm (1.3–8.2%) with
far-worst coherence (agreement 36–83%); D/C always 96–99%. Worlds genuinely
vary (B's coherence 36→83%), so the effect is not one adversarial world.
C>D (paired temptation-utility difference, 10 seeds): significant in the main
world (+0.028, 10/10, p=.0006) and world 34567 (+0.025, 10/10, p=.0016),
negligible in world 23456 (+0.004, 7/10, p=.68) — the edge is scoped to the
worlds where it replicates and the null is reported plainly.

### k-sweep in the hard world (`k_sweep_hard_results.jsonl`)
Dial tracking survives nonlinearity: D/C hit k* = intended exactly through
k=2 (0.05/0.50/1.00/2.00), slight overshoot at k=4 (4.05–5.00), ceiling at 8;
agreement 0.88–0.97. B remains erratic (k* jumps 0.05→6.05 across k; agreement
0.44–0.70) — no dial in either world.

### Data efficiency (`data_efficiency_results.jsonl`)
D/C scale gracefully: agreement 72%→96% from n=100→3000 games (utility
0.88→1.49 on temptation). **B's incoherence is not a data problem**: 40× more
preference pairs (15→609) leaves its agreement flat at ~30–40% (utility grows
but coherence doesn't). The failure is the ordinal objective, not supervision
budget. (B's harm-avoidance is present at all sizes — overshoot is cheap.)

### Supervision noise (`pair_noise_results.jsonl`)
D with 30% corrupted labels degrades gracefully (agreement 0.96→0.87, utility
2.17→2.02). B is **noise-nonmonotone**: flipping preference pairs *improves*
its agreement (0.38→0.55–0.66) and utility — the flips drag its overshooting
effective-k back toward the intended value. Nothing pins B to the specified
value, so noise moves it — sometimes accidentally "helping". Control (D_noisy)
shows graceful, monotone degradation; the contrast is the point.

### Runtime value dial (`dial_input_experiment.py`) — the capstone
ONE early-fusion model trained on mixed per-game k ∈ {0, .5, 1, 2, 4} (k as an
input feature). At inference the fed k steers the value (10 seeds, 42–51):
- k* tracks fed k (per-seed r = 0.92–0.98, mean 0.96): 0→0.34, 0.5→0.50,
  1→0.97, 2→1.62 (within 0.4 of spec through k=2; seed stds ≤0.1),
  **3 (unseen)→2.0–3.1 across seeds (interpolation works)**, 4→4.0–6.0
  (mean 4.7 — a seed-dependent overshoot; the honest seam in the tracking),
  k≥6→identification ceiling (~6.05).
- Tempted-harm falls monotonically 77%→5.6% (seed range 2–9%) as the dial
  goes 0→8 — **runtime value control with no retraining**.
- Architectural lesson (both directions tested): late-fusing a conditioning
  scalar adds only a game-independent bias and cannot express
  argmax(self + k·other) — the dial is ignored. Early fusion (k in the encoder
  input) works. Real LLM prompts/tokens are early fusion; this toy result is a
  clean minimal demonstration of why that matters for value conditioning.

### Composition: CARE + UNKNOWABLE in one model (`composition_experiment.py`)
One early-fusion model, three heads (4 actions incl. ABSTAIN, stake readout,
validity readout); trained on mixed-k valid games + mixed-corruption invalid
games {gauss, perm, zero}; tested on unseen corruptions {dropout, scale,
dimshuffle}. 10 seeds (42–51), `composition_results.jsonl`.

**Results** (abstention rates):
- **In-distribution composition works.** Valid games: 0–1% false abstention
  across seeds and fed values, 96.2%±0.4 dial agreement (heldout). Seen
  corruption (gauss): 92–100% abstention, flat across every fed k — UNKNOWABLE
  masks CARE as designed.
- **`perm` is a control, not a corruption** (discovered during analysis):
  permuting action feature-vectors is semantics-preserving (actions are
  exchangeable), so the correct behavior is to ANSWER. The model abstains on
  0–1% of heldout/micro perm games *despite perm being trained as ABSTAIN*
  (~12% on temptation games at fed k=2) — the abstention trigger is
  information-destruction, not novelty. Accidental strong control for the
  ABSTAIN mechanism's semantics.
- **Zero corruption is dial-sensitive and bimodal across seeds**: 6/10 seeds
  abstain fully at fed k=0, 4/10 still at k=2 (including one *inverted* seed
  that abstains only at k=2), none at k=4 — the fed value can override a
  trained abstention, and which way a seed resolves is not predictable from
  training alone.
- **OOD transfer is partial and family-dependent** (at fed k=2): dimshuffle
  82–90%, scale 42–49%, dropout 12–23%. Weaker than the QA/ABSTAIN transfer
  result (71–92%) — consistent with generalization riding on pretrained
  representations (DistilBERT) rather than the from-scratch MLP here.
- **OOD entanglement (limitation)**: on unseen corruptions the dial leaks into
  abstention in both directions — scale: ≤0.4% abstain at k=0 vs 96–99% at
  k=4; dimshuffle: 82–90% at k=2 collapsing to 15–27% at k=4; dropout
  transfer vanishes entirely at k=4; valid temptation games also over-abstain
  (11%±4 at k=2) — the two states are not disentangled off-manifold.

### Post-hoc CARE retrofit (`posthoc_state_experiment.py`) — combined post-hoc + token

Can the CARE channel be installed on the post-hoc pipeline (capability first,
preference tuning after) instead of trained jointly from scratch like C?
Four conditions (main hard world, k=2, 10 seeds; per-seed base model shared):

| cond | recipe | agr heldout | k* | readout ID/micro | clamp flips |
|---|---|---|---|---|---|
| A_state | selfish clone + stake BCE + conditioning | 80.6±1.0 (≈A: 80.7) | 0.00 | 97.0 / 22.8% | 0.0% |
| B_state | + sparse pairs, state BCE on pair games | 42.6±4.9 (≈B: 36.4) | 5.83 | **49.9 / 100%** | 0.5% |
| Bmix_state | + sparse pairs, state BCE on EVERY game | 39.0±3.7 (broken like B) | 5.93 | 97.4 / 26.7% | 0.3% |
| Ball_state | + dense pairs, dense state | 95.8±0.8 (≈B_all: 96.7) | 1.88 | 97.1 / 27.4% | 0.1% |

**Findings (paired stats in `significance_posthoc_state.py`):**
1. **Reserving the channel is free**: A_state ≈ A behaviorally (k*=0), readout works.
2. **Sparse preference tuning silently destroys the readout**: 49.9% ID = the
   always-on constant (~50% of heldout games are staked); degenerate 100% on
   all-stakes micro. Mechanism: sparse pairs exist only where experts disagree
   = only on staked games → state BCE sees only positive examples.
3. **Two independent coverage spectra**: Bmix_state (sparse behavior, dense
   state) fully restores the readout (+47.5pp vs B_state, p<1e-13, 10/10)
   while behavior stays broken. The readout's fate is set by state-supervision
   coverage, behavior by behavioral-pair coverage.
4. **Dense retrofit is essentially free**: Ball_state ≈ B_all (−0.9pp agr,
   p=.003), k*=1.88, readout intact — a post-hoc pipeline with dense pairs
   gets calibrated values + auditability without joint training. It does NOT
   recover from-scratch C's coherence edge (95.8 vs 97.9, p<1e-5) — consistent
   with C>D being a training-time representation effect.
5. **Mirror, not steering wheel**: clamp flips ≤0.5% in every variant (same
   as from-scratch C).

Lesson: state labels are free by construction — supervise the readout
everywhere even when behavioral pairs are sparse. And an audit readout can
fail silently (100% on all-stakes is collapse, not calibration).

### Goodhart pressure test (`goodhart_pressure.py`) — the safety-relevant experiment

Models are trained as usual, then subjected to **optimization pressure**:
full-batch REINFORCE on self-payoff reward (the misaligned task proxy — the
setting of Qi et al. 2023's fine-tuning-breaks-alignment, in miniature).
10 seeds (42–51), checkpoints at steps {0,50,100,200,400,800};
`goodhart_results.jsonl` (+ `goodhart_ball_results.jsonl` for the dense
control).

**No method is immune — but the ordering is stark:**

| model | harm @0 → @800 | k* @0 → @800 | agr-caring @800 |
|---|---|---|---|
| A (selfish ceiling) | 90.3 → 91.3% | 0 → 0 | 80.7% |
| **B (sparse post-hoc)** | **0.9 → 49.3%** | **5.85 → 1.25** | 83.5% |
| **B_all (dense post-hoc)** | 40.3 → **44.5%** | 1.94 → 1.62 | **93.0%** |
| D (explicit value) | 37.6 → 49.2% | 2.00 → 1.36 | 89.5% |
| C (state-gated) | 34.6 → 44.8% | 2.00 → 1.45 | 91.6% |
| Dial (default k=2) | 48.5 → 64.4% | — | — |

Findings:
1. **Sparse post-hoc safety is the thinnest armor**: B's harm rate explodes
   +48.3pp (0.9% → 49.3%) — the fastest collapse by far (vs D: paired
   t=33.3, p<1e-10). Its safety was a patch, not a value.
2. **The dense control is the most durable value of all**: B_all erodes only
   +4.2pp (significantly less than D's +11.6; t=13.9, p=2e-7) and keeps the
   best final agreement (93.0%). Durability tracks supervision *density*;
   recoverability tracks the *encoding channel*.
3. **B's paradox**: its agreement *improves* under pressure (43 → 84%)
   because pressure drags its unanchored k*≈5.9 overshoot down toward
   selfishness — passing *through* the intended k=2 on the way. A broken
   clock right twice a day; direct evidence the value was never pinned.
4. **The dial retains a recovery lever**: after 800 adversarial steps,
   feeding k=8 still cuts tempted-harm from 81.2% (fed k=0) to 31.1% (a
   50.1pp steering spread, down from 72.0pp). Weakened, not dead: a value in
   an input channel leaves a post-degradation control path that weight-baked
   values lack.
5. Honest negatives: everything degrades (no immunity); the Dial's
   *default-setting* behavior degrades most among value-carrying conditions
   (harm@fed-k2 64.4% @800).

## Findings (honest)

0. **The P0 richness control reframed everything (2026-09-12).** The B-vs-D
   comparisons confounded objective *form* with supervision *coverage*: B's
   preference pairs covered only the ~20% of games where the experts disagree,
   while D received a target on every game. Dense-preference controls
   (`b_enriched_sweep.py` → B_all = pairs on every game; B_card = cardinal
   margin weights) **calibrate like cloning** (r = 0.96–0.97 vs sparse-B 0.50
   in the hard world), **stay coherent** in the hard world (10 seeds: 96.7%
   agreement, best micro-stakes coherence of all conditions —
   `b_all_full.py`), **match cloning's data efficiency** at every size
   (`b_all_dataeff.py`), and **survive optimization pressure best of all
   conditions**. The pathologies attributed to preference tuning are
   *sparse-supervision* pathologies. What the explicit input channel still
   uniquely buys: auditability by construction, a runtime dial from a single
   training run (r = 0.96 fed at inference, 10 seeds), and a steering lever that
   survives degradation.
1. **The retrofit conditions extend the coverage law to the token channel**
   (10 seeds): a CARE readout installed on the post-hoc pipeline survives
   dense preference tuning (97%) but is silently destroyed by sparse tuning
   (50%, the always-on constant) — because sparse pairs cover only staked
   games, so the state supervision sees one class. Supervising the readout on
   every game fixes it (+47.5pp) with zero behavioral change: state labels are
   free by construction, so keep them dense.

1. **The initial hypothesis was not supported.** In this easy (linear, fully
   observable) world, post-hoc preference tuning generalized fine — it was the
   *most* harm-averse condition OOD. A generalization advantage for state-gated
   values needs a harder world to even be measurable.

2. **Sparse post-hoc tuning has no value dial.** B overshot the intended tradeoff
   everywhere: in-distribution help 92.7% vs intended 80.6%, and micro utility
   0.7 vs intended 1.8 — it sacrifices over twice as much as the specified
   values call for. Preference pairs encode "more caring ≻ less caring", not
   *how much* care; there is no k to set. D/C track the intended utility exactly
   on every set — the value lives in an explicit, inspectable constant. (Dense
   B_all fixes the calibration; it still needs one training run per level.)
3. **The state token buys auditability, not behavior.** C = D behaviorally, but
   C's readout lets you *ask the model when care is active* (100% in-distribution)
   — and it exposed its own granularity failure as a measurable number (26% on
   micro stakes), the concrete instantiation of the risk that detection failures
   silently disable value-gating. In this toy the behavioral damage happened to
   be absorbed by lucky extrapolation of the ungated policy head.
4. **The dangerous baseline is real**: pure capability training harms whenever
   profitable (100%), confirming the setup's premise.

## Limitations / next steps

- ~~World too easy~~ **DONE** (hard world above): nonlinearity flips the story —
  post-hoc tuning becomes incoherent while explicit-value training stays
  coherent; the C>D edge is resolved at 10 seeds (significant in the main
  world and one of two robustness worlds; scoped per-world in the paper).
- ~~Verify the dial claim directly~~ **DONE** (k-sweep above): r = 0.99 with exact
  tracking for D/C vs 0.64 with overshoot + saturation for B.
- ~~More seeds for C>D~~ **DONE** (10 seeds + paired tests above): significant on
  7/8 metrics, sign-consistent 9–10/10.
- Functional test: give the CARE state downstream consequences (choose to
  answer / abstain / intervene with asymmetric costs) to move from "readable
  state" to "functional state".
- Data-efficiency: post-hoc tuning vs joint training under limited preference
  pairs / limited scratch data.

## Run

```bash
python care_state_experiment.py --quick   # smoke
python care_state_experiment.py           # 3 seeds (easy world)
python summarize_care.py
python k_sweep.py                         # 3 seeds x 6 k values (easy world)
python summarize_k_sweep.py               # dial tables + correlation
python hard_world_experiment.py           # 3 seeds (nonlinear world)
python hard_world_experiment.py --seeds 45 46 47 48 49 50 51   # extend to 10
python summarize_hard.py && python significance_hard.py        # tables + paired tests
python intervention_test.py               # causal clamp test (honest negative)
python hard_world_experiment.py --world_seed 23456 --out hard_world_w23456.jsonl
python k_sweep_hard.py                    # dial in the hard world
python data_efficiency.py                 # n_train sweep
python pair_noise.py                      # pair-label noise + D-noise control
python dial_input_experiment.py           # runtime value dial (capstone)
python b_enriched_sweep.py                # P0 control: dense pairs / cardinal margins (dial)
python b_all_full.py                      # dense-B hard world (10 seeds) + pressure
python b_all_dataeff.py                   # dense-B data efficiency
python posthoc_state_experiment.py       # post-hoc + CARE retrofit (4 conds x 10 seeds)
python make_figures.py                    # regenerate all paper figures
```
