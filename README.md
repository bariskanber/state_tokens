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
Two additional frozen-encoder worlds, 3 seeds each. Replicates in all three
worlds: B always lowest tempted-harm (1.6–8.4%) with far-worst coherence
(agreement 38–83%); D/C always 96–99%. Worlds genuinely vary (B's coherence
38→83%), so the effect is not one adversarial world. Caveat: the small C>D
edge does not uniformly replicate at 3 seeds (w23456 temptation utility
C−D = −0.026); the C>D claim rests on the 10-seed main-world analysis and
should be reported per-world.

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
input feature). At inference the fed k steers the value (3 seeds):
- k* tracks fed k: 0→0.25–0.5, 1→0.75–1.0, 2→1.5–1.7, **3 (unseen)→2.05–2.55
  (interpolation works)**, 4→4–5, 6→6, 8→ceiling; agreement 0.67–0.95.
- Tempted-harm falls monotonically 77%→2–7% as the dial goes 0→8 — **runtime
  value control with no retraining**.
- Architectural lesson (both directions tested): late-fusing a conditioning
  scalar adds only a game-independent bias and cannot express
  argmax(self + k·other) — the dial is ignored. Early fusion (k in the encoder
  input) works. Real LLM prompts/tokens are early fusion; this toy result is a
  clean minimal demonstration of why that matters for value conditioning.

  clean minimal demonstration of why that matters for value conditioning.

### Composition: CARE + UNKNOWABLE in one model (`composition_experiment.py`)
One early-fusion model, three heads (4 actions incl. ABSTAIN, stake readout,
validity readout); trained on mixed-k valid games + mixed-corruption invalid
games {gauss, perm, zero}; tested on unseen corruptions {dropout, scale,
dimshuffle}. 3 seeds, `composition_results.jsonl`.

**Results** (abst@k=2 / validity-readout):
- **In-distribution composition works.** Valid games: 0.6% false abstention,
  95.9% dial agreement (heldout). Seen corruption (gauss): 97.9% abstain,
  97.6% readout. Dial stays ~flat on seen-invalid games (94/98/99% abstain at
  k=0/2/4): UNKNOWABLE masks CARE as designed.
- **`perm` is a control, not a corruption** (discovered during analysis):
  permuting action feature-vectors is semantics-preserving (actions are
  exchangeable), so the correct behavior is to ANSWER. The model abstains on
  only 1% of perm games *despite perm being trained as ABSTAIN* — the
  abstention trigger is information-destruction, not novelty. Accidental
  strong control for the ABSTAIN mechanism's semantics.
- **OOD transfer is partial and family-dependent**: dimshuffle 80–90%,
  scale 28–46%, dropout 12–25%. Weaker than the QA/ABSTAIN transfer result
  (71–92%) — consistent with generalization riding on pretrained
  representations (DistilBERT) rather than the from-scratch MLP here.
- **OOD entanglement (limitation)**: on unseen corruptions the dial leaks
  into abstention (scale: 0.5% abstain at k=0 vs 99% at k=4) — the two
  states are not disentangled off-manifold.

### Goodhart pressure test (`goodhart_pressure.py`) — the safety-relevant experiment

Models are trained as usual, then subjected to **optimization pressure**:
full-batch REINFORCE on self-payoff reward (the misaligned task proxy — the
setting of Qi et al. 2023's fine-tuning-breaks-alignment, in miniature).
3 seeds, checkpoints at steps {0,50,100,200,400,800}; `goodhart_results.jsonl`.

**No method is immune — but the ordering is stark:**

| model | harm @0 → @800 | k* @0 → @800 | agr-caring @0 → @800 |
|---|---|---|---|
| A (selfish ceiling) | 90.1 → 91.3% | 0 → 0 | — |
| **B (post-hoc pref.)** | **1.1 → 49.1%** | **5.72 → 1.32** | 43.0 → 83.6% |
| D (explicit value) | 37.8 → 49.7% | 2.00 → 1.33 | 97.5 → 90.0% |
| C (state-gated) | 34.4 → 46.1% | 2.00 → 1.38 | 98.7 → **92.3%** |
| Dial (default k=2) | 48.8 → 63.6% | 1.73 → 0.77 | 93.6 → 84.5% |

Findings:
1. **Post-hoc safety is the thinnest armor**: B's harm rate explodes +48pp
   (1.1% → 49.1%) — the fastest collapse by far. Its safety was a patch, not
   a value.
2. **Explicit values degrade gracefully**: D/C lose ~12pp harm and retain
   ≥90% coherence; C retains the best agreement of all (92.3%) and its
   stake-readout survives at 84.4% (from 100%).
3. **B's paradox**: its agreement *improves* under pressure (43 → 83.6%)
   because pressure drags its unanchored k*≈5.7 overshoot down toward
   selfishness — passing *through* the intended k=2 on the way. A broken
   clock right twice a day; direct evidence the value was never pinned.
4. **The dial retains a recovery lever**: after 800 adversarial steps,
   feeding k=8 still cuts tempted-harm from 79.5% to 30.0% (a 49.6pp
   steering spread, down from 70.2pp). Weakened, not dead: a value in an
   input channel leaves a post-degradation control path that weight-baked
   values lack.
5. Honest negatives: everything degrades (no immunity); the Dial's
   *default-setting* behavior degrades most (harm@k2 63.6% @800).

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
   training run (r = 0.97 fed at inference), and a steering lever that
   survives degradation.

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
  coherent; C>D edge appears but needs more seeds.
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
python make_figures.py                    # regenerate all paper figures
```
