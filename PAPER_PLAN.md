# PAPER PLAN — State Tokens (working title options below)

Target: arXiv preprint. Companion/experiment repo: `state_tokens/`.
Prior work to build on: "All We Also Need Is ABSTAIN" (the ABSTAIN token as an
existence proof of a constructible, externalized internal state).

## Title options (post-2026-09-11 reframing: tokens need not represent text)
1. "Tokens Need Not Represent Text: State Tokens as an Interface for Values, Knowledge Limits, and Control"
2. "From ABSTAIN to CARE: State Tokens with Constructed Semantics"
3. "The Vocabulary as API: Protocol Tokens, Control Codes, and State Tokens"

## One-sentence thesis
A token can externalize an *internal state* of the model — epistemic
(ABSTAIN: "unanswerable-for-me"), evaluative (CARE: "welfare at stake"),
parametric (a value dial k), operational (UNKNOWABLE: "inputs corrupted") —
and when the state's truth condition is *constructed* in the training data
rather than estimated from text, the resulting semantics is calibrated,
auditable, runtime-controllable, and degrades gracefully under optimization
pressure, where post-hoc preference tuning is none of these.

## §2 Conceptual framework: the token taxonomy (NEW, 2026-09-11)

Cut: **what fixes the token's meaning?**

- **Protocol tokens** ([PAD]/[SEP]/[BOS]/[CLS]/[EOS]): semantics is purely
  structural — a fact about the organization of the text stream ([SEP]: a
  boundary occurs here; [CLS]: designated readout slot; [EOS]: halt). Content-
  less; identical meaning in any domain; no truth conditions.
- **Control codes / soft prompts** (CTRL, prompt tuning): semantics *assigned*
  and behavior-influencing, but descriptive — estimated from corpora; not
  auditable against ground truth.
- **State tokens** (this paper): semantics is a proposition about the model's
  relation to the task, FIXED BY CONSTRUCTED SUPERVISION — we corrupt the
  input / know the payoffs / set the dial ourselves, so the token's meaning is
  verifiable. Bidirectional: emittable as a report (readout) and consumable as
  a control (conditioning).

Spectrum table for the paper:

| | [PAD]/[SEP]/[BOS] | [CLS] | [EOS] | CTRL codes / soft prompts | state tokens |
|---|---|---|---|---|---|
| Content beyond text structure | no | weak | no ("stop") | yes (learned) | yes (constructed) |
| Verifiable truth condition | no | no | no | no | **yes, by construction** |
| Emission is an act | no | no | halts only | no | yes (abstain = refusal) |
| Conditions behavior as input | no | no | no | yes | yes |
| Normative/epistemic content | no | no | no | no | yes |

One-liner: *protocol tokens organize the medium; state tokens externalize the
model's relation to the task.* Vocabulary as API: protocol tokens are
transport-layer headers; state tokens are application-layer calls — verbs
with truth conditions. The QA/ABSTAIN paper + this suite = small feasibility
study of one such API design pattern.

## Experimental sections (experiment → claim → artifact)

| § | Experiment | Claim | Status |
|---|---|---|---|
| 4 | Easy world (linear): A/B/D/C, OOD sets | Landscape + honest concession: post-hoc fine in easy regimes | done (3 seeds) |
| 5 | k-sweep (linear) | Explicit k is a dial (r=.99, exact); preference tuning is off-manifold (r=.64, overshoot, saturation at ceiling) | done (3 seeds, fig) |
| 6 | Hard world (nonlinear), 10 seeds + paired tests | Post-hoc → "safe but broken" (agr 36% vs 97%, p<.001); explicit stays coherent; C>D significant 7/8 metrics | done |
| 7 | Causal interventions on the binary stake state | HONEST NEGATIVE: late-fused state not causally used at inference (<0.5% flips); C>D is a training-time representation effect | done (3 seeds) |
| 8 | Runtime value dial (early-fusion k input, ONE model) | Runtime-controllable values: k* tracks fed k incl. interpolation (k=3→2.05–2.55); tempted-harm steerable 77%→2–7% without retraining; late-vs-early fusion lesson | done (3 seeds) |
| 9 | World-seed robustness (3 encoder seeds) | Safe-but-broken + D/C coherence replicate in all 3 worlds (B agr 38–83%, D/C 96–99%); C>D does NOT uniformly replicate at 3 seeds — report per-world, lean on 10-seed main | done (2×3 seeds) |
| 10 | Data efficiency (n_train sweep) | D/C scale gracefully (72→96% agr, n=100→3000); B flat ~30–40% agr at ALL sizes — failure is the ordinal objective, not supervision budget | done (3 seeds) |
| 11 | Supervision noise (pair flips, D-noise control) | D_noisy degrades gracefully (0.96→0.87 @30%); B is noise-NONMONOTONE (noise improves it: overshoot accidentally corrected) = unanchored | done (3 seeds) |
| 12 | Composition: CARE + UNKNOWABLE states in one game (bridge to ABSTAIN) | In-distribution composition works (0.6% false abst + 98% abst on seen corruption + flat dial); `perm` accidental control: abstention is information-driven, not novelty-driven (1% abstain on semantics-preserving perturbation trained as ABSTAIN); OOD transfer partial/family-dependent (12–90%); OOD dial-abstain entanglement = limitation | done (3 seeds) |
| **13** | **Goodhart pressure test** (REINFORCE on self-reward, 800 steps) | **PILLAR**: post-hoc safety thinnest (harm 1.1%→49.1%; broken-clock "improvement" as pressure drags overshoot through k=2); explicit values degrade gracefully (~12pp, agr ≥90%); input-channel values retain a recovery lever (harm@k0 79.5% vs @k8 30.0% after 800 steps) | done (3 seeds) |

## Planned figures/tables
- F0: taxonomy spectrum table (§2, conceptual — table not figure)
- F1: k_sweep.png (exists, linear) + hard-world version → "the dial" money figure
- F2: hard-world bars: agr + utility vs harm (safe-but-broken quadrant plot)
- F3: dial input: k* vs fed k (one model), + tempted-harm vs fed k (steering curve)
- F4: **Goodhart survival curves (PILLAR FIGURE)**: harm vs pressure steps, all 5
  subjects; inset/second panel: dial steering spread (harm@k0 − harm@k8) vs steps
- F5: data-efficiency curves (utility/agr vs n_train, B vs D/C)
- F6: noise robustness (agr vs eps, B vs D_noisy vs clean)
- T1: main hard-world table (10 seeds, ± std)
- T2: paired significance table (C>D, D>B)
- T3: world-seed robustness table
- T4: Goodhart table (harm/k*/agr @0 vs @800, + dial spread)

## Related work to cover
- **Special tokens & conditioning spectrum** (the §2 spine): BERT [CLS]/[SEP],
  GPT BOS/EOS; CTRL control codes; soft prompts/prompt tuning (Lester et al.
  2021); classifier-free guidance as conditioning precedent. Position: all
  either contentless or descriptive; none constructed+auditable.
- Preference tuning: DPO, CPO (and the 3-action displacement pathology we hit —
  methodologically relevant), Razin et al. 2024 (likelihood displacement).
- Fine-tuning breaks alignment: Qi et al. 2023 (our pressure test is the
  mechanistic miniature), plus reward-hacking/Goodhart literature.
- Constitutional/value learning; reward-model hacking.
- Interpretability: probes as readouts (mirror) vs causal interventions
  (steering) — our negative result on the binary state connects directly.
- Affective computing / emotion in agents (appraisal theory framing; functional
  not phenomenal states — the anti-overclaim paragraph).
- Conditioning mechanics: FiLM/early fusion; our late-vs-early fusion finding
  as a design lesson for value conditioning.
- Damasio (somatic markers), care ethics (Gilligan/Noddings) — philosophical
  motivation, kept brief and functional.

## Limitations to state plainly
- Toy scale, behavioral cloning + one RL-pressure phase, single value
  dimension + epistemic state, 3-action games, constructible ground truth
  (the easy part of alignment).
- All claims functional/calibrated; no phenomenal-experience claims.
- B could in principle be hyperparameter-tuned toward a target k — but only via
  outcome measurement + re-tuning; no semantic knob (argued in §5).
- Pressure test is REINFORCE full-batch on a proxy; not adversarial attack,
  not deceptive optimization; degradation ordering is the claim, not immunity.
- State-token semantics only as good as the construction; OOD readout fails
  (micro stakes 27–31%) and dial-abstain entangle off-manifold (§12).

## Submission checklist
- [x] All queue results in + summarized
- [x] Composition experiment run
- [x] Goodhart pressure experiment run (10 seeds, paired stats)
- [x] Figures regenerated from final jsonl (F1–F6; dense-B series in F1h/F2/F4/F5)
- [x] Manuscript FIRST DRAFT (paper/main.tex, 12pp, 0 warnings, all numbers verified vs jsonls)
- [x] External review accommodated (10 items; P0 richness confound → dense controls)
- [x] P0 RESOLVED: story narrowed — supervision richness (not objective form) drives
      calibration/coherence/pressure-survival; state token uniquely buys runtime dial +
      auditability. Manuscript rewritten end-to-end (17pp, 0 warnings).
- [x] Composition § rewritten honestly (zero-corruption dial-fragility disclosed;
      dial–abstention interference both directions)
- [ ] B_all pressure 10-seed → tab:pressure row + figure regen (RUNNING)
- [ ] Author review: title choice (current: "The Vocabulary as API..."), section depth, C>D phrasing
- [ ] Repro README polish (env, commands, runtimes)
- [ ] arXiv category: cs.LG (primary), cs.AI; cross-list philosophy optional

## Runtime notes
All experiments CPU-only (GTX 960 idle is fine); full queue ≈ 4 h sequential.
Each script is standalone, append-only jsonl output, resumable by seed deletion.
