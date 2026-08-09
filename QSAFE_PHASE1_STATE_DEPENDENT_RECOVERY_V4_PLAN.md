# Objective 1 state-dependent recovery Q_safe v4 plan

Status: **historical preregistration; V4 terminated tooling-invalid without a
scientific decision**

This file is retained as the immutable historical V4 design. V4 must not be
resumed: its six seeds are consumed and its first discovery merge failed before
aggregation or audit authorization. See
`QSAFE_STATE_DEPENDENT_RECOVERY_V4_TECHNICAL_FAILURE.md`. The fresh successor is
`QSAFE_PHASE1_STATE_DEPENDENT_RECOVERY_V5_PLAN.md` with a distinct root, seeds,
Stage-A RNG domain, and protocol identity.

Protocol name: `objective1_state_dependent_recovery_qsafe_v4`

Objective: obtain reproducible, statistically credible evidence that a frozen,
deployable Q_safe which selects a persistent recovery behavior reduces falls
during SAC training from zero at 0.30 m/s. Phase 2 speed-range expansion remains
blocked until the complete Objective-1 evidence compiler passes.

## 1. Why this is a new protocol

The consumed v3 audit rejected the discovery-selected global fixed backup. Its
effect was `3.105 pp` with one-sided 95% LCB `-2.161 pp`, and it failed the
all-age and all-seed direction checks. V3 therefore authorized no model
training. See `QSAFE_PHASE1_CLOSED_LOOP_RECOVERY_V3_RESULT.md`.

V3 also exposed a hierarchy-blocked exploratory diagnostic: a
discovery-locked per-state K9 rule had `36.437 pp` audit reduction and pair
agreement `0.81185`, but `tested=false`, `checks=null`, and `pass=null`. V4
treats this only as hypothesis generation. It does not reuse a v3 state,
replica, label, prediction, threshold fit, or seed.

The plan follows the separation in
[SQRL](https://arxiv.org/abs/2010.14603) and
[Recovery RL](https://arxiv.org/abs/2010.15920): state risk and recovery choice
are learned separately from the task actor, while the recovery program remains
fixed during confirmation. The benefit head follows the intervention-advantage
view in [SAILR](https://proceedings.mlr.press/v139/wagener21a.html). Ensemble
uncertainty and abstention follow the conservative-safety-critic lesson that
unsupported actions must fail closed rather than win by extrapolation. Replay
stores the action actually selected by the shield, consistent with the action
attribution requirement emphasized by the
[shielding review](https://doi.org/10.1145/3715958).

## 2. Non-negotiable boundaries

1. No v4 outcome is generated until the machine-readable protocol, collector,
   validator, statistics, one-shot markers, and synthetic tests are committed.
2. V3 artifacts are permanent falsification/exploration evidence only. They
   are forbidden inputs to v4 fitting, normalization, calibration, selection,
   early stopping, or confirmation.
3. Every role uses a physically separate file and a seed namespace assigned
   before outcomes. No role is made by slicing replicas after inspection.
4. A failed gate stops the dependent ladder. There is no top-up, seed removal,
   runner-up audit, threshold change, or heldout rerun.
5. Protected path components beginning with `formal` or `sealed` remain
   unreadable. Heldout markers are published before the corresponding file is
   opened.
6. All evidence commands require a clean, stable commit and use atomic
   no-clobber/report-last publication.
7. Phase 2 is mechanically impossible to authorize unless the final Phase-1
   compiler reports `phase1_pass=true`.

## 3. Claim ladder

```text
A. fresh state-dependent causal-headroom confirmation
   -> B. option-aware Q_safe train/calibration/heldout model gate
      -> C. fresh paired persistent-option closed-loop gate
         -> D. fresh-from-zero SAC 3-arm online gate
            -> Objective 1 pass
               -> Phase 2 may begin
```

Only the immediately preceding passing gate authorizes the next outcome-bearing
stage. Engineering and synthetic testing may run earlier, but model fitting is
forbidden until Stage A passes, and online SAC is forbidden until Stage C
passes.

## 4. Stage A — fresh direct confirmation of state-dependent headroom

### 4.1 Locked mechanism

Stage A retains the v3 K9 behavior definitions and H96 continuation exactly:

1. nominal early actor;
2. mature actor for L10;
3. mature actor for L25;
4. mature actor for L50;
5. joint brake for L10;
6. halfway-to-neutral for L10;
7. halfway-to-neutral for L25;
8. ramp-to-neutral for L25;
9. ramp-to-crouch for L25.

The source actor resumes after the option. Reselection inside an option is
forbidden. Failure, 50/500 Hz timing, ten low-level substeps, base-body height,
tilt threshold, PD gains, action projection, source/mature actor fingerprints,
and the recursive MJCF digest remain identical to v3.

### 4.2 Fresh cohort and claim scope

- actor ages: `25,438`, `50,030`, `100,359` steps;
- source seeds by age: `8401/8402`, `8411/8412`, `8421/8422`;
- exactly 64 admitted states per seed, G=384 total;
- admission: R32 nominal replicas, retain 6--26 falls inclusive;
- candidates: K9;
- discovery: R64, physical discovery files;
- audit: R64, physical audit files and disjoint preassigned RNG domains;
- horizon: H96 policy steps;
- one state maximum per source trajectory;
- zero branch disturbance after the snapshot;
- same admission-positive conditional estimand as v3.

All six seeds are new. The v3 seeds `7801/7802/7811/7812/7821/7822` and v2
seeds `7601/7602/7603` are explicitly consumed and forbidden.

The three early policies are the hash-locked checkpoints from SAC training
seed 42. Stage A is therefore a fixed-actor, admission-positive mechanism
confirmation. Its six `source_seed` values are environment/trajectory RNG
roots, not six independently trained SAC actors. Neither the hierarchical
interval nor the all-six direction check supports a cross-training-seed claim.
Cross-actor generalization is tested with physically separate actor-training
seeds in Stages B and C, while Objective 1 still depends on the 24 fresh
from-zero training seeds in Stage D.

Stage A copies the following v3 source-state mechanics without tuning:
maximum 100 policy steps per source episode; at most 4,096 proposals and 2,048
source trajectories per source seed; five-policy-step cooldown after a rejected
proposal; exactly two settle steps (`0.04 s`); a source impulse every ten policy
steps with linear standard deviation `1.0 m/s` and angular standard deviation
`4.0 rad/s`; no post-snapshot branch impulse at steps 24/48/72; and a pre-screen
requiring a nonfailed state with tilt `>=0.10 rad` or base height `<=0.32 m`.
All proposals, including rejects and exhausted attempts, remain in the ledger.
The attempt marker is atomically published before the first simulator step and
the collection report is published last.

V4 randomness has a new domain and an injective 64-bit production encoding.
Bit 63 is fixed to one, while every historical V2/V3 low-63 seed has that bit
cleared, so the protocols are numerically disjoint by construction. The next
15 bits are the low 15 bits of SHA-256 of the literal domain
`qsafe_state_dependent_recovery_v4_seed\0`, followed without truncation by
fixed-width fields `(source_seed[14], role_tag[8], identity[18], namespace[2],
index[6])`. The machine protocol locks the exclusive cap of every field; any
configured source seed, proposal limit, trajectory-step product, replica count,
or role tag that could overflow is rejected during preflight, before an attempt
marker or outcome. Consequently two valid production tuples cannot
share a seed by construction. Admission, discovery, audit, source reset, source
policy, candidate, rollout, and perturbation streams have distinct
role-tag/namespace pairs fixed in the machine protocol. Per-step draws use
`SeedSequence([rollout_seed, 0,
absolute_policy_step])`. Tests compare every branch stream element with an
independent reference packing, cover field boundaries and overflow, preserve
the historical V3 golden, and verify pairwise role disjointness plus lossless
uint64 lock/audit round trips. The claim no longer relies on a sampled
collision test or a truncated hash.

### 4.3 Selection and primary audit

For each state, discovery selects the set of candidates attaining minimum
empirical discovery fall risk. Exact ties are retained and scored by their
uniform expectation; column order cannot affect the result. Audit evaluates
that locked per-state rule once.

Stage A passes only if all checks pass:

- absolute audit reduction from nominal `>=0.10`;
- hierarchical-bootstrap one-sided 95% LCB `>=0.07`;
- discovery/audit all-K9 pair agreement `>=0.70`;
- pair-agreement one-sided 95% LCB `>=0.65`;
- effect strictly positive at all three actor ages;
- effect strictly positive for all six source seeds;
- discovery nominal risk is informative overall `[0.15,0.75]` and within each
  age `[0.10,0.90]`;
- the exact G384/K9/R64/H96 and RNG-domain data gate passes.

The bootstrap uses complete trajectory groups, equal groups within source seed,
equal six source seeds, 50,000 PCG64 replicates, seed `20260810`, linear
quantiles, and a chunk/draw order fixed in the machine protocol. It estimates
uncertainty conditional on the seed-42 actor chain; it does not treat the six
environment seeds as independently trained actor clusters.

If Stage A fails, stop v4 with `no_model_training`. A favorable fixed candidate,
same-replica oracle, tie fraction, or full-replica reanalysis cannot override
the primary gate.

The selection lock and the failure report form a crash-recoverable two-file
transaction. If the process dies after publishing a selection lock whose
`audit_authorized`, data-gate pass, and discovery-informativeness pass values are
all false but before publishing the report, the only permitted recovery entry
point is `resume-denied-report`. It reads only the canonical protocol and the
existing hash-bound selection lock and reconstructs the one canonical Stage-A
failure report. It cannot rewrite the lock. A repeated call accepts an existing
report only when its bytes are identical to that reconstruction. Parsing,
resolving, `stat`-ing, hashing, mapping, loading, or opening an audit path is
forbidden on this route; therefore a crash cannot turn a denied audit into an
outcome probe.

## 5. Stage B — option-aware Q_safe

### 5.1 Required representation fix

The existing deployable Q_safe action view is only the 36D ordered tuple
`requested || executed || q_target`. It is invalid for K9: for example,
`mature_actor_L10/L25/L50` share the same first application tuple but define
different future programs. A learner using only 36D cannot distinguish them.

V4 introduces the immutable candidate-program view `recovery_program_v1`:

```text
requested[12]
|| executed[12]
|| q_target[12]
|| behavior_id_one_hot[9]
|| behavior_steps_over_H[1]
= 46 deployable candidate features
```

The model action descriptor is 82D, not 46D:

```text
common current SAC nominal (requested || executed || q_target)[36]
|| candidate recovery_program_v1[46]
= 82 model action features
```

Every candidate row, including every non-nominal row, therefore carries the
same current SAC nominal proposal explicitly. This lets the relative-benefit
head represent the value of a recovery program relative to different nominal
SAC proposals. Candidate zero contains that same common tuple plus the nominal
program tuple, nominal one-hot, and zero duration.

The one-hot order is the exact K9 order above. The duration scalar is exactly
`candidate_behavior_steps / 96`, with nominal duration zero. The grouped-data
manifest uses schema `qsafe.recovery_program_features.v1` and binds the full K9
names, order, durations, execution laws, H96 denominator, and recovery-library
fingerprint. `TorchGroupedView`, artifact save/load, and runtime all reject
missing behavior steps, a non-integer or wrong-shape duration array, a changed
K9 name/order/duration, a fingerprint mismatch, or a width other than 82. No
fallback to `application_concat` or the old 36D runtime is permitted.

The recovery-library manifest includes the complete candidate protocol, mature
actor identity, action-projection vectors and filter/slew settings, corrected
input boundary, and privileged-input prohibition. Its canonical SHA-256 is
locked as
`fcfb1fa541acf316f87dacf82b1fdeb9188d7a4b9df7f69544b567fb2c5d1045`
and is carried unchanged by every dataset, training view, model artifact, and
runtime provider.

For a claim-bearing training view, the recorded normalization content hash is
not sufficient by itself: mean and population standard deviation are recomputed
from the fit split in float64, floored at `1e-6`, cast to float32, and compared
bit-for-bit. The action-application contract has one exact eight-field keyset,
including `max_joint_delta=null` and `use_action_filter=false`. Every valid K9
row is replayed through the locked float32 `action_to_qpos` and
`qpos_to_action` laws so requested, q-target, and executed values cannot be
mixed across candidates while retaining a self-consistent manifest.

Synthetic tests prove that L10/L25/L50 remain distinguishable even when their
first requested/executed/q-target tuples are identical. Dataset/runtime
ingestion requires the canonical K9 order and rejects any whole or partial
manifest permutation; after canonical feature construction, jointly permuting
the descriptor, mask, and outcome candidate axes is model/metric equivariant.
Candidate zero remains exactly centered, and training and runtime construct
bit-identical float32 tensors. These tests
and the feature builder are committed before any Stage-A outcome, even though
Q_safe fitting remains forbidden until Stage A passes.

Privileged simulator state is permitted only in a diagnostic model. The
claim-bearing model sees five corrected 46D observation frames and the 82D
action descriptor. Command speed remains fixed at 0.30 m/s for Objective 1;
mixed-speed training is forbidden because command velocity is absent from the
current observation.

### 5.2 Fresh actor bank and role-separated data

Data collection begins only after Stage A passes. Before candidate collection,
the unfiltered actor bank trains every preregistered SAC seed below to fixed
budgets and retains all checkpoints regardless of falls or returns. Each actor
contributes checkpoints at exactly 25,000, 50,000, and 100,000 policy steps;
failure to produce any checkpoint aborts the role rather than substituting a
seed or nearby checkpoint. Actor performance is not an inclusion criterion.
Resolved checkpoint and actor hashes are locked before the first candidate
outcome for that role.

All states are new and assigned to a physical role before any candidate
outcome:

| Role | SAC actor-training seeds | Actor-age-balanced environment source seeds | Groups | R/K | Use |
|---|---|---|---:|---:|---|
| fit | `43..46` | `8501..8504`, `8511..8514`, `8521..8524` | 1,536 (128/source) | 32 | normalization and fitting |
| probability calibration | `47/48` | `8601/8602`, `8611/8612`, `8621/8622` | 384 (64/source) | 32 | member temperatures only |
| uncertainty calibration | `49/50` | `8631/8632`, `8641/8642`, `8651/8652` | 384 (64/source) | 32 | simultaneous conformal residual bounds only |
| selector calibration | `51/52` | `8661/8662`, `8671/8672`, `8681/8682` | 384 (64/source) | 32 | lock selector and placebo kernel only |
| model test | `53..56` | `8701..8704`, `8711..8714`, `8721..8724` | 768 (64/source) | 64 | one-shot model gate |

Within each row, the nth source seed in each age block belongs to the nth
actor-training seed in that row. Thus no actor checkpoint, source trajectory,
or environment seed crosses a role. The model-test outer bootstrap unit is the
four independently trained actors (`53..56`), within which complete age/source
trajectory groups are resampled. Per-age and per-actor direction checks are
both mandatory. Stage-B claims are limited to this actor-bank distribution;
Stage D supplies the independent from-zero confirmation.

The complete actor-to-source assignment is machine data, not an inference from
the table: actor 43 owns `8501/8511/8521`, actor 44 owns
`8502/8512/8522`, and so on through actor 56 owning
`8704/8714/8724`. Every actor has exactly one source seed at each of the three
locked ages. A missing, duplicated, or cross-owned checkpoint/source pair aborts
the role.

Every role uses R32 nominal admission with the same 6--26 inclusive rule. The
admission labels are screening-only and are never training/evaluation labels.
Fit, probability-calibration, uncertainty-calibration, and
selector-calibration use physically separate R32 candidate-label files;
model-test uses a physically separate R64 label file. Admission and label files
have different seed domains, CRN IDs, rollout seeds, perturbation seeds, and
candidate seeds. Across the five roles, policy-training seeds, actor/checkpoint
hashes, state and trajectory fingerprints, CRN IDs, rollout/perturbation seeds,
and candidate seeds are pairwise disjoint over all ten role pairs. A canonical
`stage-b-split-disjointness-report.json` proves zero collisions before fitting.
Each role is one-shot and report-last. The model-test file is committed by hash
before normalization or fitting and is opened only after an atomic consumption
marker.

Normalization is computed internally from complete fit-role deployable groups
in `stage-b/fit/labels-r32-deployable.npz` only. Caller-supplied statistics,
candidate/outcome weighting, calibration groups, and privileged features are
forbidden. The report binds the fit array SHA-256, exact group-ID SHA-256,
float32 mean/std hashes, and absence of privileged inputs. These statistics are
frozen before probability calibration and before model-test consumption.

The Stage-B artifact inventory is rooted at `stage-b/`. Each role has canonical
attempt marker, per-source R32 admission shards, per-source R32 or R64 label
shards, merged admission/label/privileged arrays, `steps.jsonl`, collection
manifest, no-clobber completion marker, and report. Derived canonical artifacts
are the actor-bank manifest, split-disjointness and fit-only-normalization
reports, Q_safe artifact, three calibration/search reports, frozen selector and
placebo bundles, model-test commitment/consumption markers, and
`state-dependent-recovery-stage-b-report.json`. The YAML enumerates every path;
each completion marker binds all predecessor bytes by SHA-256 and every report
is published last.

The model-test firewall order is exact: its collector report commits the file
and content SHA-256; the training process may read that report but must not
`stat`, hash, map, load, or open the model-test file; normalization, fitting,
temperature/conformal calibration, selector calibration, placebo construction,
and frozen artifact/selector publication complete using the four development
roles; then a no-clobber `model-test-consumed.json` is published; only an
independent evaluator may first resolve, hash, and open model test, and it
publishes its report last. Interruption after the marker consumes the test.
Generic dataset loaders and aliases enforce the same rule, and the legacy
trainer that opens test before training is forbidden for v4.

### 5.3 Locked model and loss

- Selective Advantage Q_safe with a five-frame encoder, nominal state-risk
  head, relative option-benefit head, time-to-failure/max-tilt/min-height
  auxiliary heads, and selective-advantage anchoring at nominal;
- five trajectory-bootstrap ensemble members;
- 100 epochs, batch size 64, AdamW learning rate `3e-4`, weight decay `1e-5`,
  gradient clip 5, model seed `20260810`;
- hidden widths 128/128/128;
- memberwise temperature calibration on the probability-calibration split
  only;
- complete-group macro losses and metrics with equal groups inside actor/source
  strata; candidate rows are never independent samples;
- no model architecture or hyperparameter change after model-test consumption.

Nested fit-only learning curves at 384, 768, and 1,536 groups are diagnostic.
Only the full 1,536-group deployable run is claim eligible. State-only,
pointwise, privileged, and 36D duration-collapsed models are diagnostics and
cannot be promoted after seeing model-test outcomes.

The primary estimand remains conditional on admission-positive states. The
sequential proposal process does not provide known source-stream inclusion
propensities, so `acceptance_probability` is fixed to the unit conditional
analysis weight and no IPW population claim is made.

For candidate `k`, the target risk is the mean of its R binary fall replicas.
Relative-risk target is `risk_k - risk_nominal`; positive benefit always means
`risk_nominal - risk_k`. Selective anchoring subtracts the action-head score of
the complete nominal 82D descriptor, making candidate-zero risk exactly the
state-risk head. Loss weights are fixed to absolute risk `1.0`, state risk
`0.5`, relative risk `1.0`, pair ranking `0.5`, time-to-failure `0.1`, maximum
tilt `0.1`, and minimum height `0.1`. Ranking includes a pair only when its
empirical gap is at least `1/R`; ties score `0.5` in metrics. Time-to-failure is
`clip((first_failure_step-1)/96,0,1)`, with a nonfailure encoded as step 97;
auxiliary tilt/height/TTF targets use the nominal replicas only.

Each optimization batch contains up to 64 complete groups and never slices a
candidate or replica axis. Ensemble member `i` uses seed
`20260810 + 1009*i` and a trajectory-cluster bootstrap of the fit role. Each
temperature uses 100 Adam steps at learning rate `0.05`, log-temperature clamp
`[-4,4]`, and complete probability-calibration groups. Model-test bootstrap
uses 50,000 PCG64 draws, seed `20260812`, linear quantiles, with actor-training
seed outermost and complete trajectory groups innermost. Pair accuracy uses all
non-tied K9 pairs; ECE uses ensemble-mean risk against R64 empirical risks,
equal group mass divided equally among valid candidates, and ten stable
equal-mass bins. Oracle-gap capture is the equal-actor/group mean achieved
reduction divided by the corresponding mean empirical K9-oracle reduction;
zero or negative oracle opportunity fails rather than yielding a favorable
ratio.

### 5.4 Model and calibration gates

The one-shot model test passes only if:

- group-macro pair accuracy `>=0.60`, 95% cluster-bootstrap low `>=0.55`;
- strong-pair accuracy `>=0.62` for empirical risk gaps `>=0.25`;
- learned top-1 audit reduction from nominal `>=0.05`, one-sided 95% LCB
  `>=0.03`;
- frozen selector audit reduction from nominal `>=0.03`, one-sided 95% LCB
  strictly above zero, with intervention rate `<=0.35`;
- both learned top-1 and frozen-selector effects are positive at every actor
  age, all four model-test actor-training seeds, and all twelve model-test
  source seeds;
- oracle-gap capture `>=0.25`;
- equal-mass ten-bin ECE `<=0.08`;
- data and identity-disjointness gates pass.

Temperature fitting, uncertainty-bound construction, selector search, and model
testing use four physically different roles. The uncertainty-calibration role
constructs split-conformal one-sided residual offsets after temperatures are
fixed. Risk-upper bounds form one Bonferroni family with family-wise alpha
`0.05` and per-option alpha `0.05/8`; benefit-lower bounds form a second,
separate family with the same correction. No joint 16-bound 95% coverage claim
is made, and intersecting a risk UCB with a benefit LCB in the selector does not
create one. The nominal trigger is a separate marginal alpha-0.05 bound.
All three limitations are serialized in the calibration report rather than
hidden behind one ambiguous `familywise_alpha` field.
The offsets, score definitions, finite-sample rank rule, ties, clipping, and
hash are machine-locked. Ensemble standard deviation and temperature scaling
are diagnostics and are never labeled as confidence bounds.

Concretely, for option `k>0`, risk score is `empirical_risk_k - p_k` and
benefit score is `predicted_benefit_k - (empirical_risk_0-
empirical_risk_k)`. Its upper/lower offsets are the sorted-score element at
one-based rank `min(n, ceil((n+1)*(1-0.05/8)))`; risk UCB is
`clip(p_k+q_risk_k,0,1)` and benefit LCB is
`clip(predicted_benefit_k-q_benefit_k,-1,1)`. Nominal-trigger LCB analogously
uses score `p_0-empirical_risk_0` at alpha `0.05`. Each K-specific calibration
set contains all complete groups in stable actor/source/trajectory order;
there is no candidate pooling or randomized tie handling. These are marginal
split-conformal prediction bounds under the locked calibration distribution,
not confidence intervals for the final population effect.

Selector calibration searches only the preregistered finite grid from the v4
machine protocol; the old H32/K16 `config/qsafe_evidence_protocol.yaml` is not a
v4 protocol and cannot authorize any v4 stage. For every grid point, the
reported reduction interval uses a simultaneous max-statistic actor/trajectory
bootstrap across the complete grid (`50,000` draws, PCG64 seed `20260811`). The
selected configuration must obtain at least `3 pp` reduction with simultaneous
one-sided low above zero and intervention rate `<=0.35`. Lexicographic choice
among feasible settings is largest simultaneous lower bound, then lower
intervention rate, then machine-listed grid order. If none is feasible, Stage B
fails without opening the model test.

The exact 100-point grid is nominal-risk LCB threshold
`[0.10,0.20,0.30,0.40,0.50]`, minimum benefit LCB
`[0.00,0.02,0.05,0.08,0.12]`, and maximum candidate risk UCB
`[0.25,0.40,0.55,0.70]`. Fixed support limits are ensemble probability standard
deviation with population convention `ddof=0` `<=0.20`, first requested-action
RMS delta `<=0.50`, and first q-target RMS delta `<=0.25`; these are
support/slew heuristics, not confidence bounds. Executed-action distance is
recorded but is not an additional selector gate.

The selector requires a nominal-state trigger, positive conformal benefit
lower bound, conformal risk upper bound, epistemic-support predicate,
and first-action/q-target slew limits. V4 has no task-Q/reward-Q gate. The final
model-test interval is computed once for the fully locked model and selector,
so model-test outcomes perform no selector search.

All conformal order-statistic offsets are signed finite values; no unregistered
zero truncation is applied. Trigger comparison is `>=`, benefit is strictly
`>`, and risk, standard-deviation, requested-action, and q-target limits are
`<=`. Among eligible non-nominal candidates the machine order is lowest risk
UCB, then largest benefit LCB, then locked K9 index. An empty eligible set
deterministically abstains. The complete offsets, selected grid point,
probability/uncertainty/selector report hashes, comparison and candidate-choice
semantics, and `ddof=0` convention form one canonical
`qsafe.recovery_selector_bundle.v1`; its SHA-256 is stored in the Q_safe
artifact. Runtime accepts that bundle only and has no threshold/offset override.

## 6. Persistent option runtime, continuation, and replay semantics

The deployed selector returns a K9 behavior index, not merely its first action.
Every collection, calibration, paired, placebo, smoke, and online path uses one
identical deterministic state machine:

```text
episode reset -> idle
idle + accepted trigger -> option(locked behavior, exact L)
option + no fall after L steps -> spent_until_reset
option + fall -> terminal
spent_until_reset -> SAC actor only until terminal/reset
```

There is at most one option start per episode and at most one option in any H96
branch. Reselection, early shortening, extension, and nested intervention are
forbidden. While `idle` or `spent_until_reset`, the current SAC actor owns the
action; after option completion the actor resumes. Reset alone returns the
machine to `idle`. The mature recovery actor and Q_safe ensemble remain frozen
throughout every claim-bearing run. Any future repeated-intervention controller
requires a new protocol and new labels.

An idle claim-bearing Q_safe step accepts only a decision proof emitted by the
artifact-bound frozen selector bundle, never a naked K9 index. The proof binds
the newest observation history, recovery-library fingerprint, selector-bundle
hash, and all nine first-step requested/executed/q-target previews. After
projection, candidate zero or the selected option must match that preview
bit-for-bit before an action can be emitted. The projection provider is bound
to the same full recovery-library action-projection manifest.

The recovery artifact is loaded only with the canonical manifest SHA-256
authorized by the accepted Stage-B compiler report. Component hashes declared
by the artifact itself are necessary but are not a trust root. The authorized
hash remains part of the live artifact identity used by inference and the
controller. The mature frozen recovery actor is likewise loader-issued and
rechecks its current state dict, module/callable surface, hooks, mode, counters,
and manifest before and after action evaluation.

At every 50 Hz step the replay transition stores the shield-selected requested
action, its executed action, and absolute q target. The rejected SAC proposal is
logged for attribution but is never written as the transition action. During a
persistent option, each actual recovery action is therefore attributed to the
state from which it was sent. Deadline misses, option starts, duration, reason
for abstention, and intervention duty cycle are recorded.

Replay records and runtime-step records are controller-issued proofs with live
content digests. A step cross-checks every action and behavior field against its
replay proof and the exact nominal proposal before insertion. The proposal log
binds its counter-key payload, derived seed, reproducible external-noise and
action hashes, static provider fingerprint, and current actor snapshot. Stage C
freezes that snapshot after the first proposal; Stage D permits only monotonic
weight versions, requires an identical snapshot at equal versions, and requires
new actor-state and update-chain identities when a version advances. A concrete
collector adapter that calls these live checks remains a mechanical prerequisite
before the first Stage-D online outcome.

The matched-random placebo is a separately serialized kernel learned only from
selector-calibration inputs after the Q_safe selector is frozen. At a placebo
arm's own current deployable state, it uses the same nominal-risk trigger,
support mask, slew checks, state machine, and one-start-per-episode limit. It
never reads the treatment arm's state or outcome and never uses Q_safe option
ranking. A fixed table indexed by nominal-risk decile samples (a) abstain versus
intervene and then (b) a duration `{10,25,50}` and first-action-distance quartile
using a preassigned PCG64 stream. It samples uniformly among eligible options
in the selected cell; an empty cell deterministically abstains, with no fallback
to another cell.

The table is fitted once to selector-calibration Q_safe decisions to minimize,
in order, absolute intervention-rate error, duration-histogram total variation,
and first-action-distance empirical-CDF distance. Ties use lexicographic table
order. Eligibility requires calibration mismatch `<=0.02` in intervention
rate, `<=0.05` duration total variation, and `<=0.10` distance KS. Otherwise
Stage B fails. The complete table, bin edges, empty-cell policy, fit metrics,
RNG domain, and SHA-256 are locked before model test, Stage C, or Stage D. No
outcome-based reweighting is allowed.

## 7. Stage C — fresh paired closed-loop gate

After Stage B passes, evaluate the frozen selector on exactly 1,200 new
age-balanced exact states from four independently trained SAC actors. Actor
seeds `57..60` contribute one source seed at each age: `8801..8804` (25k),
`8811..8814` (50k), and `8821..8824` (100k), exactly 100 admitted states per
source seed and no top-up. For each state, nominal, Q_safe, and matched-random
branches share the complete snapshot and all disturbance/continuation RNG.
Q_safe and placebo can launch at most one persistent option; their state
machines and calibrations are locked and no paired outcome feeds back into
them.

Actor-to-source ownership is exact: actor seeds 57, 58, 59, and 60 respectively
own sources `8801/8811/8821`, `8802/8812/8822`,
`8803/8813/8823`, and `8804/8814/8824`. The Stage-C actor-bank manifest binds
each fixed 25k/50k/100k checkpoint path, actor and state-dict hashes, policy and
checkpoint fingerprints, policy-config hash, and generator commit before the
first admission outcome. Each nominal proposal proof binds the corresponding
frozen actor fingerprint; provider mutation is forbidden.

Continuation randomness is counter-based by `(state_hash, replica_index,
absolute_policy_step, stream_kind)` for disturbances and stochastic actor
sampling. All branches invoke or shadow-consume the actor draw at every absolute
policy step, including steps owned by an option, so an option cannot shift the
post-option RNG stream. This CRN contract is tested with unequal option lengths
before collection.

The actor-shadow provider is stateless. Both stages begin with the common
literal domain `qsafe.recovery_actor_shadow.v1\0`; Stage C then hashes tag
`stage_c\0` and ordered key `(state_hash_sha256, replica_index,
absolute_policy_step, stream_kind, draw_index)`, while Stage D hashes tag
`stage_d\0` and ordered key `(training_seed, absolute_exposure_step,
stream_kind, draw_index)`. Each tag occurs exactly once. A stateful provider is
forbidden even if a sequential smoke test happens to align.
The only valid actor stream kind is exactly `nominal_actor`. The static provider
manifest/fingerprint and the canonical four-field `qsafe.actor_snapshot.v1`
manifest are checked before, between, and after the equal-input determinism
challenge on every proposal.

Stage-C states use the same R32 nominal admission rule (6--26 falls inclusive),
pre-screen, proposal caps, and one-state-per-trajectory rule as Stage A, so its
estimand is explicitly admission-positive rather than natural-incidence. Arm
labels then come from a new physical evaluation file with exactly R64 replicas
for each of `[nominal, qsafe, placebo]`. Admission and evaluation use distinct
literal seed domains and have no shared CRN/rollout/perturbation/candidate seed;
within evaluation, the three arms share a CRN for each state/replica. Branch
seeds are assigned before outcomes and every three-arm replica is indivisible.
Both Q_safe-minus-
nominal and Q_safe-minus-placebo intervals reuse the same 50,000 hierarchical
bootstrap draws (PCG64 seed `20260813`, linear quantile): actor-training seed
outermost, then age/source and complete state group. No CLI override exists.

Placebo balance is re-evaluated on the realized Stage-C trajectories, not copied
from selector calibration. With equal actor, age/source, and state-group
weighting, the absolute intervention-rate mismatch over all 1,200 groups must
be `<=0.02`; conditional on realized interventions, duration-histogram total
variation must be `<=0.05` and requested-action RMS-distance ECDF KS must be
`<=0.10`. Zero Q_safe or placebo interventions fails rather than making the
conditional distances vacuous. The compiler recomputes these quantities from
`stage-c/paired-steps.jsonl`; a stored boolean is never accepted.

Canonical Stage-C evidence comprises separate admission and paired-evaluation
shards/arrays (`[1200,3,64]` arm outcomes), both step logs, state roster, frozen
actor-bank manifest, Q_safe/selector/placebo manifests, realized-balance report,
completion marker, and `state-dependent-recovery-stage-c-report.json`, all under
the YAML-enumerated `stage-c/` inventory and hash-bound report-last.

Stage C requires:

- Q_safe versus nominal H96 fall reduction `>=0.05` with one-sided 95% LCB
  `>=0.03`;
- positive effect at all three actor ages, all four actor-training seeds, and
  all twelve paired source seeds;
- improved state pairs outnumber worsened state pairs;
- Q_safe versus matched-random reduction has LCB above zero;
- no action-attribution, timing, manifest, support, or data gate failure.

Failure stops before online SAC. No paired-test threshold retuning or second
model is allowed.

## 8. Stage D — Objective-1 online confirmation

The primary v4 route is fresh SAC training from zero at 0.30 m/s. Use the 24
predeclared confirmation seeds `201..224`, exactly 500,000 policy steps per seed
and arm, with no optional stopping:

1. pure SAC baseline;
2. SAC plus frozen option-aware Q_safe;
3. SAC plus intervention-rate/duration/action-distance-matched random placebo.

For a given seed, all three arms start from bit-identical actor, critic, target,
optimizer, and reward-normalizer tensors. Counter-based, role-tagged streams
cover the full 500k exposure for environment disturbances, resets, actor
exploration, replay indices, update noise, and initialization; the placebo draw
has a separate stream. Common streams use `(training_seed,
absolute_exposure_step, stream_kind, draw_index)` so trajectory divergence does
not shift later random numbers. Inference runs at 50 Hz and must meet the
runtime deadline. Evidence is computed at the paired actor-training-seed level;
episodes are not treated as independent samples.

Unlike Stage C, the Stage-D SAC actor is an evolving provider. Its identity is
the seed/arm initialization manifest plus a hash chain covering every optimizer
update. Each action record binds absolute exposure step, actor weight version,
current actor-state hash, update-chain hash, and nominal-proposal hash. Initial
actor/critics/targets/optimizer/reward-normalizer bytes must be identical across
the three arms of a seed, while frozen Q_safe, selector, and placebo hashes
remain constant for the full exposure.

The primary baseline/treatment comparison is physically randomized within each
training seed before any Stage-D simulator step or outcome. A locked 256-bit
OS-CSPRNG master seed and the independent domain
`qsafe_state_dependent_recovery_v4_stage_d_lane_assignment\0` derive one
Bernoulli-half bit for each seed 201..224 by SHA-256. Bit zero maps pure SAC to
lane A and Q_safe to lane B; bit one swaps them. Placebo is preassigned to lane
C and is not part of the primary pair randomization. The exact 24-bit vector and
its SHA-256 are already in the protocol. A no-clobber assignment manifest binds
the derivation seed hash, domain, bits, per-seed lane map, placebo lane, creation
time before the first outcome, and its own hash. Outcome-dependent reassignment
is forbidden.

The online primary fall is the first 50 Hz post-action boundary in an episode
where base-body origin height is strictly `<0.18 m` or absolute roll/pitch is
`>=0.523599 rad`. It counts once, forces a terminal transition, and is assigned
to the quartile containing that action's absolute exposure index. Other
terminations count zero falls. Each arm's 500,000 exposure steps are exactly the
50 Hz agent-issued transitions, including a fall-causing terminal transition;
reset, stand-up, and simulator-settle substeps are excluded. There is no
refractory interval because termination/reset follows the first fall. This is
the same predicate as the H96 offline label; the runtime hardware supervisor is
still not claimed deployment-equivalent because it lacks the height predicate
and uses different timing.

During the initial replay warmup the primary selector uses only frozen Q_safe
risk/benefit/support/slew gates; no task-Q margin is present anywhere in v4.
This avoids making the intervention law depend on an untrained SAC critic.
Every option transition participates in the ordinary one-step SAC critic and
actor update schedule using the actual shield-selected requested/executed/q
target action. Updates are not paused during an option; terminal falls have no
bootstrap; one-step targets never cross reset; rejected SAC proposals are log
only and never training actions. Baseline, treatment, and placebo use identical
update counts and hyperparameters.

The online gate requires every condition in the independent canonical
`config/qsafe_state_dependent_recovery_v4.yaml`:

- at least 20% relative reduction in cumulative falls;
- at least `0.40` fewer falls per 1,000 fixed-exposure policy steps;
- seed-cluster 95% lower bound for treatment reduction above zero;
- treatment-versus-placebo reduction lower bound above zero;
- exact all-`2^24` paired label-swap p-value `<=0.05`;
- signed-safe return noninferiority within 5% of baseline magnitude;
- forward-velocity error increase at most `0.03 m/s`;
- runtime deadline-miss rate below `0.001`;
- all 72 seed/arm exposures exactly complete and provenance verified.

The primary paired difference for seed `i` is pure-SAC falls/1,000 minus
Q_safe-arm falls/1,000; larger is favorable, and the observed statistic is the
arithmetic mean of 24 differences. The exact one-sided sharp-null test enumerates
unsigned masks `0..2^24-1` with seed 201 as the least-significant bit, swapping
baseline/treatment labels inside each seed. Zero differences are retained and
both signs count; randomized statistics tied with or greater than observed are
in the tail. Thus `p = count(T_random >= T_observed)/2^24`, with no plus-one
correction. Interpretation requires within-seed label exchangeability under the
sharp null. That exchangeability is supplied by the 24 independently derived
fair physical-lane assignment bits above; the `2^24` enumeration is exactly the
randomization distribution of those bits, rather than an ungrounded symmetry
assumption about observational paired differences.

All seed-level online intervals use 50,000 PCG64 paired bootstrap draws, seed
`20260814`, linear 5% quantile, resampling complete three-arm seed tuples with
one shared `[50000,24]` index matrix. Relative fall reduction is
`(baseline-treatment)/baseline`; a nonpositive pooled baseline makes this gate
false. Return is reward per 1,000 fixed exposure steps, not an episode-average
ratio. Its gate is
`treatment-baseline >= -0.05*max(abs(baseline),1)`, which remains meaningful for
negative or zero returns. Velocity error is the mean absolute body-frame
`|vx-0.30|` over the 500k agent steps, excluding reset/stand-up/settle ticks; the
24-seed mean increase must be `<=0.03 m/s`. A deadline miss is inference wall
time strictly over `0.02 s`; the Q_safe arm's pooled miss rate must be strictly
below `0.001`, and missing/nonfinite timing fails.

Stage D also repeats the realized placebo-balance audit over the two complete
12-million-step Q_safe/placebo exposures. Intervention-rate mismatch must be
`<=0.02`; conditional duration TV must be `<=0.05`; conditional requested-action
RMS-distance KS must be `<=0.10`. Policy steps are equally weighted inside each
of 24 equally weighted training seeds, while duration/distance use realized
option starts. Zero starts in either arm fails, and no outcome-based reweighting
is permitted.

There are exactly `24 × 3 = 72` exposures. For every seed/arm, the canonical
inventory includes attempt and initialization manifests, transition and episode
logs, fall/exposure array, actor-update hash chain, RNG manifest, intervention
and timing logs, and no-clobber completion marker. A per-seed marker binds the
complete three-arm tuple. The 72-entry roster, `[24,3]` paired metric array,
sign-flip/bootstrap/balance reports, global completion marker, and
`state-dependent-recovery-stage-d-report.json` are published only after all
underlying hashes and exact 500k exposures verify.

The prospective sizing rule assumes an adverse paired-seed standard deviation
of `0.75` falls per 1,000 steps for a true absolute reduction of `0.40` per
1,000. A one-sided alpha-0.05 paired-normal planning approximation gives about
`81%` power at 24 seeds (and only about `47%` at ten); this calculation is
planning metadata, not the analysis test. Actual authorization uses the locked
seed bootstrap and exact sign-flip test. No result-dependent top-up is allowed.

Fall counts, fall timing by training quartile, raw exposure, return, velocity
tracking, intervention/abstention rates, behavior histogram, option duty cycle,
and deadline misses are all reported. There is no outcome-bearing mechanics
pilot. Integration is exercised only by synthetic state-machine/RNG tests and
short simulator smoke runs whose fall, return, velocity, and termination
summaries are neither persisted nor shown. Smoke runs occur only after all
scientific thresholds, model bytes, selector, and placebo hashes are locked. If
a smoke run requires any change to controller semantics, selector, RNG, labels,
or analysis, v4 is consumed and a new protocol is required before confirmation.

Objective 1 passes exactly when:

```text
stage_A_headroom
AND stage_B_data_model_calibration
AND stage_C_paired_closed_loop
AND fresh_030_online
```

### 8.1 Authorization compiler

V4 uses a new fail-closed compiler, never the legacy CLI that accepts caller
booleans. It accepts only the canonical Stage-A, Stage-B, Stage-C, and Stage-D
report paths named by the machine protocol. For every predecessor it verifies
the exact schema version, v4 protocol file/content hashes, generator commit,
artifact and raw-log hashes, report-last completion record, one-shot consumed
marker, predecessor authorization hash, exact seed/arm roster, and fixed
exposure. It recomputes every metric and placebo-balance check from immutable
arrays/step logs; `common_gates=true` or `placebo_matching_verified=true` is not
a valid input.

Compilation proceeds strictly A→B→C→D. At the first valid failing stage it
fully validates that stage and recomputes its gate, then requires every path in
all later-stage canonical inventories to be absent. Later reports are not
required after a valid predecessor failure; if any later array, log, marker, or
report exists, the evidence is malformed and compilation raises without output.
A valid first failure publishes one authorization report with
`objective1_pass=false`, `phase1_pass=false`, and `phase2_authorized=false`.
Missing or malformed evidence for the stage being evaluated still raises rather
than being treated as a scientific failure.

Bootstrap seeds, replicate counts, quantile rules, and thresholds come only
from the canonical v4 protocol and have no CLI override. The compiler writes
`phase1_pass=true` and `phase2_authorized=true` together if and only if all four
canonical stages pass. Every other valid outcome writes both false; malformed
or missing evidence raises without publishing an authorization report.

## 9. Phase 2 remains blocked

No v4 offline effect, paired result, or favorable subset unlocks speed
expansion. Only the complete Objective-1 compiler may set
`phase2_authorized=true`. After that, and only after that, evaluate symmetric
ranges `0.25--0.35`, `0.20--0.40`, then `0.10--0.50 m/s`, requiring at least
80% retention of the Phase-1 relative reduction and at least 16% relative fall
reduction with positive confidence bounds. A shared cross-speed model must add
the command as an explicit deployable feature; otherwise use separately named
per-speed critics.

## 10. Commit and stop boundaries

1. **v3 result commit:** immutable failure report and serialization regression
   fix. Completed by commit `607dee7`.
2. **v4 preregistration commit:** this plan, machine protocol, Stage-A direct
   conditional audit, option-aware feature/runtime contract, and synthetic
   tests. No v4 outcome before this commit is clean.
3. **Stage-A result commit:** pass/fail report. Model training begins only after
   a pass.
4. **Stage-B implementation/result commits:** fresh role collectors, artifact,
   model/calibration outcome and heldout report.
5. **Stage-C result commit:** paired persistent-option evidence.
6. **Stage-D result commit:** 24-seed online compiler and Objective-1 decision.
7. **Phase-2 commits:** forbidden until Objective 1 passes.
