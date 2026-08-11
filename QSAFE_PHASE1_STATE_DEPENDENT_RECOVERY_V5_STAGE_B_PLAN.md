# V5 Stage B — option-aware Q_safe execution plan

Status: **implementation in progress; no Stage-B outcome has been produced**

Parent protocol: `config/qsafe_state_dependent_recovery_v5.yaml`

Result-blind execution supplement:
`config/qsafe_state_dependent_recovery_v5_stage_b_execution.yaml`

Stage-A authorization report SHA-256:
`e7ea56546bf8006cfc4d8ade4f5b2c26dbfcbc132e0568e054a98c2be3174b2e`

Stage-A disposition commit:
`959605d621163f2122d6947bbb8fd657a51d5f7f`

## 1. Question and claim boundary

Stage A established a large, independently replicated state-dependent recovery
signal for the fixed seed-42 SAC actor. Stage B asks the next narrower question:

> Can a deployable, option-aware Q_safe learn enough of the frozen K9/H96
> same-state recovery ordering to choose safer recovery options on actors,
> source trajectories, states, and branch replicas that are all absent from
> fitting and calibration?

A Stage-B pass is a model-generalization result. It authorizes only Stage C's
fresh paired closed-loop test. It does not prove fewer falls during SAC
training, does not pass Objective 1, and cannot authorize a wider speed range.

The evidence ladder remains strictly ordered:

```text
Stage A causal headroom (PASS)
  -> Stage B learned Q_safe on one-shot model test
    -> Stage C fresh nominal / Q_safe / placebo paired closed loop
      -> Stage D 24-seed three-arm SAC-from-zero confirmation
        -> Objective 1 PASS
          -> Phase 2 wider command-speed range
```

Any failed gate stops at its current arrow. No favorable diagnostic, Oracle,
nearby checkpoint, extra source, or alternate bootstrap can override it.

## 2. Method and literature rationale

The design keeps the parts of SQRL that match the scientific question: sparse
failure supervision, a safety critic separate from task reward, explicit
evaluation during learning, and transfer only across related state/action
distributions. It differs from the original scalar action critic where the Go2
data showed that state risk can dominate noisy action ordering.

Following Recovery RL and SAILR, the model factorizes nominal state risk from a
relative intervention advantage. Following conservative safety-critic work,
the selector abstains outside calibrated support rather than treating the
minimum network output as safe. The matched-random placebo preserves trigger,
duration, and first-action-distance behavior so that later closed-loop evidence
tests learned option ranking rather than merely reduced action magnitude or
intervention frequency. Executed shield actions, rather than rejected nominal
actions, remain the replay actions as required by correct post-shield learning.

The deployable input is five corrected 46D observation frames plus an 82D
candidate descriptor:

- 36D current nominal application tuple;
- 36D candidate requested/executed/q-target application tuple;
- K9 behavior one-hot;
- behavior duration divided by H96.

The five-member Selective Advantage ensemble predicts nominal/state risk,
candidate-relative risk, and auxiliary TTF/tilt/height targets. No reward-Q or
task-Q gate is present in V5. A candidate must pass a nominal-risk trigger,
signed conformal benefit/risk bounds, ensemble-support limit, and first-action
requested/q-target slew limits.

## 3. Frozen population and physical splits

Fourteen independently trained SAC-from-zero actors are retained without
performance filtering. Each has exact policy-only checkpoints at 25,000,
50,000, and 100,000 committed environment transitions. A checkpoint is emitted
after transition N and its scheduled learner update, before transition N+1;
episode-boundary or nearby checkpoints are invalid.

| Role | Actor seeds | Source seeds | Groups/source | Labels | Total groups |
|---|---|---|---:|---:|---:|
| fit | 43–46 | 8501–8504, 8511–8514, 8521–8524 | 128 | R32 | 1,536 |
| probability calibration | 47–48 | 8601–8602, 8611–8612, 8621–8622 | 64 | R32 | 384 |
| uncertainty calibration | 49–50 | 8631–8632, 8641–8642, 8651–8652 | 64 | R32 | 384 |
| selector calibration | 51–52 | 8661–8662, 8671–8672, 8681–8682 | 64 | R32 | 384 |
| one-shot model test | 53–56 | 8701–8704, 8711–8714, 8721–8724 | 64 | R64 | 768 |

All roles use a physically separate R32 admission screen requiring 6–26
nominal falls inclusive. Exactly one accepted state comes from a complete
source trajectory. Admission outcomes never become labels. The fixed total is
G=3,456 and 1,216,512 K9 candidate-label rollouts, or at most 116,785,152 H96
policy steps excluding state proposal/admission work.

The ten role×partition RNG domains are physically distinct. Their low-15
SHA-256 prefixes and the remaining source/role/identity/namespace/index fields
are injectively packed into 64-bit seeds with the high bit set. This separates
source, admission, label, CRN, rollout, perturbation, and candidate streams.

Before fitting, an outcome-blind compiler must prove zero overlap for all ten
role pairs across actor seed and hashes, state/trajectory fingerprints, and all
branch seed identities. The model-test producer report commits paths and byte
hashes without reporting or exposing outcome values.

## 4. Work packages and gates

### B0 — execution lock and red-team tests

Deliverables:

- immutable Stage-B operational supplement bound to the parent V5 hashes and
  Stage-A PASS report;
- exact source roster and RNG golden vectors;
- exact actor snapshot hook and 42-entry manifest validator;
- five-role path capabilities and pre-consumption model-test firewall;
- pure signed-conformal, simultaneous selector, placebo, and hierarchical
  bootstrap functions;
- synthetic, mutation, crash, leakage, and no-clobber tests.

Exit: full repository tests pass from one clean commit. No actor training or
branch outcome may begin before this boundary.

### B1 — actor bank

Train seeds 43–56 under the frozen 0.30 m/s SAC configuration to 100k committed
transitions. Export all three exact checkpoints and preserve every seed. The
actor-bank compiler validates 42 unique identities and binds actor bytes,
state_dict, policy/config, checkpoint fingerprint, generator commit, and exact
step. Missing, duplicated, nearby, or selectively retained checkpoints stop
Stage B.

Exit: one no-clobber `qsafe...stage_b_actor_bank.v1` manifest and a clean
tracked result record containing only identity/timing—not returns or falls.

### B2 — five-role collection and blind merge

For every frozen role/source assignment:

1. publish a source attempt marker before its first transition;
2. mine complete source trajectories with the assigned exact actor checkpoint;
3. evaluate only physical R32 admission;
4. for an admitted state, evaluate K9 once on its separate R32/R64 label domain;
5. publish deployable and privileged arrays, identity-rich step log, completion
   marker, and outcome-free report last;
6. never top up, substitute, restart, or summarize candidate outcomes.

Role merge validates source counts, shapes, hashes, policy identities, one
state per trajectory, recovery feature parity, admission/label separation, and
all ten inter-role collision checks. Model-test merge is blind. It creates the
commitment before normalization or fitting and then loses permission to probe
model-test label paths.

Exit: five complete role reports, split-disjointness report, and immutable
model-test commitment. Any consumed source remains consumed after failure.

### B3 — fit and probability calibration

Fit-only normalization uses float64 population moments over complete fit groups
and publishes float32 mean/std with a 1e-6 floor. The Q_safe ensemble is exactly
five deterministic CPU members, seeds `20260810 + 1009*i`, widths 128×3, 100
epochs, group batch 64, AdamW 3e-4, weight decay 1e-5, gradient clip 5. Each
member bootstraps complete fit trajectory clusters.

Probability calibration uses only actors 47–48. Each member gets its own
temperature: 100 Adam steps, lr 0.05, log-temperature clamp [-4,4]. The report
binds the model and split hashes and contains no selector/model-test access.

Exit: frozen normalization, five-member weights, and probability report.

### B4 — signed uncertainty calibration

Actors 49–50 provide a physically separate split. For each option k=1..8:

- risk score = empirical risk k − predicted risk k;
- benefit score = predicted benefit k − empirical benefit k;
- finite-sample rank = `min(n, ceil((n+1)*(1-alpha)))`;
- alpha is 0.00625 separately for the eight-option risk and benefit families.

Nominal trigger uses alpha 0.05. Stable ties select the value at the exact
rank. Offsets remain signed; zero truncation is forbidden. Risk is clipped to
[0,1], benefit to [-1,1]. The two Bonferroni families are not described as one
joint 16-bound guarantee.

Exit: immutable uncertainty report bound to temperatures/model/split.

### B5 — selector and matched-random placebo

Actors 51–52 determine one selector from the exact 5×5×4=100 grid. Trigger uses
`>=`, benefit uses strict `>`, and risk/std/requested-action/q-target limits use
`<=`. Eligible candidates sort by lowest risk UCB, largest benefit LCB, then K9
index. Empty sets abstain.

For each configuration, point reduction uses equal actor→source→complete-group
weight. Fifty thousand PCG64 seed-20260811 hierarchical draws produce a
non-studentized simultaneous max-stat band across all 100 configurations. A
configuration is feasible only with reduction >=0.03, simultaneous LCB >0,
and intervention rate <=0.35. Choice is largest simultaneous LCB, lower
intervention rate, then machine grid order. No feasible point fails Stage B
without consuming model test.

After the selector is frozen, fit the outcome-free placebo on the same states.
It matches intervention rate, option duration, and first requested-action
distance using nominal-risk deciles × durations {10,25,50} × distance
quartiles. It never reads Q_safe option ranking or outcomes; a selected empty
cell abstains with no fallback. Required calibration errors are <=0.02 rate,
<=0.05 duration TV, and <=0.10 distance KS.

Exit: selector report/bundle, placebo bundle, and final Q_safe artifact, all
canonically hash-bound. Failure leaves model test unopened.

### B6 — irreversible model test

The evaluator first validates every frozen prerequisite, then atomically
publishes `model-test-consumed.json`; only afterward may its process perform the
first `resolve/exists/stat/open/hash/map/load` operation on a model-test label
path. A crash after the marker permanently consumes the test and cannot be
rerun or replaced.

Fifty thousand PCG64 seed-20260812 draws sample four actors with replacement,
retain the three registered source/age strata for each sampled actor, and
sample complete trajectory groups within each source. Pair-accuracy uses the
2.5th percentile lower endpoint of a two-sided 95% percentile interval;
top-1/selector reductions use one-sided 5th-percentile lower bounds.

All of these must pass:

- pair accuracy >=0.60 and lower endpoint >=0.55;
- strong-pair accuracy >=0.62 at empirical gaps >=0.25;
- learned top-1 reduction >=0.05 and one-sided LCB >=0.03;
- frozen-selector reduction >=0.03, one-sided LCB >0, intervention <=0.35;
- top-1 and selector effects strictly positive at all three ages, all four
  actors, and all twelve sources;
- oracle-gap capture >=0.25 with positive oracle opportunity;
- stable ten-bin equal-mass ECE <=0.08;
- all identity, data, calibration, selector, placebo, hash, and one-shot gates.

Exit: canonical report says either `authorize_stage_C_only` or
`no_further_stage`. Stage B can never set `objective1_pass` or
`phase2_authorized` true.

## 5. Implementation and evidence boundaries

The execution supplement requires the exact same clean generator commit for
every Stage-B operation. Therefore Stage-B must not advance `HEAD` between the
first actor-bank attempt and the one-shot decision. The auditable boundaries
are:

1. **Stage-B result-blind implementation commit** — execution lock, actor
   snapshots, role/firewall schemas, statistical primitives, workflows, and
   tests. This clean commit becomes the generator identity.
2. **One uninterrupted evidence ledger under that commit** — produce the full
   42-identity actor bank, all five role collections and blind commitments,
   then the frozen model/calibration/selector/placebo artifacts. Each step is
   atomic, no-clobber, hash-bound, and may be monitored outcome-blind, but no
   intermediate Git commit or checkout is permitted.
3. **One-shot Stage-B decision under that same commit** — validate the nine
   frozen prerequisites, publish the irreversible consumption marker, and
   evaluate exactly once.
4. **Post-decision evidence commit** — only after Stage B has terminated,
   record the immutable identity ledger, hashes, and accepted/failed report.

The repository already contains more than the user's required three milestone
commits. Keeping the entire Stage-B evidence run on one generator identity is
more important than manufacturing extra Git boundaries inside that run.

## 6. Cost, monitoring, and failure policy

Stage A observed roughly 22.8 proposals per accepted group. Extrapolated to
G=3,456, Stage B is about 359 million simulator policy steps including labels
and expected admission; the registered proposal cap gives a much larger worst
case. Ten to twelve isolated collectors are expected to require roughly
14–24 hours, while model fit/calibration/bootstrap should be minutes rather
than hours. Actor-bank time is measured from the first required seed and then
scheduled without mixing its runs with saturated branch collection.

Monitoring may inspect process liveness, exit status, source steps, proposals,
accepted-group counts, hashes, and report existence. It may not inspect fall
rates, candidate outcomes, selector performance, model-test bytes, or partial
claim metrics before the registered barrier. Tooling failure consumes any
started source or one-shot test; scientific thresholds never change in
response.

## 7. Current checklist

- [x] Stage-A PASS and Stage-B-only authorization recorded.
- [x] Parent protocol and recovery-library hashes bound.
- [x] Result-blind Stage-B execution supplement drafted before outcomes.
- [ ] Exact actor checkpoint code and 42-entry validator green.
- [ ] Five-role path scopes and model-test first-probe firewall green.
- [ ] Signed conformal, simultaneous selector, placebo, and bootstrap tests green.
- [ ] Single-role collector/merger and split proof green.
- [ ] Full repository suite green; clean implementation commit created.
- [ ] Actor bank executed and frozen.
- [ ] Five roles collected/merged; model test committed but unopened.
- [ ] Model, calibrations, selector, placebo, and artifact frozen.
- [ ] One-shot model test consumed and Stage-B decision recorded.
- [ ] Stage C remains blocked unless the preceding item passes.
- [ ] Objective 2 remains blocked until Objective 1 passes Stage D.
