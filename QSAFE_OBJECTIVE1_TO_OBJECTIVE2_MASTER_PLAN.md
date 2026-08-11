# Q_safe Objective 1 → Objective 2 Master Plan

**Repository:** `onrobot-safe-rl`
**Scope:** Go2 MuJoCo safety critic and SAC training evidence
**Status:** natural-PPO proposal route frozen for implementation; Objective 1 and Objective 2 are not yet passed
**Primary route:** SAC-from-zero at `0.30 m/s`, followed by mechanically gated speed expansion

## 1. Research conclusion

The local feasibility report concludes that large-batch PPO/MJX or MuJoCo-Warp
is technically useful, but PPO transitions alone do not identify the target

\[
P(\mathrm{fall}_{H32}\mid s,a,\pi_{\mathrm{filtered\ SAC}}).
\]

PPO will therefore be used for broad state coverage, representation pretraining,
and boundary-state proposal. Claim-bearing Q_safe labels must come from
same-state multi-action branching with a frozen continuation policy, common
random numbers across candidates, and independent replicas.

The fixed-impulse Stage-B source attempt was terminated before any source
completed.  Its ten source attempts are preserved under
`stage-b/aborted-fixed-perturbation-db761ac` and are permanently excluded from
future data.  The replacement protocol is
`config/qsafe_natural_ppo_falls_v1.yaml`: PPO-from-zero runs on flat terrain
with ordinary domain randomization and a constant forward command of
`+0.4 m/s`, but no external push, impulse, or artificial velocity injection.
Every realized command is recorded.  Every independent first fall
is retained together with its preceding 64 policy steps; PPO outcomes remain
proposal metadata only.  Objective-1 target SAC remains fixed at `+0.30 m/s`.

This follows complementary lessons from [SQRL](https://arxiv.org/abs/2010.14603),
[Recovery RL](https://arxiv.org/abs/2010.15920),
[SAILR](https://proceedings.mlr.press/v139/wagener21a.html),
[Conservative Safety Critics](https://openreview.net/forum?id=iaO86DUuKi), and
[near-future safety shielding](https://proceedings.neurips.cc/paper/2021/hash/73b277c11266681122132d024f53a75b-Abstract.html).

The selected method is a **Selective Advantage Q_safe**:

1. five corrected deployable observation frames;
2. a state-risk head;
3. nominal/candidate relative safety-benefit heads;
4. a five-member ensemble with uncertainty;
5. conformal risk/benefit bounds;
6. a state trigger plus behavior-support and action-distance gates;
7. a persistent multi-frame recovery option with closed-loop trigger checks;
8. deterministic abstention when no candidate has a reliable positive benefit.

If privileged-state learning passes while deployable learning fails, the
Objective-1 fallback is a versioned short-horizon predictive rollout shield.
If both fail, the correct conclusion is that the candidate/label mechanism or
observability is inadequate; more epochs or outcome-selected data are not an
authorized remedy.

## 2. Evidence already established

The [parallel Go2 report](RESEARCH_PARALLEL_GO2_SAFETY_DATA.md) establishes a
feasible GPU simulation path and a strong same-state branching rationale. The
[V5 Stage-A result](QSAFE_PHASE1_STATE_DEPENDENT_RECOVERY_V5_STAGE_A_RESULT.md)
passed on fresh audit replicas with approximately `34.47 pp` absolute recovery
risk reduction and a one-sided LCB of approximately `31.35 pp`.

Stage-A proves only that the frozen K9/H96 recovery library contains a reusable
state-dependent signal. It does **not** prove that a deployable Q_safe can learn
the signal, reduce paired closed-loop falls, or reduce SAC-from-zero falls.
Objective 2 remains forbidden until Objective 1 is authorized by the final
compiler.

The current Stage-B implementation is not frozen: the latest focused suite
has `44 passed / 3 failed`, with failures in the transition to an
outcome-inaccessible split identity view. No actor-bank or Stage-B outcome
collection may begin before those failures and the full-suite checks are green.

## 3. Objective 1 execution ladder

### B0 — Result-blind implementation and freeze

Complete and test the existing Stage-B implementation.

- Make the split compiler consume only a dedicated identity view. It may read
  identity arrays and raw-byte/content hashes, but may not index `fall`,
  `max_tilt_rad`, `min_height_m`, or derive any outcome statistic before the
  Model-Test commitment.
- Store a real role/source-independent trajectory fingerprint: the SHA-256
  compound snapshot captured immediately after `reset_standing` settles and
  before the first source impulse, observation, or source-policy action. Do not
  alias `trajectory_id`; its role/source prefix is operational metadata only.
- Persist admission and label CRN, rollout, perturbation, and candidate seeds.
  Validate all ten `role × partition` domains, all four namespaces within each
  partition, and all 45 unordered partition-pair seed unions.
- Add file and parent-directory fsync to every irreversible marker and staged
  publication. A partial marker or fsync failure permanently consumes the
  attempt and fails closed.
- Freeze actor-bank production dimensions to `46D` observation and `12D`
  action; remove production CLI dimension overrides.
- Add regression tests for outcome indexing, trajectory aliasing, admission-only
  seed collision, marker durability, and report publication ordering.

Exit: focused tests, full tests, `py_compile`, and `git diff --check` pass on a
clean worktree; create the single generator commit used by the entire Stage-B
evidence run.

### B1 — Fixed SAC actor bank

Train exactly 14 SAC-from-zero seeds and export exactly 25k, 50k, and 100k
policy-step policy-only checkpoints, for 42 identities total.

- Include every preregistered seed/checkpoint pair.
- Never filter by return, fall count, stability, or checkpoint quality.
- Export after the scheduled update and before the next transition.
- Bind actor bytes, state dict, policy configuration, run contract, and clean
  generator commit in the actor-bank manifest.

### B2 — Five-role same-state data

Use physically separate roles:

| Role | Groups | Replicas | Use |
|---|---:|---:|---|
| Fit | 1,536 | R32 | Q_safe and normalization |
| Probability calibration | 384 | R32 | temperature calibration |
| Uncertainty calibration | 384 | R32 | conformal offsets |
| Selector calibration | 384 | R32 | selector and placebo |
| Model-Test | 768 | R64 | one-shot held-out test |

Each group is one admission-positive state from one source trajectory. It
contains K9 candidates, H96 continuation, five corrected observation frames,
requested/executed/q-target action representations, per-replica outcomes,
CRN identities, and a physically separate privileged diagnostic view.

PPO-generated states may enter representation pretraining or proposal mining,
but final Q_safe evidence must retain target-SAC actor/source identities and
same-state labels.

### B3 — Fit, calibrate, and freeze

Fit only on the Fit role. Freeze, in order:

1. fit-only normalization;
2. five-member ensemble and temperature calibration;
3. signed conformal risk/benefit offsets;
4. exact 100-point selector grid and simultaneous bootstrap;
5. matched-random placebo bundle;
6. self-describing Q_safe artifact.

Model-Test outcomes cannot change any model, temperature, threshold, selector,
placebo, or action protocol.

### B4 — One-shot Model-Test gates

The Model-Test is consumed atomically before its outcome bytes can be loaded.
It must pass all gates:

- pair accuracy `>=0.60`, 2.5% LCB `>=0.55`;
- strong-pair accuracy `>=0.62` for empirical risk gaps `>=0.25`;
- top-1 reduction `>=0.05`, one-sided LCB `>=0.03`;
- frozen-selector reduction `>=0.03`, one-sided LCB `>0`;
- intervention rate `<=0.35`;
- oracle-gap capture `>=0.25`;
- stable ten-bin ECE `<=0.08`;
- strictly positive top-1 and selector effects at all three checkpoint ages,
  four held-out actor seeds, and twelve source seeds.

Failure is terminal for this one-shot artifact. A failed Model-Test cannot be
re-run with changed thresholds or a different seed.

### C — Paired closed-loop causal confirmation

Only a passing B4 authorizes C. Evaluate the same initial state and the same
continuation randomness under:

1. nominal SAC;
2. frozen Q_safe selector;
3. action-distance/intervention-rate-matched random placebo.

Use at least 1,200 independent groups with R64 paired replicas. Require Q_safe
reduction `>=5 pp`, one-sided LCB `>=3 pp`, positive comparison against placebo,
more improved than worsened pairs, positive effects in every registered
subgroup, and valid placebo balance.

### D — SAC-from-zero Objective 1 confirmation

Only a passing C authorizes D. Run 24 fresh seeds (`201..224`) with three
arms—pure SAC, frozen Q_safe SAC, and matched-random placebo—at exactly 500k
policy steps per seed/arm.

Require all of the following:

- relative fall reduction `>=20%`;
- absolute reduction `>=0.40 falls/1,000 steps`;
- seed-cluster fall-reduction LCB `>0`;
- Q_safe versus placebo LCB `>0`;
- exact paired label-swap test `p<=0.05`;
- return non-inferiority within the preregistered 5% margin;
- velocity tracking degradation `<=0.03 m/s`;
- deadline-miss rate `<0.001`;
- complete provenance for all 72 exposures.

The authorization compiler must recompute all stages and emit
`objective1_pass=true` before any speed-range data or model is created.

## 4. Objective 2 speed expansion

After Objective 1 only, test symmetric ranges in this order:

1. `0.25–0.35 m/s`;
2. `0.20–0.40 m/s`;
3. `0.10–0.50 m/s`.

Each range receives fresh boundary mining, same-state labels, fit/calibration/
test roles, paired closed-loop evaluation, and fixed-exposure online testing.
Stop at the first failed range.

The shared cross-speed critic must receive an explicit deployable `command_vx`
feature and a new model/schema version. A 46D critic without command input may
be reported only as a single-speed critic, not as a cross-speed generalizer.

Each successful range must retain at least 80% of Objective-1 relative fall
reduction, achieve at least 16% relative reduction, keep a positive fall LCB,
and pass the return, velocity, runtime, and placebo gates.

## 5. Verification, artifacts, and commit policy

Required invariants include bit-exact snapshot restore, no cross-role state or
trajectory collision, no role-prefix fingerprint alias, no precommit outcome
indexing, durable one-shot markers, exact actor roster, correct rejected-action
attribution, and fixed-exposure fall accounting.

Required evidence artifacts are versioned grouped datasets, trajectory/RNG
identity views, actor-bank manifest, Q_safe artifact, selector/placebo bundles,
Stage A/B/C/D reports, and the final Objective-1/Objective-2 authorization
reports. Large NPZ/checkpoint/step-log files stay outside Git; manifests,
hashes, reports, and protocol files are committed.

Milestone commits:

1. `docs: consolidate qsafe objective plan`;
2. `feat: freeze V5 Stage B result-blind evidence pipeline`;
3. `docs: record one-shot V5 Stage B decision`;
4. `docs: record Stage C paired closed-loop decision`;
5. `docs: record Stage D Objective 1 authorization`;
6. one protocol/result pair for each authorized Objective-2 speed range.

Between the generator commit and the one-shot Stage-B decision, do not change
HEAD, checkout, or create a code commit. Do not use outcome-dependent seed
filtering, top-up, optional stopping, threshold changes, or manual inspection
of protected outcome arrays. No subagent is created unless the user explicitly
requests one.
