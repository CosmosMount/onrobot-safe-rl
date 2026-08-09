# Objective 1 state-dependent recovery Q_safe V5 execution plan

Status: **Stage A passed; Stage B model fitting authorized; Objective 1 not yet passed**

Protocol identity: `objective1_state_dependent_recovery_qsafe_v5`

Machine protocol: `config/qsafe_state_dependent_recovery_v5.yaml`

Artifact root: `saved/qsafe_development/state_dependent_recovery_v5`

Last updated: `2026-08-10` (Asia/Shanghai)

This is the tracked recovery plan after V4 ended tooling-invalid. Its first
objective is unchanged: produce reproducible evidence that a frozen,
option-aware Q_safe reduces falls during SAC training from zero at `0.30 m/s`.
Speed-range expansion is Phase 2 and remains mechanically blocked until all
four Objective-1 stages pass.

## 1. V4 disposition and immutable evidence

V4 produced no scientific Stage-A decision. All six collectors and the
admission merge completed, but the first discovery merge stopped before any
discovery aggregation, selection, audit authorization, or audit access. The
cause was deterministic: the producer wrote the canonical four-field RNG
manifest (`domain_hex`, `role_tags`, `algorithm`, `stream_mapping`), while a
wrapper-local merge predicate compared it with a two-field dictionary.

The following record is immutable. It remains traceable through the committed
V5 plan, the V5 machine protocol's parent record, and the clean generator
commit; it is not redundantly embedded in every V5 artifact:

| Evidence | Value |
|---|---|
| V4 terminal-record commit | `439c525d8e065071838f5d7022031b351fd8f52e` |
| V4 collection/generator commit | `8cb6a3d038c23361fc142e20a6a0d2ad42c9df7f` |
| V4 canonical protocol SHA-256 | `101484a5df78b22941a8988f9936c7fb40b4569ed5c555273843484275dcc977` |
| V4 raw protocol SHA-256 | `dc11b0267042448434076caf359dac7da039e9eec256cb8d04fa09d87d32505f` |
| V4 cohort-lock SHA-256 | `e1d6d697d410ac4820672cd49d89a2f280d221462ba7e8abf3522e363d686cff` |
| V4 collection-readiness SHA-256 | `b4bb13554d6b815f798f9f039a75c5f06bf67f067e5a2aa8ad4875cd10095201` |
| Outcome-blind RNG-contract repair commit | `04e040bc5179682c7c435dd15f5411445278c022` |

V4 seeds `8401`, `8402`, `8411`, `8412`, `8421`, and `8422` are consumed.
V4 admission, discovery, audit, ledgers, manifests, locks, labels, model files,
and derived summaries are forbidden inputs to V5. No V4 file may be copied,
linked, merged, normalized, fitted, calibrated, selected, or audited under a
V5 name. The repair commit is reusable code provenance, not reusable outcome
evidence, and cannot retroactively repair or resume V4.

## 2. V5 identity, isolation, and RNG contract

V5 has a new protocol name, machine-protocol path, artifact root, report schema
identity, cohort lock, attempt markers, consumption markers, and report hashes.
The V5 root must be absent before preflight; a preflight may not create it.

The primary seed-domain literal is exactly:

```text
qsafe_state_dependent_recovery_v5_seed\0
```

Its bytes are
`71736166655f73746174655f646570656e64656e745f7265636f766572795f76355f7365656400`,
its SHA-256 is
`23b2e3f1b75adf44e3cf0a01815d3f05e7fce4fa7912e4aee08ebca564dc52dd`,
and the locked little-endian low-15-bit prefix used by the existing injective
packer is `12835`. V4 used prefix `18561`, so the V5 and V4 packed domains are
disjoint. The encoding remains the high tag bit followed by fixed-width fields
`domain_prefix[15]`, `source_seed[14]`, `role_tag[8]`, `identity[18]`,
`namespace[2]`, and `index[6]`; all exclusive caps and overflow rejection
remain unchanged.

The Stage-B admission/label domains, placebo domain, Stage-C
admission/evaluation domains, Stage-D lane-assignment domain, and
`qsafe.objective1_authorization_compiler.v4` schema remain byte-for-byte frozen
from the preregistration. Their `v4` text names the unchanged downstream RNG or
schema contract, not a reusable V4 outcome. Those stages never ran under V4,
and V5's protocol hash, generator commit, authorization chain, and artifact
root provide the campaign identity. A V4 **Stage-A** domain, split, collection
version, source seed, or artifact path in a V5 artifact is a hard failure.

The public dataset/ledger loaders and generic merger are not authorized V5
readers. They reject reserved V3/V4/V5 basenames and every descendant of the
three canonical workflow roots before probing a supplied artifact. Dedicated
V3/V5 code must enter a process-local scope bound to the exact workflow, role,
and lexical path; the scope cannot authorize V4, ambiguous aggregate-audit
names, a different path or role, or bypass the audit-consumption marker,
symlink, and hard-link checks.

### Mandatory producer-consumer regression gate

Before any V5 preflight or simulator step:

1. One canonical helper must construct and validate the exact four-field RNG
   manifest. The collector, shard validator, merger, selection-lock validator,
   and audit validator must call that helper; local partial reconstructions and
   exact comparisons against ad-hoc dictionaries are forbidden.
2. A producer-to-merger integration test must serialize a collector-produced
   leaf and pass it through the real merger validation path. It must prove the
   complete four-field object survives serialization and is accepted.
3. Negative tests must independently reject a missing, extra, renamed, or
   mutated `domain_hex`, `role_tags`, `algorithm`, or `stream_mapping` field;
   changed role/namespace mappings, V4 Stage-A literals, integer narrowing, and
   non-lossless `uint64` round trips must also fail.
4. The same object must pass selection-lock and audit-validator round trips.
   The historical V3 golden and pairwise role-stream disjointness tests remain
   required.
5. The complete V5 suite and the focused producer-consumer suite must pass on
   the exact clean commit later used by every preflight and collector.

Any regression failure consumes no seed because it occurs before outcomes, but
it blocks collection. A failure after a V5 attempt marker consumes that seed
and terminates V5; no in-place repair or rerun is allowed.

## 3. Claim ladder and hard stage gates

```text
A: fresh causal-headroom confirmation
  -> B: option-aware Q_safe fit/calibration/one-shot model test
    -> C: fresh paired persistent-option closed-loop test
      -> D: 24-seed, three-arm SAC-from-zero online confirmation
        -> Objective 1 pass
          -> Phase 2 speed expansion may start
```

- Engineering and synthetic testing may be prepared early, but Stage-B data
  collection or model fitting requires the canonical Stage-A report to pass.
- Stage C requires the canonical Stage-B compiler report to pass.
- Stage D requires the canonical Stage-C report to pass and a complete live
  online collector adapter with provenance checks.
- Phase 2 requires one fail-closed compiler to validate and recompute A, B, C,
  and D in order and emit both `objective1_pass=true` and
  `phase2_authorized=true`.
- At the first valid scientific failure, every later canonical artifact must
  be absent. Missing or malformed evidence raises; it is not converted into a
  favorable or ordinary scientific failure.
- No top-up, seed substitution/removal, optional stopping, runner-up audit,
  threshold change, heldout rerun, or post-outcome implementation change is
  permitted.

## 4. Stage A — fresh state-dependent causal headroom

### 4.1 Fresh cohort and unchanged mechanism

| Actor age (policy steps) | Fresh V5 source seeds | Groups/source |
|---:|---|---:|
| `25,438` | `8901`, `8902` | 64 |
| `50,030` | `8911`, `8912` | 64 |
| `100,359` | `8921`, `8922` | 64 |

The three source actors and mature recovery actor retain the V4-locked
checkpoint/content fingerprints. The new source seeds are environment and
trajectory RNG roots, not independent actor-training seeds. The claim remains
conditional on the seed-42 actor chain; cross-training-seed confirmation is
reserved for Stages B--D.

The scientific design is exactly unchanged from V4:

- `G=384`: exactly 64 admitted states for each of six seeds, at most one state
  per source trajectory;
- admission `R=32`, retaining 6--26 nominal falls inclusive;
- `K=9` in this exact order: nominal early actor; mature actor for L10, L25,
  L50; joint brake L10; halfway-to-neutral L10 and L25; ramp-to-neutral L25;
  ramp-to-crouch L25;
- physically separate discovery `R=64` and audit `R=64` replicas;
- `H=96` policy steps, 50 Hz policy / 500 Hz low level, ten substeps, source
  actor resumes after a non-nominal option, and no reselection;
- maximum 100 source policy steps/episode, 4,096 proposals, 2,048 source
  trajectories, five-step rejected-proposal cooldown, two settle steps,
  impulse every ten policy steps, pre-screen tilt `>=0.10 rad` or height
  `<=0.32 m`, and zero post-snapshot branch disturbance;
- unchanged failure predicate, state/action projection, PD gains, source and
  mature actor identities, recursive MJCF digest, conditional estimand, and
  action-attribution semantics.

### 4.2 Selection and one-shot gate

Discovery retains every candidate tied for minimum empirical fall risk at each
state and scores a tie by its uniform expectation. Candidate column order may
not affect selection. Audit evaluates that locked per-state rule once.

Stage A passes only if all V4 thresholds remain true:

- audit fall-risk reduction from nominal `>=0.10`;
- hierarchical-bootstrap one-sided 95% LCB `>=0.07`;
- all-K9 discovery/audit pair agreement `>=0.70` and its one-sided 95% LCB
  `>=0.65`;
- effect strictly positive at all three actor ages and all six source seeds;
- discovery nominal risk overall in `[0.15,0.75]` and at every age in
  `[0.10,0.90]`;
- exact G384/K9/R64/H96, identity, physical split, and RNG data gates pass.

The bootstrap is unchanged: 50,000 PCG64 draws, seed `20260810`, linear
quantile, complete trajectory groups, equal groups within source seed, and
equal weight for the six source seeds. No diagnostic fixed candidate, oracle,
tie subset, or full-replica reanalysis can override the primary decision.

### 4.3 Exact preflight, collection, merge, lock, and audit order

The following order is normative; completing a later item early invalidates
V5.

1. Commit the V5 plan, machine protocol, implementation, regression tests, and
   synthetic tests. Record a clean immutable generator commit and all hashes in
   Section 8.
2. Run the complete suite on that commit. Then run `--preflight-only` exactly
   once for `8901`, `8902`, `8911`, `8912`, `8921`, and `8922`. Every run must
   report zero simulator steps and must create no artifact root, cohort lock,
   attempt marker, shard, or report.
3. Reconfirm the same clean HEAD and protocol hashes. Atomically publish the
   V5 cohort lock before the first simulator step.
4. Launch each of the six collectors once. Each writes its attempt marker
   before its first simulator step and its collection report last. Complete all
   six report-last records before any analysis; do not inspect discovery or
   audit outcomes while collection is in progress.
5. Merge and validate admission only. The admission report must bind all six
   leaves and report both `candidate_outcomes_opened=false` and
   `audit_opened=false`.
6. Merge discovery only from the six physical discovery shards. The merger
   must validate the canonical four-field RNG manifest through the shared
   helper before aggregation. It may not parse, resolve, stat, hash, map, load,
   or open an audit path.
7. Using admission plus discovery only, run the data gate and discovery
   informativeness gate, then atomically publish the selection lock. The lock
   binds the chosen tied sets and exactly one `audit_authorized` decision.
8. If authorization is false, publish only the canonical no-model-training
   report (or use the hash-bound `resume-denied-report` transaction after a
   crash) and stop. Audit remains untouched.
9. If authorization is true, publish the no-clobber audit-consumption marker
   first. Only then may the independent evaluator first resolve/hash/open the
   six audit shards and run the single preregistered audit. Publish its report
   last. Interruption after the marker consumes the audit; no rerun is allowed.
10. Commit the canonical Stage-A pass/fail record. Start Stage B only if the
    report explicitly passes every gate.

## 5. Stage B — unchanged option-aware Q_safe science

All V4 Stage-B scientific settings remain frozen. No V4 outcome file is an
input; every role below is collected fresh under the unchanged preregistered
Stage-B role domains after Stage A passes.

| Role | Actor seeds | Environment source seeds by 25k/50k/100k age | Groups; labels |
|---|---|---|---|
| fit | `43..46` | `8501..8504`, `8511..8514`, `8521..8524` | 1,536; R32 |
| probability calibration | `47,48` | `8601,8602`, `8611,8612`, `8621,8622` | 384; R32 |
| uncertainty calibration | `49,50` | `8631,8632`, `8641,8642`, `8651,8652` | 384; R32 |
| selector calibration | `51,52` | `8661,8662`, `8671,8672`, `8681,8682` | 384; R32 |
| one-shot model test | `53..56` | `8701..8704`, `8711..8714`, `8721..8724` | 768; R64 |

Each role has separate R32 admission, actor/checkpoint identities, source
trajectories, CRN/candidate/rollout/perturbation streams, physical files, and
report-last completion. Model test is committed by hash before fitting but is
not probed until the frozen model, all three calibrations, selector, and placebo
are published and a no-clobber consumption marker exists.

The deployable model remains five corrected 46D observation frames plus the
82D descriptor (36D current nominal tuple plus the 46D K9 recovery-program
view). It is the five-member Selective Advantage Q_safe ensemble with the same
heads, losses, group bootstrap, fit-only normalization, 100 epochs, batch 64,
AdamW `3e-4`, weight decay `1e-5`, clip 5, widths 128/128/128, and model seed
`20260810`. The recovery-library fingerprint remains
`fcfb1fa541acf316f87dacf82b1fdeb9188d7a4b9df7f69544b567fb2c5d1045`.

The V4 model gate is unchanged: pair accuracy `>=0.60` with LCB `>=0.55`,
strong-pair accuracy `>=0.62`, learned top-1 reduction `>=0.05` with LCB
`>=0.03`, frozen-selector reduction `>=0.03` with LCB `>0`, intervention rate
`<=0.35`, positive top-1 and selector effects at every age/all four actors/all
twelve sources, oracle-gap capture `>=0.25`, ECE `<=0.08`, and all identity/data
gates. Calibration uses the same signed conformal rules, two separately
Bonferroni-corrected eight-option families, nominal alpha `0.05`, and the same
100-point selector grid. Selector max-statistic search uses 50,000 PCG64 draws
with seed `20260811`; model test uses 50,000 hierarchical draws with seed
`20260812`. Placebo fitting and balance limits remain `0.02` intervention-rate
mismatch, `0.05` duration TV, and `0.10` action-distance KS.

## 6. Stage C — unchanged fresh paired closed-loop science

Only a passing Stage B authorizes Stage C. Actors `57..60` own, respectively,
sources `8801/8811/8821`, `8802/8812/8822`, `8803/8813/8823`, and
`8804/8814/8824`. Collect exactly 100 admitted states per source (`G=1,200`),
R32 admission, then a physically separate R64 three-arm evaluation of nominal,
frozen Q_safe, and matched-random placebo. Each arm may start at most one
persistent option and shares state/replica continuation randomness without
shifting the actor stream.

The paired bootstrap remains 50,000 PCG64 draws, seed `20260813`, linear
quantile, actor seed outermost and complete state group innermost. Stage C
requires Q_safe-versus-nominal reduction `>=0.05` with LCB `>=0.03`; positive
effect at all three ages, four actor seeds, and twelve sources; improved pairs
outnumber worsened pairs; Q_safe-versus-placebo LCB `>0`; realized placebo
balance within `0.02/0.05/0.10`; and no runtime, attribution, manifest, support,
timing, or data failure. Failure stops before online SAC.

## 7. Stage D — unchanged SAC-from-zero confirmation

Only a passing Stage C and a complete live collector/provenance adapter
authorize Stage D. Use seeds `201..224`, three arms (pure SAC, SAC plus frozen
Q_safe, matched-random placebo), and exactly 500,000 50-Hz exposure steps for
each of 72 seed/arm runs. Within each seed the three initial actor, critic,
target, optimizer, and reward-normalizer states are bit-identical; Q_safe,
selector, and placebo remain frozen. Counter-based streams, evolving actor hash
chains, transition attribution to the executed shield action, and exact
exposure accounting remain unchanged.

The physical A/B lane assignment still uses the already-preregistered,
independently domain-separated Bernoulli-half bit per seed and lane C for
placebo. Its frozen V4-labeled domain, exact 24-bit vector, vector SHA-256, and
no-clobber assignment-manifest contract remain unchanged and must be verified
before the first Stage-D outcome. This label is a retained subprotocol schema,
not permission to consume a V4 artifact; the randomization model, bit
derivation, seed order, lane mapping, and exact `2^24` analysis are unchanged.

The online gate remains: relative fall reduction `>=20%`; absolute reduction
`>=0.40` falls/1,000 steps; seed-cluster treatment LCB `>0`;
treatment-versus-placebo LCB `>0`; exact one-sided label-swap `p<=0.05`;
signed-safe return noninferiority within 5% of baseline magnitude; velocity
error increase `<=0.03 m/s`; Q_safe deadline-miss rate `<0.001`; realized
placebo balance within `0.02/0.05/0.10`; and all 72 complete exposures with
verified provenance. The paired bootstrap remains 50,000 PCG64 draws, seed
`20260814`, linear 5% quantile, using one shared matrix of complete three-arm
seed tuples. There is no optional stopping or result-dependent top-up.

## 8. Hash, cleanliness, and commit ledger

Before the first V5 outcome, resolve every pending binding below. Values that
would make this tracked file self-referential (its own commit or raw hash) are
computed from the committed bytes and recorded in the external execution
ledger before preflight. They need not be duplicated in every machine artifact.
A value recorded from a dirty worktree, a different commit, or after an outcome
is invalid.

| Item | Required evidence | Status |
|---|---|---|
| V4 terminal record | commit and hashes in Section 1 | complete |
| RNG producer-consumer repair | commit `04e040bc5179682c7c435dd15f5411445278c022` | complete; reverified by the clean V5 suite and six successful collectors |
| V5 prereg/generator commit | full clean commit SHA | complete: `1452a112c35d94eed87bafb0c3f2bf73ab907324` |
| V5 plan | raw SHA-256 | complete for preregistered bytes: `06a5df8f64c306ce515483e95fd6f5dadec99c867c6bed2ac6d81f5298ec973d` in external ledger |
| V5 machine protocol | raw / canonical SHA-256 | `f4c3e796004d124574df3d35ef344f6a4a766d9099acb5792c1d78b8361b49b0` / `1e8667aa17ab361c323771d5deb51258644cce37bd26bed00599ca08d7545ea5` |
| V5 RNG helper/collector/merger/auditor/tests | path/content hashes bound by generator commit | complete; bound by `1452a112c35d94eed87bafb0c3f2bf73ab907324` |
| V5 recovery library | canonical fingerprint | locked to `fcfb1fa541acf316f87dacf82b1fdeb9188d7a4b9df7f69544b567fb2c5d1045` |
| V5 six preflights | seed, commit, protocol hashes, zero simulator steps, no created root | complete; all six exit 0 on the exact generator commit |
| V5 cohort lock | SHA-256 and publication time before first simulator step | complete: `495360ceece0573ba168cb10676441771bc6a7ab0208e1583f5b679e95f9fd12` |
| Six collection reports | attempt/report hashes and exact counts | complete: G384, 8,946 proposals, 52,696 source steps, 1,818 trajectories |
| Admission/discovery/selection/audit | input/output hashes and report-last markers in exact order | complete; Stage-A report `e7ea56546bf8006cfc4d8ade4f5b2c26dbfcbc132e0568e054a98c2be3174b2e` passes |

Each machine artifact must bind the generator commit, protocol hashes,
predecessor hashes, and exact source/actor roster required by its schema. The
committed plan and machine protocol provide the transitive V4 lineage; the
external execution ledger records the committed plan hash and published
artifact hashes. Artifact publication is atomic, no-clobber, and report-last.

Commit boundaries are mandatory and provide more than the requested minimum of
three independently reviewable stages:

1. `V5 preregistration + implementation + producer-consumer regression tests`
   — must precede every V5 preflight and outcome.
2. `V5 Stage-A result` — canonical pass/fail or tooling-invalid record only.
3. `V5 Stage-B implementation/result` — only after Stage A passes; separate
   implementation and outcome commits when implementation is nontrivial.
4. `V5 Stage-C paired result` — only after Stage B passes.
5. `V5 Stage-D online result and Objective-1 compiler decision` — only after
   Stage C passes.
6. Phase-2 commits — forbidden until the Objective-1 compiler authorizes them.

The earlier implementation stages are also retained as provenance:
`d9f32eba1576e7317229620df55d1d28c64b4fe8`,
`88549cae5aee5e7d185fd86120ce2e14d7ef0744`, and
`8cb6a3d038c23361fc142e20a6a0d2ad42c9df7f`. They satisfy historical staging
but do not replace the fresh V5 preregistration/result boundaries above.

## 9. Tracked execution checklist

- [x] Preserve the V4 tooling-invalid terminal record and consume all six V4
  seeds.
- [x] Land an outcome-blind shared RNG-manifest repair foundation.
- [x] Commit V5 identity/root/domains, fresh Stage-A seeds, producer-consumer
  tests, and this plan on one clean preregistration commit.
- [x] Pass the full suite and focused RNG round-trip suite on that exact commit.
- [x] Pass six outcome-free preflights without creating the V5 root.
- [x] Collect six fresh Stage-A sources exactly once and complete all reports.
- [x] Run admission merge, discovery merge, selection lock, and (only if
  authorized) the one-shot audit in the exact order in Section 4.3.
- [x] Commit the passing Stage-A disposition and authorize Stage B only.
- [ ] If Stage A passes, execute and commit Stage B.
- [ ] If Stage B passes, execute and commit Stage C.
- [ ] If Stage C passes, execute and commit Stage D and compile Objective 1.
- [ ] Begin Phase 2 only if the compiler emits `phase2_authorized=true`.

## 10. Phase 2 remains blocked

No Stage-A headroom, model-test subset, paired offline result, or diagnostic
unlocks speed expansion. Only complete A+B+C+D authorization permits the
unchanged sequence `0.25--0.35`, `0.20--0.40`, then `0.10--0.50 m/s`, with at
least 80% retention of the Phase-1 relative reduction, at least 16% relative
fall reduction, and positive confidence bounds. A shared cross-speed model must
add commanded velocity as an explicit deployable feature; otherwise use
separately named per-speed critics.
