# Q_safe Objective 1 → Objective 2 master plan

**Task:** Unitree Go2 at constant `vx=+0.30 m/s`, `vy=0`, `yaw_rate=0`

**Status:** Objective 1 not passed; Objective 2 mechanically blocked

**Data route:** target-aligned MuJoCo-Warp PPO-from-zero, natural falls only

**Q_safe route:** direct state-risk trigger plus a separately validated recovery mechanism

## 1. Claim and priority

Objective 1 asks whether a pretrained safety trigger reduces falls while SAC is
trained from zero on the existing `+0.30 m/s` task.  It must beat both pure SAC
and an intervention-rate-matched random placebo without materially damaging
return or velocity tracking.  Objective 2 may begin only after the final
Objective-1 compiler emits both `objective1_pass=true` and
`phase2_authorized=true`.

The fixed-impulse source protocol is retired.  Its ten started attempts and
zero completed sources are isolated under
`saved/qsafe_development/state_dependent_recovery_v5/stage-b/aborted-fixed-perturbation-db761ac`
and cannot enter any new dataset.

The replacement data flow is:

```text
PPO-from-zero, +0.30 m/s, no push
        ↓
all independent natural falls + matched natural normal states
        ↓
direct supervision of Q_safe state risk
        ↓
SAC-only calibration and protected transfer test
        ↓
Q_safe danger trigger + frozen recovery mechanism
        ↓
paired SAC-only causal test
        ↓
24 fresh SAC seeds × pure/Q_safe/random arms
```

PPO labels answer only whether the current state led to a fall under the PPO
trajectory.  They supervise `Q_safe(s)` directly.  PPO actions are stored for
provenance and diagnostics, but no unexecuted action is assigned a fabricated
label.  Recovery outcomes never feed back into these PPO state labels.

## 2. Recovery contract and SQRL interpretation

The requested first recovery candidate is the repository's original non-policy
controller:

- implementation: `runtime/control/go2/motions/src/recovery.cpp`;
- configuration: `runtime/control/go2/go2.yaml`;
- control rate: 500 Hz;
- stages: `Fold -> Above -> SwingDown -> Push`;
- gains: `kp=100`, `kd=8`;
- no learned recovery actor and no policy network;
- exact absolute joint targets, timing, leg masks and joint-reach gate are bound
  into the recovery artifact.

This uses the SQRL separation at a coarse level: Q_safe decides whether normal
task control may continue; rejection transfers control to another action
source.  Here that source is a deterministic motion program rather than a
recovery policy.

The fall predicate remains active during recovery.  A recovery-induced height
or orientation threshold crossing counts as a fall; it cannot be hidden as an
"intentional" motion.  Before fitting or deploying a trigger, the recovery
mechanism must pass two result-blind preflights:

1. it must not manufacture a fall in the registered stable-standing negative
   control;
2. on natural SAC pre-fall snapshots, same-state fixed-recovery continuation
   must show positive paired fall headroom versus nominal SAC.

Failure rejects this recovery mechanism for fall prevention.  It does not
invalidate the natural-PPO archive or authorize relabeling falls.  An
alternative mechanism—such as an action-conditioned SQRL resampling shield or
predictive rollout shield—requires a new frozen protocol and the same online
Objective-1 gates.

## 3. Phase A — target mechanics, parity and capacity

Production PPO must match the target SAC task:

- initial pose `hip=0.05`, `thigh=0.70`, `calf=-1.40`;
- normalized action scale `0.20/0.40/0.40`;
- position targets with `kp=60`, `kd=5` and target effort limits;
- 2-ms MuJoCo steps, ten low-level steps per 50-Hz policy step;
- target collision geoms, friction, solver and fall predicate;
- constant `+0.30 m/s` command and no push, impulse or velocity injection.

Required gates:

1. hash-bound target model/config contract;
2. 100 states × 100 steps native/Warp parity corpus;
3. 256/512/1024/2048 capacity ladder, each measured for at least five minutes;
4. select the largest tier whose throughput gain is at least 15%, peak VRAM is
   at most 20 GiB, and which has no OOM, NaN, reset/RNG cross-talk or kernel
   failure;
5. run the selected tier continuously for at least 30 minutes with no sustained
   resource growth.

The experiment process owns its resource sampler; it exits with the runner.
No external watcher, automation, retry loop or periodic user notification is
allowed.

## 4. Phase B — 30M natural-PPO collection

Train official-size clipped PPO with GAE, critic loss, entropy, minibatches and
multiple update epochs for exactly 30M policy-environment steps.  Save fixed
exposure checkpoints at `0, 1M, 2M, 5M, 10M, 20M, 30M`.  Do not stop, extend,
select or discard a checkpoint based on its fall count or return.

Every environment follows these rules:

- the first terminal fall is the episode's only counted fall;
- reset occurs in the same vector step;
- retain short episodes with availability masks;
- keep the preceding 64 policy steps and offsets `1/2/4/8/16/32/64`;
- store integration state, qpos/qvel/ctrl, corrected `5×46` history, requested
  and executed action, absolute q-target, command, randomization and RNG/
  environment/episode/step identities;
- verify runtime force arrays remain zero;
- sample matched normal states by a frozen hash rule, at least 96 steps from a
  fall or terminal, stratified by PPO age and realized randomization;
- shard atomically, fail closed on corruption or duplicate identity, and
  publish the manifest last.

Acceptance requires
`recorded_falls == independently_terminated_fall_episodes`, unique identities,
zero external force, coverage at every registered PPO age, and descriptive
fall/normal distribution reports.  Large shards and checkpoints stay outside
Git; manifests, hashes and reports are committed.

## 5. Phase C — state-risk fitting and SAC-only transfer

Fit a five-member state-risk ensemble using only the Fit role.  Split by whole
PPO episodes; a trajectory may never cross roles.  The production loss contains
no executed-action or candidate-action head.

PPO-held-out metrics are development diagnostics.  Probability calibration,
uncertainty calibration, trigger selection and protected Model-Test use only
SAC-proposed states.  No PPO state may enter the protected SAC Model-Test.

Frozen Model-Test gates:

- state AUROC `>=0.60`, bootstrap LCB `>=0.55`;
- ten-bin ECE `<=0.08` after SAC-only calibration;
- intervention rate `<=35%`;
- positive risk separation and trigger direction at every registered SAC age
  and held-out source seed;
- fixed-trigger paired fall reduction `>=3 pp`, one-sided LCB `>0`.

The protected result is one-shot.  Outcomes cannot change the model,
temperature, uncertainty offset, threshold or placebo bundle.

## 6. Phase D — at least 1,200 paired SAC-only states

Only a passing recovery preflight and Phase-C Model-Test authorize this phase.
For at least 1,200 independent SAC-only snapshots, use identical initial state
and common continuation randomness for:

1. nominal SAC;
2. frozen Q_safe trigger plus frozen recovery;
3. intervention-rate-matched random trigger plus the same recovery.

Require at least 5-pp absolute Q_safe fall reduction, one-sided LCB at least
3 pp, positive Q_safe-versus-placebo LCB, more improved than worsened pairs,
valid placebo balance and positive effects in every registered subgroup.

## 7. Phase E — SAC-from-zero Objective 1

Only passing Phases C and D authorize 24 fresh seeds (`201..224`) with three
500k-policy-step arms per seed:

- pure SAC;
- SAC + frozen Q_safe/recovery;
- SAC + matched-random recovery placebo.

Objective 1 passes only if all 72 fixed exposures complete and all gates hold:

- cumulative falls reduced by at least 20% relative;
- absolute reduction at least `0.40 falls/1000 steps`;
- seed-cluster bootstrap LCB above zero;
- Q_safe-versus-placebo LCB above zero;
- paired label-swap `p<=0.05`;
- mean return at least 95% of pure SAC;
- velocity-error increase at most `0.03 m/s`;
- deadline-miss rate below `0.1%`;
- complete immutable provenance.

## 8. Objective 2 — forbidden until authorization

After Objective 1 only, test `0.25–0.35`, then `0.20–0.40`, then
`0.10–0.50 m/s`.  Each range requires new boundary data, calibration,
protected test, paired causal evidence and fixed-exposure online evidence.  A
cross-speed critic must receive the command as an explicit deployable input.
Stop at the first failed range.

## 9. Staged commits

Keep at least these durable boundaries:

1. `protocol: replace fixed perturbations with natural PPO fall collection`;
2. `feat: add parity-gated parallel PPO natural-fall collector`;
3. `test: validate terminal reset snapshot export and resource accounting`;
4. `docs: record PPO capacity and natural-fall collection result`;
5. `feat: train state-risk qsafe and bind frozen recovery`;
6. `docs: record Objective 1 paired and online decision`.

Run focused tests before every commit and the full repository suite before each
claim-bearing collection generator is frozen.  Objective 2 commits are
forbidden until the final Objective-1 authorization report passes.
