# Q_safe fall-reduction execution roadmap

Status: active, Phase 1D Iteration 5 Stage B (Objective 1 not yet passed)
Consumed Iteration-1 protocol: `config/qsafe_evidence_protocol.yaml`

Consumed no-headroom triage: `config/qsafe_recovery_option_triage_v2.yaml`

Consumed V3 preregistration: `config/qsafe_closed_loop_recovery_triage_v3.yaml`

Terminal tooling-invalid V4 record:
`QSAFE_STATE_DEPENDENT_RECOVERY_V4_TECHNICAL_FAILURE.md`

Active preregistration: `config/qsafe_state_dependent_recovery_v5.yaml`

Active execution plan: `QSAFE_PHASE1_STATE_DEPENDENT_RECOVERY_V5_PLAN.md`

Active Stage-B implementation plan:
`QSAFE_PHASE1_STATE_DEPENDENT_RECOVERY_V5_STAGE_B_PLAN.md`

Result-blind Stage-B execution supplement:
`config/qsafe_state_dependent_recovery_v5_stage_b_execution.yaml`

Branch: `codex/qsafe-evidence-pipeline`

## 1. Objective and ordering constraint

The project has two ordered objectives:

1. Produce reproducible evidence that a learned safety mechanism reduces falls while SAC is trained from scratch, or under a small command-speed change around 0.30 m/s.
2. Only after Objective 1 passes its preregistered gates, expand the speed range while retaining most of the fall reduction.

Objective 2 is mechanically blocked. A promising plot, training-set score, Oracle result, or pointwise AUROC cannot unlock it. The phase transition requires a machine-readable Phase 1 evidence report that passes every gate in the protocol.

## 2. Evidence inherited from the repository

The starting facts are deliberately narrow:

- A pure SAC 500k checkpoint is the stable reference policy.
- A local one-frame Q_safe selector did not reduce fresh H32 falls.
- Short receding recovery improved the Oracle reduction from about 1.72 pp at one frame to 3.75 pp at two frames and 4.22 pp at three frames.
- Repeated closed-loop Oracle reduced failure from 49.53% to 36.72%, a 12.81 pp reduction with 95% CI [9.22, 16.56] pp; matched-random repeated selection improved only about 0.47 pp.
- The filtered-policy Q_safe development set had only 360 independent groups and produced near-random target-distribution action discrimination.

These facts show that useful actions exist and that receding selection matters. They do **not** show that the current learned critic can find those actions, or that more data alone will fix it.

## 3. Lessons taken from prior work

### SQRL

[Learning to be Safe: Deep RL with a Safety Critic](https://arxiv.org/abs/2010.14603) and its [author code](https://github.com/krishpop/sqrl) learn a safety critic from sparse failure labels, use rejection filtering, and transfer the critic to related tasks. The relevant lessons are:

- safety must be evaluated during learning, not only after convergence;
- Q_safe is policy-conditional because its Bellman continuation follows the constrained/filtered policy;
- a small recent constrained-policy safety replay is intentional; an all-history SAC mixture can be pessimistic for the wrong continuation;
- boundary exploration samples actions just below the safety threshold;
- transfer is only justified across related tasks/dynamics;
- cumulative safety incidents and task performance must be reported together.

The paper is a design prior, not proof that a one-frame local action critic is identifiable for this Go2 observation.

### Recovery RL

[Recovery RL](https://arxiv.org/abs/2010.15920) and its [author code](https://github.com/abalakrishna123/recovery-rl) separate task and safety objectives, initialize the safety critic from offline data, continue updating it under the composite policy, and invoke a recovery policy inside a learned recovery set. For this project, separation suggests that a state-level trigger plus a recovery/action-benefit model may be easier to learn than asking one scalar Q to solve state risk, action ranking, and recovery simultaneously.

### Conservative safety critics

[Conservative Safety Critics for Exploration](https://openreview.net/pdf?id=iaO86DUuKi) addresses distribution shift by intentionally pessimistic safety estimates. The useful idea here is abstention under action/support uncertainty. Blind conservatism is not sufficient: always selecting the nominal/previous action could reduce exploration and task learning without finding recoveries.

[SAILR](https://proceedings.mlr.press/v139/wagener21a.html) is especially relevant to online learning: it defines an intervention advantage and explicitly accounts for the backup intervention in the learning MDP. This supports the relative action-benefit head and requires an intervention-rate/action-distance-matched random placebo.

### Near-future model-based shielding

[Safe Reinforcement Learning by Imagining the Near Future](https://proceedings.neurips.cc/paper_files/paper/2021/file/73b277c11266681122132d024f53a75b-Paper.pdf) uses short model rollouts and evaluates return against cumulative violations over five seeds. This supports keeping a predictive-rollout fallback if direct action ranking remains unidentifiable.

### Shield integration

The [2025 shielding review](https://doi.org/10.1145/3715958) emphasizes that post-shielding can corrupt action/reward association unless learning attributes the reward to the action selected by the shield rather than the rejected nominal action. In this repository the SAC action-space value is the shield-selected `action_requested`; runtime clipping/filtering is part of the environment projection and yields `action_executed` plus absolute `action_q_target`. Phase 1 keeps the current actor/Q action-space contract while recording and testing all three values. It must never train the transition against the rejected nominal action.

## 4. Proposed method: Selective Advantage Q_safe

The primary method is not another pointwise BCE critic. It factorizes the problem:

```text
five corrected observation frames
          |
     temporal encoder
      /          \
state-risk V(s)   predictive auxiliary heads
      |            (TTF, max tilt, min height)
      + nominal/candidate/delta-action
                    |
       relative safety advantage A(s,a|a_nom)
                    |
       ensemble mean + epistemic uncertainty
                    |
 state trigger + support + conformal risk/benefit gates
                    |
       repeated closed-loop selection at 50 Hz
```

The training target for the action head is the same-state difference

\[
\Delta r(s,a)=P(F_{H32}\mid s,a,\pi_c)-P(F_{H32}\mid s,a_{nom},\pi_c),
\]

with common-random-number replicas and a locked continuation \(\pi_c\). Predicting a relative effect discourages the network from winning pointwise AUROC by learning state risk alone.

The selector intervenes only when all conditions hold:

1. the state-risk trigger says intervention is warranted;
2. candidate action is inside measured behavior support;
3. the lower confidence bound on benefit is positive;
4. ensemble support uncertainty remains within its preregistered limit;
5. action/q_send change satisfies local RMS and slew constraints.

If no candidate passes, it abstains. It never substitutes “minimum predicted risk” merely because a minimum exists.

### Fallback method

If privileged and deployable action ranking both remain flat, the target or candidate protocol is wrong; repair labels before changing networks. If privileged rises but deployable saturates, switch to a predictive dynamics/rollout shield or a state-triggered multi-frame recovery policy. This pivot is allowed inside Objective 1, but it receives a new protocol/model name and must pass the same causal and online gates.

## 4.1 Phase-0 blockers found by repository audit

The first audit found correctness defects that block all claim-bearing data collection:

1. The P17 collector/evaluator carries the previous normalized action into `build_observation`, although the corrected 46D contract requires the previous absolute q target actually sent (`q_send`).
2. The initial capture→step→restore check measured about 0.0196 observation mismatch. Follow-up proved the integration state and duplicate step were exact: the mismatch compared stale post-`mj_step` derived sensor fields with post-`mj_forward` fields. Explicit forward synchronization gives zero raw-observation error; sensor/frame-correct reads are still required for runtime parity.
3. The snapshot evaluator defaults to `kd=5`, while another project configuration exposes `kd=10`; actual runtime gains and torque clipping must be fingerprinted rather than inferred.
4. Two test modules currently fail at import because they refer to removed `SACInferencePolicy` and `SQRLActionDecision` names. A green partial test run is not a green safety stack.
5. Action-filter histories are outside MuJoCo's integration state and are not cloned by the legacy evaluator.

Consequences:

- Existing P17 NPZ files are legacy diagnostics only and are disallowed as final evidence inputs.
- Commit 2 begins with observation/action/filter parity and test repair, then adds the grouped schema.
- Native and accelerator backends must both pass duplicate-branch determinism plus corrected-observation parity before producing evidence data.

Phase 1A resolution: native snapshots now include MuJoCo integration state, requested/executed/q-target action state, five observation frames, and Butterworth history. The Go2 SDK-bridge sensors are used, the Python config explicitly matches runtime `kp=60, kd=5`, torque/filter coefficients are fingerprinted, and real-MJCF duplicate branches are bit-exact in regression tests.

## 4.2 Iteration-1 falsification and Iteration-2 pivot

Iteration 1 has now been executed once under its locked protocol. The full
result and immutable hashes are in `QSAFE_PHASE1_ITERATION1_RESULT.md`. The
claim-eligible deployable model achieved held-out pair accuracy `0.5062` and
top-1 reduction `-0.319 pp` with a confidence interval spanning zero. The
pointwise and privileged diagnostics were also at chance. Selector
calibration, paired closed-loop evaluation, and online evaluation were
therefore not authorized.

The decisive diagnostic was independent-replica evaluation. Although selecting
and evaluating on all R=8 replicas reported a pooled empirical Oracle reduction
of `6.896 pp`, fixed 4/4 replica cross-fit produced `-0.119 pp` with 95% CI
`[-0.369, 0.124]`; candidate-effect half correlations were approximately zero
and pair-order agreement was approximately `0.500`. The network's full-train
pair score was `0.778`, but trajectory-out-of-bag inference fell to `0.520`.
Thus the action head learned Monte Carlo ordering noise rather than a reusable
safety effect.

Iteration 2 consequently precedes all model training with a label-reliability
gate. It uses fresh seeds, 32 discovery plus 32 audit replicas, raw audit falls,
order-invariant uniform handling of discovery ties, and trajectory-cluster
bootstrap. The default pivot is a 1--4 step linearly decayed residual recovery
option. A one-step high-replica route remains available only if its independent
audit effect passes; a short predictive model is allowed only if a multistep
option has independently verified causal headroom but a direct scorer later
fails. None of these development outcomes alone can unlock Phase 2.

## 5. Work breakdown and commit boundaries

### Phase 0 — preregistration and leakage barrier (commit 1)

Deliverables:

- technical feasibility report;
- this roadmap;
- machine-readable protocol;
- protected-path rules that reject every path component beginning with `sealed` or `formal`;
- exact definitions of group, fall, H32, top-1 reduction and online fall count.

Exit criterion: another researcher can determine whether a result passes without choosing thresholds after seeing it.

### Phase 1A — grouped data and metrics (commit 2)

Implement a backend-neutral `safety_data` package:

- versioned grouped dataset schema;
- legacy P17 NPZ adapter that is opt-in and never silently fills critical fields;
- content hashes and simulator/policy fingerprints;
- group/source/seed-disjoint split audit;
- per-group candidate/replica validation;
- pair accuracy, strong-pair accuracy, AUROC, Brier, ECE, top-1 reduction, oracle-gap capture and group bootstrap CIs;
- a CLI that fails closed on leakage, duplicates, malformed groups and protected paths.

Before schema/export work, repair and test:

- absolute `q_send` history propagation;
- sensor/frame-correct body velocity and IMU reconstruction;
- complete application-state capture (including filter histories);
- runtime/snapshot PD gain and torque-limit fingerprints;
- broken Paper SQRL and async collector test imports.

Tests use synthetic grouped datasets where every expected metric is analytically known.

### Phase 1B — data generation and model learning

Implement data collection in two increments:

1. Native MuJoCo reference generator based on `MujocoSnapshotEnv`, used for correctness and the first 1k groups.
2. Isolated MjLab/MuJoCo-Warp or MJX generator after native parity passes.

State sources:

- PPO final and early checkpoints for broad coverage;
- frozen SAC at multiple training ages;
- perturbed frozen SAC boundary states;
- filtered/repeated-SAC states for target fine-tuning.

Every retained state is cloned over K candidates and R CRN replicas. The clone contains physics state plus q_send, action-filter history, five observation frames, command, perturbation program and RNG state.

Train five ensemble members with group-disjoint bootstrap samples. Compare:

- deployable history + action;
- privileged state + action diagnostic;
- state-only diagnostic;
- pointwise Q_safe baseline;
- Selective Advantage Q_safe;
- optional predictive rollout fallback.

Run nested learning curves at 360, 1k, 3k, 10k, then 30k/100k only while target-only curves still improve.

### Historical Phase 1C/1D baseline (superseded for V5)

The following was the original v1 route. It is retained as design history, not
as current authorization: its H32/K16/36D assumptions, small-shift alternative,
ten-seed minimum, IPW wording, and legacy compiler are incompatible with V5.
V4 ended tooling-invalid before discovery aggregation and remains available
only through its terminal record and historical commits. The canonical current
route is `QSAFE_PHASE1_STATE_DEPENDENT_RECOVERY_V5_PLAN.md` plus
`config/qsafe_state_dependent_recovery_v5.yaml`: K9/H96, an 82D
nominal-conditioned recovery-program descriptor, fresh actor-role splits, an
exact 1,200-state persistent-option gate, and 24 from-zero online seeds. No v4
or V5 tool may use the legacy protocol below, and no V5 tool may consume a V4
outcome artifact.

#### Phase 1C — closed-loop and online evidence (commit 3)

Implement:

- exact-snapshot nominal versus repeatedly shielded paired H32 evaluation;
- group bootstrap confidence intervals;
- fresh-SAC A/B runner with matched seeds and identical environment/action RNG before interventions diverge;
- fixed policy-step budgets and executed-action replay semantics;
- return, velocity tracking, replacement, abstention, deadline and support metrics;
- an evidence compiler that produces `phase1_pass: true|false` and explains every failed gate.

#### Phase 1D — execute Objective 1

Run in increasing cost order:

1. unit/smoke tests;
2. 1k group native PoC;
3. 360→1k deployable/privileged learning slope;
4. 1k paired repeated-closed-loop development evaluation;
5. independently fine-tune target actors at 0.27 and 0.33 m/s from the locked 0.30 m/s source, then run four-seed 100k-step mechanics screens; these are not evidence;
6. run ten new paired seeds at 500k steps at **both** 0.27 and 0.33 m/s, with matched-random placebos; this small-shift route is the primary confirmatory route because the Q_safe continuation is less nonstationary than fresh-from-zero SAC;
7. run fresh 0.30 m/s training as the secondary route when mechanics support it. Its actor/Q_safe continuation changes throughout training and therefore requires checkpoint-age-conditioned data or a frozen recovery continuation.

Objective 1 passes only if data, model, paired closed-loop, and online training gates all pass. The exact route expression is `common_mechanism_gates AND (fresh_030_online OR (shift_027_online AND shift_033_online))`; neither a favorable single endpoint nor a mechanics pilot can unlock Phase 2.

### Phase 2 — speed expansion (blocked)

After a valid Phase 1 pass, expand symmetrically:

1. 0.25–0.35 m/s;
2. 0.20–0.40 m/s;
3. 0.10–0.50 m/s.

At each range:

- mine new boundary groups without touching the held-out evaluation seeds;
- calibrate uncertainty/thresholds on calibration data only;
- keep the architecture and primary endpoint fixed;
- require at least 80% of the Phase 1 relative fall reduction and at least 16% relative reduction in absolute terms;
- stop at the first failed range and report the supported envelope.

`move_speed` is not part of the current 46D observation and does not by itself change snapshot physics. Each evaluated speed therefore needs an independently fine-tuned/conditioned target-policy checkpoint. A shared cross-speed Q_safe is allowed only if it receives a deployable velocity-command feature; otherwise use and name per-speed critics rather than silently mixing incompatible continuation targets.

## 6. Dataset contract

The independent state group is the unit of splitting, sampling and bootstrap. Required fields are:

- identity: schema version, group/candidate/replica IDs, source seed and CRN ID;
- deployable state: five corrected 46D observations;
- application state: previous requested/executed action, absolute q_send and filter history;
- candidate: requested/executed action, q target, delta and generator;
- policy: source, nominal and continuation checkpoints plus hashes;
- perturbation/terrain/domain-randomization metadata;
- simulator fingerprint: backend, version, MJCF hash, timestep, solver and gains;
- per-replica outcomes: H32 fall, first-fall step, max tilt, min height and recovery;
- privileged diagnostic state in a physically separate feature view;
- selection probability/risk bin for active-mining correction.

No row-level random split is permitted. No group may share an episode, state hash, perturbation CRN or source trajectory across train and evaluation splits.

## 7. Phase 1 evidence gates

### Data gate

- at least 1,000 independent groups from at least three source seeds;
- at least eight candidates/group;
- at least 25% mixed-outcome groups;
- zero duplicate state fingerprints across splits.

### Independent-replica label gate (Iteration 2 and later)

This gate runs before normalization, model fitting, or creation of any
held-out-consumption marker. Replica roles are assigned before candidate
outcomes and stored in the hashed collection manifest.

- discovery and audit replica sets are nonempty, disjoint, exhaustive, and
  fixed on the replica axis;
- discovery outcomes select all tied empirical minima; audit risk is averaged
  uniformly across those ties, so candidate column order cannot change the
  result;
- the primary effect is one-way discovery-to-audit nominal risk minus selected
  risk; reverse/symmetric cross-fit and same-replica Oracle values are
  diagnostics only;
- confidence intervals resample complete `trajectory_id` clusters and never
  candidates or replicas;
- a future claim-bearing training protocol requires at least 3 pp raw audit
  reduction with 95% CI low above zero, pair-order agreement at least 0.55,
  and its CI low above 0.50;
- the development recovery-option triage uses the stricter 5 pp causal-headroom
  decision specified in its separate locked protocol.

A failed label gate prohibits model training. More epochs, a different
network, or a favorable full-replica empirical Oracle cannot override it.

### Model gate

- pair accuracy ≥0.60 and group-bootstrap CI low ≥0.55;
- strong-pair accuracy ≥0.62;
- top-1 absolute fall reduction ≥3 pp with CI low >0;
- oracle-gap capture ≥25%;
- ECE ≤0.08 on the naturally weighted target calibration set.

AUROC and Brier are reported but cannot pass the action model by themselves.
The model gate is not evaluated unless the independent-replica label gate has
already passed on the applicable fresh protocol.

### Paired closed-loop gate

- at least 1,000 independent exact-state pairs;
- repeated shield reduces H32 falls by ≥3 pp with group-bootstrap CI low >0;
- improved pairs outnumber worsened pairs;
- candidate and disturbance RNG are common across matched branches.

### Online training gate

Across ten preregistered fresh confirmation seeds and a fixed 500k policy steps/seed:

- ≥20% relative reduction in cumulative falls;
- positive lower CI for fall reduction;
- mean return at least 95% of SAC;
- forward-velocity error worsens by no more than 0.03 m/s;
- runtime deadline miss rate below 0.1%;
- replacement/abstention and fall timing are reported by training quartile.
- an intervention-rate/action-distance-matched random placebo does not reproduce the reduction.

The primary outcome is falls per fixed policy-step budget, not falls per completed episode, because a shield can change episode length.

## 8. Statistical protocol

- Training seeds, model seeds and environment/perturbation seeds are different namespaces.
- Exact-state comparisons use paired differences and group bootstrap.
- Online experiments use the seed as the clustering unit; episode rows are not treated as independent replicates.
- Ten confirmation seed pairs are the minimum; a mechanics pilot cannot be promoted to evidence and optional stopping is forbidden.
- Pair accuracy is group-macro; a strong pair has empirical risk gap at least 0.25; ECE uses ten equal-mass bins with natural weights or recorded-acceptance IPW.
- The online primary rate ratio pools fall counts at fixed exposure but inference remains at the paired training-seed level: seed-cluster bootstrap CI plus an exact paired label-swap test (1,024 assignments for ten pairs).
- Report raw counts, absolute pp change, relative change, 95% CI and the number needed to shield where meaningful.
- Hyperparameters are chosen on development/calibration data. Confirmation data are evaluated once.
- Failed experiments remain in the manifest; no favorable-seed filtering.

## 9. Engineering guardrails

- Do not modify real-robot observation semantics for a simulator policy.
- Do not place JAX/MjLab in the real-time runtime; communicate via immutable datasets.
- Do not start a 100k-group generation run before the 1k/3k learning slope justifies it.
- Do not enable a new shield in a claim-bearing experiment before paired closed-loop evidence passes.
- Do not call a learned empirical filter a formal safety guarantee.
- Preserve requested/executed/q_send action distinctions end to end.
- Keep at least three milestone commits; experiment artifacts remain outside Git except small manifests/reports.

## 10. Immediate execution queue

1. Commit the report, roadmap and protocol. **Done: `981acf3`.**
2. Repair corrected-observation/q_send snapshot parity, application-state cloning and broken imports. **Done: `020a8af`; real-MJCF restore is bit-exact.**
3. Implement protected-path validation, group schema and deterministic metrics. **Done: `020a8af`; legacy row-flat files are diagnostic-only.**
4. Add synthetic tests and audit the available development NPZ formats without opening protected paths. **Done; train/calibration/test identity leakage fails closed.**
5. Implement the native group exporter and collect a small smoke dataset. **Implemented; a real 500k-SAC/MJCF K16×R8×H32 smoke is bitwise reproducible, and 56 pre-outcome-selected development groups locked the `native_poc_v1` boundary profile (nominal risk 0.410, mixed 0.719, oracle reduction 0.082 on the selected 32-group profile). This is mechanism tuning, not fall-reduction evidence.**
6. Add the factorized model trainer and privileged/deployable learning-curve CLI. **Implemented; the production action view is the ordered 36D requested/executed/q-target tuple, with requested-only retained only as an explicit ablation.**
7. Implement the paired repeated-closed-loop evaluator and evidence compiler. **Online three-arm seed statistics and the development compiler are implemented; the exact-state repeated-shield runner remains next.**
8. Create commit 3, then run the Objective 1 experiment ladder. **Done: `0cddc1d`; the reviewed native collector is the commit-4 boundary before the 1k PoC. Phase 2 remains blocked until every Objective 1 gate passes.**
9. Execute the four locked Iteration-1 model runs exactly once. **Done; the primary and all action-ranking diagnostics failed. The consumed outcomes are preserved in `QSAFE_PHASE1_ITERATION1_RESULT.md`.**
10. Audit candidate labels with independent replica halves and trajectory OOB prediction. **Done; the apparent 6.9 pp Oracle is winner's-curse bias and cross-fit action ordering is chance.**
11. Preregister and implement `objective1_recovery_option_triage_v2`. **Done: commit `a30f3ed`.**
12. Collect exactly 384 fresh groups on source seeds 7601--7603 with K29 and R64, then apply the locked triage once. **Done: the one-shot audit found less than 0.3 pp effect for every duration and fired the no-headroom stop; see `QSAFE_PHASE1_RECOVERY_TRIAGE_RESULT.md` and commit `b177ede`.**
13. Preregister v3 before observing any replacement-library outcome. **Done: early SAC policy ages, a closed-loop K9 library, admission sampling, physical discovery/audit isolation, and one-shot gates are fixed in `QSAFE_PHASE1_CLOSED_LOOP_RECOVERY_V3_PLAN.md` and `config/qsafe_closed_loop_recovery_triage_v3.yaml`.**
14. Implement and test v3, then collect the six fresh source-seed shards exactly once. **Done at commit `7681886`; all six G64 shards completed under the clean one-shot collector.**
15. Apply the v3 discovery lock and consume its audit once. **Done; the selected fixed global backup failed all primary gates and the locked decision is `report_inconclusive_no_model_training`. The hierarchy-blocked per-state diagnostic is exploratory only; see `QSAFE_PHASE1_CLOSED_LOOP_RECOVERY_V3_RESULT.md` and commit `607dee7`.**
16. Preregister v4 as a fresh direct confirmation of state-dependent K9 headroom before fitting any model. **Terminated tooling-invalid without a scientific decision; all six seeds are consumed. Collection and admission completed, but a producer/consumer RNG-manifest mismatch stopped the first discovery merge before aggregation, selection, or audit access. See `QSAFE_STATE_DEPENDENT_RECOVERY_V4_TECHNICAL_FAILURE.md` and commit `439c525`.**
17. Preregister V5 as a fresh, outcome-isolated successor after repairing the shared RNG-manifest contract. **Done at `1452a11`; V5 uses a new protocol/root/Stage-A domain and six new seeds while preserving the scientific design. No V4 artifact entered V5.**
18. Execute the V5 Stage-A same-state causal-headroom gate. **Passed once on the frozen audit: absolute reduction 0.3447 with one-sided 95% LCB 0.3135; pair agreement 0.7920 with LCB 0.7817; all three age and all six source-seed effects are positive. See `QSAFE_PHASE1_STATE_DEPENDENT_RECOVERY_V5_STAGE_A_RESULT.md`. This authorizes Stage B only.**
19. Implement and execute the frozen V5 Stage-B model/calibration/selector/placebo pipeline. **Now active. Stage C, online SAC, Objective 1, and Phase 2 remain blocked unless every Stage-B gate passes.**
