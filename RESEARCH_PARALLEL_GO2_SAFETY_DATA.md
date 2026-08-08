# Parallel Go2 safety data feasibility study

**Date:** 2026-08-08 (Asia/Shanghai)  
**Repository:** `onrobot-safe-rl`, `main@59aacf8`  
**Decision:** **Decision B — `feasible_but_only_for_pretraining`**  
**Engineering Go/No-Go:** **GO for a bounded parallel-branching PoC; NO-GO for assuming PPO data alone can train the deployable filtered-policy Q_safe, and NO-GO for a 100k-group production run before the PoC gates pass.**

## Executive conclusion

Large-batch MuJoCo is technically feasible on this machine and is likely the fastest way to determine whether the present 360 independent groups are in the sample-starved regime. The strongest implementation candidate is now **Unitree's official `unitree_rl_mjlab` (Go2 + MuJoCo Warp + RSL-RL PPO)**, not a new Go2 PPO implementation. A public Go2 checkpoint also exists, so PPO training is not on the critical path of the first PoC.

The crucial qualification is statistical, not computational. The target is

\[
P(\mathrm{fall\ within\ H32}\mid s,a,\pi_{\mathrm{filtered\ SAC}}),
\]

whereas PPO supplies states from \(d^{\pi_{PPO}}(s)\). Raw PPO transitions do not identify same-state action ordering under the filtered-SAC continuation. PPO rollouts are therefore recommended for broad representation/dynamics pretraining and as an **auxiliary state proposal distribution**. Final Q_safe fitting and validation must retain substantial frozen-SAC, filtered-SAC, and repeated closed-loop data, with labels produced by exact same-state branching and the target continuation policy.

The measured RTX 3090 MJX-JAX upper bound is already adequate: 4096 Go2 worlds produced about **303k physics steps/s**, equivalent to **30.3k 50-Hz policy-environment steps/s** or **948 H32 candidate rollouts/s** before policy, screening, replicas, and export overhead. The PoC should primarily answer identifiability and transfer, not whether the simulator can generate enough rollouts.

## 1. What the repository actually implements

### 1.1 Agents and runtime

- The repository contains DroQ, SafeDroQ, paper-SQRL, FlashSAC and LiveSAC implementations. The current safety path is `rl/agents/safe_droq`; the paper reproduction is separate under `rl/agents/paper_sqrl`.
- The live policy/runtime interface is fixed-rate. The formal safety overlays use **50 Hz policy control**; the low-level controller runs at **500 Hz**.
- `runtime/inference/observations.py` constructs the corrected 46D deployable observation as:
  - 12 joint positions;
  - 12 joint velocities;
  - 3 IMU angular velocities;
  - 3 body-frame linear velocities;
  - 4 normalized quaternion values;
  - 12 previous **absolute q targets actually sent**.
- Normalized actions are mapped to `init_qpos + action * action_offset`, joint-clipped, optionally slew-limited and Butterworth-filtered, and finally converted back to the executed normalized action. Distinguishing `action_requested`, `action_executed`, and `action_q_target` is mandatory in generated data.
- The 50-Hz configuration uses joint order `FR, FL, RR, RL × hip, thigh, calf`, `init_qpos=[0.05,0.7,-1.4]×4`, and action offsets `[0.2,0.4,0.4]×4`.

### 1.2 Existing exact-state branching

`train/mujoco_snapshot_env.py` already demonstrates the relevant primitive with native MuJoCo:

- lossless `mjSTATE_INTEGRATION` capture/restore;
- identical-state action branching;
- base velocity impulses;
- H32 fall/near-fall measurement;
- 50-Hz action with multiple 2-ms physics substeps.

**Post-report implementation audit (2026-08-09):** the integration-state primitive is present, but the current corrected-observation/application-state path is not yet evidence-safe. In the `oss` environment with MuJoCo 3.11.0, capture→step→restore produced about 0.0196 maximum observation error in reconstructed body velocity. The legacy P17 path also supplies normalized previous action where the corrected 46D contract requires absolute `q_send`, and it does not clone action-filter history. Therefore “lossless” above applies only to MuJoCo's selected integration-state fields; legacy P17 datasets remain diagnostic-only until observation, q_send, filter-state and PD-gain parity tests pass.

`scripts/collect_p17_counterfactual_dataset.py` already implements nominal-risk-first screening, policy-local candidates, exact-state restore, H32 rollout, observation history and grouped NPZ export. This is the semantic reference for a parallel generator, but its continuation and application state must be made explicit (Section 6).

### 1.3 Current evidence and bottleneck

The supplied formal results establish that the action space is not empty of recoveries:

- frozen SAC H32 failure: 49.53%;
- repeated closed-loop Oracle: 36.72%;
- reduction: 12.81 pp, 95% CI [9.22, 16.56] pp;
- matched-random repeated improvement: about 0.47 pp.

Yet the policy-consistent filtered-policy Q_safe used only 360 independent training groups and failed to generalize: pair accuracy 0.624, strong-pair 0.567, AUROC 0.507, and top-1 reduction -0.28 pp. The repository's older P17 work independently shows that pointwise state risk can be learned while cross-seed same-state candidate ordering remains near chance. Thus **more transitions are not the intervention**; more independent, mixed-outcome, same-state groups with stable continuation labels are.

## 2. Current parallel MuJoCo choices (verified 2026-08-08)

| Candidate | Verified revision / recency | Framework and PPO | Parallelism / Go2 | License | Recommendation |
|---|---|---|---|---|---|
| [MuJoCo MJX](https://mujoco.readthedocs.io/en/latest/mjx.html) | local MuJoCo 3.8.1; official docs current | MJX-JAX or MJX-Warp; Brax JAX PPO can train MJX envs | Batched `Model`/`Data`, GPU/TPU; Menagerie has `unitree_go2/scene_mjx.xml` | Apache-2.0 | Physics/branching substrate; prefer Warp for contact-heavy production after parity check |
| [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) | `d5e6b475`, 2026-08-06 | MJX-JAX and MJX-Warp; JAX PPO, plus RSL-RL route | Thousands of envs; locomotion examples, but **no Go2 task at inspected HEAD** | Apache-2.0 | Reuse training, randomization and reset patterns; do not claim direct Go2 support |
| [Unitree RL MjLab](https://github.com/unitreerobotics/unitree_rl_mjlab) | `1425b15f`, 2026-04-13 | `mjlab` over MuJoCo Warp; RSL-RL PPO | Explicit Go2 flat/rough tasks, batched Torch tensors, pushes, friction and terrain randomization | Apache-2.0 | **Best direct PoC base** |
| [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go2) | `c1a4eeb8`, 2026-08-04 | Model collection, not PPO | Official-quality Go2 `go2.xml` and MJX-specialized `go2_mjx.xml` | Go2 asset-specific BSD-3-Clause; repository Apache-2.0 | Canonical neutral model for parity tests |
| [Unitree MuJoCo](https://github.com/unitreerobotics/unitree_mujoco) | active; local copy available at `/home/xyz/code/unitree_mujoco` | Native C++/Python MuJoCo, no trainer | Go2, SDK2-compatible low-level runtime; no GPU vectorization | BSD-3-Clause | Preserve as deployment-semantics oracle |
| [Brax training](https://github.com/google/brax) | active training library; README says only training is actively maintained | JAX PPO/SAC/ARS/ES; MJX examples | Accelerator parallelism; not a maintained Go2 environment | Apache-2.0 | Use its PPO only if MjLab checkpoint cannot be reused |
| [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) | release v3.9.0.1 dated 2026-05-27 | Warp engine; integrates with Playground, MjLab/Newton | NVIDIA GPU, massively batched; closer feature coverage than old MJX-JAX | Apache-2.0 | Preferred production backend after exact parity test |
| [Newton](https://github.com/newton-physics/newton) | v1.2 current in 2026 | Warp ecosystem with `SolverMuJoCo` | GPU batching and MJCF/USD import | Apache-2.0 | Useful upstream infrastructure, but adds an unnecessary abstraction for PoC |

Important corrections to common assumptions:

1. MJX is not merely “MuJoCo on GPU”: MJX-JAX uses a reimplementation and has documented sharp edges in contact broadphase and large meshes. MJX-Warp improves these paths but is still a distinct backend whose fall/contact parity must be tested.
2. MuJoCo Playground currently has Go1/G1-style locomotion examples, not an inspected Go2 environment. The Go2 model existing in Menagerie does not create a reward/action/observation task by itself.
3. `unitree_rl_mjlab` is MuJoCo Warp/Torch/RSL-RL, not MJX-JAX. It nevertheless solves the practical “Go2 + PPO + thousands of worlds” requirement with fewer adapters.
4. Genesis can import MJCF-like assets and is fast, but changes engine/contact semantics. It should not be used for labels intended to predict the current MuJoCo runtime. The same caution applies to replacing MuJoCo with a generic Warp/Genesis dynamics stack rather than MuJoCo Warp.

## 3. Existing Go2 PPO implementations

### 3.1 Recommended: Unitree RL MjLab

The inspected task contains:

- RSL-RL PPO with actor/critic MLP `(512,256,128)`, 24 rollout steps/env, clipped PPO, GAE, adaptive learning rate;
- joint-position action around the default pose;
- actor observation: base angular velocity, projected gravity, commanded velocity, gait phase, relative joint positions/velocities, last action, plus height scan on rough terrain;
- asymmetric critic additions: base linear velocity, foot heights, air time, contacts and contact forces;
- 5–6 s interval pushes in linear and angular velocity, startup friction randomization, terrain curriculum and illegal-contact termination;
- a 20-ms deployment step (50 Hz).

A real public checkpoint is available at [diasAiMaster/unitree-go2-velocity-flat](https://huggingface.co/diasAiMaster/unitree-go2-velocity-flat). Its model card reports RSL-RL PPO, 8192 environments, 628k–738k steps/s on 10 RTX A4000 GPUs, and provides PyTorch and ONNX checkpoints plus exact deploy/env/agent YAML. These are self-reported results and should be locally reproduced before being used as capacity evidence. The checkpoint is nevertheless the fastest state-generator candidate.

### 3.2 Native CPU alternatives

- Unitree's `unitree_mujoco` is a strong simulator/deployment bridge but has no PPO trainer.
- Small Stable-Baselines3 Go2 repositories exist, but many use torque action, privileged height, different joint model or non-vectorized CPU simulation. For example, [cagataydev/sac-unitree-go2-mujoco](https://huggingface.co/cagataydev/sac-unitree-go2-mujoco) uses a 37D observation and torque control, so it is not action-compatible.
- The official [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) trains in Isaac Gym and only uses MuJoCo for sim-to-sim deployment. It is useful as a policy source, not as the requested MuJoCo data generator.

No other inspected project offers a better combination than Unitree RL MjLab of official maintenance, Go2, MuJoCo physics family, parallel PPO, action-position semantics and reusable deployment configuration.

## 4. Observation/action mapping

| Current `onrobot-safe-rl` | Unitree MjLab Go2 PPO | Match? | Conversion cost / consequence |
|---|---|---:|---|
| 12 absolute joint q | 12 relative joint q | Transformable | Add/subtract each environment's randomized/default pose |
| 12 joint dq | 12 relative joint dq | Mostly | Check frame, units and joint ordering |
| IMU gyro (3) | base angular velocity (3) | Mostly | Verify body frame and quaternion convention |
| body linear velocity (3) | actor omits it; privileged critic includes it | No | Current corrected observation requires computing it from MJX state |
| normalized quaternion (4) | projected gravity (3) | No | Compute current quaternion feature; PPO policy cannot be plugged in unchanged |
| previous absolute q target (12) | last normalized PPO action (12) | No | Must record post-projection `q_send`, including slew/filter state |
| no velocity command in 46D | command (3) | No | Fix command for state generation or keep as generator metadata only |
| no gait phase | sine/cosine phase (2) | No | PPO phase is privileged relative to final Q_safe; never leak it into deployable input |
| normalized action about `[0.05,0.7,-1.4]` with per-joint `[0.2,0.4,0.4]` | position action about `[-0.1,0.9,-1.8]` with scale `0.25` | No | Convert both through absolute q target; candidate generation must be in current action space |
| `Kp≈60, Kd≈5` in snapshot evaluator | deployment `[20,20,40]`, `[1,1,2]` | No | Dynamics-changing mismatch; use current gains for final labels or establish parity empirically |
| policy 50 Hz / low level 500 Hz | policy 50 Hz; backend substep configured by MjLab | Partial | Match exactly 10×2-ms substeps for target labels |
| joint order FR,FL,RR,RL | deploy config remaps indices `[3,4,5,0,1,2,9,10,11,6,7,8]` | No | Explicit immutable permutation and round-trip test required |

Conclusion: use the PPO policy to move and perturb the robot, then reconstruct the exact current 46D observation and generate **current-space candidate q targets**. Do not train final Q_safe on the PPO actor's observation/action tuple.

## 5. Measured hardware and throughput

### 5.1 Machine

| Component | Detected value |
|---|---|
| CPU | Intel Core Ultra 7 265K, 20 online logical CPUs |
| RAM | 62 GiB usable, no swap |
| GPU | NVIDIA RTX 3090, 24,576 MiB |
| Driver / reported CUDA | 580.173.02 / CUDA 13.0 |
| Python | 3.10.12 |
| MuJoCo | 3.8.1 installed |
| JAX / jaxlib | 0.6.2 / 0.6.2, CUDA device visible |
| PyTorch | 2.12.0+cu130, CUDA operational |
| Initially absent | `mujoco-mjx`, MuJoCo Warp, Brax, Playground, Gymnasium |

For the benchmark only, `mujoco-mjx==3.8.1` was installed under `/tmp/go2-mjx-pkgs`; no project dependency or source file was changed.

### 5.2 MJX-JAX GPU benchmark

Command basis: official `mjx-testspeed`, Menagerie `unitree_go2/scene_mjx.xml`, 2-ms physics step, Newton solver, one solver iteration, four line-search iterations, 100 steps/world. Values exclude JIT compilation and do not include PPO inference, observation construction, resets, branch labels or dataset writes.

| Worlds | JIT | Aggregate physics steps/s | 50-Hz policy-env steps/s (÷10) | H32 candidate rollouts/s (÷320) |
|---:|---:|---:|---:|---:|
| 1 | 8.29 s | 1,320 | 132 | 4.1 |
| 64 | 9.94 s | 64,898 | 6,490 | 202.8 |
| 256 | 9.61 s | 161,066 | 16,107 | 503.3 |
| 1,024 | 10.44 s | 248,051 | 24,805 | 775.2 |
| 4,096 | 11.11 s | 303,286 | 30,329 | 947.8 |

The 3090 had 480 MiB already allocated by the desktop before testing. Peak process VRAM was not reliably sampled during the short timed section, so this report deliberately does **not** invent a VRAM number. The 4096 test completed without OOM on 24 GiB; the PoC must record peak VRAM with a persistent sampler.

### 5.3 Native MuJoCo CPU benchmark

Menagerie Go2, same 2-ms physics timestep:

| Mode | Worlds | Aggregate physics steps/s | Notes |
|---|---:|---:|---|
| one `MjData` | 1 | 55,287 | long steady loop |
| serial Python loop | 64 | 36,987 | Python traversal overhead |
| serial Python loop | 256 | 21,877 | RSS 564 MiB |
| serial Python loop | 1,024 | 19,796 | RSS 1.46 GiB |
| 16 subprocesses | 64 | 358,515 | includes per-task model setup; short test optimistic |
| 16 subprocesses | 256 | 107,986 | process-local models/data |
| 16 subprocesses | 1,024 | 79,532 | memory and scheduling pressure |

Using the repository-compatible Unitree model, capture + restore + forward + 32×10 native MuJoCo steps measured **104.3 H32 branches/s** in one process; the integration state was 1,552 bytes. This excludes policy inference and observation construction.

Interpretation:

- Native CPU is excellent for semantic validation and can already generate a 1k-group pilot if parallelized across physical cores.
- `AsyncVectorEnv` uses subprocesses and pipes/shared memory, as documented by [Gymnasium](https://gymnasium.farama.org/api/vector/async_vector_env/). It is easy but adds IPC and makes exact K-way cloning awkward.
- A persistent C++ worker pool or process pool with multiple `MjData` per worker is preferable to one subprocess per environment. Expect useful scaling to roughly the physical-core count, not to 256 independent CPU cores.
- MJX/MuJoCo Warp becomes preferable when thousands of homogeneous branches must execute simultaneously. Single-world MJX is, as official documentation warns, much slower than native MuJoCo.

## 6. Same-state parallel branching design

PPO should propose states; it must not supply the final action labels. The core batched layout is:

```text
B selected states
  → repeat each state K candidates × R stochastic/perturbation replicas
  → shape [B, K, R, ...]
  → first action differs over K
  → continuation is locked to π_filtered_SAC and recomputed every 50-Hz step
  → CRN key/perturbation stream shared across K within each (B,R)
  → H32 outcomes and continuous severity are reduced over R
```

MJX data are JAX pytrees. Exact cloning is a batched `take/repeat` of every dynamic leaf, followed by `.replace(...)` of candidate controls/application state. This is cheap device memory movement compared with H32 physics. Vectorized reset is a masked tree selection between reset and stepped data. External impulses can be applied by controlled changes to free-joint velocity, or force pulses through the supported applied-force fields; the method and units must be fingerprinted.

**Physics state alone is insufficient.** A clone must include:

- `qpos`, `qvel`, actuator state, time and warm-start/integration fields required by the backend;
- current absolute `q_send` and normalized executed action;
- action filter input/output histories and slew limiter state;
- observation history (last five corrected deployable frames);
- continuation-policy hidden state, if any;
- command, terrain ID, randomization parameters and RNG key;
- pending force pulse/delay/actuator-weakening schedule.

CRN means that candidate `k=0...K-1` shares the same future disturbance/noise realization for a given replica, not that all replicas share it. Report both per-replica binary outcome and the estimated probability/severity.

### Throughput estimate for `4096`, `K=16`, `R=8`, `H=32`

There are 128 H32 rollouts per independent state group and exactly 4096 worlds can evaluate 32 groups per wave. The measured no-policy ceiling is about 948 candidate-replica evaluations/s, or **7.4 groups/s**. A practical generator should budget:

- MJX-JAX on this 3090: 300–600 evaluations/s after filtered-policy inference, observation building, reset, screening and export; **2.3–4.7 groups/s**;
- MuJoCo Warp/MjLab: plausibly higher for contact-rich batches, but no number is claimed until locally measured;
- conservative production planning: **1–3 accepted boundary groups/s**, because rejection/screening and repeated labels dominate.

At 1–3 accepted groups/s, 10k accepted groups take 0.9–2.8 h of branch compute and 100k take 9–28 h, plus state mining and validation. These are planning figures, not benchmark results.

## 7. Producing informative falls and boundaries

A converged locomotion PPO suppresses failure, so perturbation must be an active experimental design:

1. Roll out a mixture of early/mid/final PPO checkpoints and a small amount of action noise. Early checkpoints add failures but must not dominate the state distribution.
2. Randomize lateral/longitudinal velocity impulses, finite force pulses, angular impulses, friction, low-amplitude terrain, command changes, temporary actuator weakening and one-step action delays.
3. Keep perturbations separately identifiable. Do not combine every corruption at once; otherwise candidate differences become uninterpretable.
4. Estimate nominal H32 risk with 2–4 cheap replicas. Keep most states in an empirical 20–80% interval, while reserving calibration strata outside it.
5. After the K-way branch, prioritize mixed-outcome states and high action-severity spread, but retain selection probability so training/evaluation can correct sampling bias.

### GPU boundary miner

```text
rollout pool → cheap nominal replicas → risk bins
  0–5%: retain small calibration sample
  5–20%: retain difficult-normal sample
  20–80%: high-priority branching pool
  80–95%: retain recoverable/unrecoverable discriminator sample
  95–100%: retain small certain-fall calibration sample
```

Within the 20–80% pool, allocate candidate budget adaptively: start K=4, expand to K=16/32 only when outcomes or continuous severity differ. This can multiply accepted groups per GPU-hour.

## 8. What PPO data can and cannot train

### 8.1 Direct PPO transitions: no

A transition `(s,a,s')` from PPO estimates behavior under `π_PPO` and normally provides only one action at a state. It confounds state risk, behavior policy and action effect. Treating it as final Q_safe supervision would create both covariate shift \(d^{\pi_{PPO}}\neq d^{\pi_{SAC/filter}}\) and continuation-policy shift.

### 8.2 Useful roles for PPO data

- self-supervised deployable-history encoder pretraining;
- privileged-to-deployable representation distillation;
- short-horizon dynamics and contact-phase representation;
- general fall/near-fall state-risk prior;
- proposal states for same-state branches labeled under the target continuation;
- learning which perturbations reach the recovery boundary.

### 8.3 Recommended staged training

```text
large PPO/perturbation trajectories
  → representation + dynamics + state-risk pretraining
  → grouped action-ranking training on mixed policy-source branches
  → frozen-SAC / filtered-SAC / repeated-SQRL fine-tuning
  → target-distribution-only model selection and paired evaluation
```

For Level-3/4 grouped training, a defensible initial **minibatch sampling** allocation is:

- 30% PPO-proposed boundary states labeled with target continuation;
- 35% frozen-SAC boundary states;
- 25% filtered-SAC/repeated-closed-loop states;
- 10% calibrated hard-normal and nearly unrecoverable groups.

This is not a claim about the natural dataset frequency. Store all sources, then tune sampling weights on a development set whose state source is SAC/filter only. The final test must contain no PPO-proposed states. If PPO-source gradients hurt target-only validation, reduce them to representation pretraining rather than forcing a fixed mixture.

## 9. Dataset hierarchy and schema

Do not balance individual fall/non-fall rows 50/50. The statistical unit is an independent simulator state group.

| Level | Content | Main use | Priority now |
|---|---|---|---|
| L1 | normal/fall trajectory transitions | encoder/dynamics pretraining | useful, cheap |
| L2 | boundary state + nominal multi-replica outcome | calibrated state risk | useful for screening |
| L3 | boundary state + K exact-state candidates | action discrimination | **highest** |
| L4 | K candidates + repeated closed-loop target continuation | deployed SQRL target | **highest and final** |

Recommended columnar group schema (Parquet/Zarr or sharded NPZ for the PoC):

- identifiers: `dataset_version`, `group_id`, `candidate_id`, `replica_id`, `seed`, `crn_id`;
- deployable input: five corrected 46D frames, current requested/executed action, absolute current `q_send`;
- candidate: requested action, executed action, absolute q target, delta and generator/kind;
- policy: state-source policy/checkpoint, nominal policy/checkpoint, continuation policy/checkpoint, filter version and thresholds;
- perturbation: type, magnitude, duration, start/end step; terrain/material/domain-randomization values;
- physics: MuJoCo/backend/version, MJCF SHA-256, solver/integrator/timestep/iterations, gains, joint permutation;
- outcomes: per-replica H32 fall, first-fall step, max tilt, min height, contact/severity trace summaries, recovery success;
- evaluation: nominal outcome, oracle best, selected action/outcome and oracle-gap capture;
- diagnostic privileged state: full qpos/qvel, base pose/twist/COM, contacts/forces, foot state and terrain state; access-controlled away from deployable training features;
- selection: screen score, risk bin, acceptance probability and rejection reason.

The PyTorch loader should consume exported immutable shards. JAX/Torch zero-copy interoperability is unnecessary for the first PoC and would couple the online learner to the generator.

## 10. Observability upper-bound experiment

Train identical-capacity, identical-split models:

- **A deployable:** last five corrected 46D frames + current candidate action/q target;
- **B privileged:** A plus full qpos/qvel, contact flags/forces, exact world velocity, COM, base pose/twist and terrain/contact state;
- optional **C state-only privileged:** B without action, to quantify how much AUROC is merely state risk rather than action discrimination.

Use group-disjoint splits by rollout seed, checkpoint, perturbation program and episode. Evaluate on target-policy groups only.

Interpretation gates:

- privileged improves pair accuracy/oracle capture strongly while deployable saturates: observability/POMDP bottleneck;
- both improve with log data size: sample-complexity bottleneck;
- both remain near chance while oracle spread is real: noisy/incorrect continuation label, too-short action effect, or candidate construction problem;
- both have high AUROC but weak within-group ranking: state-risk shortcut, not a useful Q_safe.

## 11. Scaling experiment

Use nested, seed-fixed training subsets of **360, 1k, 3k, 10k, 30k, 100k independent groups**. Keep architecture, optimizer steps per seen group, candidate construction and target-policy validation fixed. Repeat at least five training seeds.

Report:

- all-pair and strong-pair accuracy, with strong pairs defined before examining results;
- pointwise AUROC, Brier and ECE;
- paired nominal-to-selected H32 risk reduction with group bootstrap CI;
- top-1 regret and oracle-gap capture `(nominal-selected)/(nominal-oracle)` where denominator is positive;
- source-stratified metrics and risk-bin calibration;
- train/validation gap and learning-curve slope versus log group count.

Decision rule for “360 was sample-starved”: the target-only pair accuracy and top-1 reduction must improve monotonically enough that the 1k/3k CIs exclude the 360 result, with continued positive slope at 10k. A flat deployable curve plus rising privileged curve rejects the “just add data” hypothesis. Do not unlock a 100k run merely because training loss falls.

## 12. Minimal 1–2 hour PoC

The original six-part target is too broad if “PPO can basically walk” requires fresh training. It is realistic in 1–2 hours only by reusing the public MjLab checkpoint.

### Phase 0: 15 minutes — environment and parity

1. Create an isolated `simulation/parallel_go2` worktree/module or temporary checkout; do not change the current runtime.
2. Load 1024/4096 MjLab Go2 worlds and the public checkpoint.
3. Verify joint permutation, default pose, q-target conversion and 50/500-Hz stepping against native MuJoCo for 100 unperturbed steps.

Gate P0: no unexplained base pose/joint divergence, and fall predicate/contact counts agree on a fixed corpus within a declared tolerance.

### Phase 1: 20 minutes — clone/branch test

1. Save 32 stable and 32 perturbed full application states.
2. Clone each to K=16 with identical CRN continuation.
3. Verify candidate 0 repeated twice is bitwise equal within a backend and statistically consistent across backends.

Gate P1: duplicate branches match; mixed-candidate outcomes exist; no history/filter state leakage.

### Phase 2: 30–60 minutes — 1k boundary groups

Mine PPO states, but convert candidates to the current action/q_send semantics. Use K=8 initially and R=2 for screening; spend R=8 only on retained boundary groups. Label continuations with the frozen SAC first, then filtered continuation if already available without opening sealed Formal Test material.

Gate P2: ≥1000 independent accepted groups, ≥30% mixed-outcome, useful oracle spread, and dataset/source fingerprints complete.

### Phase 3: 20 minutes — learning signal

Train the existing Q_safe architecture on nested 360 and 1000 groups, deployable and privileged. Compare target-only pair accuracy and paired top-1 reduction. This is a diagnostic development run, not a Formal Test.

PoC success requires all of:

- throughput ≥200 H32 candidate-replica evaluations/s;
- no semantic parity failure;
- 1000 groups within the time budget;
- positive 360→1000 slope on target-distribution pair accuracy;
- privileged/deployable gap is measurable and interpretable;
- model-selected risk does not worsen on the development set.

If only throughput passes, the architecture is feasible but the Q_safe hypothesis is not yet supported.

## 13. Integration architecture

```text
onrobot-safe-rl current runtime / native MuJoCo oracle
                    |
          versioned safety dataset schema
                    ^
                    |
 simulation/parallel_go2 (isolated generator)
   ├── MjLab/MuJoCo-Warp PPO state proposer
   ├── current-observation + q_send adapter
   ├── batched boundary miner / exact brancher
   └── parity and fingerprint exporter
```

Start as `simulation/parallel_go2` in this repository so schemas/tests can evolve atomically. Split into `onrobot-safe-rl-data-generator` only if dependencies or release cadence become burdensome. Do not import JAX/MjLab into the real-time runtime.

### JAX/PyTorch boundary

- **A — recommended:** JAX or Torch/Warp handles simulation, PPO and export; current PyTorch Q_safe trains offline from shards.
- **B — not for PoC:** JAX simulation calling PyTorch inference introduces synchronization and device-copy/interop complexity. Export SAC/filtered policies to ONNX or reimplement the small forward pass in the simulator framework only after parity tests.
- **C — reject:** migrating the entire project to JAX changes the deployed learner/runtime for no benefit to this question.

Since MjLab already uses PyTorch/RSL-RL with MuJoCo Warp, it may avoid JAX entirely. The architectural principle remains the same: generator and deployed learner communicate through data, not through a shared training framework.

## 14. Engineering effort

| Work item | Estimate | Main risk |
|---|---:|---|
| Isolated MjLab/checkpoint bring-up | 0.5–1 day | dependency/CUDA version pinning |
| Current observation/action/q_send adapter | 1–2 days | frame/order/filter semantics |
| Full-state clone + K×R CRN brancher | 2–4 days | application state beyond physics |
| Boundary miner and perturbation programs | 2–3 days | sampling bias / too-easy states |
| Versioned exporter + PyTorch loader | 1–2 days | schema drift and shard throughput |
| Native-vs-Warp parity suite | 2–4 days | contact/fall divergence |
| Scaling/privileged experiment harness | 2–3 days | group leakage and repeated-seed statistics |

Expected first trustworthy 10k-group development dataset: roughly **2–3 engineer-weeks**, dominated by semantic validation rather than simulation speed.

## 15. Risks and mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| PPO/SAC covariate shift | impressive auxiliary metrics, no deployed benefit | target-only validation, source weighting, final fine-tune on SAC/filter |
| backend physics shift | labels do not match current MuJoCo runtime | fixed snapshot parity corpus and native relabel subset |
| state-risk shortcut | high AUROC, random candidate ranking | group-normalized/ranking losses and within-group metrics |
| H32 cutoff noise | one-step timing changes flip binary labels | replicas plus TTF/severity targets; report binary separately |
| incomplete clone | candidates receive unequal histories/noise | clone application state and duplicate-candidate determinism tests |
| active-mining bias | miscalibrated risk | record acceptance probability; preserve natural calibration strata |
| privileged leakage | undeployable model appears strong | physically separate feature views and enforce loader allowlist |
| easy normal/certain fall dominance | wasted throughput | staged 20–80% boundary screening and adaptive K/R |

## 16. Final recommendation

Choose **Decision B: `feasible_but_only_for_pretraining`** with the following precise meaning:

1. **Parallel MuJoCo branching is strongly recommended.** It is fast enough on the current RTX 3090 and exact-state batching directly addresses the missing independent same-state groups.
2. **PPO is useful but not the label policy.** Reuse Unitree MjLab's Go2 PPO/checkpoint as a broad, perturbable state generator and for representation/dynamics pretraining.
3. **Final Q_safe requires target-policy data.** Frozen SAC, filtered SAC and repeated closed-loop states and continuations must remain a majority of target-specific fine-tuning/validation signal.
4. **Do the 1k-group PoC before system-building.** Its purpose is to measure the 360→1000 learning slope and privileged/deployable gap. It must not read the sealed Formal Test or enable masking in a formal run.
5. **Do not promise that 100k solves Q_safe.** If target-only candidate ranking is flat while privileged improves, invest in observability/predictive dynamics. If both are flat, repair labels/candidate horizon. Only if the curves rise should 10k/30k be authorized.

This is a Go for a bounded generator PoC and a No-Go for direct PPO-transition Q_safe training or immediate large-scale production.

## Sources

Primary sources were preferred and all named repositories were opened or shallow-cloned at the revisions shown above:

- [MuJoCo MJX documentation](https://mujoco.readthedocs.io/en/latest/mjx.html)
- [MuJoCo simulation/threading documentation](https://mujoco.readthedocs.io/en/latest/programming/simulation.html)
- [Google DeepMind MuJoCo](https://github.com/google-deepmind/mujoco)
- [Google DeepMind MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)
- [Google DeepMind MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)
- [Google DeepMind MuJoCo Menagerie Go2](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go2)
- [Unitree RL MjLab](https://github.com/unitreerobotics/unitree_rl_mjlab)
- [Unitree MuJoCo](https://github.com/unitreerobotics/unitree_mujoco)
- [Brax](https://github.com/google/brax)
- [Newton](https://github.com/newton-physics/newton)
- [Gymnasium AsyncVectorEnv](https://gymnasium.farama.org/api/vector/async_vector_env/)
- [Public Unitree MjLab Go2 PPO checkpoint](https://huggingface.co/diasAiMaster/unitree-go2-velocity-flat)
