# Go2 Safe RL Research Pipeline

## Invariants

- The converged SAC/DroQ reward learner remains the baseline.
- Safety learning is auxiliary until the evaluation gate passes.
- Scripted standup/recovery is a reset and emergency supervisor, not policy
  competence. Recovery transitions never enter reward replay.
- `python -m train --mode in_process --config config/go2.yaml` remains valid.

## Completed foundation

1. Safety costs, unsafe/near-failure labels, termination reasons and
   intervention masks are recorded in every policy transition.
2. Safety replay is split into recent, boundary, failure and recovery buffers.
3. An independent sigmoid Q_safe critic is trained without modifying the SAC
   actor, reward critic or executed action.
4. `safety_eval` exports rollout CSV, JSON metrics and an SVG report, and blocks
   shielding unless AUROC and pre-failure warning-rise gates pass.
5. `safety_collect` freezes the reward learner and collects reproducible
   perturbed-policy data.

## Q_safe propagation improvement (current)

The first held-out disturbed evaluation produced AUROC 0.739 but only a 0.0277
increase in mean Q_safe before failure, below the required 0.05. The following
changes target slow risk propagation without changing the baseline learner.

### 1. H-step future-failure supervision

For transitions within `H=32` steps of a hard failure, replay supplies
`future_failure_labels=1`. Q_safe adds an auxiliary binary cross-entropy loss:

```text
L_safe = L_n_step_td + beta * BCE(Q_safe(s,a), future_failure_within_H)
```

Initial `beta=0.5`. This directly teaches early warning while retaining the
Bellman objective.

### 2. Eight-step safety target

Replay incrementally records the next observation, unsafe event and bootstrap
mask over up to eight same-episode policy transitions:

```text
y_8 = I[t:t+7] + (1-I[t:t+7]) gamma_safe^n
      mask_n Q_safe_target(s_t+n, a_t+n)
```

Partial sequences at the replay frontier use their available `n <= 8` steps.
No target crosses an episode or recovery boundary.

### 3. Behavior-matched backup actions

Safety collection stores its Gaussian action-noise standard deviation. Target
actions use the same behavior distribution:

```text
a' = clip(sample(pi(.|s')) + Normal(0, behavior_noise_std), -1, 1)
```

Ordinary training and clean evaluation use zero noise. This prevents disturbed
current actions from being backed up under an unrealistically clean policy.

## Validation gate

After collecting at least 30 diverse failures across several seeds and noise
levels, evaluate on held-out seeds. Action masking remains blocked unless:

- both future-failure positive and negative samples exist;
- Q_safe AUROC is at least 0.70;
- pre-failure mean Q_safe exceeds normal mean Q_safe by at least 0.05;
- return and failure metrics are based on independent evaluation rollouts.

Use `--rollout-seed` to keep collection and evaluation noise streams disjoint.
Safety collection checkpoints after every completed episode so an interrupted
multi-episode batch retains all completed data.

## Stage 2: SQRL Route A (active)

**Decision (2026-07-25 updated): implement paper SQRL, not heuristic mask.**

The withdrawn Stage-2.5 candidate-scoring shield is replaced by Route A:

1. Constrained sampling \(\bar\pi\): sample \(K\) actions from \(\pi\), keep
   \(Q_{\mathrm{safe}}\le\varepsilon\), importance-sample by \(\log\pi\); if
   empty, \(\arg\min Q_{\mathrm{safe}}\) (`jaxrl/agents/sqrl.py`).
2. Train \(Q_{\mathrm{safe}}\) on recent on-policy FIFO (`sample_recent`).
3. Pretrain: constrained collect + SAC actor (no \(\nu\)).
4. Finetune: same sampling + actor loss
   \(\alpha\log\pi - Q_r + \nu(Q_{\mathrm{safe}}-\varepsilon)\) with dual
   ascent on \(\nu\ge 0\).

### Paper-style protocol (from-scratch + slow→fast transfer)

To strictly claim “learn safety from scratch + transfer” (Minitaur analogue in
Srinivasan et al. 2020):

- **Scene**: keep `scene_empty.xml` for both stages (do **not** use empty→stairs
  as the primary claim; that violates the paper’s coverage assumption).
- **\(T_{\mathrm{pre}}\)**: `move_speed=0.30`, `--mode sqrl_pretrain --from-scratch`
  (no SAC `12584` warm-start; π and \(Q_{\mathrm{safe}}\) from step 0).
- **\(T_{\mathrm{target}}\)**: `move_speed=0.40`, `--mode sqrl_finetune` with ν.
- Orchestrator + SAC transfer control:
  `scripts/run_sqrl_transfer_scratch.py`

```bash
# Full paper-style run (long). Prefer empty scene already loaded in sim.
PYTHONPATH=. python scripts/run_sqrl_transfer_scratch.py \
  --pre-speed 0.30 --ft-speed 0.40 \
  --pretrain-steps 12000 --finetune-steps 4000

# Smoke (short budgets)
PYTHONPATH=. python scripts/run_sqrl_transfer_scratch.py \
  --pretrain-steps 200 --finetune-steps 100 --skip-sac --skip-eval
```

CLI building blocks:

```bash
python -m train --mode sqrl_pretrain --from-scratch --move-speed 0.30 \
  --config config/go2.yaml --save-dir saved/checkpoints_sqrl_transfer_pre

python -m train --mode sqrl_finetune --move-speed 0.40 \
  --config config/go2.yaml \
  --checkpoint saved/checkpoints_sqrl_transfer_pre/training_snapshot_*.pkl \
  --save-dir saved/checkpoints_sqrl_transfer_ft
```

Early \(Q_{\mathrm{safe}}\) uses `sqrl_activation_steps` / ε anneal for stability;
document as an engineering approximation, not a change to the transfer claim.

**Q_safe collapse fix (2026-07-26):** from-scratch transfer run produced a
dead \(Q_{\mathrm{safe}}\) (all outputs ≈0, AUROC≈0.48) despite ~6% labeled
positives in \(D_{\mathrm{safe}}\). Causes: (1) uniform `sample_recent` +
early all-negative BCE; (2) `log(clip(q,1e-6))` BCE **zeros gradients** once
sigmoid saturates, so the head cannot recover. Fixes: skip updates until the
failure buffer is non-empty; `sample_recent_balanced` (~50% positives); BCE
with logits via `optax.sigmoid_binary_cross_entropy`. Offline, 500 balanced
updates revive the collapsed `16000` ft head to AUROC≈0.97. Logs now report
label-wise `pos`/`neg` Q means.

**SQRL activation gate:** constrain \(\pi\) only after
`sqrl_activation_steps` **and** batch AUROC / pos−neg gap pass
(`sqrl_min_auroc`, `sqrl_min_pos_neg_gap`). Finetune may boot-trust a restored
critic until the first measured batch.

**Held-out action noise (critical):** default `--noise-mode candidate` adds
noise to SQRL candidates *before* \(Q_{\mathrm{safe}}\) filtering so the
constraint can reject unsafe disturbed actions. `--noise-mode post` (noise
after selection) defeats SQRL and must not be used to claim safety transfer.
Default \(\varepsilon=0.20\); actor Lagrange \(\nu\) is capped at 2.0.

**v2 from-scratch transfer (12k@0.30 → 4k@0.40):** \(Q_{\mathrm{safe}}\)
healthy (AUROC≈1.0, large pos/neg gap). Fair eval (candidate noise 0.50,
\(\varepsilon=0.20\), seeds 9010/9011, 2 ep): SQRL falls=0 / return≈3440;
SAC falls=0 / return≈3006; SQRL `no_safe`≈0.017. Artifacts under
`saved/checkpoints_*_xfer_v2_*` and
`saved/safety_evaluation/sqrl_xfer_v2_*`. Keep `sqrl_train_candidate_noise_std=0`
by default (v3 train-time candidate noise degraded held-out stability).

### Legacy same-task + SAC warm-start (not paper transfer)

The earlier A/B (`scripts/run_sqrl_vs_sac.py`) warm-starts from pure SAC
`12584` and keeps the same `move_speed` / scene for “pretrain” vs “finetune”.
That validates the Route A mechanism but **must not** be described as
paper-style task transfer. Prefer `scene_empty.xml`. Default save dir:
`saved/checkpoints_sqrl/`.

```bash
# Pretrain (fresh Q_safe on top of converged SAC)
python -m train --mode sqrl_pretrain --config config/go2.yaml \
  --checkpoint saved/checkpoints_58d/training_snapshot_000000012584.pkl \
  --save-dir saved/checkpoints_sqrl

# Finetune (requires pretrain snapshot with safety_critic_state)
python -m train --mode sqrl_finetune --config config/go2.yaml \
  --checkpoint saved/checkpoints_sqrl/training_snapshot_<STEP>.pkl \
  --save-dir saved/checkpoints_sqrl
```

Config knobs: `sqrl_epsilon`, `sqrl_num_candidates`, `sqrl_lagrange_lr`,
`sqrl_qsafe_recent_only`. Legacy `--safety-mask` remains rejected.

### Historical heuristic-mask experiment log (withdrawn)

The following records the old candidate-scoring shield experiments. They are
**not** the active Route A path.

### Masking validation status

An early K=16 experiment appeared to reduce falls by 25%, but each candidate
was incorrectly given an independently sampled target disturbance before
selection. That let the selector choose a favorable noise realization and is
an oracle comparison, so the result is withdrawn.

The corrected evaluation selects the commanded action first and applies one
held-out disturbance afterward, identically to the unmasked baseline. On seed
9010 with action noise 0.50 and epsilon 0.10:

| metric | unmasked | structured K=32 |
|---|---:|---:|
| falls | 2/2 | 2/2 |
| mean episode length | 200.0 | 179.5 |
| mean return | 1425.2 | 1304.9 |
| no-safe-candidate rate | 0% | 17.8% |

Structured candidates and temporal scoring are implemented, but this fair
experiment does not improve safety. The likely next requirement is training
Q_safe on the masked-policy distribution and reducing its false
no-safe-candidate rate before claiming a shielding benefit.

`safety_collect --safety-mask` closes this distribution loop while keeping the
reward learner frozen. It selects the command with the structured shield,
applies one target disturbance afterward, stores resulting transitions in
safety replay, and updates Q_safe online. Completed successful trajectories
provide hard negatives; masked failures retain H-step positive backfill.

The first loop-closure trial added two masked failures and one 400-step masked
success trajectory. On held-out seed 9010, however, Q_safe AUROC fell to 0.537
and masking retained 2/2 falls. This small update set is insufficient and
temporarily worsens calibration; checkpoint 15,327 remains the reference
critic. Further loop closure should be performed as a larger, balanced dataset
(not a few sequential online episodes) before another shielding claim.

---

## 项目交接记录（2026-07-25）

本节用于新对话直接接手项目。开始工作前应先阅读本节和上面的算法说明。

### 仓库与 Git 状态

- 本地仓库：`/home/xyz/Desktop/xluo/onrobot-safe-rl`
- GitHub：`CosmosMount/onrobot-safe-rl`
- 当前分支：`codex/safe-rl-pipeline`
- 分支起点：`cb3e573 feat : better training`
- 已推送提交：
  `aa3678d feat: add staged safe RL pipeline`
- 远端分支：`origin/codex/safe-rl-pipeline`
- 当前仅有一个与项目功能无关的未提交文件：
  `.cursor/debug-d88d84.log`。不要提交或覆盖它。
- 最近一次完整测试：`43 passed`

### 环境与运行条件

- Python/JAX 环境：

  ```bash
  micromamba activate oss
  ```

- MuJoCo simulator 和 C++ controller 由用户在外部启动。历史运行进程为：
  `unitree_mujoco` 和 `go2_control`。
- 普通训练入口必须继续保持可用：

  ```bash
  python -m train --mode in_process --config config/go2.yaml
  ```

- 不要重写环境或大改 controller 主流程。继续复用 Transition、costs、
  termination reason、standup/recovery 和现有 replay/update 结构。

### 已完成的阶段与历史修改

#### Step 0：建立开发分支

- 从 `feat : better training` 建立 `codex/safe-rl-pipeline`。
- 后续所有 safe RL 工作均位于该分支。

#### Step 1：确认 SAC/DroQ baseline

- 在 Go2 MuJoCo + C++ controller 上连续运行约 15 分钟。
- 12,584 steps 后保持稳定行走。
- episode return 约 3900。
- rolling x velocity 约 0.62–0.66 m/s。
- upright ratio 约 1.0。
- baseline checkpoint：
  `saved/checkpoints_58d/training_snapshot_000000012584.pkl`
- 该 checkpoint 只包含早期 SAC 状态，不含后续训练的 Q_safe。

#### Step 2：Safety logging 与标签

- `RobotState` 增加 `joint_tau`，DDS 从 `tau_est` 读取，缺失时置零。
- `Transition` 增加：
  - `unsafe_label`
  - `near_failure_label`
  - `safety_replay_dict()`
- legacy `replay_dict()` 六字段保持不变。
- 每步记录九类 cost：
  - tilt
  - joint limit
  - joint velocity
  - torque
  - power
  - impact
  - slip（无传感时保留为 0）
  - intervention
  - communication
- unsafe reason覆盖 excessive tilt、joint limit、motor fault、
  recovery failed、belly-up和 hard fall。
- near-failure 使用 tilt、up cosine、angular velocity、base height、
  joint margin和 recovery状态。
- 增加了安全指标日志和兼容 CLI `--config`。

#### Step 3：Multi-buffer safety replay

- 新增：
  - `D_safe_recent`
  - `D_safe_boundary`
  - `D_safe_failure`
  - `D_recovery`
  - safety all mirror
- 混合采样比例为 `40/30/20/10`。
- failure 前 H 步回填到 failure buffer。
- recovery transition单独保存，不进入普通 SAC replay。
- safety replay支持 checkpoint保存与恢复。

#### Step 4：旁路 Q_safe

- 新增独立 sigmoid safety critic和 target network。
- Q_safe更新不会修改：
  - SAC actor
  - reward critic
  - temperature
  - controller
  - executed action
- 初始单步 Bellman target后又扩展为：
  - future-failure BCE auxiliary loss
  - 8-step safety target
  - behavior-noise-matched backup action
- 当前主要配置：

  ```yaml
  safety_critic_n_step: 8
  safety_future_loss_weight: 0.5
  safety_discount: 0.99
  safety_critic_tau: 0.005
  ```

- 日志包括 TD loss、future BCE、Q_safe分类均值、AUROC、AP、ECE、
  n-step mean和 backup noise。

#### Step 5：Non-invasive evaluation

- 新增 `--mode safety_eval`。
- 导出：
  - `saved/safety_evaluation/safety_rollout.csv`
  - `saved/safety_evaluation/safety_evaluation.json`
  - `saved/safety_evaluation/safety_evaluation.svg`
- gate条件：
  - 同时有 future-failure正负样本
  - AUROC ≥ 0.70
  - pre-failure mean Q_safe − normal mean Q_safe ≥ 0.05
- 新增 `--mode safety_collect`，冻结 SAC，只收集扰动数据并训练 Q_safe。
- 支持：
  - `--action-noise-std`
  - `--rollout-seed`
  - 每 episode checkpoint
- 多 seed、多扰动数据后，reference Q_safe在 held-out seed 9010 得到：
  - AUROC：0.704
  - normal Q_safe：0.119
  - boundary Q_safe：0.741
  - failure Q_safe：0.960
  - pre-failure Q_safe：0.333
  - warning delta：0.214
- gate首次通过的 reference checkpoint：
  `saved/checkpoints_step5/training_snapshot_000000015327.pkl`

#### Step 6：SQRL-style action masking（已撤回）

- 新增 `jaxrl/agents/action_masking.py`（现仅作历史/单测；运行时禁用）。
- **2026-07-25：主动作路径只保留 Q_safe；`--safety-mask` 会被拒绝。**
- 历史行为：wrapper默认关闭，只在显式传入 `--safety-mask` 时启用。
- 当前候选结构：
  - policy mean
  - previous action
  - contracted previous action
  - policy mean附近局部样本
  - ordinary policy samples
- 当前默认 `K=32`。
- safe candidate内部使用归一化 reward Q、risk penalty和 action-delta
  penalty联合评分。
- 无安全候选时：
  - previous action风险低于 emergency threshold则复用；
  - 否则选最低 Q_safe。
- 重要实验修正：
  target disturbance必须在 shield选完 commanded action后统一施加。
  早期“每个 candidate独立噪声”的实验属于 oracle，不公平；其
  25% fall reduction结论已撤销，不能引用。
- 公平 K32 实验没有提高安全：
  - baseline seed 9010：2/2 falls，mean length 200，mean return 1425
  - structured K32：2/2 falls，mean length 179.5，mean return 1305
  - no-safe rate：17.8%

### Masked-policy loop closure 最新状态

`safety_collect --safety-mask` 已实现，可冻结 reward learner并用 masked
policy收集数据。首轮数据：

| seed | noise | epsilon | result | steps |
|---:|---:|---:|---|---:|
| 606 | 0.50 | 0.10 | failure | 116 |
| 707 | 0.50 | 0.20 | failure | 213 |
| 808 | 0.40 | 0.15 | success | 400 |

- 这组数据生成的实验 checkpoint：
  `saved/checkpoints_step5/training_snapshot_000000016056.pkl`
- held-out seed 9010复测：
  - falls：2/2
  - mean episode length：138
  - mean return：949
  - no-safe rate：9.1%
  - Q_safe AUROC：0.537
  - warning delta：0.030
- 结论：三条顺序在线轨迹导致 calibration下降，不能替代 reference。
- `15327` 仍是 reference Q_safe；`16056` 仅是失败的 loop-closure实验。

### Checkpoint 使用说明

| checkpoint | 用途 | 状态 |
|---|---|---|
| `12584` in `checkpoints_58d` | 收敛 SAC baseline | 有效；无 Q_safe |
| `15327` in `checkpoints_step5` | 当前 reference SAC + Q_safe | 有效 |
| `16056` in `checkpoints_step5` | 小样本 masked loop closure | 实验失败，不作基线 |
| `13797` in `checkpoints_step5` | 从倒地启动造成污染 | 禁止使用 |

其余 `checkpoints_step5` 文件是中间实验状态。新实验应明确记录起始
checkpoint、seed、noise、epsilon和是否启用 mask。

### 常用命令

普通 baseline：

```bash
python -m train --mode in_process --config config/go2.yaml
```

冻结 policy 收集 safety data：

```bash
python -m train --mode safety_collect \
  --config config/go2.yaml \
  --checkpoint saved/checkpoints_step5/training_snapshot_000000015327.pkl \
  --play-episodes 1 \
  --action-noise-std 0.45 \
  --rollout-seed 1001
```

公平 non-invasive evaluation：

```bash
python -m train --mode safety_eval \
  --config config/go2.yaml \
  --checkpoint saved/checkpoints_step5/training_snapshot_000000015327.pkl \
  --play-episodes 2 \
  --action-noise-std 0.50 \
  --rollout-seed 9001
```

Masked collection：

```bash
python -m train --mode safety_collect \
  --config config/go2.yaml \
  --checkpoint saved/checkpoints_step5/training_snapshot_000000015327.pkl \
  --play-episodes 1 \
  --action-noise-std 0.45 \
  --rollout-seed 1002 \
  --safety-mask \
  --safety-mask-epsilon 0.15
```

Masked held-out evaluation：

```bash
python -m train --mode safety_eval \
  --config config/go2.yaml \
  --checkpoint saved/checkpoints_step5/training_snapshot_000000015327.pkl \
  --play-episodes 2 \
  --action-noise-std 0.50 \
  --rollout-seed 9002 \
  --safety-mask \
  --safety-mask-epsilon 0.15
```

### 下一步建议（已更新：SQRL Route A）

启发式 mask（Stage 2.5）已撤回。当前主动路径是 **SQRL Route A**：

1. **论文式主声明**：`scripts/run_sqrl_transfer_scratch.py` ——
   `--from-scratch` 慢速 pretrain → 快速 finetune（+ν），对照 SAC transfer
2. **机制验证（非迁移）**：`--mode sqrl_pretrain` 可热启动 SAC `12584`；
   见 `scripts/run_sqrl_vs_sac.py`
3. 评估默认 `scene_empty.xml`；**不要**把 `15327` 当 SQRL \(Q_{\mathrm{safe}}\)；
   **不要**用 empty→stairs 作为论文式迁移主声明
4. CSAC-LB 仍为后续备选，不与 Route A 混用

历史启发式 mask 结论（仅存档）：empty-scene 上 ε+hold 未降 falls 且伤 return。

### Stage 2.5 计划：压低 false no-safe / fallback（已归档；shield 撤回）

目标指标（reference `15327`，公平 disturbance：先选动作再加噪声）：

- `no_safe_candidate_rate` 先压到 < 5%
- 显著降低 `fallback_min_risk_rate`
- return / episode length 相对 unmasked 不明显 collapse
- 再观察 falls 是否下降

执行顺序：

1. **ε 扫描**：在 held-out seed 9010、noise 0.50、2 episodes 上扫
   `0.10 / 0.15 / 0.20 / 0.25 / 0.30 / 0.40`，记录 no-safe、fallback、
   return、length、falls、AUROC。脚本：
   `scripts/sweep_safety_mask_epsilon.py`
2. **改 fallback**：无 safe 时不再默认执行 `argmin(Q_safe)`（仍可能高风险）。
   优先 previous → contracted previous → policy mean；若都高于
   `fallback_emergency_risk`，则 **hold previous**，而不是跳到陌生高风险动作。
   默认 `allow_min_risk_fallback=False`。
3. **对比**：同一 seed/noise 下对比旧 fallback vs 新 fallback（可用最佳 ε）。
4. 只有 no-safe/return 改善后，再补 masked success 或做早停重训。
5. CSAC barrier 仍排在屏蔽稳定有效之后。

#### Stage 2.5 当前进度（2026-07-25）

已完成：

- fallback 改为 conservative hold（`jaxrl/agents/action_masking.py`）
- ε 扫描脚本：`scripts/sweep_safety_mask_epsilon.py`
- 扫描结果：
  `saved/safety_evaluation/epsilon_sweep_ref15327_holdfallback/`

相对旧 A/B（ε=0.10、旧 min-risk fallback：no-safe≈19%、
fallback_min_risk≈15%）：

| setting | falls | mean len | return | no-safe | min-risk | hold |
|---|---:|---:|---:|---:|---:|---:|
| unmasked (this sweep) | 2 | 137 | 997 | 0 | 0 | 0 |
| ε=0.10 + new fallback | 2 | 97.5 | 630 | **0.041** | **0** | 0.021 |
| ε=0.15 | 2 | 63 | 183 | 0.183 | 0 | 0.151 |
| ε=0.20 | 2 | 39.5 | 126 | 0.215 | 0 | 0.203 |
| ε=0.30 | 2 | 32.5 | 95 | 0.154 | 0 | 0.123 |
| ε=0.40 | 2+ | 30.5 | 74 | 0.361 | 0 | 0.361 |

解读：

- **fallback 改动有效**：ε=0.10 时 no-safe 从 ~19% 降到 ~4%，
  `fallback_min_risk` 归零。
- **尚未获得 fall reduction**；mask 仍降低 return/length。
- 本轮 unmasked 基线偏弱（len 137 vs 历史 ~200），且扫描后期多次
  belly-up standup，**高 ε 结果不可直接横向比较**。

#### Fallback A/B（hold vs legacy min-risk）

脚本：`scripts/compare_mask_fallback.py`
结果：`saved/safety_evaluation/fallback_ab_ref15327_eps010/`
条件：reference `15327`，noise 0.50，ε=0.10，每 cell 前独立 stabilize，
seeds 9010/9011，各 2 episodes。

| cell | falls | mean len | return | no-safe | min-risk | hold |
|---|---:|---:|---:|---:|---:|---:|
| 9010 unmasked | 2 | 97.5 | 621 | 0 | 0 | 0 |
| 9010 hold | 2 | 67 | 353 | 0.231 | **0** | 0.179 |
| 9010 min-risk | 2 | 71 | 373 | 0.134 | 0.120 | 0 |
| 9011 unmasked | 2 | 71.5 | 368 | 0 | 0 | 0 |
| 9011 hold | 2 | 61 | 194 | 0.156 | **0** | 0.139 |
| 9011 min-risk | 2 | 42 | 184 | 0.190 | 0.179 | 0 |

解读：

- hold fallback **消除了 min-risk 有害跳变**，但 **没有减少 falls**，
  也没有稳定抬高 return。
- 当日 unmasked 基线明显差于历史（~200 len / ~1400 return），且 Q_safe
  gate 多次 BLOCK；评估环境可能偏“累/脏”。后续公平对比前应先确认
  unmasked 能回到历史水平（必要时重启 mujoco/controller）。
- 下一步优先：恢复干净评估基线 → 在干净基线上复测 ε∈{0.10,0.20,0.30}
  + hold fallback → 若 no-safe 仍高，再扩候选/降 Q_safe 假阳性。

#### 重启后干净复测（部分完成）

目录：`saved/safety_evaluation/clean_eps_hold_sweep/`
历史对照（seed 9010 unmasked）：len≈200，return≈1400–1500，AUROC≈0.74。

| cell | falls | mean len | return | AUROC | warn Δ | no-safe | hold |
|---|---:|---:|---:|---:|---:|---:|---:|
| 9010 unmasked | 2 | **202** | **610** | **0.600** | 0.136 | 0 | 0 |
| 9011 unmasked | 2 | 67 | 342 | 0.656 | 0.052 | 0 | 0 |
| 9010 ε=0.10 hold | 2 | 60.5 | 226 | 0.480 | -0.039 | 0.289 | 0.215 |
| 9010 ε=0.20 hold | 2 | 30.5 | 80 | 0.772 | 0.061 | 0.098 | 0.082 |
| 9010 ε=0.30 / 9011 mask | — | — | — | — | — | — | blocked |

结论：

- **未完全回到历史水平**：9010 的 episode length 已回到 ~200，但 return
  只有 ~610（历史 ~1400+），AUROC 0.60（历史 ~0.74，gate BLOCK）。
  9011 仍然偏弱。
- 在此“半干净”基线上，ε=0.10/0.20 + hold **仍 2/2 falls**，且明显伤
  return；ε=0.20 的 no-safe 较低（~10%）但 episode 很短。
- 复测中途 standup 卡死（joint_error 在 0.3–1.4 振荡，非 belly-up），
  ε=0.30 与 9011 mask 未跑完。需要再次重启 mujoco/controller 后继续。

#### Empty scene 复测（确认台阶干扰）

场景：`scene_empty.xml`（无台阶）。
目录：`saved/safety_evaluation/empty_scene_eps_hold_sweep/`

| cell | falls | mean len | return | AUROC | no-safe | hold |
|---|---:|---:|---:|---:|---:|---:|
| 9010 unmasked | 1 | **200.5** | **1635** | 1.0* | 0 | 0 |
| 9011 unmasked | 2 | 65.5 | 359 | 0.680 | 0 | 0 |
| 9010 ε=0.10 hold | 2 | 98 | 512 | 0.647 | 0.270 | 0.219 |
| 9010 ε=0.20 hold | 2 | 31 | 89 | 0.328 | 0.145 | 0.097 |
| 9010 ε=0.30 hold | 2 | 29.5 | 88 | — | 0.186 | 0.169 |
| 9011 ε=0.10 hold | 2 | 44.5 | 203 | 0.624 | 0.169 | 0.135 |
| 9011 ε=0.20 hold | 2 | 63 | 271 | 0.637 | 0.246 | 0.190 |
| 9011 ε=0.30 hold | 2 | 41.5 | 158 | 0.649 | 0.193 | 0.181 |

\*9010 unmasked 有一条 400-step success + 一条 1-step 失败，AUROC/warn 统计不可靠，但
return/length 已回到历史水平。

结论：

- **是的，带台阶的 `scene.xml` 会污染评估**；空场景下 9010 unmasked
  恢复到 return≈1635 / len≈200。
- 后续评估/采集默认应使用 `scene_empty.xml`。
- 在干净空场景上，ε=0.10/0.20/0.30 + hold **仍未降低 falls**，且全面伤
  return；no-safe 仍偏高（约 15–27%）。屏蔽有效性仍未建立。
- **启发式 Stage 2.5 已撤回**；主动 Stage 2 为 SQRL Route A
  （`sqrl_pretrain` / `sqrl_finetune`）。

### 重要注意事项

- 不要引用已撤销的早期 25% fall reduction。
- 不要使用污染 checkpoint `13797`。
- 不要把 scripted recovery成功当作 policy safety。
- hard fall触发后 episode必须 terminated；recovery transition继续与 SAC
  replay隔离。
- `safety_eval` 的 gate 只表示监测质量；SQRL 部署用 `sqrl_*` 模式。
- 不要对 `safety_eval` / `safety_collect` 传 legacy `--safety-mask`。
- SQRL 热启动用 `12584`；不要用 `15327` 的 \(Q_{\mathrm{safe}}\) 初始化 SQRL。
- 评估/采集优先 `scene_empty.xml`（stock `scene.xml` 含台阶）。
- 当前尚未实现 CSAC-LB actor barrier。
- 当前尚未迁移到 FlashSAC backbone。

## TODO：多速度 SQRL 诊断后的改进路线（2026-07-28）

诊断基线：

- 完整报告：
  `saved/safety_evaluation/qsafe_diagnostics/report.json`
- `256×256` 在独立 episode validation 上优于更深网络：
  AUROC `0.912`、ECE `0.074`、Brier `0.130`。
- `512×512×256` 虽然训练 loss 更低，但 validation ECE 升至 `0.151`，
  Brier 升至 `0.167`，因此当前问题不是网络深度不足。
- policy candidate 的 no-safe rate：
  `v0.60=80.3%`、`v0.80=85.2%`、`v1.00=100%`。
- uniform candidate 的 no-safe rate：
  `v0.60=28.3%`、`v0.80=59.0%`、`v1.00=95.3%`。
  这说明 `0.60–0.80` 主要有 actor support 问题，`1.00` 还存在标签和
  Q_safe 饱和。

按依赖顺序执行以下 TODO；前一项通过验收后再进入下一项。

### P0：标签、episode 元数据和独立验证

- [x] 在 safety replay 中持久化：
  `episode_id`、`cmd_speed`、`policy_version`、`time_to_failure_steps`。
- [x] 保持原 `future_failure_labels` 向后兼容，同时增加
  `future_failure_h8/h16/h32`。
- [x] validation 必须按完整 episode 和 command speed 切分，不允许随机
  transition 泄漏。
- [x] 自然分布 validation 固定记录 AUROC、AP、ECE、Brier、log-loss、
  positive prior。
- [x] 验收：旧 checkpoint 可迁移加载；普通 SAC replay schema 不变；
  safety replay checkpoint round-trip 和 episode split 测试通过。

### P1：SQRL 启用 gate

- [x] 不再只依赖 balanced training batch 的 AUROC/pos-neg gap。
- [x] gate 增加自然分布 ECE、Brier、normal-state no-safe rate、
  candidate Q_safe range/std。
- [x] finetune warm-start 不再无条件把 Q_safe 视为 ready。
- [x] 验收：Q_safe 全饱和、全零、无验证正负样本、normal no-safe 过高时，
  SQRL 保持 inactive 并明确记录阻塞原因。

### P2：reward policy 能力和速度 curriculum

- [x] 使用不大于 `0.05 m/s` 的速度增量。
- [x] 只有 rolling velocity、episode length、fall rate 同时达到稳定条件后
  才升速。
- [x] 高速新阶段降低 exploration scale，并在稳定后逐步恢复。
- [ ] 验收：SAC-only 在每个目标速度先形成可用动作分布；未达标时不得进入
  SQRL fall-reduction 对比。

执行：

1. `train/speed_curriculum.py` 新增 episode 级
   `PerformanceSpeedCurriculum`；默认使用 8 个当前 frontier episode，
   同时检查平均速度跟踪比例、平均 episode length 和 fall rate。
2. `train/env.py` 改为离散速度 stage；至少 50% episode 直接采样当前
   frontier，低速 episode 不参与 frontier 晋级判定。
3. 每次晋级把 stochastic policy action 相对 deterministic mean 的偏差
   缩放到 0.5；只在不摔倒且跟踪/长度达标的 frontier episode 后恢复，
   失败会重置恢复进度。
4. 记录 `curriculum/upper_speed`、三项晋级统计、
   `exploration_multiplier` 和 `promoted`。

验证：

- `tests/test_speed_curriculum.py` 覆盖最大增量、三条件门控、低速样本隔离、
  探索恢复和失败重置。
- Go2 与 simulation 两套配置均能加载新字段。
- 代码级验收已通过；上面的 SAC-only 线上验收留到 P5 实验执行。

### P3：结构化候选和 no-safe fallback

- [x] 候选集合加入 policy mean、previous action、contracted previous、
  已知稳定 checkpoint/gait proposal 和局部扰动。
- [x] `no_safe_candidate` 时不再只执行 `argmin(Q_safe)`；优先稳定动作，
  最后进入明确的 emergency supervisor/recovery。
- [x] 记录 policy/local/structured candidate 各自的 min/mean/max/std 和
  safe coverage。
- [ ] 验收：在 `0.60–0.80` 的自然 validation 状态上，normal-state
  no-safe rate 显著低于当前基线，并且不增加 fall rate。

执行：

1. active SQRL 候选由三组组成：原 policy samples；围绕 policy mean /
   previous action 的 local samples；policy mean、previous、contracted
   previous 和外部 `proposal_action` 四个 structured proposals。
2. proposal 接口默认使用 neutral action；后续可直接注入稳定 checkpoint
   或 gait generator 的动作，无需改 selector。
3. no-safe 时只在 structured proposals 内选低风险动作；全部超过
   `sqrl_fallback_emergency_risk` 时保持 contracted previous，并置
   `sqrl_emergency_supervisor=1`，不执行任意 sampled argmin。
4. 输出三组独立风险统计、safe coverage、selected group、structured
   fallback 和 emergency supervisor 指标。

验证：

- `tests/test_sqrl.py` 已验证三组指标、动作边界以及 no-safe 不再走
  min-risk sampled fallback。
- normal-state no-safe 的离线 A/B 和真实 fall rate 仍需 P5 rollout 验收。

### P4：校准、采样修正和不确定性

- [x] balanced safety batch 使用 importance weighting，或将排序训练与概率
  校准分成两个 head。
- [x] 在自然 held-out episode 上做 temperature calibration；禁止直接把
  balanced batch sigmoid 当作 failure probability。
- [x] 保持单个 critic 为 `256×256`；优先实现小型 ensemble，而不是继续
  加深单网络。
- [x] OOD/ensemble disagreement 高时交给 supervisor，不允许高置信度
  mask 任意动作。
- [ ] 验收：相对当前 `256×256`，独立 validation ECE/Brier 改善，且
  no-safe rate 与实际 failure rate 的关系可校准。

执行：

1. safety replay 按 recent-policy 自然 positive prior 为 balanced/mixed
   batch 附加 importance weights；TD 与 future-failure BCE 同时加权。
2. 每 100 step 可在自然 recent sample 上拟合一个正温度参数；SQRL
   selector 和在线 gate 都使用校准后的概率。完整、按 episode 隔离的
   calibration 仍由离线诊断/复训流程负责。
3. `SafetyCritic` 支持 `safety_critic_ensemble_size>1` 的共享 backbone
   bootstrap heads，并输出预测分歧；默认保持 1，保证旧 checkpoint 和
   baseline 不变。
4. selector 使用 `mean risk + uncertainty_penalty * disagreement`；
   高分歧会保守地推高风险，并最终进入 structured/emergency 路径。
5. checkpoint loader 会给旧 Q_safe snapshot 自动补 neutral
   `calibration_temperature=1`。

验证：

- temperature fitting、importance-weight batch、3-head disagreement、
  selector 与旧 snapshot `16056` 的迁移恢复均已测试。
- 当前完整测试集：`71 tests passed`。
- ECE/Brier 的数值验收必须使用新采集的独立 episode，在 P5 汇总。

### P5：重新运行最小实验矩阵

- [x] 先跑 `0.50` 的 SAC 能力 gate 和 3 rollout seeds；因 SQRL 在首个
  速度未通过，按 gate 停止，不扩到 `0.60 / 0.80`。
- [x] 对比 SAC、Q_safe logging、SQRL structured masking。
- [ ] 只有 SQRL 在 fall rate 上稳定优于 SAC 且 return 不 collapse，才扩到
  `1.00 m/s` 和 FlashSAC。

#### P5 实际执行结果（2026-07-28）

SAC curriculum：

- seed 42，从零训练 `0.30→0.80`，首段 15k step 用默认
  `frontier_probability=0.50`：
  - `0.30` 在 step 5141 晋级；
  - `0.35` 在 step 11213 晋级；
  - 15k 时处于 `0.40`。
- checkpoint resume 会从 safety replay 恢复最高 command stage；续训
  15k→30k 使用 `frontier_probability=0.75`：
  - `0.40` 在 step 19200 晋级；
  - `0.45` 在 step 25741 晋级；
  - `0.50` 在 step 29200 通过并晋级 `0.55`。
- 固定 `0.50` SAC-only held-out：1200 step、0 fall、平均
  `x_velocity=0.491`。因此 `0.50` 基础能力 gate 通过。
- checkpoint：
  `saved/checkpoints_multispeed_v2_seed42/sac_pre/`
  `training_snapshot_000000030000.pkl`
- 日志：
  `saved/safety_evaluation/multispeed_v2_seed42/`

Q_safe：

- 从 30k SAC checkpoint 导出 103 个完整 episode：
  67 success / 36 failure，共 30,000 transition。
- 82 episode 训练、21 episode validation；3-head、每 head
  `256×256`，训练 5,000 gradient steps。
- 独立 validation：
  AUROC `0.935`、AP `0.693`、temperature `1.664`、
  ECE `0.0048`、Brier `0.0150`、log-loss `0.0643`，
  natural positive prior `3.06%`。
- 固定 `0.50` logging-only：1200 step、0 fall、平均
  `x_velocity=0.520`、平均 Q_safe `0.0034`。
- checkpoint：
  `saved/checkpoints_multispeed_v2_seed42/qsafe_ensemble3/`
  `training_snapshot_000000035000.pkl`

隔离压力矩阵：

- 每个 cell 均重启 `scene_empty.xml` MuJoCo 和 controller；
  actor/Q_safe 冻结；固定 `0.50`；action disturbance `σ=0.20`；
  3 rollout seeds；每格 1200 policy steps；recovery 时间不计入。
- 原 structured selector：

| seed | SAC falls / vel | SQRL falls / vel | no-safe | emergency |
|---:|---:|---:|---:|---:|
| 9300 | 0 / 0.525 | 0 / 0.637 | 0.3% | 0.1% |
| 9301 | 11 / 0.437 | 27 / 0.100 | 10.2% | 2.3% |
| 9302 | 34 / 0.203 | 43 / 0.154 | 14.9% | 4.4% |
| 合计/均值 | **45 / 0.388** | **70 / 0.297** | **8.44%** | **2.28%** |

SQRL fall 比 SAC 增加 `55.6%`，速度下降 `23.5%`，未通过。

定位并修复 selector 的侵入性问题：旧实现即使 policy mean 安全，也优先
随机选择 safe policy sample；新实现依次保留 safe policy mean、previous、
contracted previous、proposal，之后才搜索随机候选，并给所有候选组施加
一致 disturbance。

修正后快速复测（9301/9302，每格 600 step）：

- SAC：12 falls，平均速度 `0.415`。
- SQRL：19 falls，平均速度 `0.336`，no-safe `7.42%`，
  emergency `2.00%`。
- 相比旧 selector，collapse 减轻，但 fall 仍增加 `58.3%`。

结论：

- 网络加深不是瓶颈；校准与 ensemble 也没有自动带来 control safety。
- 主要剩余问题是 **counterfactual action generalization**：Q_safe 在自然
  behavior action 上校准很好，但不能可靠判断从同一状态采样的 OOD action。
- P5 在 `0.50` 正确停止；当前不得宣称 fall reduction，不进入
  `0.60/0.80/1.00`、CSAC 或 FlashSAC。
- 下一轮应优先采集同状态 action perturbation / short-horizon branch
  rollout，或训练 action-local Lipschitz/contrastive ranking；之后再复测
  masking，而不是继续扩大网络。

## TODO：反事实 Q_safe 与可拒绝 SQRL（P6–P10）

P5 已证明自然 behavior-action validation 不能作为 action masking 的充分
条件。以下任务按依赖顺序执行；默认配置必须继续保持普通 SAC 行为不变，
只有显式开启相应选项时才启用新功能。

### P6：MuJoCo branch rollout 反事实数据

- [x] 从自然 SAC rollout 保存完整 MuJoCo integration state，不用近似
  `RobotState` 代替可恢复的物理状态。
- [x] 对同一状态构造 `nominal`、`nominal + delta_i`、`previous`、
  `contracted_previous` 四类候选。
- [x] 每个候选从同一 snapshot 恢复，首步执行候选，后续使用冻结 SAC
  deterministic policy，统一向前模拟 32 policy steps。
- [x] 从同一条 branch trajectory 派生 H=8/16/32 标签，避免三个 horizon
  重复模拟。
- [x] 记录 failure、near-failure、time-to-failure、base tilt/height、
  contact、相对 nominal 的 safety improvement、action distance、
  command speed 和完整 candidate family。
- [x] branch 数据使用独立 artifact/schema，不进入 ordinary SAC replay。
- [x] 验收：snapshot restore 后重复 rollout 数值一致；同一 snapshot 的
  所有候选共享完全相同初始物理状态；artifact 可 round-trip；普通
  `python -m train --mode in_process` 无行为变化。

实现与验证：

- `learner/counterfactual_dataset.py`：独立 schema、候选构造、一次最长
  rollout 派生多 horizon 标签、原子 artifact。
- `train/mujoco_branch.py`：使用 `mjSTATE_INTEGRATION` 精确
  capture/restore，复用 Go2 action→joint target、PD 参数和 safety label。
- `scripts/collect_mujoco_branches.py`：从冻结 SAC checkpoint 采集自然
  rollout 和反事实 branches。
- `tests/test_counterfactual_dataset.py`：fake backend 起点一致性、
  artifact round-trip、真实 MuJoCo restore bitwise determinism。
- smoke：30k multi-speed SAC checkpoint，5 natural steps、1 snapshot、
  5 branches 成功生成 `/tmp/branch_smoke.pkl`。

### P7：support-aware selector 与 abstention

- [x] 使用 behavior-policy log likelihood 和相对 nominal action distance
  作为 action-local support 指标。
- [x] 只有同时满足 support 与 risk threshold 的候选才允许参与 Q_safe
  排序。
- [x] unsupported candidate 只能被拒绝，不能因为低 Q_safe 被当作安全。
- [x] 无受支持的安全替代动作时 abstain：保留 nominal/previous/
  contracted action，必要时通知 supervisor，而不是 sampled
  `argmin(Q_safe)`。
- [x] 记录 support coverage、unsupported rate、abstention rate、
  selected behavior log-prob 和 selected action distance。
- [x] 验收：构造“极低 Q_safe 但明显 OOD”的候选时 selector 必须拒绝；
  默认关闭时保持旧 checkpoint 与普通 SAC 行为兼容。

实现与验证：

- selector 对全部候选重新计算 SAC behavior log-prob，并使用每维
  log-prob 与相对 policy mean 的 RMS action distance 联合定义 support。
- `sqrl_support_gate_enabled=false` 为默认值；关闭时 support mask 全为真，
  保留历史路径。
- support gate 开启且无候选通过时固定 contracted-previous abstention，
  不允许低风险 OOD candidate 进入 sampled/structured argmin。
- `tests/test_sqrl.py` 使用不可能通过的 behavior threshold 构造低风险 OOD
  场景，验证 support coverage=0、abstention=1 且执行 contracted previous；
  SQRL 11 项测试通过。

### P8：CSC/CQL-style conservative safety critic

- [x] 加入可选 conservative loss，默认权重为 0。
- [x] 注意风险 critic 的符号：数值越高越危险，因此最小化目标应使用
  `alpha * (E_D[Q_safe] - E_pi[Q_safe])`，等价于在 CSC 原式中对
  policy/OOD action 提高风险。不能直接使用
  `+alpha*(E_pi[Q]-E_D[Q])`，否则梯度方向会把 OOD 风险压低。
- [x] 记录 conservative loss、data risk、policy/OOD risk、risk gap 和
  saturation rate。
- [x] 提供 alpha sweep，至少包含 `0` 基线和多个小正值；若 normal 与
  unsafe 全部饱和则判失败。
- [x] 验收：合成 batch 上 conservative 项的梯度确实提高 policy/OOD
  action 风险；`alpha=0` 与现有 loss 数值兼容。

#### P8 实现与验证（2026-07-28）

1. `SafetyCritic` 新增默认关闭的 `conservative_weight` 与多动作采样。
   优化项采用 `alpha*(E_D[Q]-E_pi[Q])`；配置为 0 时不执行额外采样。
2. 新增 `safety_conservative_loss/raw`、data/policy risk、risk gap 和
   saturation rate 日志。
3. `python scripts/sweep_conservative_qsafe.py ...` 默认依次运行
   `alpha=0,0.01,0.03,0.1`，每组输出隔离 checkpoint 与 manifest。
4. `tests.test_safety_critic` 共 7 项通过，包括梯度符号、alpha=0 数值
   分解、训练指标有限性及 checkpoint round-trip。
5. 100-step 工程 sweep 已保存至 `saved/conservative_qsafe_sweep_v2`。
   alpha 从 `0→0.1` 时，held-out policy-minus-data risk gap 从
   `-0.0500→-0.0269`，说明 policy action 风险确实被相对推高；但
   policy saturation 仍约 `0.72–0.75`，且 validation Brier 从
   `0.00599→0.00661` 变差。当前只支持继续测试 `0.01/0.03`，不支持直接
   采用 `0.1`，更不能仅凭自然 AUROC 宣称 conservative critic 已解决问题。

### P9：独立 critic 的选择/验证解耦

- [x] 新 ensemble 使用独立 backbone/optimizer parameter tree；旧的共享
  backbone snapshot 保持可加载，但不能作为唯一的 double-estimator
  证据。
- [x] critic A 只负责从受支持候选中选动作，critic B 独立验证。
- [x] nominal 安全时原样保留；替代动作必须被 A、B 都判断为安全，并且
  相对 nominal 的风险改善超过配置 margin。
- [x] B 验证失败时 abstain，不继续利用 B 在同一批候选中搜索最小值。
- [x] 记录 A/B risk、A/B disagreement、replacement rate、
  validation-reject rate 和 improvement margin。
- [x] 验收：人为制造 A 的低估候选时 B 能阻止替换；独立 member 参数不
  共享；单 critic/旧 snapshot 路径保持兼容。

#### P9 实现与验证（2026-07-28）

1. `sqrl_double_critic_enabled` 默认关闭；启用时创建不同 seed 的完整
   `SafetyCritic B`，拥有独立 backbone、target、optimizer 和 RNG。
2. A 完成一次候选选择后，B 只验证 A 的选择。nominal 被 A/B 同时判安全
   时保持 nominal；否则只有 A/B 都满足安全阈值和 improvement margin
   才替换，验证失败直接使用 contracted-previous abstention。
3. checkpoint 新增可选 `safety_validator_state`；旧 snapshot 和默认单
   critic 路径不受影响。
4. 新增 A/B risk、disagreement、replacement、validation reject 和
   improvement 指标。SQRL+Q_safe 相关 20 项测试通过。

### P10：control-facing 评估与 SQRL gate

- [x] 保留自然 episode AUROC/AP/ECE/Brier/log-loss，但不再把它们作为
  masking 的充分条件。
- [x] 基于 P6 branch ground truth 新增：
  same-state pairwise risk-ranking accuracy、selected false-safe rate、
  selected-action ECE/Brier、top-1 safety regret、
  risk-coverage/abstention curve、相对 nominal failure reduction。
- [x] 对 K 做子采样曲线，检查 K 增大时 false-safe rate 是否上升。
- [x] control gate 必须同时满足 counterfactual ranking、false-safe、
  coverage 和实际 nominal-relative improvement；任一缺失时只允许
  Q_safe logging。
- [x] 验收：构造自然 calibration 完美、但候选排序反转的数据时 gate
  必须阻止 SQRL；报告同时输出自然指标与 control-facing 指标。

#### P10 实现与首轮结果（2026-07-28）

1. `learner/control_evaluation.py` 基于同 snapshot branches 计算 pairwise
   ranking、selected false-safe、selected ECE/Brier、top-1 regret、
   risk-coverage/abstention、nominal-relative failure reduction 与 K 曲线。
2. `scripts/evaluate_branch_qsafe.py` 同时报告 checkpoint recent replay 的
   自然 AUROC/AP/ECE/Brier/log-loss 和 branch control 指标；支持 behavior
   likelihood/action-distance support gate。
3. SQRL gate 默认要求 `sqrl_control_metrics_path`。缺少 branch 报告、指标
   非有限、ranking/false-safe/coverage/improvement 任一不达标时，只允许
   Q_safe logging，不执行 masking。
4. 合成反转排序测试通过：自然 calibration 即使完美也不能绕过 control
   gate。80 snapshots/1200 branches 的端到端报告保存为
   `saved/safety_evaluation/control_metrics_050.json`：
   - natural AUROC `0.9474`、ECE `0.0071`；
   - branch pairwise accuracy `0.5530`；
   - selected false-safe rate `0.80`、coverage `0.3125`；
   - nominal-relative failure reduction `0.0625`；
   - 因此当前 Q_safe **不允许进入控制**。
5. 采集器已改为自然 failure 后 reset 并继续，最终完成 400 policy steps、
   22 failures。仍需注意：当前 reset 后每段自然 episode 较短，导致
   H=32 的 nominal failure rate 为 1.0；该数据非常适合检验危险边界排序，
   但最终 SAC-vs-SQRL 统计还需加入更长的稳定自然片段。

### P11：Branch-supervised Q_safe 首轮实验（2026-07-28）

实现：

1. `SafetyCritic.update_counterfactual` 加入两类反事实监督：
   - 平衡采样的 branch future-failure BCE；
   - 同一 simulator snapshot 内的 pairwise logit ranking，物理后果更差
     （failure、较短 time-to-failure、near-failure）的动作风险必须更高。
2. `learner/branch_supervision.py` 构造平衡 point batch 和 same-state
   riskier/safer pairs，并支持 snapshot 或完整 natural episode 隔离切分。
3. `scripts/train_branch_qsafe.py` 从同一 Q_safe checkpoint 重启每个变体，
   交替执行 natural replay Bellman anchor 与 branch update；所有模型只在
   train episodes 上拟合 temperature，最后在未见 episodes 上比较。
4. branch collector 新增直接从 `pi(a|s)` 采样 candidate 的模式。新的
   `counterfactual_branches_050_policy_v1.pkl` 包含 80 snapshots、
   1840 branches；behavior support coverage 从 local Gaussian 数据的
   `32.5%` 提升到 `80.2%`。支持域内安全动作覆盖：
   H8 `764` 条/45 snapshots，H16 `280` 条/26 snapshots，
   H32 `83` 条/17 snapshots。

#### 训练协议

- source：`qsafe_ensemble3/training_snapshot_000000035000.pkl`
- command speed：`0.50 m/s`
- natural steps：400，natural failures：22
- 训练：300 branch updates；每步配 1 次 natural replay Q_safe update
- point/pair batch：256/256
- ranking weights：`0, 0.01, 0.05, 0.10, 0.25`
- 最终复验：H16、ranking `0.10/0.25`，按完整 episode 切分，
  split seeds `42/43/44`

#### 严格 episode-split 结果（H16，ranking=0.25）

| split seed | natural AUROC | branch AUROC | support pair accuracy | false-safe | coverage | nominal fall | selected fall | failure reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.896 | 0.968 | 0.670 | 0.000 | 0.200 | 0.933 | 0.800 | 0.133 |
| 43 | 0.913 | 0.958 | 0.689 | 0.333 | 0.214 | 0.929 | 0.857 | 0.071 |
| 44 | 0.883 | 0.990 | 0.566 | 0.000 | 0.214 | 0.857 | 0.786 | 0.071 |
| mean ± std | 0.897 ± 0.012 | 0.972 ± 0.013 | 0.642 ± 0.054 | 0.111 ± 0.157 | 0.210 ± 0.007 | 0.906 ± 0.035 | 0.814 ± 0.031 | 0.092 ± 0.029 |

对照：

- ranking `0.10`：support pair `0.595±0.054`、false-safe
  `0.333±0.272`、failure reduction `0.024±0.034`，明显差于 `0.25`。
- 原始 Q_safe natural AUROC 为 `0.947`；branch 训练后约 `0.897`，存在
  可测的自然分布遗忘，虽然 natural Brier 只从约 `0.0066` 变为
  `0.0070–0.0072`。
- H32 的严格 episode split 即使 ranking `0.25`，support pair 仅
  `0.622`、false-safe `1.0`、failure reduction `0`；32-step 预测仍不可用。

结论：

- branch BCE+ranking 确实能把未见 episode 上的 action ranking 和实际
  failure reduction 推高；“仅加深网络”不是当前最有效方向。
- H16/ranking=0.25 只在 3 个 split 中有 1 个同时通过现有 control gate；
  均值也略低于 pair `0.65`、略高于 false-safe `0.10` 门槛。因此结果有
  研究价值但尚不稳健，**仍保持 logging-only**。
- 下一步应优先用独立 critic B 验证 seed-43 的低风险 false-safe 动作，
  并采集更多独立自然 episodes；不能通过调高 epsilon 或放宽 support
  threshold 掩盖问题。

### P12：Independent critic B 验证实验（2026-07-28）

实现：

1. `evaluate_double_critic_control` 严格执行 A-select/B-validate：
   B 只验证 A 的唯一选择，不允许 B 在候选集中重新搜索；拒绝后执行固定
   `contracted_previous` abstention。
2. 指标明确分解：
   replacement rate、validation reject、abstention rate、false-safe，
   以及 replacement 与 abstention 各自贡献的 failure reduction。
3. `scripts/train_branch_validator.py` 使用不同 seed 从零创建完整 critic B，
   具备独立 backbone、target、optimizer 和 RNG。B 先 natural pretrain
   3000 steps，再做 300 steps branch adaptation（每步保留 natural anchor）。
4. 在 H16 三个严格 episode splits 上扫描
   `epsilon={0.10,0.15,0.20}` 与 `margin={0,0.02,0.05}`。

#### 同 horizon B（A=H16，B=H16）

B 的自然 AUROC 约 `0.821–0.833`，branch AUROC 约 `0.958–0.988`。
在 `epsilon=0.20` 时三 split 均值：

| 指标 | A only | A+B |
|---|---:|---:|
| false-safe | 0.111 | 0.167 |
| coverage | 0.210 | 0.186 |
| selected failure rate | 0.814 | 0.721 |
| total failure reduction | 0.092 | 0.186 |
| validated replacement contribution | — | 0.068 |
| abstention contribution | — | 0.117 |
| validation reject rate | — | 0.024 |
| abstention rate | — | 0.814 |

解释：

- selected failure 虽下降，但约 63% 的 reduction 来自固定 contracted
  fallback，不是 B 验证后的动作替换。
- B 没有拒绝 seed-43 的关键 false-safe；反而因 coverage 分母缩小，
  false-safe rate 从总体 `0.111` 变为 `0.167`。
- `margin=0/0.02/0.05` 几乎无变化，说明 A/B 对错误候选的风险差具有高度
  相关性。独立初始化并没有自动产生独立错误。

#### 长 horizon verifier（A=H16，B=H32）

| epsilon | false-safe | coverage | replacement rate | abstention rate | total reduction | replacement contribution | abstention contribution |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.000 | 0.000 | 0.000 | 1.000 | 0.210 | 0.000 | 0.210 |
| 0.15 | 0.000 | 0.022 | 0.022 | 0.978 | 0.210 | 0.022 | 0.187 |
| 0.20 | 0.333 | 0.046 | 0.046 | 0.954 | 0.186 | 0.022 | 0.163 |

结论：

- H32 B 在低 epsilon 下只是“总是否决器”；看似更安全，实际几乎完全依赖
  scripted contracted fallback，不能算 policy safety 或可靠 action filtering。
- epsilon 提高后 seed-43 false-safe 再次出现，说明长期 verifier 没有解决
  根本的 selected-action OOD/aliasing 错误。
- 当前 A+B 路径仍不得开启在线控制。下一步应建立 selected false-safe hard
  negative buffer，并让 B 使用与 A 不同的数据 bootstrap/目标或 conformal
  upper bound；继续增加相同数据、相同目标下的网络数量收益有限。

### P13：Hard-negative、data bootstrap 与 conformal upper bound（2026-07-28）

实现：

1. 从 A 在训练 episode 中实际接受的动作里挖掘
   `predicted safe && branch failure`，按 `(snapshot, candidate)` 建立
   selected false-safe hard-negative 集；branch point/pair sampler 支持
   独立的 oversampling fraction 与 loss weight。
2. critic B 不再使用 A 的全部训练 episode：先保留 20% 为 B-only
   conformal calibration episodes，再从其余 episode 随机抽取 80% 用于
   B branch fitting；外部 validation episodes 始终不参与训练、temperature
   fitting 或 conformal calibration。
3. 在 B-only calibration 集上计算 one-sided finite-sample additive
   upper bound：
   `Q_upper = clip(Q_B + quantile(y - Q_B), 0, 1)`；评估
   `alpha={0.05,0.10,0.20}`，不改变在线控制默认值。
4. 新增 hard-negative mining/sampling 与 conformal coverage 单元测试；
   全量测试 `91/91` 通过。

#### H16 三 split 结果（epsilon=0.20，margin=0）

| 方法 | false-safe | coverage | replacement | abstention | selected fall | total reduction |
|---|---:|---:|---:|---:|---:|---:|
| P12 原始独立 B | 0.167±0.236 | 0.186±0.031 | 0.068±0.054 | 0.814±0.031 | 0.721±0.117 | 0.186±0.088 |
| bootstrap B + hard negatives，raw | 0.111±0.157 | 0.186±0.031 | 0.092±0.065 | 0.814±0.031 | 0.697±0.091 | 0.210±0.059 |
| 上述方法 + conformal α=0.20 | **0.000±0.000** | 0.090±0.083 | 0.044±0.063 | 0.910±0.083 | 0.697±0.091 | 0.210±0.059 |

逐 split 的 conformal α=0.20：

| seed | offset | false-safe | coverage | replacement | abstention | reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.026 | 0.000 | 0.200 | 0.133 | 0.800 | 0.200 |
| 43 | 0.161 | 0.000 | 0.071 | 0.000 | 0.929 | 0.143 |
| 44 | 0.182 | 0.000 | 0.000 | 0.000 | 1.000 | 0.286 |

hard-negative 诊断：

- seed 42/44 的 B-fit episodes 内没有 selected false-safe；seed 43 只挖到
  2 条（A 在 7 个有覆盖状态中错误接受 2 个）。这表明主要错误发生在未见
  episode，训练内错误重放的数据量很有限。
- seed 43 同拆分消融：关闭 hard weighting 时 branch AUROC `0.959`、
  raw false-safe `0.333`、reduction `0.143`；启用后 AUROC 反而降至
  `0.935`，raw false-safe/reduction 仍为 `0.333/0.143`。因此当前
  hard-negative weighting **没有证明能改善泛化或排序**。
- hard weighting 让 seed-43 calibration residual 变大，α=0.20 offset 从
  `0.041` 增至 `0.161`，conformal gate 因而拒绝更多动作并把 false-safe
  降为 0。这个收益来自 uncertainty-aware abstention，不应解释为 Q_safe
  本身更准确。
- α=0.05 的 offset 为 `0.557–0.672`，基本退化为总是否决器；即使 α=0.20，
  平均 coverage 也只有 9%。当前 conformal 结果可作为安全降级基线，但
  不能作为可部署 action selector。

结论：

- “独立训练数据 + 风险上界”能消除当前三 split 的已观察 false-safe，但
  代价是 91% abstention，且多数 fall reduction 仍来自 contracted fallback。
- 当前仍保持 logging-only。下一优先级应扩大独立 episode 与 boundary
  coverage，并采用能区分 epistemic uncertainty 的 ensemble/conformal
  calibration；在 coverage 和 replacement contribution 明显提高之前，
  不开启实时 masking。

### P14 TODO：主动反事实采集与 episode-bootstrap ensemble

当前阻塞不是单网络容量，而是 selector 决策边界和未见 episode 的反事实
覆盖不足。以下任务必须按顺序完成；P14 全程保持 SQRL control
`logging-only`。

#### P14.1：Decision-boundary active branch collection

- [x] branch collector 支持加载 frozen critic A 和独立 critic B，不修改
  SAC actor/controller。
- [x] 每个自然状态先对 policy/local/previous/contracted candidates 做轻量
  probe；优先保存以下 exact-state snapshots：
  - A 或 B 风险接近 `epsilon_safe`；
  - A/B disagreement 超过阈值；
  - candidate 接近 behavior support 边界；
  - 当前或前一步为 near-failure；
  - selector 可能替换 nominal 或即将 abstain。
- [x] 为 stable normal snapshots 保留独立配额，防止数据再次被 failure
  states 主导。
- [x] artifact metadata 记录 selection reason、A/B candidate risk range、
  disagreement、support coverage、episode id、policy step 和 command speed。
- [x] 默认 interval collector 行为保持兼容；active mode 必须显式开启。
- [x] 采集侧验收：
  - 至少 `100–200` 个独立自然 episodes；
  - stable/boundary/disagreement/near-failure 均有非零覆盖；
  - normal snapshots 不少于 25%；

实现与 smoke test：

- 新增 `learner/active_branch_sampling.py`，将 near-failure、A/B
  disagreement、risk boundary、support boundary、selector decision 和
  stable normal 分成独立优先级/配额；默认 normal 配额 `80`、其余每类
  `40`，满配额时 normal 占 `28.6%`。
- `collect_mujoco_branches.py` 只有显式提供
  `--active-selector-checkpoint` 才启用 active mode；可选
  `--active-validator-checkpoint`。旧 interval 模式参数与行为保持不变。
- 20 natural-step MuJoCo smoke test 得到 9 snapshots/81 branches：
  near-failure `2`、disagreement `2`、risk-boundary `2`、
  selector-decision `2`、normal `1`；support-boundary 作为重叠 trigger
  出现，但因更高优先级原因占用该 snapshot，主类别计数为 0。
- artifact：
  `saved/safety_datasets/active_branch_smoke_seed42.pkl`。每个 snapshot
  已记录全部 triggers、A/B nominal/min/max risk、最大 disagreement、
  support coverage、replacement/abstention 预测及 episode/policy step。
- active sampling 与全量回归测试 `94/94` 通过。

正式采集（2026-07-29）：

1. 为避免确定性 reset/actor 产生重复 episode，collector 新增默认关闭的
   `--natural-action-noise-std`，并将自然轨迹 RNG 与 branch candidate RNG
   完全分离。新增默认关闭的 `--natural-episode-max-steps`，达到采集 horizon
   时只记录 `truncated=true` 并 reset，不产生 failure label。
2. `0.03` 噪声的 long-stable 分层集
   `active_branches_p14_seed142.pkl`：3 episodes、3000 natural steps、
   243 snapshots、3645 branches；normal `32.9%`；H16 branch failure
   positive rate `8.75%`。其中一个 episode 连续稳定 2952 步，适合补充
   stable-normal states，但 episode 数不足，不能单独用于 bootstrap。
3. `0.03` 噪声的 failure-heavy 集
   `active_branches_p14_seed143_ep100.pkl`：100 episodes、223 snapshots、
   3345 branches；91 failures、9 truncated；100 个 episode 的 action-mean
   signature 在 `1e-3` 精度下全部不同。normal 仅 `10.3%`，因此只作为
   rare-failure memory，不作为平衡主训练集。
4. 最终平衡 active 集
   `active_branches_p14_seed144_ep100_balanced.pkl`：
   - 100 episodes、2683 natural steps、200 snapshots、3000 branches；
   - 92 failures、8 truncated；100/100 action signatures 唯一；
   - 主类别：normal `50`，其余 disagreement/risk-boundary/
     support-boundary/selector-decision/near-failure 各 `30`；
   - normal 占比恰为 `25%`，来自 35 个不同 episodes；
   - H16 branch failure positive rate `42.7%`，没有全失败饱和。
5. 平衡集 H16 分层 failure rate：
   normal `4.3%`、risk-boundary `20.0%`、support-boundary `39.8%`、
   disagreement `41.6%`、selector-decision `76.4%`、near-failure
   `100%`。类别梯度符合主动采样预期，说明 selection reasons 具有实际
   物理区分度。
6. 该平衡集适合进入 episode-bootstrap 训练，但自然 episode failure rate
   高，不能作为部署表现评估集。P14.2 必须先固定 train/calibration/
   validation episode split；validation episodes 严禁进入任何 member
   fitting、temperature fitting 或 conformal calibration。

#### P14.2：Episode-bootstrap uncertainty ensemble

- [x] 训练 `3–5` 个完整独立 Q_safe，每个 member 使用不同 episode
  bootstrap，不共享 backbone、target、optimizer 或 RNG。
- [x] 每个 member 保留 natural replay anchor；branch BCE/ranking 只使用其
  bootstrap episodes。
- [x] ensemble mean 表示风险，member disagreement/variance 表示 epistemic
  uncertainty；高 disagreement 必须 abstain。
- [x] calibration episodes 与所有 member fitting episodes 隔离；在其上
  校准 ensemble/conformal upper bound。
- [x] validation episodes 与所有 member fitting、temperature calibration
  和 conformal calibration 完全隔离。
- [x] 比较单 critic、独立 A/B、ensemble mean、ensemble upper bound，不得
  只报告最优 seed。

实现与首轮结果（2026-07-29）：

1. 新增 `learner/episode_bootstrap.py`：
   - 按完整 episode 构造 fit/temperature/conformal/validation 四路互斥
     split；
   - fit episodes 有放回抽样；
   - 重复抽中的 episode 重新映射 snapshot group，防止复制品形成错误的
     cross-copy ranking pairs。
2. 新增 `scripts/train_branch_ensemble.py`。首轮训练 3 个完整独立
   `ensemble_size=1` Q_safe；每个 member 拥有独立 backbone、target、
   optimizer、model RNG、natural replay sampler RNG 和 branch bootstrap。
3. 数据中 100 个自然 episodes 只有 53 个实际被 active sampler 保存了
   snapshots，因此实际 branch split 为：
   fit `32 episodes/1860 branches`、temperature `5/285`、
   conformal `5/195`、validation `11/660`。三个 bootstrap 分别覆盖
   21、15、19 个不同 fit episodes，证明 member 数据确实不同，但有效
   episode 数仍偏少。
4. 每个 member 执行 3000 natural-anchor updates + 300 branch updates。
   最终 branch pair batch accuracy 为 `0.855/0.754/0.852`。
5. 完全独立 validation 上：

| 方法 | point AUROC | pair accuracy | false-safe | coverage | failure reduction |
|---|---:|---:|---:|---:|---:|
| 旧 critic A | 0.955 | 0.526 | 0.214 | 0.318 | 0.000 |
| 旧独立 critic B | 0.954 | 0.538 | 0.143 | 0.318 | 0.023 |
| member 0 | 0.918 | 0.525 | 0.200 | 0.114 | 0.023 |
| member 1 | 0.940 | 0.550 | 0.500 | 0.045 | 0.000 |
| member 2 | 0.905 | 0.456 | 0.000 | 0.068 | 0.000 |
| ensemble mean | 0.946 | 0.541 | **0.000** | 0.068 | 0.000 |

6. Ensemble disagreement：mean `0.086`、median `0.068`、P90 `0.183`、
   max `0.286`。使用 std gate：
   - threshold `0.10`：coverage `2.3%`、false-safe `0`；
   - threshold `0.20`：coverage `6.8%`、false-safe `0`；
   - threshold `0.02/0.05`：coverage `0`。
7. `mean + beta*std` 在 `beta=1` 时 point AUROC 提升到 `0.954`，但 coverage
   降到 `2.3%`；`beta=2` coverage 为 0。Conformal α=0.20 offset
   `0.207`，α=0.10/0.05 offset `0.392/0.427`，三者 control coverage
   全为 0。
8. 同 validation 的旧 A+B：coverage `29.5%`、false-safe `15.4%`、
   replacement `6.8%`、total reduction `2.3%`；全部 reduction 仍来自
   contracted fallback，validated replacement contribution 为 0。

结论：

- Episode bootstrap 提高了 point-level ensemble 稳定性，ensemble AUROC
  高于 2/3 members，并把 observed false-safe 降到 0；但是收益来自强烈
  保守化，coverage 只有 `6.8%`，没有产生任何 replacement failure
  reduction。
- 当前瓶颈仍是同状态 action ranking，而不是 failure classification。
  Pair accuracy `0.541` 与旧 A/B 接近，远低于 P14.3 的可控要求。
- 只有 53 个 episodes 含 branch snapshots，且 bootstrap member 仅覆盖
  15–21 个不同 fit episodes，是 ensemble disagreement 偏大和 conformal
  offset 过高的直接数据原因之一。
- P14.2 工程实现通过，但控制效果未通过。保持 logging-only，不进入在线
  masking。下一步应先让 active collector 在每个 episode 保底保存 normal
  snapshot，并增加 same-state ranking supervision 的 episode 覆盖，再做
  P14.3 五 split 验收。

#### P14.3：Control-facing 独立验收

- [ ] 至少 5 个完整 episode splits，固定 seeds 和候选集合。
- [ ] 分开报告 validated replacement 与 contracted fallback 的 reduction。
- [ ] 进入在线 non-invasive rollout 的最低门槛：
  - selected false-safe `<= 5%`；
  - coverage `>= 30%`；
  - replacement rate `>= 15%`；
  - replacement contribution 为正；
  - fallback contribution 不超过 total reduction 的 50%；
  - K 增大时 false-safe 无显著上升。
- [ ] 只有上述门槛在 5 个 splits 上稳定满足，才允许进行 frozen-policy
  SQRL masking rollout；只有 return 不 collapse 且 fall rate 下降，才进入
  CSAC barrier/FlashSAC。

最终实验顺序：

1. P6 离线 branch dataset 与 snapshot determinism；
2. P7 support-only A/B；
3. P8 `alpha` sweep；
4. P9 independent A/B critic；
5. P10 固定 `0.50 m/s`、相同 disturbance/seeds 的 SAC vs SQRL；
6. 只有 fall rate 稳定下降且 return 不 collapse，才恢复
   `0.60/0.80/1.00`、CSAC 和 FlashSAC 路线。

#### P14.4：在线 masking 诊断对照（0.50 m/s，2026-07-29）

本节是失败机理诊断，不代表 P14.3 已验收。使用相同的 30k SAC checkpoint，
分别继续训练 4000 步；两组均关闭 command curriculum。

| 方法 | fall / episodes | Q_safe active | action replacement | final x velocity | wall time |
|---|---:|---:|---:|---:|---:|
| 纯 SAC | 0 / 10 | — | — | 0.610 | 260.8 s |
| SQRL masking + actor Lagrange | 2 / 10 | 3527 / 4000 | 12 | 0.586 | 491.1 s |
| SQRL masking-only | 5 / 13 | 3800 / 4000 | 42 | 0.622 | 538.0 s |

为消除组合基线混淆，新增 `sqrl_actor_lagrange_enabled`，默认 `false`。
masking-only 时 SAC actor loss 不含 safety penalty，Q_safe 只可改变 executed
action。门控新增迟滞，避免单个 calibration batch 缺类时反复关闭；每次
replacement 记录 step、nominal/selected risk、epsilon 和 candidate group。

masking-only 的 5 次 fall 位于 step
`30020/32000/32026/32048/32070`。第 1 次发生在 gate 启用前。gate
启用后的首次 fall（32000）前 replacement 为 0，说明 Q_safe 没有提前把
nominal 判为危险。此后 42 次 replacement 全部集中在该 fall 后的 reset
不稳定区间：27 步没有任何 supported safe candidate，21 次替换动作风险
仍高于 epsilon，7 步触发 emergency；candidate group 为 policy sample
9 次、structured fallback 33 次。大量接管仍未阻止随后 3 次快速 fall。

结论：当前在线 SQRL 没有产生 fall reduction；与 SAC 的绝对 fall 差为
`+5`，因 SAC 为 0，relative fall reduction 无定义。结果同时证实两个
失败模式：

1. 危险状态到来前存在 false negative，selector 不接管；
2. 进入危险/reset 状态后 safe set 经常为空，structured fallback 不具备
   recovery 能力。

原始和派生结果：
`saved/safety_evaluation/sqrl_mask_only_050_full/comparison_with_sac.json`。
在 P14.3 多 split 达标前保持 logging-only，不将此 gate 用于正式训练。

### P15：多速度共同 Actor — SAC vs SQRL 正式协议（实现完成，待 pilot）

P14.4 的 `0.50 m/s` 表只保留为**非配对 diagnostic**：它没有执行
“每个 seed 先训练唯一 common SAC，再从同一个 actor/reward checkpoint
分出 SAC 与 SQRL”的 P15 流程，因此不得作为正式 SAC/SQRL 结论。

#### P15.1：共同 SAC pretrain

- [x] 新增 `performance_then_balanced` curriculum。
- [x] performance 阶段从 `0.30` 开始，每次只增加 `0.05 m/s`；frontier
  采样概率为 `0.75`。
- [x] 到达 `1.00` 不等于成功；`1.00` 自身必须满足最近 8 episodes、
  平均长度 `>=300`、速度比例 `>=0.75`、fall rate `<=12.5%`。
- [x] 通过 `1.00` 后进入 balanced round-robin；15 档速度每档必须有
  `>=1600` policy transitions 和 `>=4` episodes。
- [x] performance 最多 100k steps，balanced 最多额外 30k；任何门槛
  未达到都直接失败，不生成合格 common checkpoint。
- [x] checkpoint metadata 和独立
  `speed_coverage_manifest.json` 保存 speed bins、逐档
  transitions/episodes/falls、frontier history、phase 和 coverage 状态。
- [x] common SAC 只训练 actor/reward critics；Q_safe 参数保持初始化状态，
  safety replay 只负责收集标签，不参与动作选择或 SAC loss。

#### P15.2：同源 Q_safe pretrain

- [x] 新增 `scripts/collect_p15_multispeed_branches.py`。它在 15 个速度上
  分别收集至少 1600 natural steps 和 40 exact-state snapshots，再合并
  artifact；候选包含 nominal、policy samples、previous、
  contracted previous 和 nominal 局部扰动，horizon 为 `8/16/32`。
- [x] 合并时重新映射 snapshot/episode id，避免不同速度之间 id 冲突。
- [x] 新增 `scripts/train_p15_qsafe.py`：
  - actor、reward critics 和 reward replay 从 common checkpoint 恢复并冻结；
  - natural safety data 与 branch data 都按完整 episode 做
    `70% train / 15% calibration / 15% validation`；
  - 每个速度独立划分，禁止同一 episode 跨 split；
  - Q_safe batch 保留 recent/boundary/failure/all 的
    `40/30/20/10`，并在每个可用 source 内按 command speed 均衡采样；
  - 保留现有 MLP、Bellman/future-label loss 和 same-state ranking loss；
  - calibration 只使用 calibration natural episodes，validation 不参与
    fitting 或 temperature calibration。
- [x] 每个 snapshot 自动写入稳定 SHA-256 `agent_state_hash`。branch artifact、
  common checkpoint 和 Q_safe checkpoint 的 hash 必须完全一致；不一致时
  collector/trainer/composer 都拒绝继续。
- [x] recovery 数据继续只位于 `D_recovery`，不进入 reward replay，也不是
  Q_safe mixed batch 的采样源。

#### P15.3：逐速度 gate 与 frozen paired evaluation

- [x] `0.40/0.50/0.80/1.00` 分别产生 gate JSON，统一要求：
  AUROC `>=0.80`、ECE `<=0.10`、Brier `<=0.15`、pair accuracy
  `>=0.65`、selected false-safe `<=5%`、coverage `>=30%`、
  replacement `>=15%`、replacement failure contribution `>0`，
  且 fallback contribution 不超过总 reduction 的 `50%`。
- [x] gate evaluation 在相同 MuJoCo snapshots 上比较 nominal 与
  selector action，并单独报告 replacement 和 structured fallback 的真实
  failure contribution；结果写入 `frozen_paired_evaluation.json`。
- [x] 任一目标速度 gate 失败时，runner 保存 logging-only checkpoint/report
  后停止，不启动 masking finetune。

#### P15.4：固定速度 SAC/SQRL finetune

- [x] 新增 `scripts/run_p15_common_actor.py`，seed 内只有一个 common SAC。
  SAC 从 common checkpoint 开始；SQRL 从相同 agent/reward replay 加
  Q_safe 的 checkpoint 开始。
- [x] 两组固定速度均训练 4000 policy steps，关闭 curriculum，使用相同
  seed、actor loss、UTD、batch 和 replay 起点。
- [x] primary SQRL 为 masking-only：
  `sqrl_actor_lagrange_enabled=false`、`K=32`、`epsilon=0.20`、
  behavior-support gate 开启、Q_safe 冻结。
- [x] 已通过离线 gate 的速度可从 finetune 第一步启用 masking；普通命令
  仍使用原在线 calibration/candidate-window gate，不受影响。
- [x] empty safe set 使用 structured fallback，但 replacement、fallback、
  no-safe 和 emergency 分开统计。
- [x] 每次 replacement 保存 nominal/selected Q_safe、action distance、
  candidate group，以及随后 8/16/32 步 failure/near-failure/censor 状态。
  同时统计 fall 前 H 步没有 replacement 的 false-negative falls。
- [x] 每个 run 写 `training_summary.json`，包含 falls/1000 steps、
  episode fall rate、return、episode length、velocity、near-fall、
  intervention、critic timing、SQRL active/replacement/no-safe/emergency、
  replacement outcome rates 和 wall-clock。
- [x] 多 seed 结果按 seed 保存原始值和配对差；汇总输出均值及
  95% bootstrap CI。SAC falls 为 0 时百分比 reduction 写 `null`，只保留
  绝对差值。

#### P15.5：验收与运行

已通过的纯代码验收包括：

- `1.00` 必须自身晋级后才能切 balanced；
- balanced 速度 round-robin 和逐档 transition/episode 双门槛；
- multi-buffer + speed-stratified sampling；
- episode 三 split 无泄漏与 fingerprint 稳定；
- actor hash 稳定、参数变化可检测；
- recovery replay 隔离；
- P15 gate 全条件检查；
- structured fallback contribution 独立统计；
- prevalidated gate 第一步启用；
- replacement 8/16/32 outcome 与 false-negative fall 统计。

解析正式 pilot（不启动训练）：

```bash
micromamba run -n oss python -m scripts.run_p15_common_actor \
  --dry-run --out-root saved/p15
```

运行 seed 42 pilot：

```bash
micromamba run -n oss python -m scripts.run_p15_common_actor \
  --seeds 42 --out-root saved/p15 --wandb
```

pilot 全部通过后才运行正式 seeds：

```bash
micromamba run -n oss python -m scripts.run_p15_common_actor \
  --seeds 42,43,44,45,46 --out-root saved/p15_formal \
  --reuse-complete --wandb
```

当前状态：实现和离线单测已完成；尚未产生 seed 42 的 0.30–1.00 正式训练
结果。runner 会在 common coverage、Q_safe 数据覆盖或任一逐速度 gate
失败时停止，因此不会再自动复用旧 30k/mixed checkpoint。

真实 controller/MuJoCo 两档 smoke（2026-07-29）：

- `0.30/0.35` common curriculum 在 120 policy steps 完成；
- 两档 frontier 均实际通过后才切 balanced；
- balanced 每档 `40 transitions / 2 episodes / 0 falls`，coverage manifest
  与 common actor hash 已落盘；
- multi-speed branch collector 每档完成
  `60 natural transitions / 3 episodes / 3 snapshots / 57 branches`，
  合并后为 `6 snapshots / 114 branches`，snapshot/episode id 无冲突；
- 该极短 smoke 的 branch train split 没有任何 failure label，Q_safe data
  gate 按设计拒绝训练。此结果只验证工程连接和 fail-fast，不是学习结果，
  也不能降低正式 pilot 的 `1600 transitions / 40 snapshots` 门槛。

smoke 输出位于 `saved/p15_smoke_030_035/`。

### P16：固定 0.30 m/s 的 Q_safe 跨 actor 复用实验

P16 回答一个比 P15 更窄的问题：先在 `0.30 m/s` 训练一条
`SAC + auxiliary Q_safe` 源 run，然后只取出 Q_safe，能否减少另一条
从零训练、同为 `0.30 m/s` 的 SAC 在学习过程中的摔倒。源 run 只负责
生产 Q_safe，不作为目标对照组，也不把源 actor、reward critics 或
reward replay 迁移到目标 run。

#### P16.1：配对设计

- 源 seed 默认为 `142`，目标 pilot seed 为 `42`，两者分离。
- 目标 seed 先生成唯一的 step-0 checkpoint，包含随机初始化
  actor/reward critics、空 reward replay 和空 safety replay。
- A 组从该 checkpoint 运行纯 SAC。
- C 组从同一 checkpoint 加入冻结 Q_safe，但只评分和记录，永不改动作。
- B 组从同一 checkpoint 加入同一个冻结 Q_safe，并从第 0 个 policy
  update 起启用 masking；actor Lagrange penalty 关闭。
- 三组固定 `0.30 m/s`，训练步数、seed、SAC 超参数和初始 agent hash
  完全相同。唯一允许跨 source/target 边界复制的是
  `safety_critic_state`。

`scripts/compose_qsafe_transfer_checkpoint.py` 专门执行这次跨 actor
迁移。它保留目标 actor/reward critic/replay，记录 source/target 两个
agent hash，并明确写入四个 transfer 标志。P15 的
`compose_control_checkpoint.py` 仍然拒绝不同 actor hash，二者不可混用。

#### P16.2：跨 actor 启用门槛

随机初始化的目标 actor 与成熟源 actor 的动作分布不同，所以不能直接用
源 replay 上的 AUROC 决定 masking 是否安全。P16 先冻结目标 actor，从
它的自然 rollout 保存 MuJoCo snapshots，对 nominal、policy samples、
previous、contracted previous 和局部扰动动作执行 `8/16/32` 步分支。

`scripts/evaluate_p16_transfer.py` 只用这些目标 actor 分支计算：

- point AUROC、ECE 和 Brier；
- same-state pairwise ranking accuracy；
- behavior-support coverage；
- selected false-safe rate、replacement rate；
- replacement 与 structured fallback 各自带来的真实 failure reduction。

门槛沿用 P15 的 control-facing 条件，但输出协议为 `P16`。未通过时仍运行
A 组与 C 组，以判断“仅计算 Q_safe”是否意外改变训练；B 组不运行，状态
写为 `transfer_gate_failed`。这样不会在已经证实 critic 无法迁移时强行
执行危险 masking。

#### P16.3：结果与执行

每组输出完整 `training_summary.json`，P16 汇总额外保存累计 fall 曲线、
每次 fall 的 policy step、首次 fall step、falls/1000 steps、return、
episode length、velocity，以及 masking 的 active/replacement/no-safe/
emergency 和 replacement 后 `8/16/32` 步结果。

手动运行 pilot：

```bash
micromamba run -n oss python -u -m scripts.run_p16_qsafe_reuse \
  --source-seed 142 --target-seeds 42 \
  --source-steps 15000 --target-steps 15000 \
  --out-root saved/p16_qsafe_reuse
```

当前执行顺序为 P15 seed-42 pilot 完整结束后再启动 P16，避免单个
simulator/controller 被两个在线训练同时占用。P16 的总结果写入
`saved/p16_qsafe_reuse/p16_results.json`。
