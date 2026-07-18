# 单机 Go2 SAC 改进清单

本文档把 Sabatini et al. (2026) 的结论转换为当前仓库的单环境、
在线 SAC 改进任务。目标不是照搬 8192 环境的配置，而是修复与并行度无关、
并且对单机/真机安全和样本效率有直接影响的问题。

## 当前判断

| 优先级 | 项目 | 当前状态 | 主要位置 |
| --- | --- | --- | --- |
| P0 | 缩小训练前随机探索 | 未解决；前 1000 步为 `[-1, 1]` 均匀采样 | `train/loop.py`, `config/*.yaml` |
| P0 | 小方差 actor 初始化 | 未解决；初始 Gaussian `std=1` | `jaxrl/distributions/tanh_normal.py` |
| P0 | reset 必须达到稳定姿态 | 已实现首版；失败时终止而不是放行倒置 policy transition | `train/env.py` |
| P0 | 倒地 reward/terminal penalty | 未解决；当前 reward 没有 upright gate 或失败惩罚 | `train/obs.py`, `train/env.py` |
| P0 | 稳定的 tanh log-prob | 未解决；当前从 clipped action 反算 pre-tanh 值 | `jaxrl/distributions/tanh_transformed.py` |
| P0 | 确保 checkpoint 定期保存 | 有缺陷；当前保存逻辑位于 eval 分支内，`no_eval: true` 时不会定期保存 | `train/loop.py` |
| P1 | n-step return | 未实现；当前是 one-step TD | `jaxrl/data/replay_buffer.py`, `jaxrl/agents/sac/droq/learner.py` |
| P1 | action bounds 校准 | 部分完成；已有手工 `action_offset`，但没有从 soft joint limits 推导非对称边界 | `train/config.py`, `train/env.py` |
| P1 | requested/projected/executed action 诊断 | schema 已有，但 replay 和日志没有完整利用 | `common/transition.py`, `train/loop.py` |
| P2 | SAC 超参数复核 | 与论文差异较大，需要消融而非直接照抄 | `config/*.yaml` |
| 已完成 | timeout-aware bootstrap | 基本正确；truncation 保持 bootstrap，termination 才清零 mask | `common/transition.py` |

## P0：训练安全性和可恢复性

### 0. 禁止从不稳定 reset 开始 policy rollout

已加入 `abort_on_unstable_reset: true`。standup/recovery 在
`reset_hold_steps` 内不能连续满足稳定条件时，训练会抛出
`UnstableResetError`，且不会发送离开 standup FSM 的 policy target。
当前等待窗口为 220 个 20 Hz step（11 秒），覆盖 recovery、standup 和
连续 10 帧稳定性确认；旧值 100 步只等于 standup 动作本身的 5 秒时长。

保留 `abort_on_unstable_reset: false` 只用于复现旧 baseline，不应用于真机。

下一步需在仿真中验证：

- 成功恢复后能正常继续 rollout；
- 恢复失败时日志包含 roll、pitch、joint error 和 belly-up；
- 不再出现一次 fall 后紧跟一个固定 21 步的倒置 episode。

### 1. 替换全范围 uniform warmup

当前行为：

```python
if i < start_training:
    action = Uniform(-1, 1)
```

这不是“采样不均匀”，而是探索范围可能过宽。对单台机器人，全范围均匀动作会增加
摔倒、安全投影和无效 transition 的比例。

建议：

- 增加 `exploration_mode: truncated_normal`。
- 增加 `exploration_std`，初始建议 `0.15`，可比较 `0.10/0.15/0.20`。
- 探索分布以归一化 action 0 为中心，即以 `init_qpos` 为中心。
- 保留 `uniform` 作为诊断选项，但不作为真机默认值。
- 统计 requested、projected、executed 三种 action，而不只统计请求值。

验收：

- 12 个 requested action 的均值接近 0，标准差接近配置值。
- clipping 比例、intervention 比例和摔倒率显著低于原 uniform baseline。
- random 阶段仍能覆盖所有关节，但不持续触及 `±1`。

### 2. 实现论文式 actor 初始化

当前 mean/log-std 输出头均使用普通 Dense 初始化；零 observation 下实测初始
`mean=0, std=1`。

建议：

- mean head：小尺度非零权重、零 bias。
- log-std head：零权重、bias=`log(initial_action_std)`。
- 增加配置 `initial_action_std: 0.15`。
- 保留 log-std clamp。
- 添加初始化测试，直接断言零 observation 上各维 std 接近配置值。

注意：只修改 actor 初始化仍不够，因为当前前 1000 步根本不使用 actor；应与上一项一起改。

### 3. 将 checkpoint 保存从 eval 解耦

当前 `save_training_snapshot()` 只在 evaluation 分支执行，而默认配置是
`no_eval: true`，长时间在线训练可能没有周期 checkpoint。

建议：

- 新增独立的 `checkpoint_interval`。
- 无论是否启用 eval，都按间隔保存 agent + replay + step + metadata。
- 使用临时文件加原子 rename，避免中断时产生半写 snapshot。
- 保留最近若干个 snapshot，避免磁盘无限增长。

验收：

- `no_eval: true` 时仍按间隔生成 snapshot。
- 中断并重启后从相同步数、agent 和 replay size 恢复。

### 3.1 修复倒地状态的奖励语义

当前 reward 只使用 pitch、前向速度和 yaw rate。roll 接近 180° 时可能仍满足
`cos(pitch) ≈ 1`，使倒置滑动获得正向奖励。

建议：

- 使用完整 body-up cosine 作为 upright gate；
- true termination 增加明确且配置化的 terminal penalty；
- 分别记录 task reward、upright gate 和 terminal penalty；
- 先修 reward，再启用 n-step，避免更快传播错误奖励。

### 3.2 使用稳定的 tanh-squashed Gaussian log-prob

当前实现从 `tanh` 后并被 clip 的 action 通过 `arctanh` 反推 latent。动作饱和时会
丢失原始 latent，可能导致 entropy 和 actor loss 数值爆炸。

建议 sample API 同时返回 action 与原始 pre-tanh latent，并直接用 latent 计算
Gaussian log-prob 和稳定的 tanh Jacobian，不再执行
`arctanh(clip(action))`。

## P1：学习目标和动作语义

### 4. 实现 timeout-aware 5-step return

当前 timeout 语义已经基本正确：

- `terminated=True`：mask 为 0，不 bootstrap。
- `truncated=True`：mask 为 1，从 reset 前末状态 bootstrap。

缺少的是 n-step return。建议：

- replay 保留 episode 连续性和 termination/truncation 标志。
- 采样时构造最多 5 步的累计折扣 reward。
- 真正 termination 截断累计回报并禁止之后 bootstrap。
- timeout 在对应 reset 前 observation 上 bootstrap，不跨 episode 累加 reward。
- episode 尾部不足 5 步时正确处理可用 horizon。
- 将 `n_step` 配置化，先做 `1 vs 3 vs 5` 消融。

验收测试至少覆盖：

1. 普通连续 5 步；
2. 第 3 步真实 termination；
3. 第 3 步 timeout；
4. replay 环形覆盖边界；
5. episode 尾部不足 n 步。

### 5. 校准每关节 action bounds

当前动作映射为：

```text
q_target = init_qpos + normalized_action * action_offset
```

这已经比直接使用机械硬极限安全，但上下界固定对称，并且依赖手工 offset。

建议：

- 明确配置 soft joint limits，不直接使用硬件 hard limits。
- 根据默认关节角到 soft limits 的上下距离，得到每关节非对称动作范围。
- 如果继续使用对称 `action_offset`，记录它相对于 soft limits 的安全余量并加启动校验。
- actor 分布若映射到非对称范围，log-prob 必须包含 affine transform 的
  change-of-variables 修正。

单机仓库不必为了复现论文而立即改成非对称边界；应先验证当前
`[0.2, 0.4, 0.4] × 4` 是否限制步态或导致频繁安全投影。

### 6. 对齐“学习动作”和“执行动作”

当前 replay 使用 `requested_action`，但环境可能经过 joint delta、filter 或 controller
投影后执行另一动作。critic 若学习请求动作的 Q，却观察执行动作导致的转移，会产生
action-transition mismatch。

需要通过实验选择并记录清楚：

- actor 输出 requested action；
- 安全层输出 projected action；
- controller 实际目标对应 executed action；
- critic 应以哪一个 action 为条件；
- intervention transition 是否保留、降权或单独建模。

验收：

- 记录每关节 requested/projected/executed 的直方图和差值。
- 记录 intervention rate、clipping rate。
- 检查大 action mismatch 是否对应 critic loss 尖峰或摔倒。

## P2：单环境超参数实验

论文参数来自 8192 个并行环境，不能整体复制。建议每次只改变一项：

| 参数 | 当前值 | 首轮候选 |
| --- | ---: | --- |
| initial actor std | 1.0 | 0.15 |
| pre-training exploration | uniform `[-1,1]` | truncated Gaussian, std 0.15 |
| n-step | 1 | 5 |
| initial temperature | 0.1 | 0.01、0.001 |
| target entropy | `-action_dim/2 = -6` | 保留 baseline，再比较约 `-2` |
| discount | 0.99 | 保留 baseline，再比较 0.97 |
| UTD | 20 | 5、10、20，同时检查实时频率 |

每次实验至少记录：

- episode return、x velocity、episode length；
- critic loss、Q 均值、actor loss、entropy、temperature；
- requested/projected/executed action；
- intervention、fall、recovery 和 timeout 次数；
- world x/y/z，用于区分走出相机、台阶接触和物理异常；
- environment steps 和 wall-clock time；
- 至少 3 个 seed，避免把 SAC 较大的 seed variance 当作改进。

## 不需要因为“单机”而放弃的论文结论

论文实验以 8192 个并行环境为主，但以下结论与并行度无关：

- 合理动作范围；
- 小方差、站姿中心化的初始策略；
- timeout 与真实失败分离；
- n-step reward propagation。

并行度主要影响吞吐、batch 构成、UTD 和 wall-clock 对比。单机在线学习反而更强调
安全探索和样本效率，因此前三项仍有直接参考价值。

## 推荐实施顺序

1. 修复 checkpoint 独立定期保存。
2. 加入 action 三阶段诊断和 intervention 指标。
3. 将 uniform warmup 改成可配置的小方差 Gaussian。
4. 实现 actor `initial_action_std=0.15`。
5. 用相同 seed 比较 uniform baseline 与小方差探索。
6. 实现并测试 timeout-aware n-step replay。
7. 再做 temperature、target entropy、discount 和 UTD 消融。
8. 最后评估是否需要非对称 soft-limit action bounds。
