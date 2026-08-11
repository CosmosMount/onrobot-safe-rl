# PPO 多并行验证记录

## 目的

该验证回答的是“Go2 是否能以多环境并行方式运行 PPO 采样”，不是
“PPO 数据是否已经证明 Q_safe 能减少 SAC fall”。后一个问题仍然必须使用
same-state branching、目标 continuation policy 和 SAC-only 验证集回答。

## 实际执行

入口：`scripts/validate_parallel_go2_ppo.py`

```bash
python3 scripts/validate_parallel_go2_ppo.py \
  --envs 64 --iterations 2 --rollout-steps 24 \
  --output /tmp/qsafe_parallel_ppo_64.json
```

每个 worker 使用 `/home/xyz/code/unitree_mujoco/unitree_robots/go2/scene_empty.xml`，
独立 MuJoCo `MjData`、独立随机种子和独立 reset；策略步包含 10 个 2ms
低层物理步。动作采用绝对 `q_target = init_q + scale * action`，并以 PD
力矩执行。采样后执行 clipped PPO ratio、value loss 和 entropy bonus 更新。

## 结果（2026-08-11）

| 项目 | 结果 |
|---|---:|
| 并行环境 | 64 |
| PPO iterations | 2 |
| rollout steps / iteration | 24 |
| 总 policy-env steps | 3072 |
| 物理步 / policy step | 10 |
| policy-env steps/s | 768.23 |
| 观察到的 falls | 1570 |
| backend | native MuJoCo subprocess workers |

这证明了：多环境采样、批量收集、PPO 更新和 fall 统计在本机可以闭环运行。
高 fall 数来自未训练的短 smoke policy，不能解释为 Q_safe 效果。

## MjLab/Warp 状态

已在 `/tmp/go2-mjlab-pkgs` 隔离安装 Unitree MjLab 依赖并解析 Go2-Flat
PPO 入口。实际初始化 MuJoCo Warp 时出现本机版本组合的 Warp kernel
兼容错误（`sensor_vel`/`_frame_axis` codegen），因此没有把失败的 MjLab
吞吐当作结果，也没有修改实时运行时或项目依赖。下一闸门是固定兼容版本后
重复同一 parity corpus，再比较 Warp 与 native backend 的 fall/contact 一致性。

## 与 Objective 1 的关系

这项 PoC 只开放 PPO 作为 state proposer / coverage source。后续导出数据必须
在同一状态上对候选动作分支，并用 frozen SAC 或 filtered-SAC continuation
生成标签；PPO 单步 transition 不进入最终 Q_safe 的直接监督。Objective 1
仍以 SAC-from-zero paired fall reduction 为优先，Objective 2 继续锁定，直到
Objective 1 的预注册 gate 通过。
