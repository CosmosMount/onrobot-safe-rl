# Protected action-space oracle v1 result

结论：本次正式门控按预注册规则以“自然 early-pre-fall 状态不足”结束，不能
执行 K24×R32 branching，也不能训练 `Q_safe(s,a)` selector。

两个全新 SAC-from-zero actor 均在固定 `vx=+0.30 m/s`、无 Q_safe、无外力
条件下完成10k训练步。seed 57训练期间出现45个独立fall，seed 58出现35个；
两者的固定10k checkpoint随后各自在两个全新source上自然运行20k步。

| actor | source | exposure | independent fall | 48–96步early-prefall |
|---:|---:|---:|---:|---:|
| 57 | 9701 | 20,000 | 0 | 0 |
| 57 | 9702 | 20,000 | 0 | 0 |
| 58 | 9703 | 20,000 | 0 | 0 |
| 58 | 9704 | 20,000 | 0 | 0 |

这说明10k checkpoint已经能在当前无扰动平地source中连续完成40个500步
episode。计划要求每个source固定抽30组，并明确禁止根据结果增加exposure、
更换checkpoint、替换seed或自动重试，因此没有可合法进入protected branching
的状态。训练早期确实发生fall，但训练过程中的策略不断变化；这些fall不能
伪装成固定10k actor的protected source状态。

本次没有施加扰动、没有运行pre-fall recovery、没有用PPO轨迹给候选动作
打标签，也没有运行任何selector训练。保留的六个大型source/actor产物不提交
Git，manifest和哈希记录在机器可读报告中。

```text
candidate_oracle_gate_pass = false
model_training_authorized = false
objective1_pass = false
phase2_authorized = false
```

下一轮只能先回到development阶段重新定义能够在不人为推力条件下覆盖自然
SAC-from-zero失稳的状态年龄，例如直接冻结训练期间预注册的较早checkpoint；
在新的protected roster预注册并使用全新actor/source之前，不得复用本批结果
作正式确认。
