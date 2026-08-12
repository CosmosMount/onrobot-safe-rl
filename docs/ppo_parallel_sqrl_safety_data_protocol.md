# PPO并行数据增强SQRL Safety Critic协议

状态：生效。机器可读定义见
`config/qsafe_ppo_sqrl_data_v1.yaml`。本协议取代先训练候选动作oracle、再用
SAC branching标签训练critic的旧主路线；旧结果只保留为历史诊断。

## 首轮研究问题

首轮只检验：在critic结构和SQRL Bellman训练形式相同的条件下，2000环境
随机PPO采集的数据是否比现有SQRL/SAC safety数据更能学习同状态动作风险，
并能否在200组独立PPO branching中真实减少H96 fall。

PPO使用两个seed的early、boundary和mature冻结checkpoint。每个checkpoint
采集相同数量transition，形成分层嵌套的1M、3M、5M数据集。数据量不是最终
目标；state-action和失稳边界覆盖改善才是主要假设。

## 固定语义

- `Q_safe`是固定PPO reference continuation下的policy-conditional critic。
- Bellman cost只使用`c_{t+1}`，next action由冻结reference PPO随机采样。
- `critic_action`唯一指当前20 ms内真正送入PD的12维绝对关节目标。
- requested、pre-projection及旧`action_executed`字段不得进入critic。
- fall后立即terminal和reset；训练及采集阶段禁止recovery或get-up。
- PPO candidate已经按策略采样，有限SQRL rejection不得再次按策略密度加权。

## 首轮停止条件

PPO-data critic必须优于state-only、shuffled-training和现有SQRL/SAC-data
critic，并且冻结selector在独立branching上的fall-reduction LCB大于零。
否则停止，不启动SAC transfer或新的50k SAC实验。

首轮只允许输出`ppo_data_advantage_supported`与`action_signal_learnable`的
判定；其余两个flag保持false。Objective 1通过前禁止速度范围扩展。
