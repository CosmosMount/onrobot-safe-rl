# Objective 1 动作条件 Q_safe 协议（v1）

状态：生效。它取代此前“`Q_safe(s)` 识别危险后统一执行 recovery”的路线。

## 1. 研究问题

最终模型定义为：

```text
Q_safe(s, a) = P(当前五帧可部署观测 s 下先执行动作 a，
                 随后按配对 SAC continuation 运行时，H96 内发生 fall)
```

部署时 SAC 每个 50 Hz 控制周期提出 nominal action。系统生成当前状态下仍
保持站立、踏步和支撑能力的可执行候选，比较 nominal 与候选风险。nominal
安全时不干预；nominal 危险时，从满足安全门槛的动作中选择风险最低且偏离
nominal 最小的动作。下一周期重新观测和选择，不默认锁定长 recovery。

## 2. 已否定路线

以下结果仅保留为失败 baseline/诊断，不是 Objective 1 有效证据：

- state-only Q_safe 触发 joint-brake；
- halfway-neutral、ramp-neutral、ramp-crouch；
-成熟 SAC 策略固定时长接管；
- PPO 策略固定时长接管；
-基于 state-only successor risk 的 H1/H3 predictive shield；
-任何“near-fall 后统一动作或统一长序列”的 selector。

它们回答的是“危险后统一恢复是否有效”，并不回答“在同一状态下哪个具体
动作更安全”。旧产物不得进入新模型的 fit/calibration/model-test，也不得
计入 Objective 1 的通过判定。

## 3. PPO 的边界

PPO 多并行只产生自然 fall、normal、near-fall 状态。禁止外力、impulse 和
训练期间 recovery。每个环境第一次 fall 即 terminal，并在同一个 policy
step 完成 reset。完全翻倒后的 get-up 只属于 terminal 之后的独立起身流程，
不得作为 pre-fall 候选或 Q_safe 动作标签。

PPO 单轨迹只能说明实际执行动作对应的结果。它不能为未执行候选动作生成
`Q_safe(s,a)` 标签。state-only ensemble 只允许提出 early near-fall 状态、
提供辅助特征或作为 baseline。

## 4. 先验证候选空间

在训练 action critic 前必须通过 oracle gate：

1. 从自然 SAC/PPO 轨迹中选择距第一次 terminal fall 还有 48–96 个 policy
   step 的 early pre-fall 状态；state-only risk 只用于 proposal/分层。自然
   轨迹标签只能决定状态 admission，不能给任何未执行动作打标签。admission
   全程不知道 candidate branching outcome。
2. 恢复完整 native MuJoCo snapshot，不 settle，不施加扰动。
3. 同状态 branch SAC nominal、局部 SAC 动作、对称局部扰动、状态相关支撑
   与 capture-step 动作；必要的短 option 只能作为显式候选，不能是 get-up。
4. 全部候选共享配对的 SAC continuation 随机性，评估 H96 fall。
5. 事后 oracle 对每个状态查看全部候选的全部 branch outcome，选择经验
   H96 fall 率最低的具体动作；报告 nominal、逐候选、oracle-best fall 率，
   以及按 actor/source 分层的 reduction 和置信区间。它只表示候选空间的
   上界，不是可部署 selector。另行报告 discovery 选动作、audit 测风险的
   稳定性，以及“预知未来随机性”的 same-CRN realization oracle；后者只作
   诊断，不能用于正式门控。
6. protected oracle 至少包含 2 个未见 SAC actor、4 个未见 source、120 个
   state group，且每动作至少 32 个 replica。只有 oracle reduction 的下置信
   界大于零且每个 actor 方向为正，才允许训练 `Q_safe(s,a)`。失败时只能修改
   候选空间，不能调 selector 阈值。

正式 protected roster 固定为 SAC-from-zero actor 57/58 的 10k checkpoint：
actor 57 对应 source 9701/9702，actor 58 对应 9703/9704。每个 source 固定
自然运行 20k policy steps并抽取30组，每动作32个replica。不得因fall数量
更换actor/checkpoint/source、增加exposure或自动重试。PPO档案仅用于
development/fit状态覆盖；protected oracle使用natural-SAC状态。两类轨迹均
不得给未执行候选动作直接打标签。

## 5. 模型训练和验证

训练数据必须来自同状态多动作 branching。split 单位是 source state/group，
同一状态的不同动作禁止跨 train/calibration/test。模型必须通过 within-state
action sensitivity 检验，拒绝退化成 `Q_safe(s)`。

离线报告至少包含 pair accuracy、strong-pair accuracy、top-1 fall
reduction、oracle reduction、learned-selector reduction 和两者差距。最终
Model-Test 只使用未见 SAC actor/source；SAC-from-zero 三臂只使用 fresh
seed。Objective 1 未通过前，Objective 2 保持禁止。

机器可读的唯一协议源为
`config/qsafe_action_conditioned_objective1_v1.yaml`。
