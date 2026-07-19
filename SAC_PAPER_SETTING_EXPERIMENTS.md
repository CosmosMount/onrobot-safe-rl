# Go2 单机 SAC：论文 Setting 对照与逐项排错

本文档记录当前 Go2 单机 DroQ/SAC 与 *A Walk in the Park* 的 setting
差异，以及每次只改变一个因素的实验结果。目标不是把仓库改造成论文官方代码，
而是保留现有 Go2、固定翻倒自救和 controller 安全边界，以论文为参照理解
SAC 的实现、数值稳定性和控制 setting。

`SAC_SINGLE_ENV_TODO.md` 保留为历史和扩展建议，其中小方差 random、
n-step、降低 UTD 等内容不属于当前首轮论文 setting 排错。

## 固定不变量

首轮实验不修改：

- Go2 和 Unitree MuJoCo；
- `in_process` 单机训练结构；
- 固定流程翻倒自救，不改 learned reset policy；
- 当前 58 维 observation；
- 前 1000 步 `Uniform[-1, 1]` random action；
- UTD=20，即每个新 transition 更新 20 次 critic、1 次 actor、1 次温度；
- critic LayerNorm、dropout 和双 Q；
- action offset、nominal pose、one-step return；
- recovery transition 不进入 replay；
- upright reward guard 和 failure terminal penalty。

## Setting 对照

| 项目 | 论文 setting | 当前仓库 | 分类 | 对 SAC/MDP 的影响 | 当前动作 |
| --- | --- | --- | --- | --- | --- |
| UTD | 20 | 20 | 一致 | 决定每条新数据被 critic 使用的次数 | 固定 |
| critic 正则化 | LayerNorm/Dropout | LayerNorm + 0.01 dropout | 一致 | 抑制高 UTD 下 critic 过拟合 | 固定 |
| random warmup | 前 1000 步 uniform | 前 1000 步 `[-1,1]` uniform | 一致 | 决定初始 replay 覆盖范围 | 固定 |
| actor/critic 更新比 | 每步 1 actor / 20 critic | 相同 | 一致 | 不能把 UTD 误解为 actor 也更新 20 次 | 固定 |
| action mapping | nominal pose + per-joint offset | `[0.05,0.7,-1.4]` 和 `[0.2,0.4,0.4]` ×4 | 基本一致 | 定义策略可到达的 PD target | 固定 |
| action filter | 不使用更好 | 默认关闭 | 一致 | 过滤会引入历史依赖并改变 MDP | 固定 |
| 机器人 | Unitree A1 | Unitree Go2 | 故意不同 | 动力学、电机和 PD 合适区间不同 | 不改 |
| reset | learned reset policy/人工处理 | 固定 recovery→standup FSM | 故意不同 | 影响 wall time，不应污染 policy replay | 保留 |
| recovery replay | reset 数据不计训练样本 | `policy_step=False` 不插入 | 正确性保护 | 防止 critic 学习非策略控制导致的 transition | 保留并验证 |
| reward guard | 上游 dense reward | idle/upright gate + terminal penalty | E05 已对照，保留safe默认 | upstream最终能走，但零速约5分/步且收益不可直接解释为位移 | profile可切换 |
| PD damping | 仿真消融中约 `Kd=10` 较好 | policy `Kd=5` | E04 已回退 | `Kd=10` 在当前 Go2 上速度下降、跌倒和饱和显著增加 | 后续保持 5 |
| policy 频率 | 目标 20 Hz | E03真实target发送18.10 Hz，达到门槛但未满20 Hz | E03已修复到可接受 | 实际 action duration 改变即改变 MDP | 保留，记录剩余开销 |
| tanh log-prob | 稳定 squashed Gaussian | sample 时保留 pre-tanh latent，使用稳定 Jacobian | E01 已修复 | 饱和时仍保持有限 log-prob/梯度 | 保留 |
| checkpoint | 与 eval 解耦 | 原实现依赖 eval | 工程 bug | 不改变学习，但中断会丢失 agent/replay | E00 修复 |
| observation | roll/pitch、速度、关节、contact、prev action | quaternion、速度、关节、requested/executed action | 故意/待研究 | 定义不同的可观测 MDP | 只记录，不改 |
| foot contact | 4 维 contact | 缺失 | 后续可选 | 可能帮助相位和支撑腿判断 | 首轮不改 |
| target entropy 等 | Walk in the Park官方配置 | `target_entropy=-6`等 | E06已核对，一致 | 直接影响随机性与温度 | 无证据不消融 |
| 小 std、n-step、低 UTD | 非本文主 baseline | 未启用 | 非当前范围 | 会同时改变探索或学习目标 | 不做 |

## 指标怎样解释

| 指标 | 能说明什么 | 不能单独说明什么 |
| --- | --- | --- |
| critic loss | 当前 Q 与 Bellman target 的误差和数值稳定性 | loss 小不等于策略会走 |
| Q 均值 | critic 对长期软回报的估计尺度 | Q 上升不等于真实位移增加 |
| entropy / temperature | 策略随机性和自动温度调节是否正常 | 高 entropy 不必然代表有效探索 |
| actor loss | entropy 项和 Q 项合成后的优化目标 | 不能脱离 Q、温度和 action 饱和判断 |
| episode return | 当前 reward 定义下的累计表现 | reward 有漏洞时不等于走路 |
| forward velocity / world-x | 瞬时速度和真实净位移 | 单个速度尖峰不代表稳定步态 |
| upright / fall | 是否保持正常姿态、是否利用倒置漏洞 | 不跌倒也可能只是在原地站立 |
| effective Hz | 实际 action/transition 循环频率 | 配置文件中的 20 Hz 不等于真实 20 Hz |
| action saturation | tanh 输出是否长期接近 `±1` | 少量饱和不一定有问题 |
| replay size | 实际进入学习的数据量 | recovery wall time 不计有效样本 |

## 统一实验协议

- 首轮固定 `seed=42`，每项 5000 个 loop step：1000 random + 4000 train。
- 每项使用全新 agent、空 replay 和独立 `save_dir`。
- 行为实验之间不恢复 checkpoint。
- 每 100 步打印当前值和最近最多 1000 个有效 policy step 的汇总。
- 每 1000 步保存 compact agent+replay snapshot，正常结束或 SIGINT 保存最终状态。
- 明显有效的 setting 才进入 20000 步和至少 3 个 seed 的确认。

固定启动命令：

```bash
mkdir -p logs
python -m train --mode in_process --config-profile simulation \
  2>&1 | tee logs/reward_guard_flat.log
```

训练结束后将最新日志归档到
`logs/experiments/<experiment_name>.log`。`reward_guard_flat.log` 始终指向
最近一次实验。

### 硬停止条件

- policy、batch、loss、Q、entropy 或 temperature 出现 NaN/Inf；
- 数值连续爆炸且 policy 已被保护性停更；
- DDS/IPC 失联；
- reset/recovery 不能回到稳定 policy start；
- 仿真地形或物理状态异常。

环境故障不算算法实验结论；排除环境问题后使用相同配置从头运行。

## 修改队列

状态流：

```text
待修改 → 已修改待训练 → 训练中 → 已分析 → 保留/回退
```

| 实验 | 唯一变量 | 状态 | 进入下一步的条件 |
| --- | --- | --- | --- |
| E00 `e00_reward_guard_flat` | 只加观测、隔离和 checkpoint | 已分析，保留基础设施 | 完成 5000 步并形成基线摘要 |
| E01 `e01_stable_tanh_logprob` | 稳定 tanh Gaussian log-prob | 已分析，保留 | 测试证明正确；5000步无 NaN/Inf |
| E02 `e02_fused_utd20` | JIT/scan 融合 20 次 critic | 已分析，保留 | 参数语义一致，在线更新提速76.8% |
| E03 `e03_20hz_schedule` | action 周期调度 | 已分析，保留 | 真实action Hz=18.10，更新计数完整 |
| E04 `e04_kd10` | policy `Kd: 5 → 10` | 已分析，回退 | 速度-45.2%，fall+66.7%，不满足保留条件 |
| E05 `e05_upstream_reward` | upstream 与 safe reward | 已分析，主线保留safe | 速度+7.4%未达10%；upstream保留为复现profile |
| E06 | SAC setting逐项核对 | 已核对，无需改动 | 官方配置一致且日志无对应异常，不重复训练 |

## 实验记录

### E00：当前 reward-guard 基线

| 字段 | 内容 |
| --- | --- |
| 唯一修改 | 无学习行为修改；新增实验隔离、compact checkpoint、rolling metrics |
| 配置 | seed=42，max_steps=5000，UTD=20，random=1000，Kd=5 |
| reward | idle gate 0.04、upright gate cos(30°)、terminal penalty -10 |
| reset | 当前固定 recovery/standup，非 policy transition 不进 replay |
| 假设 | 当前数值爆炸和低 effective Hz 可在不改变算法的情况下被可靠测量 |
| 预期 | random 阶段约 20 Hz；train 阶段因串行 UTD 明显降频；倒置状态无正向 reward |
| 日志 | `logs/experiments/e00_reward_guard_flat.log`；W&B run `ctkcyp5r` |
| wall time | 23 分 24 秒；5000 个有效 policy transition，replay size=5000 |
| 最近 1000 步 | forward velocity=0.416 m/s；world-x delta=-11.949 m；upright=99.9%；action saturation=2.9% |
| 最近 1000 步训练量 | critic loss=16.495；Q=556.636；actor loss=-566.905；entropy=-6.418；temperature=0.176 |
| 最近 1000 步时序 | env step=50.45 ms；UTD update=63.60 ms；loop=115.29 ms；effective Hz=8.70 |
| episode/fall | 全程 49 次 fall、3 次 time-limit；最后 1000 步只有 1 次 fall |
| 数值范围 | 日志点 Q 2.64→593.04；entropy 4.91→最低 -9.73→-6.62；temperature 0.10→最低0.0915→0.1848 |
| NaN/爆炸 | 0 WARNING、0 Traceback、无 NaN/Inf；critic loss 在 step 1800 有一次 229.24 尖峰后恢复 |
| checkpoint | step 1000/2000/3000/4000/5000 均成功；最终 compact snapshot 约 6.4 MB |
| 决策 | 保留 E00 的实验隔离、rolling metrics 和 Flax state-only compact checkpoint；不把当前学习行为直接判为失败 |
| 学到的结论 | 当前 SAC 能在约3000步后形成较长episode并持续提高body-frame前向速度，但同步 UTD 使实际频率只有约8.7 Hz；早期高跌倒率使固定自救占用大量wall time |

后续每个实验都复制以上表格，并填写实际数据、与前一实验的百分比变化以及
“保留/回退”的理由。

### E00 结果分析

1. **当前不是立即数值崩溃。** 之前出现过的 `1e17` entropy/temperature
   爆炸在本次 5000 步中没有重现。entropy 自动回到 target entropy `-6`
   附近，temperature 先下降再上升，说明自动温度调节至少在本次 seed 中工作。
2. **真实控制频率与论文 setting 不同。** random 阶段只有环境等待时约
   19.2 Hz；进入训练后 50 ms 环境等待与约 64 ms UTD 更新串行相加，
   loop 约 115 ms，即约 8.7 Hz。配置写 20 Hz 并不代表策略面对的是
   20 Hz MDP。
3. **学习趋势存在但早期代价很高。** step 1000 后曾连续出现 20–50 步
   episode；到 step 3265 后完成了一个 400 步 time-limit episode，最后
   1000 步只有一次 fall，平均前向速度达到 0.416 m/s。
4. **world-x 与 body-forward 不能混用。** 最后窗口 body-frame forward
   velocity 为正而 world-x delta 为负，表示机器人朝向已改变后仍沿自身
   前方运动。后续应增加平面路径长度、初始朝向投影位移和 heading 指标，
   不能只用 world-x 的正负判定步态。
5. **E01 仍有价值，但优先级应按实验目的理解。** 本次没有触发 tanh
   log-prob 爆炸，不代表从 clipped action 反算 latent 是正确实现。E01
   属于数值正确性修复；它需要证明饱和 action 下 log-prob/梯度更稳定，
   而不能只看单次 return 是否提高。

下一步保持所有 E00 setting 不变，只进行 E01
`stable_tanh_logprob`，并使用相同 5000 步指标做对照。按逐项流程，
在用户确认前不修改 E01。

### E01：稳定 tanh Gaussian log-prob

| 字段 | 内容 |
| --- | --- |
| 唯一修改 | sample 时保留原始 pre-tanh latent；actor 和 sampled critic backup 直接用 latent 计算 Gaussian log-prob 与稳定 tanh Jacobian |
| 未修改 | seed=42、UTD=20、random=1000、网络、target entropy、temperature、reward、Kd=5、controller/recovery |
| 测试 | 15项 unittest 全通过；普通 action 与 inverse 公式一致；latent=`[100,-100,50]` 时 action、log-prob、梯度均有限 |
| 日志 | `logs/experiments/e01_stable_tanh_logprob.log`；W&B run `gyp5g2ja` |
| 环境重试 | 第一次在 step 879 因 SportModeState age=0.260s 超过0.250s硬停止；尚未训练，日志/快照保留为 `e01_attempt1_stale_dds`，第二次从空 replay 重跑 |
| wall time | 15分26秒；5000个有效 policy transition，replay size=5000 |
| 最近1000步 | forward velocity=0.644 m/s；world-x delta=-6.785 m；upright=100%；action saturation=0.67% |
| 最近1000步训练量 | critic loss=8.398；Q=741.350；actor loss=-748.267；entropy=-5.661；temperature=0.147 |
| 最近1000步时序 | env step=50.46 ms；UTD update=65.43 ms；loop=117.12 ms；effective Hz=8.56 |
| episode/fall | 全程23次 fall、6次 time-limit；后段连续完成400步episode，最后一次 fall 在step3938 |
| NaN/爆炸 | 0 NaN/Inf、0 Traceback；entropy和temperature始终有限 |
| checkpoint | step 1000/2000/3000/4000/5000 均成功；失败尝试和完整实验互不覆盖 |
| 决策 | 保留。它是公式正确性修复；单seed行为指标也没有显示回退 |
| 学到的结论 | 饱和 action 不应作为 latent 的可逆存储。保留采样 latent 后，SAC 的 entropy 梯度路径数值正确，且本次训练更稳定、更少饱和 |

### E01 与 E00 同口径比较

| 最近1000步指标 | E00 | E01 | 变化 |
| --- | ---: | ---: | ---: |
| forward velocity | 0.416 | 0.644 | +54.6% |
| action saturation | 2.90% | 0.67% | -77.0% |
| critic loss | 16.495 | 8.398 | -49.1% |
| entropy | -6.418 | -5.661 | 更接近 target -6 |
| temperature | 0.176 | 0.147 | -16.4% |
| effective Hz | 8.70 | 8.56 | -1.6%，基本不变 |
| 全程 fall | 49 | 23 | -53.1% |
| time-limit episode | 3 | 6 | +3次 |
| episode length mean | 89.0 | 163.4 | +83.6% |
| episode return mean | 403.8 | 1044.7 | +158.7% |

解释：

1. **正确性结论独立于 return。** 旧路径在 action 到达浮点 `±1` 后会丢失
   原始 latent；clip 后反解得到的是另一个值。E01 保留真实 latent，并使用
   `2(log 2 - x - softplus(-2x))` 计算稳定 Jacobian，因此极端样本也有有限梯度。
2. **本次单 seed 的行为结果同时改善。** forward velocity、episode length、
   fall 和饱和率都明显优于 E00，critic loss 约减半。它们支持保留 E01，
   但还不能证明所有改善都能跨 seed 重现。
3. **Q 变大不是数值爆炸。** E01 最近窗口 Q=741 高于 E00 的557，但对应
   episode return 和长度也明显更高，critic loss 更低，且 entropy/temperature
   有限；当前证据更符合更长软回报 horizon，而不是 critic 发散。
4. **world-x delta 不适合作为跨 reset 累计里程。** 物理 reset 会改变窗口
   起终点，机器人转向后 body-forward 与 world-x 也会分离。本次负值不否定
   0.644 m/s 的 body-frame forward velocity；后续应补充逐 transition 路径长度
   和初始 heading 投影位移。
5. **E01 没有解决频率。** update 耗时与 E00 基本相同，训练仍只有约8.6 Hz。
   这符合实验设计：E01 只修 log-prob，吞吐问题留给 E02。

下一步应为 E02 `fused_utd20`，只融合20次 critic 更新并验证算法语义不变。
按逐项流程，在用户确认前不开始修改或训练 E02。

### E02：融合 UTD=20 critic 更新

| 字段 | 内容 |
| --- | --- |
| 唯一修改 | 把20个顺序 critic minibatch update 放入一个整体 `jax.jit` + `jax.lax.scan`；actor和temperature仍在第20个 minibatch 上各更新1次 |
| 未修改 | E01 log-prob、seed=42、UTD=20、batch切分顺序、RNG顺序、target soft-update次数、random=1000、网络、reward、Kd=5、controller/recovery |
| 等价性测试 | 固定 seed/batch 下，旧Python顺序实现与fused实现的全部agent叶子、RNG和loss在`rtol=atol=2e-6`内一致 |
| 更新计数测试 | critic step=`20`（测试配置为4时等于4）；actor step=`1`；temperature step=`1`；target critic optimizer step=`0` |
| 全量测试 | 18项 unittest 全通过 |
| 离线性能 | 58维obs、12维action、batch=256、UTD=20：顺序中位20.23 ms，fused中位7.65 ms，2.65×加速 |
| 日志 | `logs/experiments/e02_fused_utd20.log`；W&B run `dyhnuzjz` |
| wall time | 8分22秒；E01为15分26秒，缩短45.8% |
| 最近1000步 | forward velocity=0.653 m/s；world-x delta=-41.008 m；upright=100%；action saturation=0.275% |
| 最近1000步训练量 | critic loss=4.128；Q=817.193；actor loss=-821.547；entropy=-5.151；temperature=0.0986 |
| 最近1000步时序 | env step=50.34 ms；UTD update=15.21 ms；loop=66.23 ms；effective Hz=15.11 |
| episode/fall | 全程10次 fall、9次 time-limit；最后一次 fall 约在step1521，后段持续完成长episode |
| NaN/爆炸 | 0 NaN/Inf、0 Traceback；训练数值有限 |
| checkpoint | step 1000/2000/3000/4000/5000 均成功 |
| 决策 | 保留。算法参数语义通过等价性测试，吞吐显著改善，在线训练无稳定性回退 |
| 学到的结论 | UTD=20本身不必降低；低频主要来自20个JIT调用的dispatch/同步开销。融合执行能保留相同Bellman更新序列并显著降低wall time |

### E02 与 E01 同口径比较

| 最近1000步指标 | E01 | E02 | 变化 |
| --- | ---: | ---: | ---: |
| UTD update | 65.43 ms | 15.21 ms | -76.8% |
| loop time | 117.12 ms | 66.23 ms | -43.4% |
| effective Hz | 8.56 | 15.11 | +76.5% |
| forward velocity | 0.644 | 0.653 | +1.5% |
| action saturation | 0.667% | 0.275% | -58.8% |
| critic loss | 8.398 | 4.128 | -50.8% |
| Q | 741.35 | 817.19 | +10.2% |
| 全程 fall | 23 | 10 | -56.5% |
| time-limit episode | 6 | 9 | +3次 |
| episode length mean | 163.4 | 248.5 | +52.1% |
| episode return mean | 1044.7 | 1802.1 | +72.5% |

解释：

1. **算法更新语义相同。** `lax.scan` 的 carry 是完整 agent，每个 critic
   minibatch 都接收前一次更新后的 critic、target critic和RNG。测试逐叶对比
   旧实现，证明没有把20次更新误改成并行平均，也没有让actor更新20次。
2. **在线提速低于离线2.65×是正常的。** 每个环境step仍固定等待约50 ms；
   fused只把学习部分从65 ms降到15 ms，所以总循环由117 ms降到66 ms。
3. **行为指标不能视为纯粹的同MDP消融。** 计算语义虽一致，但同步单机循环中
   更快的更新使action hold从约117 ms变成约66 ms，机器人面对的动力学时间尺度
   也改变了。因此E02的fall/return改善同时包含执行加速和更高控制频率的效果。
4. **15.1 Hz仍不是论文的20 Hz。** 当前结构是50 ms环境等待后再做约15 ms
   学习，两者串行相加。根据预定门槛 `effective Hz < 18`，E03条件已经满足。
5. **world-x负位移主要反映转向。** E02最后窗口body-forward约0.653 m/s，
   但world-x为负且world-y明显变化，仍需heading/路径长度指标才能判断直行性。

下一步应为 E03 `20hz_schedule`，只调整action周期与学习计算的调度，保持
E02的fused UTD和全部SAC setting不变。按逐项流程，在用户确认前不开始E03。

### E03：20 Hz action流水线调度

| 字段 | 内容 |
| --- | --- |
| 唯一行为修改 | action发送后，在它的50ms hold窗口内执行上一条transition的完整fused UTD更新；下一动作使用更新前actor，形成1次update的policy lag |
| 观察开销修改 | simulation profile把rolling聚合/W&B由每步改为每10步；transition、replay insert和SAC update仍逐步执行 |
| 未修改 | E02 fused UTD、UTD=20、batch/RNG内部顺序、E01 log-prob、random=1000、网络、reward、Kd=5、controller/recovery |
| transition时序 | `send(a_t) → hold 50ms → read(s_{t+1})/reward → queue transition t → send(a_{t+1}) → update(t) during hold` |
| recovery边界 | callback只在`policy_step=True`时执行；reset/recovery期间pending update保留到下一次正常policy hold |
| 测试 | 21项 unittest 全通过；专项验证50ms剩余hold、overrun、recovery禁更、fused语义 |
| 更新计数 | 最终snapshot：critic step=80000，actor=4000，temperature=4000，replay=5000；与4000正式训练步×UTD20严格一致 |
| 日志 | `logs/experiments/e03_20hz_schedule.log`；W&B run `bhjh5850` |
| 首次尝试 | step约2200时发现每步rolling/W&B在hold外把真实发送压到17.2 Hz；快照/日志归档为`e03_attempt1_metrics_outside_hold`，修正纯观察开销后从空replay重跑 |
| wall time | 9分31秒；policy loop更快，但比E02多5次fall/recovery，因此总wall time反而比E02多69秒 |
| 最近1000步 | forward velocity=0.657 m/s；world-x delta=-5.713 m；upright=100%；action saturation=0.142% |
| 最近1000步训练量 | critic loss=4.458；Q=810.955；actor loss=-816.532；entropy=-5.100；temperature=0.1073 |
| 最近1000步时序 | env step=50.36 ms；UTD update=11.62 ms（被hold覆盖）；loop=54.71 ms；loop Hz=18.28；真实action Hz=18.10 |
| hold overrun | 最后1000步均值0 ms；首次在线shape编译的单次长hold已滚出最终窗口 |
| episode/fall | 全程15次 fall、7次 time-limit；最后一次fall在step2730，后段保持长episode |
| NaN/爆炸 | 训练loss/Q/entropy/temperature无NaN/Inf，0 Traceback；日志唯一`nan`是首个动作没有前一发送时间时预期的`action_hz=nan` |
| checkpoint | step 1000/2000/3000/4000/5000 均成功；保存前会flush pending update，agent与replay一致 |
| 决策 | 保留。达到≥18 Hz验收且transition/update计数正确；1-update policy lag作为单机流水线与论文并行执行的工程差异明确记录 |
| 学到的结论 | 单纯减少计算只能到15 Hz；要接近20 Hz必须让learner与action hold重叠。代价是actor轻微滞后，这与异步/并行采集中的policy lag同类 |

### E03 与 E02 同口径比较

| 最近1000步指标 | E02 | E03 | 变化 |
| --- | ---: | ---: | ---: |
| UTD update | 15.21 ms | 11.62 ms | -23.6% |
| loop time | 66.23 ms | 54.71 ms | -17.4% |
| loop effective Hz | 15.11 | 18.28 | +21.0% |
| 真实action Hz | 未直接测量 | 18.10 | 新增真实发送指标 |
| control hold overrun | 未测量 | 0 ms | 最终窗口无超时 |
| forward velocity | 0.653 | 0.657 | +0.5% |
| action saturation | 0.275% | 0.142% | -48.5% |
| critic loss | 4.128 | 4.458 | +8.0% |
| Q | 817.19 | 810.95 | -0.8% |
| 全程 fall | 10 | 15 | +50% |
| time-limit episode | 9 | 7 | -2次 |
| episode length mean | 248.5 | 215.0 | -13.5% |
| episode return mean | 1802.1 | 1475.5 | -18.1% |

解释：

1. **频率目标达到但不是严格20 Hz。** 真实发送18.10 Hz满足预设≥18门槛；
   剩余约4.7 ms主要来自action采样、replay采样、Python调度及每10步一次日志。
   若要求严格20.0 Hz，需要独立collector/control线程或把更多外围工作异步化，
   超出当前单变量E03范围。
2. **transition语义保持正确。** `s_{t+1}`在`a_t`完整hold后读取，随后发送
   `a_{t+1}`；学习只占用`a_{t+1}`的hold，不会缩短transition的物理时间，
   也不会用35ms状态冒充50ms next observation。
3. **代价是1-update policy lag。** 为了不延迟下一action，`a_{t+1}`必须在
   update(t)完成前选择。这不改变每条Bellman更新，但改变了actor采样与learner
   RNG操作的跨步顺序，属于异步执行常见的轻微staleness。
4. **单seed行为没有全面优于E02。** forward velocity和饱和率略好，但fall、
   episode length和return变差。由于频率从15 Hz变为18 Hz本身改变MDP，这些
   差异不能用来否定调度正确性；后续Kd消融应基于E03的新动力学时间尺度。
5. **总wall time受recovery主导。** 虽然有效policy loop提速21%，E03多5次
   fall，固定恢复流程使5000步总时间比E02更长。这进一步说明吞吐和步态稳定性
   是两个不同指标。

下一步按队列为 E04 `Kd: 5→10`，保持E03流水线和全部SAC setting不变。
按逐项流程，在用户确认前不开始E04。

### E04：policy PD damping `Kd=10`

| 字段 | 内容 |
| --- | --- |
| 唯一行为修改 | policy phase 的 `Kd: 5 → 10`；`Kp=60` 不变 |
| 未修改 | E03流水线、UTD=20、random=1000、SAC网络/超参数、reward、action mapping、observation和replay语义 |
| 增益隔离 | policy=`Kp60/Kd10`；stand-up固定为E03的`60/5`；recovery保持`100/8` |
| 基础设施修复 | Controller支持递归YAML overlay，使仿真E04不修改真实/默认`config/go2.yaml`的`Kd=5`；stand-up增益显式独立，避免顶层policy增益污染固定恢复流程 |
| 测试 | Controller编译成功；21项unittest通过；最终checkpoint的critic=80000、actor=4000、temperature=4000、replay=5000 |
| 日志 | `logs/experiments/e04_kd10.log`；W&B run `s6lvjtjv` |
| 非正式尝试 | 两次分别在21/245步终止，暴露原顶层`Kd`同时作用于stand-up的混杂；日志和checkpoint已归档，不计入E04结果 |
| wall time | 12分52秒；5000个有效policy transition |
| 最近1000步 | forward velocity=0.360 m/s；world-x delta=-0.477 m；upright=99.3%；action saturation=3.592% |
| 最近1000步训练量 | critic loss=18.809；Q=598.370；actor loss=-606.530；entropy=-8.072；temperature=0.1590 |
| 最近1000步时序 | env step=50.34 ms；UTD update=10.94 ms；loop=54.47 ms；loop Hz=18.36；真实action Hz=18.20 |
| episode/fall | 全程25次fall、0次time-limit；episode length均值166.7，return均值650.2 |
| NaN/爆炸 | loss/Q/entropy/temperature无NaN/Inf和Traceback；日志中的唯一`nan`仍是step 0没有前一动作时间戳 |
| 决策 | **回退到policy `Kd=5`。** 速度未改善10%，反而下降45.2%；fall增加66.7%，超过允许的20%；action饱和约为E03的25.4倍 |
| 学到的结论 | 论文A1上的`Kd≈10`不能直接移植到Go2。PD damping改变的是action到状态转移的物理映射，策略会重新适应，但更高阻尼并未自动带来更平滑、更稳定的学习 |

### E04 与 E03 同口径比较

| 最近1000步指标 | E03 `Kd=5` | E04 `Kd=10` | 变化 |
| --- | ---: | ---: | ---: |
| forward velocity | 0.657 m/s | 0.360 m/s | -45.2% |
| world-x delta | -5.713 m | -0.477 m | 不作为胜负依据；受朝向和reset影响 |
| upright ratio | 100.0% | 99.3% | -0.7百分点 |
| action saturation | 0.142% | 3.592% | 25.4倍 |
| action std | 0.589 | 0.685 | +16.2% |
| critic loss | 4.458 | 18.809 | +321.9% |
| Q | 810.955 | 598.370 | -26.2% |
| entropy | -5.100 | -8.072 | 更低2.972 |
| temperature | 0.1073 | 0.1590 | +48.2% |
| 全程 fall | 15 | 25 | +66.7% |
| episode length mean | 215.0 | 166.7 | -22.5% |
| episode return mean | 1475.5 | 650.2 | -55.9% |
| 真实action Hz | 18.10 | 18.20 | 基本相同 |

解释：

1. **频率不是差异来源。** 两项真实action频率只差约0.10 Hz，E04的退化不能
   归因于重新出现的低频训练。
2. **world-x不能推翻forward velocity结论。** 机器人会转向，且reset会改变
   episode坐标连续性；E04的world-x更接近零并不等价于走得更好。局部前向速度、
   fall和episode return三者一致指向退化。
3. **SAC数值有限但学习任务更难。** E04没有NaN或熵爆炸，不过critic loss约为
   E03的4.2倍，温度升高、entropy更负且动作饱和显著增加，说明策略在更强阻尼下
   使用更激进的边界动作，同时critic面对的回报/终止分布更不稳定。
4. **单seed足以执行本轮回退规则，但不是论文值的普遍否定。** 若未来要研究
   Go2最优PD，可在`Kd=5`附近做多seed连续扫描；当前目标是逐项复现和排错，
   E04已明确不满足进入20000步确认的门槛。

下一步按队列为 E05 reward原文对照，但在用户确认前不开始；E05应从E03优胜
配置继续，即policy `Kd=5`。

### E05：upstream reward 对照

| 字段 | 内容 |
| --- | --- |
| 唯一行为修改 | safe→upstream：`reward_min_forward_vel: 0.04→null`、`reward_upright_min_cos: 0.866→-1`、`fall_terminal_penalty: -10→0` |
| 未修改 | E03流水线、policy `Kp60/Kd5`、UTD=20、random=1000、SAC网络/超参数、action mapping、observation、固定recovery和replay过滤 |
| profile基础设施 | simulation支持`reward_profile: baseline/upstream`；两个preset都显式包含三项reward字段，避免切换后残留上一个profile的值 |
| 测试 | 23项unittest通过；验证upstream倒置状态不被gate且failure penalty为0；最终checkpoint critic=80000、actor=4000、temperature=4000、replay=5000 |
| 日志 | `logs/experiments/e05_upstream_reward.log`；W&B run `8riu9rg8` |
| wall time | 8分11秒；5000个有效policy transition |
| 最近1000步 | forward velocity=0.706 m/s；world-x delta=+33.313 m；action saturation=0.250%；action std=0.610 |
| 真实姿态复核 | upstream下日志`upright_gate=1`是禁用后的常量，不能代表直立；从replay quaternion重算body-up均值0.9979、最小0.9887，100%高于cos(30°) |
| 最近1000步训练量 | critic loss=8.377；Q=814.072；actor loss=-818.796；entropy=-4.812；temperature=0.0718 |
| 最近1000步时序 | UTD update=14.30 ms；loop Hz=17.66；真实action Hz=17.42；hold overrun=0 |
| episode/fall | 原始日志：全程15次fall、8次400步time-limit；最后一次fall在step 2097附近 |
| NaN/爆炸 | loss/Q/entropy/temperature无NaN/Inf和Traceback；唯一`nan`为step 0预期的`action_hz=nan` |
| 决策 | upstream作为论文复现profile保留，但主线恢复safe baseline。速度仅提升7.4%，未达到预设10%确认门槛；单seed不足以证明upstream更优 |
| 学到的结论 | dense tolerance在零速时仍给约5分/步，甚至轻微后退也可能为正，因此早期return不能当作会走；但本seed最终没有利用倒置漏洞，并学出了正常直立前进 |

### E05 与 E03 safe reward 同口径比较

| 最近1000步指标 | E03 safe | E05 upstream | 变化 |
| --- | ---: | ---: | ---: |
| forward velocity | 0.657 m/s | 0.706 m/s | +7.4% |
| world-x delta | -5.713 m | +33.313 m | 朝向不同，不能做百分比；E05确有大幅净移动 |
| 真实直立比例 | safe gate=100% | quaternion复算=100% | 均正常直立 |
| action saturation | 0.142% | 0.250% | +76.5%，绝对值仍低 |
| action std | 0.589 | 0.610 | +3.5% |
| critic loss | 4.458 | 8.377 | +87.9%，仍有限 |
| Q | 810.955 | 814.072 | +0.4% |
| entropy | -5.100 | -4.812 | +0.288 |
| temperature | 0.1073 | 0.0718 | -33.1% |
| episode length mean | 215.0 | 213.0 | -1.0% |
| episode return mean | 1475.5 | 1705.4 | +15.6%，reward定义不同，不代表等量性能提升 |
| 全程 fall | 15 | 15 | 相同 |
| 真实action Hz | 18.10 | 17.42 | -3.8%，E05本机负载下低于18 Hz门槛 |

解释：

1. **reward漏洞存在，但本次没有发展为最终坏策略。** tolerance的目标区间是
   0.5–1.0 m/s且margin为1.0 m/s，所以零速本身得到`forward_term=0.5`，
   即未扣yaw时约5分。日志在`x_vel≈0.027`时出现reward 4.71，在轻微后退时
   也出现正reward。safe idle gate会消除这部分不可解释回报。
2. **upright gate关闭后，指标语义也变了。** `upright_gate=1`只说明阈值设成
   -1，不能证明机器人直立。因此使用checkpoint最后1000条observation的
   quaternion重算；结果表明本次最终策略确实直立，而不是倒置利用reward。
3. **upstream没有达到单变量晋级标准。** forward velocity只提高7.4%，低于
   10%；fall没有改善，critic loss更高。world-x明显为正说明它确实移动，
   但E03可能因转向沿负world-x运动，不能把符号变化直接换算成百分比优势。
4. **return不能跨reward定义直接比较。** E05 episode return高15.6%，其中包含
   零速/低速阶段的dense shaping；这正说明论文复现时应同时观察速度、姿态和
   位移，而不是只看W&B return。
5. **本次频率略低是一个小混杂。** E05真实action Hz为17.42，低于E03的18.10，
   但hold没有overrun且SAC更新计数完整。若未来要宣称reward profile性能优劣，
   需要在相同机器负载下做多seed复测；当前结论只支持“可复现且存在回报解释
   风险”，不支持upstream优于safe。
6. **修复了timeout统计字段。** E05原始日志有8次`truncated`，但旧rolling
   代码只把`standup_timed_out`计入`timeouts_total`，因此本次W&B summary错误
   显示0。该映射已修正并加测试；不影响E05训练、replay或上述原始日志计数。

当前simulation默认已恢复`reward_profile: baseline`。下一步为E06逐项核对SAC
setting；按流程在用户确认前不开始。

### E06：SAC/DroQ setting 核对

E06首先区分两个不同来源，避免把两套论文设置混在一起：

- 当前主线复现对象是 Smith、Kostrikov、Levine 的 *A Walk in the Park*，
  官方代码为 <https://github.com/ikostrikov/walk_in_the_park>；
- Sabatini et al. 2026 *Bridging the Gap* 用于检查现代SAC实现问题，但其主实验
  是8192个并行环境的IsaacLab训练，不是当前单机DroQ baseline。

#### 与 Walk in the Park 官方代码逐项比较

| setting/语义 | 官方实现 | 当前仓库 | 结论 |
| --- | ---: | ---: | --- |
| actor learning rate | `3e-4` | `3e-4` | 一致 |
| critic learning rate | `3e-4` | `3e-4` | 一致 |
| temperature learning rate | `3e-4` | `3e-4` | 一致 |
| hidden dimensions | `[256,256]` | `[256,256]` | 一致 |
| discount | `0.99` | `0.99` | 一致 |
| tau | `0.005` | `0.005` | 一致 |
| critics | 2 | 2 | 一致 |
| critic dropout | `0.01` | `0.01` | 一致 |
| critic LayerNorm | `true` | `true` | 一致 |
| initial temperature | `0.1` | `0.1` | 一致 |
| target entropy | `None→-action_dim/2=-6` | 相同 | 一致 |
| actor Q aggregation | `mean(Q1,Q2)` | `mean(Q1,Q2)` | 一致 |
| target Q aggregation | `min(Q1,Q2)` | `min(Q1,Q2)` | 一致 |
| sampled entropy backup | 开启 | 开启 | 一致 |
| UTD | 20 | 20 | 一致 |
| random warmup | 1000 | 1000 | 一致 |

当前仓库额外使用global gradient norm clip=1。原Walk in the Park代码没有该项，
但Sabatini 2026明确使用max gradient norm=1；它是稳定性保护，不是当前日志中
需要移除的疑似bug。

#### 为什么不把 Sabatini 2026 的数值直接覆盖进来

| setting | Sabatini 2026 | 当前Walk in the Park主线 |
| --- | ---: | ---: |
| environments | 8192 | 1 |
| action step | 20 ms | 目标50 ms，实测约55–57 ms |
| actor/critic/temp LR | `2e-4/2e-4/2e-5` | `3e-4/3e-4/3e-4` |
| discount/tau | `0.97/0.003` | `0.99/0.005` |
| initial temperature | `0.001` | `0.1` |
| target entropy | scale `0.167` | `-action_dim/2=-6` |
| return | 5-step | 1-step |
| network | `[1024,512,256]`, SiLU | `[256,256]` |
| observation normalization | actor/critic均开启 | 未做running normalization |
| LayerNorm | false | true |
| actor initial std | `0.15` | 沿用Walk in the Park actor |

这些参数共同依赖大规模并行采样、24-step rollout、8192 batch和5-step target。
一次只搬运其中一个数值，不是在复现Sabatini，反而会形成无法解释的混合算法。

#### 日志证据

E03 safe baseline最后1000步：

- entropy=-5.100，目标entropy=-6；
- temperature=0.1073，接近初始0.1且没有单向饱和；
- critic loss=4.458，Q=810.955；
- 0 NaN/Inf，最终策略能达到0.657 m/s。

E05更换reward后temperature降到0.0718、entropy变为-4.812，说明自动温度确实会
根据策略/reward诱导出的分布调整；这不是initial temperature或target entropy
失效的证据。

#### E06 决策

本轮不启动新的5000步训练，因为：

1. Walk in the Park官方SAC/DroQ setting与当前代码完全一致；
2. actor使用双Q均值不是bug，而是官方参考实现的明确语义；
3. E01已修复官方旧代码在tanh边界反算latent的数值问题，但没有改变目标分布；
4. entropy和temperature日志稳定，没有触发预先约定的单变量消融条件；
5. 不修改任何变量再跑一次只是在不同物理初态下重复E03，不能回答新的SAC问题。

如果后续有意转向复现Sabatini 2026，应新开一条独立实验线，至少把actor小方差
初始化、timeout next-state、5-step return、observation normalization和对应网络/
学习率作为一套设计处理，不能归入当前E06的单变量setting修补。

## 决策规则

正确性修复（checkpoint、log-prob、transition 时序、recovery replay）由测试和
实现语义决定，不根据单 seed return 回退。

性能实现修复（fused UTD、20 Hz 调度）在算法语义不变且吞吐提高时保留。

setting 消融需要同时满足：

- 最近 1000 步平均 forward velocity 或 world-x 净位移改善至少 10%；
- 跌倒率不恶化超过 20%；
- Q、entropy、temperature 没有新增异常；
- 视觉观察与数值结论一致。

每项训练和分析完成后停止，先更新本文档并与用户确认，再进入下一项。
