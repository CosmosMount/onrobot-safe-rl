# 续写（带名词解释，截至 2026-07-29）

于是开始直接检查 Q_safe 对同状态下不同 action 的判断，而不再只看自然
rollout 上的 AUROC。

名词解释：

- **Q_safe / safety critic**：输入机器人状态 \(s\) 和动作 \(a\)，输出未来
  发生摔倒或接近摔倒的风险。数值越大表示越危险。
- **自然 rollout**：不人为恢复状态或替换动作，让当前 policy 在仿真中
  正常连续运行得到的轨迹。
- **AUROC**：衡量模型能否把失败样本排在正常样本前面。0.5 接近随机，
  1.0 表示排序完美。但它不能证明模型会在同一个状态下选对动作。

## Exact-state counterfactual branch

在 MuJoCo 中保存 SAC 自然运行时的精确状态，对同一个状态分别执行：
SAC nominal action、nominal+扰动、previous action、contracted previous
action 和 policy sampled actions，再向前模拟 8/16/32 步，记录是否摔倒、
near-failure、time-to-failure、倾角、高度和接触。

名词解释：

- **MuJoCo**：机器人动力学仿真器。这里用于保存和恢复完全相同的机器人
  状态，并测试不同动作的物理后果。
- **Exact state**：MuJoCo 的完整积分状态，包括位置、速度和动力学内部
  状态。每条 branch 都从同一个 exact state 开始。
- **Counterfactual / 反事实**：实际只执行了一个动作，但通过恢复仿真状态，
  可以回答“如果当时执行另一个动作，会发生什么”。
- **Branch**：从同一个保存状态出发、执行某个候选动作形成的一条短轨迹。
- **SAC nominal action**：SAC 在当前状态下原本准备执行的确定性动作，是
  所有安全改善的对照基准。
- **Nominal+扰动**：在 nominal action 附近加入小噪声，用于检查局部是否
  存在更安全的动作，以及 Q_safe 能否识别小动作差异。
- **Previous action**：上一 policy step 实际使用的动作，是一种保持动作
  连续性的候选。
- **Contracted previous action**：把上一动作乘以约 0.9，使动作幅度略微
  收缩，是没有可信安全动作时使用的固定保守 fallback。
- **Policy sampled action**：直接从 SAC 的动作概率分布
  \(\pi(a|s)\) 中采样，因此通常比任意高斯扰动更符合 actor 学到的步态。
- **H8/H16/H32**：分别观察候选动作在未来 8、16、32 个 policy steps 内
  的后果。当前约 20 Hz，对应约 0.4、0.8、1.6 秒。
- **Near-failure**：还没有真正摔倒，但倾角、高度、接触等已经进入危险
  边界。
- **Time-to-failure**：从执行候选动作到真正失败经历的步数。越短一般
  表示动作越危险。

这种实验有效，是因为所有 branch 的初始状态完全相同，只改变第一个动作；
执行第一个动作后，又统一交给同一个冻结 policy 继续控制。因此不同结果
主要反映候选动作的影响，而不是初始姿态不同。

名词解释：

- **冻结 policy**：评估过程中 policy 参数不更新，保证所有 branch 使用
  相同的后续控制器。
- **配对比较**：同一个状态下比较多个动作，比不同 episode 之间直接比较
  更能隔离动作本身的作用。

## 第一批反事实数据

第一批 policy-sampled 数据包含 80 个状态、1840 条 branches。候选动作的
behavior support coverage 从 32.5% 提高到 80.2%，说明直接从 policy
采样比只在 nominal action 周围加高斯扰动更合理。

名词解释：

- **Behavior support / 行为支持域**：训练数据或当前 actor 真正覆盖过的
  动作区域。支持域内的动作有数据依据，支持域外动作的 Q_safe 预测通常
  不可靠。
- **Support coverage**：候选动作中有多少比例落在行为支持域内。这里越高，
  表示候选越接近 policy 真正会产生的动作。
- **高斯扰动**：从正态分布采样噪声并加到动作上。独立扰动每个关节可能
  产生 actor 从未学过的不自然组合。

## Branch-supervised Q_safe

增加 branch failure BCE 和同状态 action pair ranking 后，H16 的三组
episode split 结果为：

  branch AUROC：0.972±0.013
  同状态 pair ranking accuracy：0.642±0.054
  selected false-safe：0.111±0.157
  coverage：0.210±0.007
  nominal-relative fall reduction：0.092±0.029

名词解释：

- **BCE / Binary Cross Entropy**：二分类损失。这里让 Q_safe 预测某条
  branch 在 H16 内是否会失败。
- **Pair ranking**：对同一个状态下的一对动作进行排序，要求实际更危险的
  动作具有更高的 Q_safe。
- **Episode split**：按完整 episode 划分训练集和验证集。同一 episode 的
  相邻状态不能同时进入训练和验证，避免数据泄漏。
- **Pair ranking accuracy**：同状态动作对中，Q_safe 风险顺序与真实物理
  后果顺序一致的比例。0.5 接近随机，越高越好。
- **Selected false-safe**：selector 接受为安全的动作中，实际 branch
  rollout 会失败的比例。越低越好。
- **Coverage**：有多少状态能找到被系统接受的安全动作。过低表示系统大多
  数时候只会拒绝或 fallback。
- **Nominal-relative fall reduction**：同一个状态下，SAC nominal action
  的失败标签减去最终选中动作的失败标签，再对所有状态取平均。正数表示
  相比原 SAC 减少了失败。
- **±**：三个不同 episode split 的均值和标准差。标准差大表示结果对数据
  划分敏感。

结果说明反事实监督确实改善了 action ranking，但只有 1/3 split 通过控制
门槛，所以还不能启用在线 masking。

名词解释：

- **控制门槛 / control gate**：在允许 Q_safe 实际修改动作前必须满足的
  最低指标，包括 false-safe、coverage、pair accuracy 和真实 fall
  reduction。
- **Masking / action masking**：过滤 policy 给出的危险候选，只允许预测
  风险低于阈值的动作执行。

## Behavior support gate

同时增加 behavior support gate：只有数据支持域内的 candidate 才允许
Q_safe 排序，支持域外直接 abstain。这样可以避免 Q_safe 在整个 action
空间中搜索一个虚假的低风险 OOD action。

名词解释：

- **Support gate**：检查候选动作是否接近 actor 分布或历史数据。只有检查
  通过的动作才能参与安全比较。
- **Candidate**：selector 在同一个状态下准备比较的候选动作。
- **OOD / Out of Distribution**：超出训练数据分布的状态或动作。神经网络
  在 OOD 区域可能给出非常自信但错误的预测。
- **Abstain**：系统承认当前没有足够可信的信息，不让 Q_safe 自由选择，
  转而使用固定 fallback 或 supervisor。

## Conservative safety critic

尝试 CQL-style conservative safety loss 后，policy/OOD action 的风险会被
相对推高，但较大的 conservative weight 会损害 calibration，并没有单独
解决 selected false-safe。因此这个方向只能作为辅助正则，不能代替
counterfactual data。

名词解释：

- **CQL / Conservative Q-Learning**：对数据支持不足的动作采用更悲观的
  Q 估计。在这里是把不熟悉动作的安全风险向上推。
- **Conservative safety loss**：在原 Bellman/BCE 损失外加入保守项，使
  policy 或 OOD actions 不容易被错误预测为低风险。
- **Conservative weight**：保守损失的权重。过小可能不起作用，过大则会
  让所有动作都被预测为危险。
- **Calibration / 概率校准**：例如预测风险为 0.2 的样本中，长期来看是否
  真的约有 20% 失败。排序正确不等于概率校准正确。
- **正则 / regularization**：附加在主要训练目标上的约束，用于减少某类
  错误，但不能替代缺失的监督数据。

## 独立 critic A/B

随后增加完整独立的 critic B：critic A 负责选择动作，critic B 只验证 A
选中的动作，B 不能重新搜索 candidate。A/B 都认为风险低，并且相对
nominal action 的风险改善超过 margin，才允许替换；否则执行
contracted-previous fallback。

名词解释：

- **Critic A / selector critic**：在受支持候选中选择预测风险最低的动作。
- **Critic B / validator critic**：独立检查 A 已经选中的唯一动作，不参与
  搜索，避免两个 critic 联合利用同一批候选中的预测误差。
- **独立 critic**：A/B 不共享网络参数、target network、optimizer 和随机
  数生成状态。
- **Margin**：替代动作相对 nominal action 必须达到的最小风险改善。如果
  改善很小，不值得冒预测误差带来的风险。
- **Fallback**：Q_safe 不可信或没有安全动作时执行的固定降级动作。
- **Contracted-previous fallback**：执行缩小后的上一动作，而不是执行
  Q_safe 在 OOD 区域找到的任意最小风险动作。

独立 A+B 虽然把 selected failure rate 从 0.814 降到 0.721，但约 63% 的
fall reduction 来自 fallback，而不是被 Q_safe 验证通过的 action。
critic B 也没有稳定挡住 seed-43 的 false-safe。

名词解释：

- **Selected failure rate**：selector 最终输出动作的实际 branch failure
  比例。
- **Replacement contribution**：真正用 Q_safe 选择的替代动作带来的
  failure reduction。
- **Fallback contribution**：由于拒绝候选并执行固定 fallback 带来的
  failure reduction。它属于 supervisor safety，不能算作 policy 学会安全。
- **Seed**：随机数种子。不同 seed 会产生不同的数据划分、初始化和采样
  顺序，用于检验结果是否稳定。

## Hard-negative replay

继续挖掘 A 实际选错的 hard negatives。三个 split 中只找到 0/2/0 条；
seed-43 加权训练后，branch AUROC 从 0.959 降到 0.935，false-safe 仍为
0.333。说明主要错误发生在未见 episode，训练集内 hard-negative replay
数据太少。

名词解释：

- **Hard negative**：模型预测为安全、selector 也会接受，但真实 branch
  rollout 会失败的动作。它是最值得重点学习的错误样本。
- **Hard-negative replay**：提高这些错误样本在训练 batch 中的采样频率
  或 loss 权重。
- **加权训练**：让某些样本对总 loss 的影响更大。样本过少时，过度加权
  可能损害整体泛化。

## Conformal risk upper bound

在独立 calibration episodes 上增加 conformal risk upper bound 后，
三个 split 的 observed false-safe 可以降到 0，但 coverage 从 18.6%
降到 9.0%，abstention 达到 91%。这相当于“不确定时全部拒绝”，可以作为
安全降级机制，但还不是有效的 action selector。

名词解释：

- **Conformal calibration**：使用一批不参与训练的数据，统计模型实际
  低估风险的程度，并据此构造风险上界。
- **Risk upper bound**：不直接使用 Q_safe 的平均预测，而使用更悲观的
  上界，例如 \(Q_{\text{upper}}=Q+\text{calibration offset}\)。
- **Calibration episode**：只用于确定温度或风险上界，不参与网络参数
  训练，也不能与最终 validation episode 重叠。
- **Observed false-safe=0**：当前有限验证样本中没有发现错误安全动作，
  不等于数学上保证未来永远为 0。
- **Abstention rate**：系统拒绝使用 Q_safe 选择结果的状态比例。91%
  表示绝大多数时候都没有实际使用 selector。

## Active branch collection

因此开始主动采集 selector 最容易出错的状态，而不是固定每隔 N 步保存：

  Q_safe 接近 epsilon 的 risk-boundary；
  critic A/B disagreement；
  behavior support boundary；
  selector 准备 replacement/abstain；
  near-failure；
  stable normal。

名词解释：

- **Active collection / 主动采集**：不是均匀保存所有状态，而是优先保存
  对当前模型最有信息量或最容易出错的状态。
- **Epsilon**：SQRL 判断动作是否安全的风险阈值。
  \(Q_{\text{safe}}\le\epsilon\) 才可能被接受。
- **Risk boundary**：Q_safe 接近 epsilon 的区域。很小的预测误差就可能
  改变“安全/不安全”决定。
- **A/B disagreement**：两个独立 critic 对同一候选动作的风险预测差异，
  可作为 epistemic uncertainty 的近似。
- **Support boundary**：候选动作接近行为支持域边缘，模型开始从插值进入
  外推的位置。
- **Selector decision state**：selector 准备替换 nominal action 或准备
  abstain 的状态，直接影响实际控制结果。
- **Stable normal**：姿态稳定、风险较低的正常行走状态。必须保留这类样本，
  防止数据集只包含摔倒附近状态。

最终平衡 active 数据包含 100 个 episodes、200 个 snapshots、3000 条
branches。normal 占 25%，其余五类各 30 个 snapshots。H16 的真实
failure rate 随类别明显增加：

  normal：4.3%
  risk-boundary：20.0%
  support-boundary：39.8%
  A/B disagreement：41.6%
  selector-decision：76.4%
  near-failure：100%

名词解释：

- **Snapshot**：保存的一份 exact simulator state。一个 snapshot 可以
  产生多条不同 candidate branches。
- **平衡数据**：人为控制各类 snapshot 的配额，避免 failure 或 normal
  中某一类完全占据训练集。
- **Failure rate 随类别增加**：说明 active sampler 的类别具有真实物理
  含义，而不只是模型数值上的任意分组。

## Episode-bootstrap ensemble

在此数据上训练 3 个完整独立的 episode-bootstrap Q_safe。每个 member
使用不同的 bootstrap episodes、网络、target、optimizer 和 RNG；fit、
temperature calibration、conformal calibration、validation episodes
完全隔离。

名词解释：

- **Ensemble**：同时训练多个独立模型，再综合它们的预测。
- **Member**：ensemble 中的一个完整 Q_safe 模型。
- **Episode bootstrap**：以 episode 为单位有放回抽样。某些 episode 会被
  多次抽中，另一些不被抽中，使不同 member 看到不同训练分布。
- **有放回抽样**：每次抽样后仍允许再次抽到同一个 episode。
- **Target network**：用于构造 Bellman target 的慢更新网络，减少训练
  振荡。
- **Optimizer**：根据 loss 更新网络参数的算法及其内部状态。
- **RNG**：随机数生成器状态，影响初始化、batch 采样和动作采样。
- **Fit episodes**：真正用于更新网络权重的数据。
- **Temperature calibration**：只调整概率温度，使风险概率更接近真实
  频率，不改变动作排序。
- **Validation episodes**：只用于最终报告，不能参与训练或任何校准。

最终 ensemble mean 的 point AUROC 为 0.946，高于 2/3 单独 members；
observed false-safe 从旧 critic A 的 21.4% 降到 0。但：

  同状态 pair ranking accuracy：0.541
  coverage：6.8%
  nominal-relative fall reduction：0

名词解释：

- **Ensemble mean**：多个 member 风险预测的平均值。
- **Point AUROC**：把每条 \((s,a)\) branch 独立看作一个分类样本计算的
  AUROC。它仍然不等同于同状态动作选择能力。
- **Observed false-safe**：最终 validation 数据中，预测安全但实际失败的
  已观察比例。
- **Coverage 6.8%**：只有约 6.8% 的状态存在被系统接受的动作，说明方法
  非常保守。
- **Fall reduction=0**：最终选中动作没有比 SAC nominal action 少摔倒。

加入 ensemble disagreement gate、mean+std 或 conformal upper bound 后，
coverage 进一步降到 0~2.3%，没有产生 validated replacement fall
reduction。

名词解释：

- **Ensemble disagreement/std**：多个 member 输出的标准差。越大表示模型
  对该动作越没有共识。
- **Epistemic uncertainty**：由于数据不足或模型认知不足产生的不确定性；
  理论上可通过增加相关数据降低。
- **Disagreement gate**：member 分歧超过阈值时直接 abstain。
- **Mean+std**：用平均风险加标准差作为更悲观的风险估计。
- **Validated replacement**：替代动作经过安全模型检查后真正被执行，并在
  branch ground truth 中比 nominal 更安全。

## 当前结论

Q_safe 已经可以较好地区分“某条 transition 最终是否摔倒”，但仍不擅长
在同一个状态下比较多个相近 candidate actions。当前主要瓶颈是
same-state counterfactual action coverage，而不是网络深度、网络数量或
自然 validation AUROC。

名词解释：

- **Transition**：一个训练时间步的数据，通常包含
  \((s,a,r,s',done)\) 以及 safety labels/costs。
- **Failure classification**：判断某个状态动作样本未来是否失败。
- **Action-level ranking**：在同一个状态下判断多个动作谁更安全，是 SQRL
  masking 真正依赖的能力。
- **Counterfactual action coverage**：训练数据对“同一状态下多个不同动作
  及其真实后果”的覆盖程度。

当前所有 masking 仍保持 logging-only。下一步需要保证每个自然 episode
至少保存一个 normal/early snapshot，把有 branch 数据的 episode 从当前
53 个提高到 80~100 个，再重新训练 episode-bootstrap ensemble。

名词解释：

- **Logging-only**：只计算和记录 Q_safe，不允许它修改实际执行动作。
- **Early snapshot**：每个 episode 开始阶段保存的状态，用于保证更多
  episode 真正进入 branch dataset。
- **有 branch 数据的 episode**：至少包含一个 exact-state snapshot 和多条
  candidate rollouts 的 episode。没有 snapshot 的 episode 无法用于
  same-state ranking。

只有同时满足：

  selected false-safe <= 5%
  coverage >= 30%
  replacement rate >= 15%
  replacement 本身产生正的 fall reduction
  fallback contribution 不超过总 reduction 的 50%

才进入 frozen-policy SQRL masking rollout；在此之前不继续放宽 epsilon，
也不进入 CSAC barrier 或 FlashSAC。

名词解释：

- **Replacement rate**：最终动作不是 nominal，而是安全系统选出的替代
  动作的状态比例。
- **Fallback contribution**：总 fall reduction 中由固定降级动作贡献的
  部分；过高说明 Q_safe 没有真正学会选动作。
- **Frozen-policy masking rollout**：冻结 SAC 参数，只让 safety selector
  修改动作，用于隔离 action masking 本身的效果。
- **放宽 epsilon**：提高安全阈值会增加 coverage，但也可能放入更多危险
  动作，不能用来掩盖 Q_safe 不准确。
- **CSAC barrier**：把安全风险以 barrier penalty 的形式加入 actor loss，
  让 policy 在训练过程中主动避开高风险动作。
- **FlashSAC**：项目中计划使用的更高吞吐 SAC backbone。只有安全 pipeline
  在当前 DroQ/SAC 上验证有效后才迁移。
