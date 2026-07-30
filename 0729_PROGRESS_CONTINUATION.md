续写（截至 2026-07-29）

于是开始直接检查 Q_safe 对同状态下不同 action 的判断，而不再只看自然
rollout 上的 AUROC。

在 MuJoCo 中保存 SAC 自然运行时的精确状态，对同一个状态分别执行：
SAC nominal action、nominal+扰动、previous action、contracted previous
action 和 policy sampled actions，再向前模拟 8/16/32 步，记录是否摔倒、
near-failure、time-to-failure、倾角、高度和接触。

第一批 policy-sampled 数据包含 80 个状态、1840 条 branches。候选动作的
behavior support coverage 从 32.5% 提高到 80.2%，说明直接从 policy
采样比只在 nominal action 周围加高斯扰动更合理。

增加 branch failure BCE 和同状态 action pair ranking 后，H16 的三组
episode split 结果为：

  branch AUROC：0.972±0.013
  同状态 pair ranking accuracy：0.642±0.054
  selected false-safe：0.111±0.157
  coverage：0.210±0.007
  nominal-relative fall reduction：0.092±0.029

说明反事实监督确实改善了 action ranking，但只有 1/3 split 通过控制门槛，
结果还不稳定。

同时增加 behavior support gate：只有数据支持域内的 candidate 才允许
Q_safe 排序，支持域外直接 abstain。这样可以避免 Q_safe 在整个 action
空间中搜索一个虚假的低风险 OOD action。

尝试 CQL-style conservative safety loss 后，policy/OOD action 的风险会被
相对推高，但较大的 conservative weight 会损害 calibration，并没有单独
解决 selected false-safe。因此这个方向只能作为辅助正则，不能代替
counterfactual data。

随后增加完整独立的 critic B：critic A 负责选择动作，critic B 只验证 A
选中的动作，B 不能重新搜索 candidate。A/B 都认为风险低，并且相对
nominal action 的风险改善超过 margin，才允许替换；否则执行
contracted-previous fallback。

独立 A+B 虽然把 selected failure rate 从 0.814 降到 0.721，但约 63% 的
fall reduction 来自 fallback，而不是被 Q_safe 验证通过的 action。
critic B 也没有稳定挡住 seed-43 的 false-safe。说明仅仅增加一个不同
初始化的网络，不能自动产生独立错误。

继续挖掘 A 实际选错的 hard negatives。三个 split 中只找到 0/2/0 条；
seed-43 加权训练后，branch AUROC 从 0.959 降到 0.935，false-safe 仍为
0.333。说明主要错误发生在未见 episode，训练集内 hard-negative replay
数据太少。

在独立 calibration episodes 上增加 conformal risk upper bound 后，
三个 split 的 observed false-safe 可以降到 0，但 coverage 从 18.6%
降到 9.0%，abstention 达到 91%。这相当于“不确定时全部拒绝”，可以作为
安全降级机制，但还不是有效的 action selector。

因此开始主动采集 selector 最容易出错的状态，而不是固定每隔 N 步保存：

  Q_safe 接近 epsilon 的 risk-boundary；
  critic A/B disagreement；
  behavior support boundary；
  selector 准备 replacement/abstain；
  near-failure；
  stable normal。

最终平衡 active 数据包含 100 个 episodes、200 个 snapshots、3000 条
branches。normal 占 25%，其余五类各 30 个 snapshots。H16 的真实
failure rate 随类别明显增加：

  normal：4.3%
  risk-boundary：20.0%
  support-boundary：39.8%
  A/B disagreement：41.6%
  selector-decision：76.4%
  near-failure：100%

说明 active sampler 找到的状态类别具有真实物理风险差异，不只是 Q_safe
数值上的差异。

在此数据上训练 3 个完整独立的 episode-bootstrap Q_safe。每个 member
使用不同的 bootstrap episodes、网络、target、optimizer 和 RNG；fit、
temperature calibration、conformal calibration、validation episodes
完全隔离。

最终 ensemble mean 的 point AUROC 为 0.946，高于 2/3 单独 members；
observed false-safe 从旧 critic A 的 21.4% 降到 0。但：

  同状态 pair ranking accuracy：0.541
  coverage：6.8%
  nominal-relative fall reduction：0

加入 ensemble disagreement gate、mean+std 或 conformal upper bound 后，
coverage 进一步降到 0~2.3%，没有产生 validated replacement fall
reduction。

目前结论：

Q_safe 已经可以较好地区分“某条 transition 最终是否摔倒”，但仍不擅长
在同一个状态下比较多个相近 candidate actions。当前主要瓶颈是
same-state counterfactual action coverage，而不是网络深度、网络数量或
自然 validation AUROC。

当前所有 masking 仍保持 logging-only。下一步需要保证每个自然 episode
至少保存一个 normal/early snapshot，把有 branch 数据的 episode 从当前
53 个提高到 80~100 个，再重新训练 episode-bootstrap ensemble。

只有同时满足：

  selected false-safe <= 5%
  coverage >= 30%
  replacement rate >= 15%
  replacement 本身产生正的 fall reduction
  fallback contribution 不超过总 reduction 的 50%

才进入 frozen-policy SQRL masking rollout；在此之前不继续放宽 epsilon，
也不进入 CSAC barrier 或 FlashSAC。
