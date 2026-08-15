# SQRL-Go2 faithful reproduction

This package is an isolated Go2 adaptation of *Learning to be Safe: Deep RL
with a Safety Critic* (Srinivasan et al., 2020). Nothing under `rl/`, `train/`,
or `runtime/` depends on it. It reuses only the established Go2 transport,
observation, failure, and action-projection infrastructure.

The frozen first experiment is 0.30 m/s pre-training followed by paired 0.40
m/s target fine-tuning. Every target branch receives the exact same pre-trained
actor:

- `sac_transfer`: ordinary SAC behavior and actor loss;
- `sqrl_mask`: Q_safe rejection-sampling behavior, ordinary SAC actor loss;
- `sqrl_full`: the same mask plus the SQRL Lagrangian actor constraint.

Target task critics, task replay, and alpha are fresh. Pre-trained Q_safe is
transferred and frozen for both SQRL branches.

## Development sequence

The three-seed screening froze `N_PRE=25000` and `N_TARGET=10000` in
`config/base.yaml`. Budgets remain mandatory CLI arguments so every run is
explicit and a development invocation cannot silently become a formal run.

Start the simulator and C++ controller first. The recommended command lets the
reproduction runner own the ordered Python runtime lifecycle, avoiding queue
prefill races during CUDA initialization:

```bash
python3 -m reproductions.sqrl_go2.runners.run_pretrain \
  --config reproductions/sqrl_go2/config/pretrain_030.yaml \
  --steps 25000 --seed 0 --device cuda --launch-runtime

python3 -m reproductions.sqrl_go2.runners.run_target \
  --config reproductions/sqrl_go2/config/target_040.yaml \
  --pretrain-checkpoint saved/reproductions/sqrl_go2/seed_0/pretrain_030/final.pt \
  --branch sac_transfer --steps 10000 --seed 0 --device cuda --launch-runtime
```

Repeat the target command with `sqrl_mask` and `sqrl_full`. The development gate
has passed; see [docs/RESULTS.md](docs/RESULTS.md). Formal paired seeds are a
separate protocol and are not claimed by the current results.
Without `--launch-runtime`, start the exact matching runtime only after the
collector process is waiting; this manual mode is intended for debugging.

## Safety/action invariants

- `cost = 1` iff the transition is a true first-fall termination.
- Time-limit truncation has `cost = 0` and task-Q bootstraps across it.
- Q_safe is trained only from complete recent constrained-policy trajectories.
- Q_safe scores the normalized action returned by runtime projection, and the
  adapter fails closed if the runtime executes a different action.
- The first protocol freezes action filtering and slew limiting off, matching
  the current base Go2 configuration, so learner-side differentiable actions
  and executed normalized actions are identical.

See [docs/PAPER_ALIGNMENT.md](docs/PAPER_ALIGNMENT.md) for every paper/Go2
mapping and disclosed approximation.

The locked ten-seed study is specified in
[docs/FORMAL_PROTOCOL.md](docs/FORMAL_PROTOCOL.md). Formal execution owns the
simulator/controller lifecycle, refuses output overwrite, verifies executable
hashes before every run, and publishes aggregate statistics only after all 40
runs are complete.

## Safe policy 位置与使用方法

这里的 safe policy 不是一个单独的 `safe_policy.pt` 文件，而是下面三部分在
推理时组合得到的有限候选近似策略 `bar_pi`：

1. 目标阶段训练完成的 SAC actor，负责生成候选动作；
2. 同一个 seed 在预训练阶段得到并冻结的 `Q_safe`，负责给候选动作预测风险；
3. `SafetyPolicy`，从候选动作中选出第一个 `Q_safe <= epsilon_safe` 的动作；
   如果没有候选通过，就执行预测风险最低的动作。

### 代码位置

- 安全策略/候选动作筛选实现：
  `reproductions/sqrl_go2/algo/safety_policy.py`
- 安全 critic 网络及 checkpoint 加载：
  `reproductions/sqrl_go2/algo/safety_critic.py`
- SAC actor：`reproductions/sqrl_go2/algo/sac.py`
- 运行时组装方式：`reproductions/sqrl_go2/runners/render_comparison.py`
  中的 `_load_branch()`
- 默认参数：`reproductions/sqrl_go2/config/base.yaml`，其中
  `epsilon_safe=0.1`、`mask_candidates=100`

### 正式实验权重位置

有效的正式实验目录是：

```text
saved/reproductions/sqrl_go2/formal_v2/
```

每个 seed（10 到 19）都必须使用同一个 seed 下的一对权重。以 seed 10 的
`sqrl_mask` 分支为例：

```text
# 目标任务 actor（0.40 m/s）
saved/reproductions/sqrl_go2/formal_v2/seed_10/target_040_sqrl_mask/final_sac.pt

# 与该 actor 配套的冻结 Q_safe，保存在 pretrain checkpoint 的 ["safety"] 字段
saved/reproductions/sqrl_go2/formal_v2/seed_10/pretrain_030/final.pt
```

如果使用 `sqrl_full`，把 actor 路径改成：

```text
saved/reproductions/sqrl_go2/formal_v2/seed_10/target_040_sqrl_full/final_sac.pt
```

不要把不同 seed 的 actor 和 Q_safe 混用。`target_040_*/final_sac.pt` 只保存
目标 SAC，不包含冻结的 Q_safe；Q_safe 必须从同 seed 的
`pretrain_030/final.pt` 中加载。`formal_v1` 已失效，不应用于新结果或部署。

### 最简单的使用方式：评估并生成视频

下面的命令会自动加载同一 seed 的三个目标分支；对于两个 SQRL 分支，它会
自动把目标 actor 与预训练 Q_safe 组合成 safe policy：

```bash
python3 -m reproductions.sqrl_go2.runners.render_comparison \
  --config reproductions/sqrl_go2/config/target_040.yaml \
  --checkpoint-root saved/reproductions/sqrl_go2/formal_v2/seed_10 \
  --model /home/xyz/code/unitree_mujoco/unitree_robots/go2/scene_empty.xml \
  --output-dir saved/reproductions/sqrl_go2/formal_v2/seed_10/video_evaluation \
  --episodes 10 \
  --episode-steps 500 \
  --video-frames 1000 \
  --device cuda
```

输出目录会包含三个分支的视频和
`sqrl_go2_target_040_video_evaluation.json`。如果机器没有 CUDA，把
`--device cuda` 改为 `--device cpu`。

### 在 Python 中加载 safe policy

以下示例复现 `_load_branch()` 的关键逻辑：

```python
from pathlib import Path

import torch

from reproductions.sqrl_go2.algo.safety_policy import SafetyPolicy
from reproductions.sqrl_go2.config import load_config
from reproductions.sqrl_go2.runners.common import build_core

seed = 10
device = "cuda"
root = Path("saved/reproductions/sqrl_go2/formal_v2") / f"seed_{seed}"
cfg = load_config("reproductions/sqrl_go2/config/target_040.yaml")
sac, safety, _, _, _ = build_core(cfg, seed=seed, device=device)

# 加载目标任务 actor。
target = torch.load(
    root / "target_040_sqrl_mask/final_sac.pt",
    map_location=sac.device,
    weights_only=False,
)
sac.load_checkpoint(target)
sac.actor.eval()

# 加载同 seed 的预训练 Q_safe；目标阶段按协议冻结它。
pretrain = torch.load(
    root / "pretrain_030/final.pt",
    map_location=sac.device,
    weights_only=False,
)
safety.load_checkpoint(pretrain["safety"], load_optimizer=False)
safety.freeze()
safety.critic.eval()

safe_policy = SafetyPolicy(
    actor=sac.actor,
    safety_critic=safety.critic,
    epsilon=cfg.sqrl.epsilon_safe,
    max_candidates=cfg.sqrl.mask_candidates,
    device=sac.device,
)
```

执行动作时，传入 230 维观测（5 帧 46 维观测拼接）和运行环境的动作预览
函数：

```python
import numpy as np

class Preview:
    def __init__(self, actions):
        projections = env.action_applier.preview_many(
            actions, env.data.qpos[env.qpos_addresses]
        )
        self.requested = np.stack([x.action_requested for x in projections])
        self.critic_actions = np.stack([x.action_executed for x in projections])
        self.q_targets = np.stack([x.action_q_target for x in projections])

selection = safe_policy.select(observation, Preview)
step_result = env.step(selection.requested_action)
```

`selection.risk` 是选中动作的预测风险，`selection.accepted` 表示是否至少有
一个候选通过阈值，`selection.no_safe_candidate` 表示是否触发了“选择最低风险
候选”的回退逻辑。

### 使用边界

这套 safe policy 是论文复现用的 MuJoCo 实验策略，不是已经认证的真机安全
控制器。正式 v2 及后续机制审计没有建立 Q_safe 风险分数的可靠校准，因此不应
仅凭 `Q_safe <= 0.1` 就在真实 Go2 上关闭限位、急停、姿态保护或人工接管。
