# onrobot-safe-rl

这是一个面向 Unitree Go2 行走任务的在线安全强化学习框架。当前架构由三部分组成：
C++ 低层控制器、固定频率 Python runtime、Python 训练客户端，三者通过 socket 和共享内存通信。

默认 Go2 profile 使用 DroQ 训练 **46D observation** 策略。

## 架构

```text
Unitree DDS / MuJoCo bridge
        |
        v
runtime/control/go2      C++ 低层控制器
        |                - 500 Hz 关节 PD 控制
        |                - stand-up / recovery 状态机
        |                - policy target socket
        v
runtime/inference        Python 固定频率 runtime
        |                - 构造 observation
        |                - 执行 safety / reset gate
        |                - 计算在线 reward
        |                - 发布训练 transition
        v
train                    Python 训练客户端
        |                - 从 agent 采样 action
        |                - 写入 replay
        |                - 更新 DroQ 或 FlashSAC
        v
rl                       RL agents、buffers、distributions、工具函数
```

核心约定：

- 关节/action 顺序是 `FR, FL, RR, RL x hip, thigh, calf`。
- policy action 是归一化动作，范围为 `[-1, 1]`。
- runtime 会将 action 映射到绝对关节目标：
  `init_qpos + action * action_offset`。
- observation 里的 `previous_requested_action` 是发送给 controller 的绝对 q target。
- observation 保持 46D：
  `joint_q(12), joint_dq(12), gyro(3), body_velocity(3), quat(4), previous_requested_action(12)`。

## 环境准备

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

编译 Go2 C++ controller：

```bash
cmake -S runtime/control/go2 -B runtime/control/go2/build
cmake --build runtime/control/go2/build
```

C++ controller 依赖 Unitree SDK2 和 `yaml-cpp`。默认会使用 build 目录下复制出来的
`runtime/control/go2/build/go2.yaml`，也可以显式传入 config 路径。

## 运行方法

需要分别在三个终端启动。

1. 启动 C++ controller：

```bash
runtime/control/go2/build/go2_control \
  runtime/control/go2/go2.yaml
```

如果需要 overlay config：

```bash
runtime/control/go2/build/go2_control \
  runtime/control/go2/go2.yaml \
  config/real_robot.yaml
```

2. 启动 Python runtime：

```bash
python -m runtime.inference.runtime --config-profile go2
```

可用 profile：

- `go2`：默认在线 Go2 训练配置。
- `simulation`：仿真实验 overlay。
- `real_robot`：真机 Go2 网络配置，目前为 `domain_id=0`、`interface=eth0`，并假设 sport velocity 已经是 body frame。
- `go2_livesac`：LiveSAC 的异步采集 profile；除 distributional critic 与 reward normalization 外，更新配置沿用 DroQ。

3. 启动训练：

```bash
python -m train
```

默认使用 `config/go2.yaml` 里的 categorical-DroQ 对照实验：复用 DroQ actor 和完整
critic backbone，只把 scalar head 换成 categorical head。也可以不改 YAML，直接用命令行切换 agent：

```bash
python -m train --agent droq
python -m train --agent flashsac
python -m train --config-profile go2_livesac
```

运行 checkpoint 做 deterministic play：

```bash
python -m train --mode play --checkpoint saved/checkpoints_46d/step_00000015000
```

如果不传 `--checkpoint`，play mode 会从 `train.save_dir` 下自动加载最新 checkpoint。

## 当前默认配置

默认 `go2` profile 继承 `config/common.yaml`，并覆盖：

```yaml
train:
  agent: droq
  explore_action_scale: 0.5
  max_steps: 15000
  utd_ratio: 5
  save_dir: saved/checkpoints_46d
```

DroQ 默认参数与之前更容易收敛的配置对齐：

- `num_qs: 5`
- `num_min_qs: 2`
- `actor_q_reduction: min`
- `target_q_min: -100.0`
- `target_q_max: 1000.0`
- terminal transition 在同步和异步模式下都只写入 replay 一次
- `temp_initial_value: 0.1`

reward profile 默认是 `upstream`，对应 `walk_in_the_park` 的 dense run reward：

```text
10 * (tolerance(cos(pitch) * x_velocity) - 0.1 * abs(dyaw))
```

## Safety 与 Episode 统计

runtime 使用 roll/pitch 判断 fallen，acceleration 只作为 upside-down 的辅助信号。
导致翻倒的 policy step 仍然会作为 terminal transition 进入 replay，但不会计入 episode length。

关键指标：

- `policy_step`：本步执行了 policy action，可以进入 replay。
- `count_policy_step`：本步是有效 policy 步，没有 fallen/inverted。
- `training/length`：只统计 `count_policy_step`。
- `training/return`：统计 replay-enabled policy transition 的 reward。
- `env/reset_pose_error`：reset gate 中的 `norm(joint_q - init_qpos)`。
- `env/awaiting_reset_pose`：runtime 正在等待 policy 重新接管。

翻倒之后，runtime 会阻止 policy 立即接管，直到 reset pose 稳定：

```text
norm(joint_q - init_qpos) < reset_joint_tolerance
```

并且需要连续满足 `recovery_stable_steps` 个 runtime step。
runtime 最多等待 `reset_hold_steps`；如果超时且 `abort_on_unstable_reset=true`，
会请求 recovery/stand-up，而不是把不稳定状态交给 policy。

## 配置结构

Python 配置按层合并：

```text
config/common.yaml
config/rewards/<reward_profile>.yaml
config/<profile>.yaml
```

加载入口是 `train.config.load_app_config`。

重要文件：

- `config/common.yaml`：共享 robot、observation、reward、train、DroQ、FlashSAC 默认配置。
- `config/go2.yaml`：默认在线 Go2 profile。
- `config/simulation.yaml`：仿真 overlay。
- `config/real_robot.yaml`：真机 DDS/network overlay。
- `config/rewards/upstream.yaml`：上游 dense reward。
- `config/rewards/baseline.yaml`：带 gate 的诊断 reward。
- `runtime/control/go2/go2.yaml`：C++ controller、stand-up、recovery、PD、socket 配置。

C++ controller config 和 Python config 需要保持一致的字段：

- `domain_id`
- `interface`
- `ipc_socket`
- `state_socket`
- joint order 和 q limits

## 扩展 Agent

Agent 代码位于 `rl/agents`。

添加新 agent 的步骤：

1. 创建 `rl/agents/<name>/agent.py`，实现 config dataclass 和继承 `BaseAgent` 的 agent。
2. 在 `rl/agents/<name>/` 下添加 network/update 代码。
3. 在 `rl/agents/__init__.py` 注册新 agent。

### 50 Hz Q_safe agent

`safe_droq` 是 DroQ 的安全扩展，保留相同 actor、reward critic 和 reward
replay，并增加独立的 failure critic 与 safety replay。

- `safety_mode: logging`：只计算 SAC nominal action 的风险，不生成 32
  个候选，也不修改动作，适合测量 Q_safe 本身且尽量不影响 50 Hz 控制。
- `safety_mode: masking`：一次 batched forward 评估 policy candidates。
  nominal 安全时保持原动作；nominal 不安全且存在安全候选时才替换；没有
  安全候选时 abstain 并保持 nominal。
- Q_safe 在完整 episode 上标记 failure 前 `H` 步，并以 future-failure
  balanced batch 训练。脚本 recovery transition 不会被当成 policy
  failure action。

使用同一份 50 Hz overlay 做公平对比：

```bash
# 原始 DroQ baseline
python -m train --config config/go2_50hz_safe.yaml --agent droq

# Q_safe logging-only
python -m train --config config/go2_50hz_safe.yaml --agent safe_droq
```

启用 masking 前，通过命令行加载已经验证的 critic。这里只迁移 Q_safe，
新 run 的 DroQ actor/reward critic/replay 仍从零开始：

```bash
python -m train --config config/go2_50hz_safe.yaml \
  --agent safe_droq --safety-mode masking \
  --safety-pretrained-path SOURCE/agent/safety_critic.pt \
  --save-dir saved/experiments/go2_50hz_masking
```

三组必须使用不同 `save_dir`。
4. 在 `train/config.py` 添加默认配置。
5. 在 `config/common.yaml` 添加 YAML 配置段。

Agent 输出必须匹配归一化 action space，shape 为 `(12,)`。
如果使用 tanh-squashed Gaussian actor，优先复用
`rl/utils/normalizations.py` 里的 `NormalTanhPolicy`。

## 扩展 Observation

observation 构造集中在：

```text
runtime/inference/observations.py
train/config.py::Go2Config.obs_dim
```

当前 observation 有意保持 46D。如果要改维度：

1. 更新 `build_observation`。
2. 更新 `Go2Config.obs_dim`。
3. 修改 checkpoint save_dir，避免误加载不兼容模型。
4. 更新 README 和相关分析脚本。

不要混用 46D 和非 46D checkpoint。

## 扩展 Reward

reward profile 位于 `config/rewards`。

添加新 reward profile：

1. 新增 `config/rewards/<name>.yaml`。
2. 在 `train/config.py` 中允许该 profile 名称。
3. 在 `runtime/inference/observations.py` 中实现新的 reward 项。
4. 将诊断项放入 reward info dict，方便 wandb 记录。

reward 会直接影响在线安全性。建议保持 dense task reward 和 terminal/fall 逻辑分离，
这样 terminal transition 更容易分析。

## 扩展 Controller Motion

C++ 低层代码位于 `runtime/control/go2`。

- `controller/`：phase machine、policy socket、DDS IO。
- `motions/`：stand-up 和 recovery 动作。
- `lowlevel/`：motor command 填充和低层工具。
- `utils/`：YAML 解析。

修改 C++ 后重新编译：

```bash
cmake --build runtime/control/go2/build
```

每次重新编译后都需要重启 controller。

## 诊断

快速检查当前 profile：

```bash
python - <<'PY'
from train.config import load_app_config
robot_cfg, train_cfg, agent_cfg = load_app_config(profile='go2')
print('obs_dim', robot_cfg.obs_dim)
print('agent', train_cfg.agent, agent_cfg.agent_type)
print('utd_ratio', train_cfg.utd_ratio)
print('terminal_replay_repeats', train_cfg.terminal_replay_repeats)
PY
```

如果存在 `tools/check_go2_mjcf_order.py`，可以检查 Go2 MJCF actuator/sensor 顺序：

```bash
python tools/check_go2_mjcf_order.py
```

行为不稳定时，优先看这些 W&B 指标：

- `env/fallen`, `env/inverted`
- `env/safety_roll`, `env/safety_pitch`
- `env/reset_pose_error`, `env/awaiting_reset_pose`
- `env/count_policy_step`
- `actor/log_std_mean`, `actor/action_saturation`
- `critic/q_max`, `critic/target_q_max`
- per-leg action 和 q-target 指标

代码或配置变更后，需要重启 C++ controller 和 Python runtime。
只重启 training client 不能更新已经加载在 runtime/controller 进程里的逻辑。
