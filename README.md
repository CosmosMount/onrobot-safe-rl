# online_safe_sac

Go2 四足机器人在线 walk 训练栈。当前仓库采用 P2 结构：C++ controller 负责 500 Hz 底层控制与安全执行边界，Python/JAX 负责 actor 推理、transition 构造、replay、DroQ 更新和 checkpoint。

当前默认运行路径仍是兼容的 `in_process` 模式：collector 与 learner 在同一 Python 进程中运行，底层通过 Unix socket 向 C++ controller 发送命令。`split` 模式入口已经保留，但在 framed controller protocol 全链路启用前会复用 `in_process` 路径。

## 当前架构

```text
python -m train --mode in_process --config-profile go2
  |
  |-- train/       CLI 兼容入口
  |-- collector/   actor runner、legacy env adapter、transition builder
  |-- learner/     replay、DroQ update、完整 checkpoint、训练编排
  |-- common/      protocol、transition、config schema
        |
        | legacy Unix socket: /tmp/go2_policy.sock
        v
controller/build/go2_control
  |-- realtime/    controller、policy scheduler、state snapshot/synchronizer
  |-- transport/   IPC server、policy protocol、config loader、DDS bridge schema
  |-- command/     action mapping、projection、filter、PD command
  |-- safety/      supervisor schema、standup/recovery、IMU helpers
        |
        v
Unitree DDS: rt/lowcmd, rt/lowstate, rt/sportmodestate
```

## 目录说明

| 目录 | 说明 |
| --- | --- |
| `controller/` | C++ 控制器，按 `realtime/transport/command/safety` 分层 |
| `collector/` | Python collector 侧逻辑；当前包含 legacy env adapter |
| `learner/` | Python/JAX learner 侧逻辑、checkpoint、UTD credit |
| `common/` | Python 协议、transition、配置合并 schema |
| `train/` | 兼容 CLI 与 legacy wrappers；不再作为长期核心架构中心 |
| `jaxrl/` | DroQ/SAC 算法实现 |
| `config/` | `common.yaml` + profile overlay + reward presets |
| `mjcf/` | Go2 MuJoCo 模型资源 |

## 关键运行状态

- `python -m train --mode in_process`：当前推荐入口，collector 和 learner 同进程。
- `python -m train --mode split`：P2 分执行单元入口已保留，目前会打印提示并复用 `in_process`。
- `config/go2.yaml`：兼容入口，controller 默认读取它。
- `config/common.yaml + config/simulation.yaml`：推荐的仿真 profile。
- `config/common.yaml + config/real_robot.yaml`：推荐的真机 profile，默认 `sport_velocity_world_frame: false`。
- checkpoint 只支持 agent + replay 的完整 snapshot；旧 agent-only checkpoint 不兼容。
- observation 已包含 `previous_requested_action` 和 `previous_executed_action`，旧 checkpoint/replay 默认不兼容。

## 依赖

### Python

```bash
pip install -r requirements.txt
```

主要依赖包括 `jax`、`flax`、`optax`、`numpy`、`pyyaml`、`scipy`、`tqdm`、`wandb`、`unitree_sdk2py`。

`train/main.py` 默认设置：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

### C++ controller

- unitree_sdk2，默认 CMake prefix 为 `/opt/unitree_robotics/lib/cmake`
- yaml-cpp
- CMake >= 3.16
- C++17

## 编译 controller

```bash
cd controller
mkdir -p build
cd build
cmake ..
make -j4
```

生成文件：

```text
controller/build/go2_control
```

controller 支持可选配置路径参数：

```bash
cd controller/build
./go2_control ../../config/go2.yaml
```

不传参数时默认读取：

```text
../../config/go2.yaml
```

## 仿真部署

### 1. 启动 unitree_mujoco

```bash
cd /path/to/unitree_mujoco/simulate/build
./unitree_mujoco -r go2 -s scene.xml -i 1 -n lo
```

仿真 DDS 默认：

```yaml
domain_id: 1
interface: lo
sport_velocity_world_frame: true
```

### 2. 启动 controller

```bash
cd /path/to/onrobot-safe-rl
./controller/build/go2_control config/go2.yaml
```

如果从 `controller/build` 目录运行，则使用 `../../config/go2.yaml`。若从仓库根目录运行，则使用 `config/go2.yaml`。

### 3. 启动训练

兼容配置：

```bash
cd /path/to/onrobot-safe-rl
python -m train --mode in_process --config-profile go2
```

拆分配置：

```bash
python -m train --mode in_process --config-profile simulation
```

## 真机部署

真机 profile 默认覆盖：

```yaml
domain_id: 0
interface: eth0
sport_velocity_world_frame: false
train:
  max_joint_delta: 0.05
```

推荐启动训练：

```bash
python -m train --mode in_process --config-profile real_robot
```

controller 当前仍直接读取单个 YAML 文件。若使用 split profile，需要先生成或维护对应的 controller YAML；最稳妥的兼容方式仍是将 `config/go2.yaml` 中的 `domain_id/interface/sport_velocity_world_frame/train.max_joint_delta` 改成真机值后启动 controller。

## 配置说明

| 文件 | 用途 |
| --- | --- |
| `config/go2.yaml` | 当前 controller 与 Python 的兼容单文件配置 |
| `config/common.yaml` | P2 共享配置，包含机器人、recovery、train、DroQ 默认值 |
| `config/simulation.yaml` | 仿真 DDS/profile 覆盖 |
| `config/real_robot.yaml` | 真机 DDS/profile 覆盖 |
| `config/rewards/baseline.yaml` | 诊断基线，启用 idle velocity gate |
| `config/rewards/upstream.yaml` | 上游 dense reward，`reward_min_forward_vel: null` |

常用字段：

| 字段 | 说明 |
| --- | --- |
| `domain_id`, `interface` | DDS domain 和网络接口 |
| `sport_velocity_world_frame` | 仿真通常为 world frame，真机通常为 body frame |
| `sport_state_max_age_ms` | sport velocity 最大允许 age，过期会拒绝训练 |
| `init_qpos`, `action_offset` | nominal pose 与 action 映射 |
| `reward_min_forward_vel` | idle reward gate；诊断基线为 `0.04` |
| `train.use_action_filter` | 当前诊断基线默认关闭 Python action filter |
| `train.start_training` | 从随机探索切换到策略采样的步数 |
| `train.utd_ratio` | DroQ 更新比例 |
| `train.save_dir` | 完整训练 snapshot 保存目录 |

## Protocol 状态

P2 目标协议固定为 length-prefixed frames：

- `ObservationFrame(seq, observation, freshness, controller_mode)`
- `ActionFrame(seq, requested_action, policy_version)`
- `TransitionFrame(seq, next_observation, requested_action, projected_action, executed_q_target, timestamps, safety_flags, termination_reason)`

Python 侧 schema 在：

```text
common/protocol.py
common/transition.py
```

C++ 侧 schema 在：

```text
controller/transport/policy_protocol.hpp
```

当前 controller 主路径仍保留 legacy IPC，用于兼容 `in_process` 训练。后续启用 framed protocol 时，应让 C++ realtime scheduler 产生 observation/transition frame，Python collector 只响应 sequence-matched action。

## Checkpoint

当前 online training 不再恢复旧 Flax agent-only checkpoint。保存和恢复均使用完整训练 snapshot：

```text
agent
replay_buffer
step
metadata
```

实现位置：

```text
learner/checkpoint.py
```

如果修改 reward、observation、action mapping 或 protocol，请新建 run 并删除旧 checkpoint 目录。

## 性能基准

在配置中设置：

```yaml
train:
  benchmark_only: true
  benchmark_steps: 200
  profile: true
  save_checkpoints: false
```

启动：

```bash
python -m train --mode in_process --config-profile go2
```

参考指标：

- `timing/effective_hz >= 19`
- `timing/update_ms < 40`，UTD=20 时
- `env/x_velocity` 应出现持续正向分量
- `training/critic_loss` 不应长期爆炸

## 故障排查

| 现象 | 检查项 |
| --- | --- |
| `Timed out waiting for LowState` | unitree_mujoco/controller/DDS domain/interface 是否一致 |
| sport velocity stale | `rt/sportmodestate` 是否发布，`sport_state_max_age_ms` 是否过小 |
| socket connect 失败 | controller 是否已启动，`ipc_socket` 是否一致 |
| return 高但不走 | 检查 velocity frame、idle reward gate、action offset、是否加载旧 checkpoint |
| 恢复 checkpoint 被拒绝 | 删除旧 agent-only checkpoint，使用完整 snapshot 新 run |
| `effective_hz < 19` | 检查 GPU/update_ms，必要时降低 UTD 或改 split collector/learner |

## 当前限制

- `split` 模式尚未启用独立 collector/learner 进程通信。
- C++ framed protocol schema 已存在，但 controller 主路径仍使用 legacy IPC。
- action filter 已在 C++ command 层有结构位置，完整二阶 Butterworth 迁移仍需后续接入。
- `train/env.py` 是 legacy adapter，不应继续扩展为长期核心模块。
