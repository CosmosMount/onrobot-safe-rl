# onrobot-safe-rl

Go2 四足机器人在线 walk 训练栈。当前仓库采用 P2 结构：C++ controller 负责 500 Hz 底层控制与安全执行边界，Python/JAX 负责 actor 推理、transition 构造、replay、DroQ 更新和 checkpoint。

当前默认运行路径仍是兼容的 `in_process` 模式：collector 与 learner 在同一 Python 进程中运行，底层通过 Unix socket 向 C++ controller 发送命令。`split` 模式入口已经保留，但在 framed controller protocol 全链路启用前会复用 `in_process` 路径。

## 当前架构

```text
python -m train --mode in_process --config config/go2.yaml
  |
  |-- train/       CLI、collector、env adapter、replay/update 编排、checkpoint
  |   |-- collector/ actor runner、legacy env adapter、transition builder
  |   |-- artifacts/ 训练诊断产物
  |-- rl/          DroQ / FlashSAC 算法包和 backend 协议
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
| `train/` | Python 训练入口、Go2 gym env、collector、learner 编排、checkpoint、日志 |
| `train/collector/` | Python collector 侧逻辑；当前包含 legacy env adapter |
| `train/artifacts/` | 训练诊断产物，不作为可 import 源码模块 |
| `common/` | Python 协议和 transition |
| `rl/droq/` | walk_in_the_park 风格 DroQ/SAC 算法实现 |
| `rl/flashsac/` | FlashSAC 风格 PyTorch 算法实现 |
| `config/` | 单一 `go2.yaml`，同时供 Python train 和 C++ controller 使用 |
| `mjcf/` | Go2 MuJoCo 模型资源 |

## 关键运行状态

- `python -m train --mode in_process`：当前推荐入口，collector 和 learner 同进程。
- `python -m train --mode split`：分执行单元入口已保留，目前会打印提示并复用 `in_process`。
- `config/go2.yaml`：唯一配置入口，controller 和 Python train 默认都读取它。
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

以下命令均从仓库根目录执行；根目录本身没有 `CMakeLists.txt`，
controller 的 CMake 工程位于 `controller/`：

```bash
cd /path/to/onrobot-safe-rl
cmake -S controller -B controller/build
cmake --build controller/build -j4
```

生成文件：

```text
controller/build/go2_control
```

controller 支持可选配置路径参数：

```bash
cd /path/to/onrobot-safe-rl
./controller/build/go2_control config/go2.yaml
```

如果从 `controller/build` 目录启动，等价命令为：

```bash
cd /path/to/onrobot-safe-rl/controller/build
./go2_control ../../config/go2.yaml
```

## 仿真部署

仿真训练需要三个终端，依次启动 unitree_mujoco、controller 和 Python
训练进程。三个进程需同时保持运行。

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

启动训练：

```bash
cd /path/to/onrobot-safe-rl
python -m train --mode in_process --config config/go2.yaml
```

## 真机部署

真机也只改 `config/go2.yaml`。启动前至少确认：

```yaml
domain_id: 0
interface: eth0
sport_velocity_world_frame: false
train:
  max_joint_delta: 0.05
```

启动训练：

```bash
python -m train --mode in_process --config config/go2.yaml
```

controller 和 Python train 读取同一个 YAML，因此仿真/真机切换时只需要维护 `config/go2.yaml`。

## 配置说明

| 文件 | 用途 |
| --- | --- |
| `config/go2.yaml` | 唯一配置文件；controller 与 Python train 共用 |

常用字段：

| 字段 | 说明 |
| --- | --- |
| `domain_id`, `interface` | DDS domain 和网络接口 |
| `sport_velocity_world_frame` | 仿真通常为 world frame，真机通常为 body frame |
| `sport_state_max_age_ms` | sport velocity 最大允许 age，过期会拒绝训练 |
| `init_qpos`, `action_offset` | nominal pose 与 action 映射 |
| `reward_min_forward_vel` | idle reward gate；设为 `null` 可恢复更接近上游的 dense reward |
| `reward_upright_min_cos` | upright reward gate；设为 `-1.0` 可关闭 |
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
train/checkpoint.py
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
python -m train --mode in_process --config config/go2.yaml
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
