# online_safe_sac

Go2 四足机器人在线 walk 训练的 sim-to-real 栈。策略在 Python/JAX 中训练，底层关节控制在 C++ controller 中执行，仿真或真机通过 Unitree DDS 提供状态反馈。

训练 MDP 对齐 [A Walk in the Park](https://github.com/ikostrikov/walk_in_the_park)（DroQ、run reward、绝对关节目标 + Butterworth 低通、UTD=20）。

## 架构

```
python -m train --mode in_process
  │
  ├─ collector/  actor 推理、transition 构造
  ├─ learner/    replay、DroQ update、完整 checkpoint
  └─ common/     protocol、transition、config schema
        │
        │ Unix socket (/tmp/go2_policy.sock)
        ▼
controller/go2_control
  ├─ realtime/   500 Hz loop、policy scheduler、state snapshot
  ├─ transport/  DDS/IPC/protocol
  ├─ command/    映射、限幅、滤波、PD command
  └─ safety/     supervisor、standup/recovery
```

| 组件 | 职责 |
|------|------|
| `collector/` | actor 推理、legacy env adapter、transition 构造 |
| `learner/` | replay、DroQ update、checkpoint、训练编排 |
| `common/` | protocol、transition、配置 schema |
| `train/` | 兼容 CLI 入口；默认调用 in-process collector + learner |
| `controller/` | realtime/transport/command/safety 四层 C++ 控制器 |
| `jaxrl/` | JAX/Flax SAC（DroQ）算法实现 |
| `mjcf/` | 本地 Go2 MuJoCo 模型资源 |

## 论文对齐项

| 项 | walk_in_the_park | 本仓库 |
|----|------------------|--------|
| 算法 | DroQ | `jaxrl/agents/sac/droq` ✓ |
| Reward | `10 * (tolerance(cos_pitch*x_vel) - 0.1*|dyaw|)` | `train/obs.py` ✓ |
| Action | `init_qpos + action * [0.2,0.4,0.4]×4` | `config/go2.yaml` ✓ |
| 执行 | 绝对目标 + Butterworth @ 4Hz | 默认诊断基线关闭 filter；`train/action_filter.py` 可选 |
| UTD | 20 | `train.utd_ratio` ✓ |
| 探索 | 1000 步 | `train.start_training` ✓ |
| 控制 | 20 Hz 同步循环 | P2 结构由 C++ scheduler 接管；legacy path 在 env.step 内等待 |

## 目录结构

```
config/              common + simulation/real_robot + rewards

controller/          C++ 底层控制器 (go2_control)
  main.cpp           加载配置并启动 controller
  realtime/          controller、policy_scheduler、state_snapshot
  transport/         ipc_server、policy_protocol、config_loader、dds_bridge
  command/           lowlevel_commander、action_mapper、action_filter
  safety/            motion_service、safety_supervisor、imu_utils

collector/           actor runner、legacy env adapter、transition builder
learner/             learner orchestration、checkpoint、UTD credit
common/              protocol、transition、config schema
train/               CLI 兼容入口和 legacy wrappers
jaxrl/               RL 算法库
  agents/sac/droq/   DroQLearner
  data/              ReplayBuffer
  env/               prepare_env、BoxSpec、evaluate
```

## 依赖

### Python

```bash
pip install -r requirements.txt
```

主要包：`jax[cuda12]`（或按 GPU 选 cuda11）、`flax`、`optax`、`numpy`、`pyyaml`、`scipy`、`tqdm`、`wandb`、`unitree_sdk2py`。

建议设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false`（`train/main.py` 已默认设置）以减少 JAX 显存预占。

### C++（controller）

- unitree_sdk2（建议安装于 `/opt/unitree_robotics`）
- yaml-cpp
- CMake ≥ 3.16，C++17

### 外部程序

- [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)：仿真
- Go2 真机 + unitree_sdk2 网络环境

## 编译 controller

```bash
cd controller
mkdir -p build && cd build
cmake ..
make -j4
```

产物：`controller/build/go2_control`

## 运行

当前入口仍是 `config/go2.yaml`。新增的 `config/common.yaml`、`config/simulation.yaml`、`config/real_robot.yaml` 用于后续拆分配置；真机速度默认应使用 body frame，即 `sport_velocity_world_frame: false`。

### 1. 启动 unitree_mujoco（仿真）

```bash
cd /path/to/unitree_mujoco/simulate/build
./unitree_mujoco -r go2 -s scene.xml -i 1 -n lo
```

### 2. 启动 controller

```bash
cd /path/to/online_safe_sac
./controller/build/go2_control
```

从仓库根目录运行，自动读取 `config/go2.yaml`。

### 3. 启动训练

```bash
conda activate oss
cd /path/to/online_safe_sac
python -m train --mode in_process --config-profile go2
```

`--config-profile simulation` 使用 `config/common.yaml + config/simulation.yaml`；`--config-profile real_robot` 使用 `config/common.yaml + config/real_robot.yaml`。`go2` 保留旧兼容文件。

`--mode split` 是 P2 分执行单元入口；在 framed controller protocol 全链路启用前，它会复用 in-process 兼容路径。

### 性能基准

在 `config/go2.yaml` 中设置：

```yaml
train:
  benchmark_only: true
  benchmark_steps: 200
  profile: true
  save_checkpoints: false
```

然后 `python -m train`。

验收：`timing/effective_hz ≥ 19`，`timing/update_ms < 40`（在 UTD=20 时）。

### 真机

修改 `config/go2.yaml`：

```yaml
domain_id: 0
interface: eth0
```

然后同样运行 `./controller/build/go2_control` 与 `python -m train`。

真机安全限幅：在 `train:` 下设置 `max_joint_delta: 0.05`。

## 收敛验收

对齐后约 20k 步应观察到：

1. `training/return` 上升，`env/x_velocity` 出现持续正向分量
2. `training/critic_loss` 不再间歇性 >100；`training/q` 尺度约 0–10（非 ~180）
3. wandb 可查看 `training/return`、`training/critic_loss`、`env/x_velocity`

**注意**：改 reward / action MDP 后旧 checkpoint 不兼容。当前 online training 只允许从 agent + replay 的完整 snapshot 恢复；旧 agent-only checkpoint 会被拒绝。

## 配置说明（`config/go2.yaml`）

| 段 / 字段 | 说明 |
|-----------|------|
| `domain_id`, `interface` | DDS（仿真 1/lo，真机 0/eth0） |
| `init_qpos`, `action_offset` | 站立姿态与动作映射 |
| `fallen_risk_rad` | 倾斜超此角（rad）→ train 触发 standup |
| `success_orientation_rad` | 超过此角（rad）→ episode 终止 |
| `train.max_episode_steps` | 每个 episode 的有效 policy 步数（standup 不计入） |
| `train.reset_grace_steps` | episode 开始后多少 policy 步内不触发 standup |
| `train.reset_hold_steps` | episode reset 时等待 standup 的最长步数 |
| `train.recovery_stable_steps` | standup 后需连续稳定步数才恢复 policy |
| `train.explore_action_scale` | 探索阶段随机动作幅度（0.2 较稳） |
| `train.utd_ratio` | 每步 critic 更新次数 |
| `train.droq` | DroQ 算法超参 |

改完配置后重启 controller 与 train 即可。

## IPC / Protocol

P2 协议固定为 length-prefixed frames，并保留三类语义帧：

- `ObservationFrame(seq, observation, freshness, controller_mode)`
- `ActionFrame(seq, requested_action, policy_version)`
- `TransitionFrame(seq, next_observation, requested_action, projected_action, executed_q_target, timestamps, safety_flags, termination_reason)`

当前 legacy IPC 仍可用于 in-process 兼容路径；新代码应优先依赖 `common.protocol` 和 `controller/transport/policy_protocol.hpp`。

## 故障排查

| 现象 | 检查项 |
|------|--------|
| `Timed out waiting for LowState` | unitree_mujoco 是否运行；domain_id / interface 是否一致 |
| socket connect 失败 | controller 是否已启动；ipc_socket 路径是否一致 |
| return 高但不动 | 检查是否误开 `--max_joint_delta`；确认 `action_offset` 为论文值 |
| `q` ~180、critic_loss 暴涨 | 多为旧 MDP checkpoint；新 run + 对齐 reward |
| effective_hz < 19 | 降低 `utd_ratio` 或检查 GPU；看 `--profile` 的 `update_ms` |
