# onrobot-safe-rl

Online safe RL stack for Unitree Go2 locomotion. The current architecture uses a
standalone C++ low-level controller, a fixed-rate Python runtime, and a Python
training client connected through shared memory.

The default Go2 profile trains a 46D observation policy with DroQ.

## Architecture

```text
Unitree DDS / MuJoCo bridge
        |
        v
runtime/control/go2      C++ low-level controller
        |                - 500 Hz joint PD commands
        |                - stand-up / recovery FSM
        |                - policy target socket
        v
runtime/inference        Python fixed-rate runtime
        |                - builds observations
        |                - applies safety / reset gates
        |                - computes online reward
        |                - publishes train transitions
        v
train                    Python training client
        |                - samples actions from agent
        |                - inserts replay transitions
        |                - updates DroQ or FlashSAC
        v
rl                       RL agents, buffers, distributions, utilities
```

Key runtime contract:

- Joint/action order is `FR, FL, RR, RL x hip, thigh, calf`.
- Policy actions are normalized in `[-1, 1]`.
- Runtime maps actions to absolute q targets with `init_qpos + action * action_offset`.
- `previous_requested_action` in the observation is the absolute q target sent to the controller.
- Observation remains 46D: `joint_q(12), joint_dq(12), gyro(3), body_velocity(3), quat(4), previous_requested_action(12)`.

## Setup

Install Python dependencies in the environment used for training:

```bash
pip install -r requirements.txt
```

Build the Go2 controller:

```bash
cmake -S runtime/control/go2 -B runtime/control/go2/build
cmake --build runtime/control/go2/build
```

The C++ controller requires Unitree SDK2 and `yaml-cpp`. It defaults to the copied
`runtime/control/go2/build/go2.yaml`, but can also accept explicit config paths.

## Running

Start these in separate terminals.

1. Start the C++ controller:

```bash
runtime/control/go2/build/go2_control \
  runtime/control/go2/go2.yaml
```

For a config overlay:

```bash
runtime/control/go2/build/go2_control \
  runtime/control/go2/go2.yaml \
  config/real_robot.yaml
```

2. Start the Python runtime:

```bash
python -m runtime.inference.runtime --config-profile go2
```

Profiles:

- `go2`: default online Go2 training profile.
- `simulation`: simulation experiment overlay.
- `real_robot`: real Go2 network profile, currently `domain_id=0`, `interface=eth0`, body-frame sport velocity.

3. Start training:

```bash
python -m train
```

Default training uses DroQ from `config/go2.yaml`. To override the agent without
editing YAML:

```bash
python -m train --agent droq
python -m train --agent flashsac
```

Run a deterministic policy checkpoint:

```bash
python -m train --mode play --checkpoint saved/checkpoints_46d/step_00000015000
```

If no checkpoint is passed, play mode loads the latest checkpoint under
`train.save_dir`.

## Current Defaults

The default `go2` profile inherits `config/common.yaml` and overrides:

```yaml
train:
  agent: droq
  explore_action_scale: 0.5
  max_steps: 15000
  utd_ratio: 5
  save_dir: saved/checkpoints_46d
```

DroQ settings are aligned with the better-converging setup:

- `num_qs: 5`
- `num_min_qs: 2`
- `actor_q_reduction: min`
- `target_q_min: -100.0`
- `target_q_max: 1000.0`
- `terminal_replay_repeats: 4`
- `temp_initial_value: 0.1`

The reward profile defaults to `upstream`, matching `walk_in_the_park` dense run
reward:

```text
10 * (tolerance(cos(pitch) * x_velocity) - 0.1 * abs(dyaw))
```

## Safety And Episode Accounting

The runtime uses roll/pitch for fallen detection and uses acceleration only as
an auxiliary upside-down signal. A policy step that causes a fall still enters
replay as a terminal transition, but it does not count toward episode length.

Metrics:

- `policy_step`: a policy action was executed and may be inserted into replay.
- `count_policy_step`: a valid non-fallen, non-inverted policy step.
- `training/length`: counts only `count_policy_step`.
- `training/return`: sums rewards for replay-enabled policy transitions.
- `env/reset_pose_error`: `norm(joint_q - init_qpos)` during reset gating.
- `env/awaiting_reset_pose`: runtime is waiting before policy can take over.

After a fall, policy takeover is blocked until reset pose is stable:

```text
norm(joint_q - init_qpos) < reset_joint_tolerance
```

for `recovery_stable_steps` consecutive runtime steps. The runtime waits up to
`reset_hold_steps`; if it times out with `abort_on_unstable_reset=true`, it
requests recovery/stand-up instead of letting an unstable state enter policy.

## Configuration

Python configs are layered:

```text
config/common.yaml
config/rewards/<reward_profile>.yaml
config/<profile>.yaml
```

The selected profile is loaded by `train.config.load_app_config`.

Important files:

- `config/common.yaml`: shared robot, observation, reward, train, DroQ, FlashSAC defaults.
- `config/go2.yaml`: default online Go2 profile.
- `config/simulation.yaml`: simulation overlay.
- `config/real_robot.yaml`: real robot DDS/network overlay.
- `config/rewards/upstream.yaml`: upstream dense reward.
- `config/rewards/baseline.yaml`: gated diagnostic reward.
- `runtime/control/go2/go2.yaml`: C++ controller, stand-up, recovery, PD, socket settings.

Keep the C++ controller config and Python config consistent for:

- `domain_id`
- `interface`
- `ipc_socket`
- `state_socket`
- joint order and q limits

## Extending Agents

Agents live under `rl/agents`.

To add an agent:

1. Create `rl/agents/<name>/agent.py` with a config dataclass and an agent class
   implementing `BaseAgent`.
2. Add networks/update code under `rl/agents/<name>/`.
3. Register it in `rl/agents/__init__.py`.

### 50 Hz Q_safe agent

`safe_droq` preserves the DroQ actor, reward critics, and reward replay while
adding an episode-aware safety replay and failure critic. In `logging` mode it
scores only the exact nominal action and never changes control. In `masking`
mode it evaluates candidates in one batch, retains a safe nominal action, and
replaces it only when a safe alternative exists; an empty safe set abstains.

Use one overlay for a frequency-matched comparison:

```bash
python -m train --config config/go2_50hz_safe.yaml --agent droq
python -m train --config config/go2_50hz_safe.yaml --agent safe_droq
```

To transfer only a validated Q_safe into a fresh DroQ run:

```bash
python -m train --config config/go2_50hz_safe.yaml \
  --agent safe_droq --safety-mode masking \
  --safety-pretrained-path SOURCE/agent/safety_critic.pt \
  --save-dir saved/experiments/go2_50hz_masking
```
4. Add defaults to `train/config.py`.
5. Add a YAML section under `config/common.yaml`.

Agent actions must match the normalized action space and output shape `(12,)`.
If the agent uses a tanh-squashed Gaussian actor, prefer the shared
`NormalTanhPolicy` in `rl/utils/normalizations.py`.

## Extending Observations

Observation construction is centralized in:

```text
runtime/inference/observations.py
train/config.py::Go2Config.obs_dim
```

Current observation is intentionally 46D. If changing it:

1. Update `build_observation`.
2. Update `Go2Config.obs_dim`.
3. Update any checkpoint save directory to avoid loading incompatible models.
4. Update README and downstream analysis scripts.

Do not silently mix 46D and non-46D checkpoints.

## Extending Rewards

Reward profiles live under `config/rewards`.

To add a profile:

1. Add `config/rewards/<name>.yaml`.
2. Allow the name in `train/config.py`.
3. Implement any new reward terms in `runtime/inference/observations.py`.
4. Log diagnostic terms through the reward info dict.

Reward changes directly affect online safety. Keep fall penalties and terminal
logic separate from dense task reward so terminal transitions remain easy to
reason about.

## Extending Controller Motions

C++ low-level code lives under `runtime/control/go2`.

- `controller/`: phase machine, policy socket, DDS IO.
- `motions/`: stand-up and recovery motion primitives.
- `lowlevel/`: motor command filling and low-level helpers.
- `utils/`: YAML parsing.

After changing C++ code:

```bash
cmake --build runtime/control/go2/build
```

Restart the controller after every rebuild.

## Diagnostics

Useful quick checks:

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

Check Go2 MJCF actuator/sensor ordering if `tools/check_go2_mjcf_order.py`
exists:

```bash
python tools/check_go2_mjcf_order.py
```

When behavior looks unstable, first inspect these W&B metrics:

- `env/fallen`, `env/inverted`
- `env/safety_roll`, `env/safety_pitch`
- `env/reset_pose_error`, `env/awaiting_reset_pose`
- `env/count_policy_step`
- `actor/log_std_mean`, `actor/action_saturation`
- `critic/q_max`, `critic/target_q_max`
- per-leg action and q-target metrics

Restart the C++ controller and Python runtime after code or config changes. The
training client alone cannot update logic already loaded in a running runtime.
