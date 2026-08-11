# Target-aligned natural-PPO capacity result

**Decision:** pass; select 2,048-world capacity and use 2,000 worlds for exact
30M PPO exposure geometry.

The capacity ladder and selected-tier stability run used the target-aligned
Go2 flat task at constant `+0.30 m/s`, with no push, impulse, velocity injection
or recovery motion.  These runs measured MuJoCo-Warp simulation, target
observation construction, vector reset and official-size PPO actor inference.
They did not perform PPO optimizer updates; a fresh training/capture pilot is
therefore still required before the 30M run.

| Worlds | Measured time | Throughput (env-steps/s) | Gain | Peak total VRAM |
|---:|---:|---:|---:|---:|
| 256 | 309.94 s | 16,519 | — | 1,477 MiB |
| 512 | 317.56 s | 28,215 | 70.8% | 1,543 MiB |
| 1,024 | 313.05 s | 44,159 | 56.5% | 1,737 MiB |
| 2,048 | 317.71 s | 58,016 | 31.4% | 2,221 MiB |

Every upgrade exceeded the frozen 15% throughput-gain threshold.  The 2,048
tier was therefore selected.  Its stability run lasted 1,839.37 seconds and
processed 106,496,000 environment steps at 57,898 env-steps/s.  Mean GPU
utilization was 93.68%, peak total VRAM was 2,338 MiB, measured memory growth
was 0 MiB, and there was no OOM, nonfinite state, external force, push event,
GPU sampling error or kernel failure.

Production uses 2,000 rather than 2,048 worlds because `2,000 × 125 = 250,000`
environment steps per PPO iteration.  All registered checkpoints at
1M/2M/5M/10M/20M/30M then land on exact iteration boundaries while remaining
below the validated 2,048-world capacity.

The immutable authorization is
`saved/qsafe_development/natural_ppo/capacity-030-target-aligned/capacity-authorization-v1.json`.
It binds the four five-minute reports, the 30-minute report, generator commit
`398003174a8e311522fc88cacf8b8e4715edefc7`, target-alignment contract
`96d19c0c0898efc6e39d936f874104c6b0665957c8dec0c3dfe035d2a081ef0d`, and
MuJoCo/MuJoCo-Warp/Warp versions `3.5.0/3.5.0/1.12.0`.

This is engineering authorization for PPO training, not evidence that Q_safe
reduces SAC falls.  Objective 1 remains open and Objective 2 remains blocked.
