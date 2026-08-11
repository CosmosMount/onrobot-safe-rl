# Natural-PPO direct Q_safe pilot result

**Date:** 2026-08-11 (Asia/Shanghai)  
**Disposition:** development gate passed; not Objective-1 evidence

## Frozen pilot task

- Unitree Go2 Flat, MuJoCo-Warp 3.5.0 / Warp 1.12.0.
- 2,000 parallel environments, RSL-RL clipped PPO from zero.
- Constant command `vx=+0.30 m/s`, `vy=0`, `yaw_rate=0`.
- No push event, external impulse, or artificial velocity injection.
- Ordinary friction, encoder-bias, and base-COM randomization remains enabled.
- Exactly 1,000,000 policy-environment steps; this is a development pilot, not
  the registered 30M production exposure.

The clean collection generator was `dc85aaf2992798b0f2354daac8ff675f14b8384f`.
The run took 12.408 seconds and produced the exact 1M checkpoint
`model_3.pt` (`296fc21c...1372`).

## Natural outcomes and direct supervision

The collector recorded 388 independent target-predicate fall episodes. Each
fall was counted once immediately before vector reset. The archive contained
2,536 available registered pre-fall states and 4,238 delayed normal candidates.
No non-zero external force was admitted.

Whole PPO episodes were assigned to immutable roles before matching. Matching
was restricted to the same role, checkpoint-age bucket, and coarse
randomization stratum:

| Role | Matched pairs | Compiled samples |
|---|---:|---:|
| Fit | 1,764 | 3,528 |
| Calibration | 348 | 696 |
| Held-out PPO test | 424 | 848 |
| Total | 2,536 | 5,072 |

Each pair contains one pre-fall positive and one normal negative. The positive
label means a fall occurs within H96 under the action actually executed by PPO;
the negative survived at least 96 future policy steps. These labels directly
train the state-risk and executed-action-risk heads. They do not claim an
outcome for an unexecuted recovery action.

The deterministic compiled dataset SHA-256 is
`332c3a3e064c3d75cacbc465ff04fd6944c6c2daffaa6c0d1e35bbb7234b91cd`.
The clean compiler/trainer commit was
`dee3deef368132c8b5e45099374740c43d391550`.

## Held-out PPO result

A five-member pointwise Q_safe ensemble was fitted only on the Fit role and
temperature-calibrated only on the Calibration role. The held-out PPO test was
not used for model or threshold selection.

| Metric | State-risk head | Executed-action-risk head |
|---|---:|---:|
| AUROC | 0.9536 | 0.9492 |
| Matched-pair accuracy | 0.9410 | 0.9458 |
| Accuracy at 0.5 | 0.9092 | 0.9127 |
| 10-bin ECE | 0.0664 | 0.0595 |

This passes the development question “can natural PPO fall/normal states
directly train a learnable risk signal?” on an episode-disjoint held-out PPO
set. It does **not** prove that the model reduces SAC falls. The protected
SAC-only Model-Test, paired closed-loop comparison, and 24-seed three-arm
SAC-from-zero experiment have not been consumed; therefore
`objective1_pass=false` and Objective 2 remains forbidden.

## Visualization

The six-second, no-push, `+0.30 m/s` pilot video is stored outside Git at:

`saved/qsafe_development/natural_ppo/pilot-1m-v4-030.mp4`

SHA-256: `d4b3a81d4f4d22fb8503d3e5a705635cf6c64d5e14ee6eccc1ee9a4baca0f5a3`.

To reproduce the video from the checkpoint:

```bash
cd /tmp/go2-research.MajfHT/unitree_rl_mjlab
PYTHONPATH=/home/xyz/Desktop/xluo/onrobot-safe-rl:/tmp/qsafe-mjlab-pinned:/tmp/go2-mjlab-pkgs:/tmp/go2-research.MajfHT/unitree_rl_mjlab \
MUJOCO_GL=egl \
python3 /home/xyz/Desktop/xluo/onrobot-safe-rl/scripts/render_mjlab_go2_checkpoint.py \
  --checkpoint /home/xyz/Desktop/xluo/onrobot-safe-rl/saved/qsafe_development/natural_ppo/seed-43-pilot-1m-v4-030/model_3.pt \
  --output /home/xyz/Desktop/xluo/onrobot-safe-rl/saved/qsafe_development/natural_ppo/pilot-1m-v4-030.mp4 \
  --steps 300
```
