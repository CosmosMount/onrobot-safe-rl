"""Request controller stand-up/recovery manually."""

from __future__ import annotations

import time
import argparse

from train.config import load_app_config
from runtime.inference.dds import StateReader
from runtime.inference.ipc import PolicyClient
from runtime.inference.observations import quat_to_euler_xyz, is_pose_stable


def _acc_z(state) -> float:
    return float(state.imu_accel[2])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Request controller stand-up/recovery.")
    parser.add_argument("--with-recovery", action="store_true")
    args = parser.parse_args(argv)

    robot_cfg, train_cfg, _ = load_app_config()
    control_dt = 1.0 / train_cfg.control_frequency

    state_reader = StateReader(
        socket_path=robot_cfg.state_socket,
        sport_velocity_world_frame=robot_cfg.sport_velocity_world_frame)
    state_reader.connect()
    state_reader.wait_for_state(timeout=10.0)

    client = PolicyClient(robot_cfg.ipc_socket)
    client.connect()

    state = state_reader.get_state()
    roll, pitch, _ = quat_to_euler_xyz(state.imu_quat)
    up = _acc_z(state)
    print(f'[standup] start roll={roll:+.3f} pitch={pitch:+.3f} rad '
          f'acc_z={up:+.2f} m/s²',
          flush=True)

    stable_count = 0
    for step in range(train_cfg.standup_timeout_steps):
        client.send_standup(with_recovery=args.with_recovery,
                            q_target=robot_cfg.init_qpos)
        time.sleep(control_dt)
        state = state_reader.get_state()
        roll, pitch, _ = quat_to_euler_xyz(state.imu_quat)
        if step % 20 == 0:
            print(f'  step={step} roll={roll:+.3f} pitch={pitch:+.3f} rad '
                  f'acc_z={_acc_z(state):+.2f} m/s² ',
                  flush=True)
        if is_pose_stable(state, robot_cfg):
            stable_count += 1
            if stable_count >= train_cfg.recovery_stable_steps:
                break
        else:
            stable_count = 0

    ok = is_pose_stable(state, robot_cfg)
    roll, pitch, _ = quat_to_euler_xyz(state.imu_quat)
    print(f'[standup] done stable={ok} roll={roll:+.3f} pitch={pitch:+.3f} rad',
          flush=True)
    client.close()
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
