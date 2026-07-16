"""Trigger controller belly-up standup (recovery→standup): python -m train.recovery"""

from __future__ import annotations

import time

from train.config import load_app_config
from train.dds import DdsConfig, StateReader
from train.ipc import PolicyClient
from train.obs import (is_belly_up, is_flipped_back, is_pose_stable,
                       quat_to_euler_xyz, gravity_acc_z)


def main() -> int:
    robot_cfg, train_cfg, _ = load_app_config()
    control_dt = 1.0 / train_cfg.control_frequency

    state_reader = StateReader(
        sport_velocity_world_frame=robot_cfg.sport_velocity_world_frame)
    state_reader.init_dds(robot_cfg.domain_id, robot_cfg.interface)
    state_reader.wait_for_state(timeout=10.0)

    client = PolicyClient(robot_cfg.ipc_socket)
    client.connect()

    state = state_reader.get_state()
    roll, pitch, _ = quat_to_euler_xyz(state.imu_quat)
    up = gravity_acc_z(state)
    print(f'[standup] start roll={roll:+.3f} pitch={pitch:+.3f} rad '
          f'acc_z={up:+.2f} m/s² belly_up={is_belly_up(state, robot_cfg)}',
          flush=True)

    stable_count = 0
    with_recovery = is_belly_up(state, robot_cfg)
    for step in range(train_cfg.reset_hold_steps):
        client.send_standup(with_recovery=with_recovery,
                            q_target=robot_cfg.init_qpos)
        time.sleep(control_dt)
        state = state_reader.get_state()
        roll, pitch, _ = quat_to_euler_xyz(state.imu_quat)
        if step % 20 == 0:
            print(f'  step={step} roll={roll:+.3f} pitch={pitch:+.3f} rad '
                  f'acc_z={gravity_acc_z(state):+.2f} m/s² '
                  f'flipped={is_flipped_back(state, robot_cfg)}',
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
