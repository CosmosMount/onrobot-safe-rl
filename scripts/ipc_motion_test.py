#!/usr/bin/env python3
"""Send a large joint target via IPC and check if joints move."""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.config import load_app_config
from train.dds import StateReader
from train.ipc import PolicyClient


def main() -> None:
    cfg, _train_cfg, _ = load_app_config()
    reader = StateReader(sport_velocity_world_frame=cfg.sport_velocity_world_frame)
    reader.init_dds(cfg.domain_id, cfg.interface)
    reader.wait_for_state(timeout=10.0)

    client = PolicyClient(cfg.ipc_socket)
    client.connect()

    state0 = reader.get_state()
    q0 = state0.joint_q.copy()
    q_target = q0.copy()
    q_target[1] += 0.4  # thigh joint bump

    print(f'q0[1]={q0[1]:.4f}  target[1]={q_target[1]:.4f}  delta={q_target[1]-q0[1]:.4f}')
    for i in range(40):
        client.send_target(q_target)
        time.sleep(0.05)

    time.sleep(0.5)
    state1 = reader.get_state()
    q1 = state1.joint_q.copy()
    dq = q1 - q0
    print(f'q1[1]={q1[1]:.4f}  joint_delta[1]={dq[1]:.4f}  |dq|={np.linalg.norm(dq):.4f}')
    print(f'joint_dq_max={np.max(np.abs(state1.joint_dq)):.6f}')


if __name__ == '__main__':
    main()
