"""Versioned Go2 MDP specs and stable metadata hashes."""

from __future__ import annotations

import hashlib

from train.config import Go2Config


OBSERVATION_SPECS = {
    'legacy_58': (
        'joint_q',
        'joint_dq',
        'previous_requested_action',
        'previous_sent_action',
        'base_ang_vel',
        'body_velocity',
        'imu_quat',
    ),
    'deploy_safe_57': (
        'base_ang_vel_scaled_0.25',
        'projected_gravity_body',
        'command_scaled_2_2_0.25',
        'joint_pos_offset',
        'joint_vel_scaled_0.05',
        'previous_requested_action',
        'previous_sent_or_applied_action',
    ),
}


def observation_dim(spec: str, cfg: Go2Config) -> int:
    if spec == 'legacy_58':
        return 4 * cfg.num_joints + 10
    if spec == 'deploy_safe_57':
        return 4 * cfg.num_joints + 9
    raise ValueError(f'Unknown observation spec: {spec}')


def observation_spec_hash(spec: str) -> str:
    if spec not in OBSERVATION_SPECS:
        raise ValueError(f'Unknown observation spec: {spec}')
    payload = '|'.join((spec, *OBSERVATION_SPECS[spec]))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]
