"""Training package public conveniences.

Heavy runtime dependencies are imported lazily so utility modules such as
``train.checkpoint`` can be used in lightweight tests.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    'MUJOCO',
    'REAL',
    'DdsConfig',
    'StateReader',
    'Go2Env',
    'POLICY_PACKET',
    'POLICY_SOF',
    'PolicyClient',
]


def __getattr__(name: str) -> Any:
    if name in {'MUJOCO', 'REAL', 'DdsConfig', 'StateReader'}:
        from train import dds
        return getattr(dds, name)
    if name == 'Go2Env':
        from train.env import Go2Env
        return Go2Env
    if name in {'POLICY_PACKET', 'POLICY_SOF', 'PolicyClient'}:
        from train import ipc
        return getattr(ipc, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
