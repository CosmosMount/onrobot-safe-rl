"""Exact complete-iteration MjLab state capture for interruption resume."""

from __future__ import annotations

from typing import Any

import torch


SIM_DATA_FIELDS = (
    "time", "qpos", "qvel", "act", "ctrl", "qacc_warmstart",
    "qfrc_applied", "xfrc_applied", "mocap_pos", "mocap_quat",
)
SIM_MODEL_FIELDS = (
    "geom_friction", "body_ipos", "body_subtreemass", "dof_invweight0",
    "body_invweight0", "tendon_length0", "tendon_invweight0",
)
ACTION_MANAGER_FIELDS = ("action", "prev_action", "prev_prev_action")
ACTION_TERM_FIELDS = ("_raw_actions", "_processed_actions")
COMMAND_FIELDS = (
    "command", "command_counter", "heading_error", "heading_target",
    "is_heading_env", "is_standing_env", "time_left", "vel_command_b",
)
ROBOT_FIELDS = (
    "encoder_bias", "joint_pos_target", "joint_vel_target",
    "joint_effort_target")
MANAGER_TENSOR_FIELDS = {
    "reward": ("_reward_buf", "_step_reward"),
    "termination": ("_truncated_buf", "_terminated_buf"),
    "metrics": ("_step_count", "_step_values"),
}


def _clone_fields(obj: Any, names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    result = {}
    for name in names:
        if hasattr(obj, name):
            result[name] = getattr(obj, name).clone()
    return result


def _restore_fields(obj: Any, values: dict[str, torch.Tensor]) -> None:
    for name, value in values.items():
        destination = getattr(obj, name)
        destination.copy_(value.to(destination.device))


def _clone_tensor_mapping(values: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {name: value.clone() for name, value in values.items()
            if torch.is_tensor(value)}


def _restore_tensor_mapping(destination: dict[str, Any],
                            values: dict[str, torch.Tensor]) -> None:
    if set(values) != {name for name, value in destination.items()
                       if torch.is_tensor(value)}:
        raise ValueError("resume tensor mapping structure changed")
    for name, value in values.items():
        destination[name].copy_(value.to(destination[name].device))


def _capture_circular_buffers(manager: Any) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for group, terms in manager._group_obs_term_history_buffer.items():
        for term, buffer in terms.items():
            result[f"{group}/{term}"] = {
                "pointer": int(buffer._pointer),
                "num_pushes": buffer._num_pushes.clone(),
                "buffer": None if buffer._buffer is None else buffer._buffer.clone(),
            }
    return result


def _restore_circular_buffers(manager: Any,
                              state: dict[str, dict[str, object]]) -> None:
    available = {
        f"{group}/{term}": buffer
        for group, terms in manager._group_obs_term_history_buffer.items()
        for term, buffer in terms.items()
    }
    if set(state) != set(available):
        raise ValueError("observation history structure changed")
    for name, values in state.items():
        buffer = available[name]
        buffer._pointer = int(values["pointer"])
        buffer._num_pushes.copy_(values["num_pushes"].to(buffer._num_pushes.device))
        stored = values["buffer"]
        if stored is None:
            buffer._buffer = None
        elif buffer._buffer is None:
            buffer._buffer = stored.to(buffer._num_pushes.device).clone()
        else:
            buffer._buffer.copy_(stored.to(buffer._buffer.device))


def _capture_contact_sensors(environment: Any) -> dict[str, dict[str, torch.Tensor]]:
    result = {}
    for name, sensor in environment.scene.sensors.items():
        air = getattr(sensor, "_air_time_state", None)
        history = getattr(sensor, "_history_state", None)
        values: dict[str, torch.Tensor] = {}
        if air is not None:
            values.update({f"air/{key}": value.clone()
                           for key, value in vars(air).items()
                           if torch.is_tensor(value)})
        if history is not None:
            values.update({f"history/{key}": value.clone()
                           for key, value in history.items()})
        if values:
            result[name] = values
    return result


def _restore_contact_sensors(environment: Any,
                             state: dict[str, dict[str, torch.Tensor]]) -> None:
    for name, values in state.items():
        sensor = environment.scene.sensors[name]
        air = getattr(sensor, "_air_time_state", None)
        history = getattr(sensor, "_history_state", None)
        for key, value in values.items():
            prefix, field = key.split("/", 1)
            destination = getattr(air, field) if prefix == "air" else history[field]
            destination.copy_(value.to(destination.device))


def capture_environment_state(environment: Any) -> dict[str, object]:
    action_term = environment.action_manager.get_term("joint_pos")
    command_term = environment.command_manager.get_term("twist")
    robot = environment.scene.entities["robot"].data
    return {
        "schema_version": "ppo_sqrl_go2.mjlab_resume.v2",
        "sim_data": _clone_fields(environment.sim.data, SIM_DATA_FIELDS),
        "sim_model": _clone_fields(environment.sim.model, SIM_MODEL_FIELDS),
        "episode_length_buf": environment.episode_length_buf.clone(),
        "common_step_counter": int(environment.common_step_counter),
        "sim_step_counter": int(environment._sim_step_counter),
        "action_manager": _clone_fields(
            environment.action_manager, ACTION_MANAGER_FIELDS),
        "action_term": _clone_fields(action_term, ACTION_TERM_FIELDS),
        "command_term": _clone_fields(command_term, COMMAND_FIELDS),
        "robot": _clone_fields(robot, ROBOT_FIELDS),
        "observation_buffer": _clone_tensor_mapping(
            environment.observation_manager._obs_buffer),
        "observation_history": _capture_circular_buffers(
            environment.observation_manager),
        "reward": {
            "fields": _clone_fields(environment.reward_manager,
                                    MANAGER_TENSOR_FIELDS["reward"]),
            "episode_sums": _clone_tensor_mapping(
                environment.reward_manager._episode_sums),
        },
        "termination": {
            "fields": _clone_fields(environment.termination_manager,
                                    MANAGER_TENSOR_FIELDS["termination"]),
            "term_dones": _clone_tensor_mapping(
                environment.termination_manager._term_dones),
        },
        "metrics": {
            "fields": _clone_fields(environment.metrics_manager,
                                    MANAGER_TENSOR_FIELDS["metrics"]),
            "episode_sums": _clone_tensor_mapping(
                environment.metrics_manager._episode_sums),
        },
        "contact_sensors": _capture_contact_sensors(environment),
    }


def restore_environment_state(environment: Any, state: dict[str, object]) -> None:
    if state.get("schema_version") != "ppo_sqrl_go2.mjlab_resume.v2":
        raise ValueError("unknown MjLab resume state")
    _restore_fields(environment.sim.model,
                    state["sim_model"])  # type: ignore[arg-type]
    _restore_fields(environment.sim.data, state["sim_data"])  # type: ignore[arg-type]
    environment.episode_length_buf.copy_(state["episode_length_buf"])
    environment.common_step_counter = int(state["common_step_counter"])
    environment._sim_step_counter = int(state["sim_step_counter"])
    _restore_fields(environment.action_manager,
                    state["action_manager"])  # type: ignore[arg-type]
    _restore_fields(environment.action_manager.get_term("joint_pos"),
                    state["action_term"])  # type: ignore[arg-type]
    _restore_fields(environment.command_manager.get_term("twist"),
                    state["command_term"])  # type: ignore[arg-type]
    _restore_fields(environment.scene.entities["robot"].data,
                    state["robot"])  # type: ignore[arg-type]
    environment.sim.forward()
    environment.sim.sense()
    _restore_tensor_mapping(environment.observation_manager._obs_buffer,
                            state["observation_buffer"])  # type: ignore[arg-type]
    _restore_circular_buffers(environment.observation_manager,
                              state["observation_history"])  # type: ignore[arg-type]
    _restore_fields(environment.reward_manager,
                    state["reward"]["fields"])  # type: ignore[index,arg-type]
    _restore_tensor_mapping(environment.reward_manager._episode_sums,
                            state["reward"]["episode_sums"])  # type: ignore[index,arg-type]
    _restore_fields(environment.termination_manager,
                    state["termination"]["fields"])  # type: ignore[index,arg-type]
    _restore_tensor_mapping(environment.termination_manager._term_dones,
                            state["termination"]["term_dones"])  # type: ignore[index,arg-type]
    _restore_fields(environment.metrics_manager,
                    state["metrics"]["fields"])  # type: ignore[index,arg-type]
    _restore_tensor_mapping(environment.metrics_manager._episode_sums,
                            state["metrics"]["episode_sums"])  # type: ignore[index,arg-type]
    _restore_contact_sensors(environment,
                             state["contact_sensors"])  # type: ignore[arg-type]


def capture_rng_state() -> dict[str, object]:
    import random
    import numpy as np
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict[str, object]) -> None:
    import random
    import numpy as np
    random.setstate(state["python"])  # type: ignore[arg-type]
    np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    torch.set_rng_state(state["torch_cpu"])  # type: ignore[arg-type]
    torch.cuda.set_rng_state_all(state["torch_cuda"])  # type: ignore[arg-type]
