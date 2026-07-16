"""Python-side policy protocol datatypes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import json
import struct
import zlib

import numpy as np

LENGTH_PREFIX = struct.Struct('<I')

class MessageType(IntEnum):
    OBSERVATION = 1
    ACTION = 2
    TRANSITION = 3


@dataclass(frozen=True)
class ProtocolHeader:
    protocol_version: int
    message_type: MessageType
    sequence_id: int
    policy_version: int
    send_monotonic_ns: int
    observation_sequence: int
    payload_length: int
    checksum: int


@dataclass(frozen=True)
class ObservationFrame:
    header: ProtocolHeader
    observation: np.ndarray
    lowstate_age_us: int
    sportstate_age_us: int
    stale_mask: int
    controller_mode: int


@dataclass(frozen=True)
class ActionFrame:
    header: ProtocolHeader
    requested_action: np.ndarray


@dataclass(frozen=True)
class TransitionFrame:
    header: ProtocolHeader
    next_observation: np.ndarray
    requested_action: np.ndarray
    projected_action: np.ndarray
    executed_q_target: np.ndarray
    safety_flags: int
    termination_reason: int


def checksum(payload: bytes) -> int:
    return zlib.crc32(payload) & 0xFFFFFFFF


def encode_length_prefixed(payload: bytes) -> bytes:
    return LENGTH_PREFIX.pack(len(payload)) + payload


def decode_length_prefixed(buffer: bytes) -> tuple[bytes, bytes] | None:
    if len(buffer) < LENGTH_PREFIX.size:
        return None
    (size, ) = LENGTH_PREFIX.unpack(buffer[:LENGTH_PREFIX.size])
    end = LENGTH_PREFIX.size + size
    if len(buffer) < end:
        return None
    return buffer[LENGTH_PREFIX.size:end], buffer[end:]


def encode_header(header: ProtocolHeader) -> bytes:
    payload = {
        'protocol_version': header.protocol_version,
        'message_type': int(header.message_type),
        'sequence_id': header.sequence_id,
        'policy_version': header.policy_version,
        'send_monotonic_ns': header.send_monotonic_ns,
        'observation_sequence': header.observation_sequence,
        'payload_length': header.payload_length,
        'checksum': header.checksum,
    }
    return json.dumps(payload, separators=(',', ':')).encode('utf-8')


def decode_header(payload: bytes) -> ProtocolHeader:
    data = json.loads(payload.decode('utf-8'))
    return ProtocolHeader(
        protocol_version=int(data['protocol_version']),
        message_type=MessageType(int(data['message_type'])),
        sequence_id=int(data['sequence_id']),
        policy_version=int(data['policy_version']),
        send_monotonic_ns=int(data['send_monotonic_ns']),
        observation_sequence=int(data['observation_sequence']),
        payload_length=int(data['payload_length']),
        checksum=int(data['checksum']),
    )
