#pragma once

#include <array>
#include <cstdint>

static constexpr uint16_t POLICY_PROTOCOL_VERSION = 1;
static constexpr uint32_t POLICY_MAX_FRAME_BYTES = 4096;

enum class protocol_message_type : uint16_t
{
    OBSERVATION = 1,
    ACTION = 2,
    TRANSITION = 3,
};

enum class termination_reason : uint8_t
{
    NONE = 0,
    TIME_LIMIT = 1,
    EXCESSIVE_TILT = 2,
    JOINT_LIMIT = 3,
    COMMAND_TIMEOUT = 4,
    STATE_TIMEOUT = 5,
    MOTOR_FAULT = 6,
    MANUAL_ESTOP = 7,
    RECOVERY_FAILED = 8,
};

struct protocol_header
{
    uint16_t protocol_version{1};
    protocol_message_type message_type{protocol_message_type::ACTION};
    uint64_t sequence_id{0};
    uint32_t policy_version{0};
    uint64_t send_monotonic_ns{0};
    uint64_t observation_sequence{0};
    uint32_t payload_length{0};
    uint32_t checksum{0};
};

struct action_frame
{
    protocol_header header{};
    std::array<float, 12> requested_action{};
};

struct transition_frame
{
    protocol_header header{};
    uint64_t observation_time_ns{0};
    uint64_t action_apply_time_ns{0};
    uint64_t transition_end_time_ns{0};
    std::array<float, 12> requested_action{};
    std::array<float, 12> projected_action{};
    std::array<float, 12> executed_q_target{};
    uint32_t safety_flags{0};
    termination_reason reason{termination_reason::NONE};
    uint8_t controller_mode{0};
};
