#pragma once

#include "policy_protocol.hpp"

enum class supervisor_mode : uint8_t
{
    BOOT = 0,
    WAIT_STATE = 1,
    STAND_UP = 2,
    POLICY = 3,
    SOFT_STOP = 4,
    RECOVERY = 5,
    ESTOP = 6,
    FAULT = 7,
};

struct safety_decision
{
    bool allow_policy{false};
    bool recovery_required{false};
    bool hard_stop_required{false};
    uint32_t safety_flags{0};
    termination_reason reason{termination_reason::NONE};
    supervisor_mode mode{supervisor_mode::BOOT};
};

class safety_supervisor
{
public:
    safety_decision command_timeout(supervisor_mode mode) const
    {
        return safety_decision{
            false,
            false,
            false,
            0x1,
            termination_reason::COMMAND_TIMEOUT,
            mode,
        };
    }
};
