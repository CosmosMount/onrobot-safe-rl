#pragma once

#include <cstdint>

struct state_freshness
{
    uint64_t lowstate_age_us{0};
    uint64_t sportstate_age_us{0};
    uint32_t stale_mask{0};
};
