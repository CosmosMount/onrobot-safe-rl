#pragma once

#include <array>
#include <unitree/idl/go2/LowState_.hpp>

namespace motions
{

    struct standup_config
    {
        bool configured = false;
        static constexpr int kMaxPhases = 4;
        int num_phases = 0;
        std::array<std::array<float, 12>, kMaxPhases> keyframes{};
        std::array<float, kMaxPhases> phase_duration_s{};
        float warmup_s = 1.0f;
        float hold_s = 0.2f;
        std::array<float, 12> stable_pose{};
        float joint_tolerance = 0.15f;
        float kp = 60.f;
        float kd = 5.f;
    };

    enum class standup_stage
    {
        WARMUP,
        PHASES,
        HOLD
    };

    class standup
    {
    public:

        standup(standup_config& _config, float control_hz);
        void reset(const unitree_go::msg::dds_::LowState_& state);
        bool update(bool _state_received,
                    const unitree_go::msg::dds_::LowState_& state,
                    std::array<float, 12>& q_out);
        bool near_stable_pose(const unitree_go::msg::dds_::LowState_& state) const;

    private:

        standup_config config_;
        float ctrl_hz_;
        int warmup_ticks_;
        int hold_ticks_;
        std::array<int, standup_config::kMaxPhases> phase_ticks_{};
        standup_stage stage_{standup_stage::WARMUP};
        int tick_{0};
        int phase_index_{0};
        float phase_percent_{0.f};
        float hold_percent_{0.f};
        bool captured_start_{false};
        std::array<float, 12> start_pos_{};
        std::array<float, 12> segment_start_{};

    };


}