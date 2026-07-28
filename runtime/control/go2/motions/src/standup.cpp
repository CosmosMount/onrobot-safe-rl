#include "standup.hpp"
#include "motions.hpp"

namespace motions
{
    standup::standup(const standup_config& stand_up, float control_hz)
        : stand_up_(stand_up),
        control_hz_(control_hz),
        warmup_ticks_(seconds_to_ticks(stand_up.warmup_s, control_hz)),
        hold_ticks_(seconds_to_ticks(stand_up.hold_s, control_hz))
    {
        for (int i = 0; i < stand_up.num_phases; ++i) 
        {
            phase_ticks_[i] =
                seconds_to_ticks(stand_up.phase_duration_s[i], control_hz);
        }
    }

    void standup::reset(const unitree_go::msg::dds_::LowState_& state)
    {
        (void)state;
        tick_ = 0;
        phase_index_ = 0;
        phase_percent_ = 0.f;
        hold_percent_ = 0.f;
        captured_start_ = false;
        stage_ = standup_stage::WARMUP;
        std::cout << "[standup] Starting stand-up sequence." << std::endl;
    }

    bool standup::near_stable_pose(
        const unitree_go::msg::dds_::LowState_& state) const
    {
        if (!stand_up_.configured) 
        {
            return true;
        }
        for (int i = 0; i < 12; ++i) 
        {
            if (std::abs(state.motor_state()[i].q() - stand_up_.stable_pose[i]) >
                stand_up_.joint_tolerance) 
            {
                return false;
            }
        }
        return true;
    }

    bool standup::update(bool state_received,
                            const unitree_go::msg::dds_::LowState_& state,
                            std::array<float, 12>& q_out)
    {
        if (!state_received || !stand_up_.configured || stand_up_.num_phases <= 0) 
        {
            q_out = stand_up_.stable_pose;
            return true;
        }

        if (stage_ == standup_stage::HOLD) 
        {
            hold_percent_ =
                clamp01(hold_percent_ + 1.f / static_cast<float>(hold_ticks_));
            q_out = stand_up_.stable_pose;
            return hold_percent_ >= 1.f;
        }

        ++tick_;

        if (stage_ == standup_stage::WARMUP) 
        {
            if (tick_ < warmup_ticks_) 
            {
                for (int i = 0; i < 12; ++i) 
                {
                    q_out[i] = state.motor_state()[i].q();
                }
                return false;
            }
            stage_ = standup_stage::PHASES;
        }

        if (!captured_start_) 
        {
            for (int i = 0; i < 12; ++i) 
            {
                start_pos_[i] = state.motor_state()[i].q();
            }
            segment_start_ = start_pos_;
            captured_start_ = true;
        }

        if (phase_index_ < stand_up_.num_phases) 
        {
            const int duration = phase_ticks_[phase_index_];
            phase_percent_ =
                clamp01(phase_percent_ + 1.f / static_cast<float>(duration));
            const auto& target = stand_up_.keyframes[phase_index_];
            for (int i = 0; i < 12; ++i) 
            {
                q_out[i] = (1.f - phase_percent_) * segment_start_[i] +
                        phase_percent_ * target[i];
            }
            if (phase_percent_ >= 1.f) 
            {
                segment_start_ = target;
                phase_percent_ = 0.f;
                ++phase_index_;
            }
            return false;
        }

        stage_ = standup_stage::HOLD;
        q_out = stand_up_.stable_pose;
        return false;
    }
}