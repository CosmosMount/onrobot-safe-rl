#include "recovery.hpp"
#include "motions.hpp"

namespace motions
{
    recovery::recovery(const recovery_config& recovery, float control_hz)
        : recovery_(recovery),
        control_hz_(control_hz),
        fold_ramp_ticks_(seconds_to_ticks(recovery.fold_ramp_s, control_hz)),
        fold_settle_ticks_(seconds_to_ticks(recovery.fold_settle_s, control_hz)),
        above_ramp_ticks_(seconds_to_ticks(recovery.above_ramp_s, control_hz)),
        above_settle_ticks_(seconds_to_ticks(recovery.above_settle_s, control_hz)),
        swing_down_ramp_ticks_(
            seconds_to_ticks(recovery.swing_down_ramp_s, control_hz)),
        swing_down_settle_ticks_(
            seconds_to_ticks(recovery.swing_down_settle_s, control_hz)),
        push_ramp_ticks_(seconds_to_ticks(recovery.push_ramp_s, control_hz)),
        push_settle_ticks_(seconds_to_ticks(recovery.push_settle_s, control_hz))
    {
    }

    void recovery::reset(const unitree_go::msg::dds_::LowState_& state)
    {
        tick_ = 0;
        segment_start_tick_ = 0;
        stage_ = recovery_stage::FOLD;
        capture_jpos(state);
        std::cout << "[recover] Fold -> Above -> SwingDown -> Push" << std::endl;
    }

    bool recovery::update(bool state_received,
                            const unitree_go::msg::dds_::LowState_& state,
                            std::array<float, 12>& q_out)
    {
        if (!state_received) 
        {
            q_out = recovery_.fold_jpos;
            return false;
        }

        const int local_iter = tick_ - segment_start_tick_;

        switch (stage_) 
        {
            case recovery_stage::FOLD:
                interpolate_all(local_iter, fold_ramp_ticks_, recovery_.fold_jpos, q_out);
                if (local_iter >= fold_ramp_ticks_ + fold_settle_ticks_) 
                {
                    stage_ = recovery_stage::ABOVE;
                    begin_segment(recovery_.fold_jpos);
                }
                break;

            case recovery_stage::ABOVE:
                interpolate_active(local_iter, above_ramp_ticks_, recovery_.above_jpos, q_out);
                if (local_iter >= above_ramp_ticks_ + above_settle_ticks_) 
                {
                    stage_ = recovery_stage::SWING_DOWN;
                    apply_leg_pose(recovery_.above_jpos, recovery_.swing_legs, q_out);
                    begin_segment(q_out);
                }
                break;

            case recovery_stage::SWING_DOWN:
                interpolate_active(local_iter, swing_down_ramp_ticks_,
                                recovery_.swing_down_jpos, q_out);
                if (local_iter >= swing_down_ramp_ticks_ + swing_down_settle_ticks_ &&
                    legs_at_pose(state, recovery_.swing_down_jpos, recovery_.swing_legs)) 
                {
                    stage_ = recovery_stage::PUSH;
                    apply_leg_pose(recovery_.swing_down_jpos, recovery_.swing_legs, q_out);
                    begin_segment(q_out);
                }
                break;

            case recovery_stage::PUSH:
                interpolate_calf_push(local_iter, push_ramp_ticks_, recovery_.push_jpos, q_out);
                if (local_iter >= push_ramp_ticks_ + push_settle_ticks_) 
                {
                    ++tick_;
                    return true;
                }
                break;
        }

        ++tick_;
        return false;
    }

    void recovery::capture_jpos(const unitree_go::msg::dds_::LowState_& state)
    {
        for (int leg = 0; leg < 4; ++leg) 
        {
            for (int j = 0; j < 3; ++j) 
            {
                initial_jpos_[leg][j] = state.motor_state()[leg * 3 + j].q();
            }
        }
    }

    void recovery::begin_segment(const std::array<float, 12>& start_jpos)
    {
        for (int leg = 0; leg < 4; ++leg) 
        {
            for (int j = 0; j < 3; ++j) 
            {
                initial_jpos_[leg][j] = start_jpos[leg * 3 + j];
            }
        }
        segment_start_tick_ = tick_ + 1;
    }

    void recovery::interpolate_leg(int leg,
                                    int curr_iter,
                                    int max_iter,
                                    const std::array<float, 3>& ini,
                                    const std::array<float, 3>& fin,
                                    std::array<float, 12>& q_out) const
    {
        float b = 1.f;
        if (curr_iter <= max_iter && max_iter > 0) 
        {
            b = static_cast<float>(curr_iter) / static_cast<float>(max_iter);
        }
        const float a = 1.f - b;
        for (int j = 0; j < 3; ++j) 
        {
            q_out[leg * 3 + j] = a * ini[j] + b * fin[j];
        }
    }

    void recovery::interpolate_all(int curr_iter,
                                    int max_iter,
                                    const std::array<float, 12>& fin,
                                    std::array<float, 12>& q_out) const
    {
        for (int leg = 0; leg < 4; ++leg) 
        {
            std::array<float, 3> ini{};
            std::array<float, 3> target{};
            for (int j = 0; j < 3; ++j) 
            {
                ini[j] = initial_jpos_[leg][j];
                target[j] = fin[leg * 3 + j];
            }
            interpolate_leg(leg, curr_iter, max_iter, ini, target, q_out);
        }
    }

    void recovery::apply_leg_pose(const std::array<float, 12>& pose,
                                    const std::array<bool, 4>& leg_mask,
                                    std::array<float, 12>& q_out) const
    {
        for (int leg = 0; leg < 4; ++leg) 
        {
            const std::array<float, 12>& src =
                leg_mask[leg] ? pose : recovery_.fsold_jpos;
            for (int j = 0; j < 3; ++j) 
            {
                q_out[leg * 3 + j] = src[leg * 3 + j];
            }
        }
    }

    void recovery::interpolate_active(int curr_iter,
                                        int max_iter,
                                        const std::array<float, 12>& active_target,
                                        std::array<float, 12>& q_out) const
    {
        std::array<float, 12> blended_target{};
        apply_leg_pose(active_target, recovery_.swing_legs, blended_target);
        for (int leg = 0; leg < 4; ++leg) 
        {
            if (!recovery_.swing_legs[leg]) 
            {
                for (int j = 0; j < 3; ++j) 
                {
                    q_out[leg * 3 + j] = recovery_.fold_jpos[leg * 3 + j];
                }
                continue;
            }
            std::array<float, 3> ini{};
            std::array<float, 3> target{};
            for (int j = 0; j < 3; ++j) 
            {
                ini[j] = initial_jpos_[leg][j];
                target[j] = blended_target[leg * 3 + j];
            }
            interpolate_leg(leg, curr_iter, max_iter, ini, target, q_out);
        }
    }

    void recovery::interpolate_calf_push(int curr_iter,
                                            int max_iter,
                                            const std::array<float, 12>& push_pose,
                                            std::array<float, 12>& q_out) const
    {
        apply_leg_pose(recovery_.swing_down_jpos, recovery_.swing_legs, q_out);
        for (int leg = 0; leg < 4; ++leg) 
        {
            if (!recovery_.push_legs[leg]) 
            {
                continue;
            }
            const int calf = leg * 3 + 2;
            const float ini = initial_jpos_[leg][2];
            const float fin = push_pose[calf];
            float b = 1.f;
            if (curr_iter <= max_iter && max_iter > 0) 
            {
                b = static_cast<float>(curr_iter) / static_cast<float>(max_iter);
            }
            q_out[calf] = (1.f - b) * ini + b * fin;
        }
    }

    bool recovery::legs_at_pose(
        const unitree_go::msg::dds_::LowState_& state,
        const std::array<float, 12>& target,
        const std::array<bool, 4>& leg_mask) const
    {
        for (int leg = 0; leg < 4; ++leg) 
        {
            if (!leg_mask[leg]) 
            {
                continue;
            }
            for (int j = 0; j < 3; ++j) 
            {
                const int idx = leg * 3 + j;
                if (std::abs(state.motor_state()[idx].q() - target[idx]) >
                    recovery_.joint_reach_tol) 
                {
                    return false;
                }
            }
        }
        return true;
    }
}