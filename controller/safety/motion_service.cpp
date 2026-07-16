#include "motion_service.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <unistd.h>

#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>

namespace
{

int seconds_to_ticks(float seconds, float control_hz)
{
    return std::max(1, static_cast<int>(std::lround(seconds * control_hz)));
}

float clamp01(float v)
{
    return std::max(0.f, std::min(1.f, v));
}

}  // namespace

bool deactivate_motion_service()
{
    unitree::robot::b2::MotionSwitcherClient client;
    client.SetTimeout(10.0f);
    client.Init();
    for (int i = 0; i < 12; ++i) {
        std::string robot_form;
        std::string motion_name;
        if (client.CheckMode(robot_form, motion_name) != 0 || motion_name.empty()) {
            return true;
        }
        client.ReleaseMode();
        sleep(5);
    }
    return true;
}

recovery_fsm::recovery_fsm(const recovery_config& recovery, float control_hz)
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

void recovery_fsm::reset(const unitree_go::msg::dds_::LowState_& state)
{
    tick_ = 0;
    segment_start_tick_ = 0;
    stage_ = recovery_stage::FOLD;
    capture_jpos(state);
    std::cout << "[recover] Fold -> Above -> SwingDown -> Push" << std::endl;
}

bool recovery_fsm::update(bool state_received,
                          const unitree_go::msg::dds_::LowState_& state,
                          std::array<float, 12>& q_out)
{
    if (!state_received) {
        q_out = recovery_.fold_jpos;
        return false;
    }

    const int local_iter = tick_ - segment_start_tick_;

    switch (stage_) {
    case recovery_stage::FOLD:
        interpolate_all(local_iter, fold_ramp_ticks_, recovery_.fold_jpos, q_out);
        if (local_iter >= fold_ramp_ticks_ + fold_settle_ticks_) {
            stage_ = recovery_stage::ABOVE;
            begin_segment(recovery_.fold_jpos);
        }
        break;

    case recovery_stage::ABOVE:
        interpolate_active(local_iter, above_ramp_ticks_, recovery_.above_jpos, q_out);
        if (local_iter >= above_ramp_ticks_ + above_settle_ticks_) {
            stage_ = recovery_stage::SWING_DOWN;
            apply_leg_pose(recovery_.above_jpos, recovery_.swing_legs, q_out);
            begin_segment(q_out);
        }
        break;

    case recovery_stage::SWING_DOWN:
        interpolate_active(local_iter, swing_down_ramp_ticks_,
                           recovery_.swing_down_jpos, q_out);
        if (local_iter >= swing_down_ramp_ticks_ + swing_down_settle_ticks_ &&
            legs_at_pose(state, recovery_.swing_down_jpos, recovery_.swing_legs)) {
            stage_ = recovery_stage::PUSH;
            apply_leg_pose(recovery_.swing_down_jpos, recovery_.swing_legs, q_out);
            begin_segment(q_out);
        }
        break;

    case recovery_stage::PUSH:
        interpolate_calf_push(local_iter, push_ramp_ticks_, recovery_.push_jpos, q_out);
        if (local_iter >= push_ramp_ticks_ + push_settle_ticks_) {
            ++tick_;
            return true;
        }
        break;
    }

    ++tick_;
    return false;
}

void recovery_fsm::capture_jpos(const unitree_go::msg::dds_::LowState_& state)
{
    for (int leg = 0; leg < 4; ++leg) {
        for (int j = 0; j < 3; ++j) {
            initial_jpos_[leg][j] = state.motor_state()[leg * 3 + j].q();
        }
    }
}

void recovery_fsm::begin_segment(const std::array<float, 12>& start_jpos)
{
    for (int leg = 0; leg < 4; ++leg) {
        for (int j = 0; j < 3; ++j) {
            initial_jpos_[leg][j] = start_jpos[leg * 3 + j];
        }
    }
    segment_start_tick_ = tick_ + 1;
}

void recovery_fsm::interpolate_leg(int leg,
                                   int curr_iter,
                                   int max_iter,
                                   const std::array<float, 3>& ini,
                                   const std::array<float, 3>& fin,
                                   std::array<float, 12>& q_out) const
{
    float b = 1.f;
    if (curr_iter <= max_iter && max_iter > 0) {
        b = static_cast<float>(curr_iter) / static_cast<float>(max_iter);
    }
    const float a = 1.f - b;
    for (int j = 0; j < 3; ++j) {
        q_out[leg * 3 + j] = a * ini[j] + b * fin[j];
    }
}

void recovery_fsm::interpolate_all(int curr_iter,
                                   int max_iter,
                                   const std::array<float, 12>& fin,
                                   std::array<float, 12>& q_out) const
{
    for (int leg = 0; leg < 4; ++leg) {
        std::array<float, 3> ini{};
        std::array<float, 3> target{};
        for (int j = 0; j < 3; ++j) {
            ini[j] = initial_jpos_[leg][j];
            target[j] = fin[leg * 3 + j];
        }
        interpolate_leg(leg, curr_iter, max_iter, ini, target, q_out);
    }
}

void recovery_fsm::apply_leg_pose(const std::array<float, 12>& pose,
                                  const std::array<bool, 4>& leg_mask,
                                  std::array<float, 12>& q_out) const
{
    for (int leg = 0; leg < 4; ++leg) {
        const std::array<float, 12>& src =
            leg_mask[leg] ? pose : recovery_.fold_jpos;
        for (int j = 0; j < 3; ++j) {
            q_out[leg * 3 + j] = src[leg * 3 + j];
        }
    }
}

void recovery_fsm::interpolate_active(int curr_iter,
                                      int max_iter,
                                      const std::array<float, 12>& active_target,
                                      std::array<float, 12>& q_out) const
{
    std::array<float, 12> blended_target{};
    apply_leg_pose(active_target, recovery_.swing_legs, blended_target);
    for (int leg = 0; leg < 4; ++leg) {
        if (!recovery_.swing_legs[leg]) {
            for (int j = 0; j < 3; ++j) {
                q_out[leg * 3 + j] = recovery_.fold_jpos[leg * 3 + j];
            }
            continue;
        }
        std::array<float, 3> ini{};
        std::array<float, 3> target{};
        for (int j = 0; j < 3; ++j) {
            ini[j] = initial_jpos_[leg][j];
            target[j] = blended_target[leg * 3 + j];
        }
        interpolate_leg(leg, curr_iter, max_iter, ini, target, q_out);
    }
}

void recovery_fsm::interpolate_calf_push(int curr_iter,
                                         int max_iter,
                                         const std::array<float, 12>& push_pose,
                                         std::array<float, 12>& q_out) const
{
    apply_leg_pose(recovery_.swing_down_jpos, recovery_.swing_legs, q_out);
    for (int leg = 0; leg < 4; ++leg) {
        if (!recovery_.push_legs[leg]) {
            continue;
        }
        const int calf = leg * 3 + 2;
        const float ini = initial_jpos_[leg][2];
        const float fin = push_pose[calf];
        float b = 1.f;
        if (curr_iter <= max_iter && max_iter > 0) {
            b = static_cast<float>(curr_iter) / static_cast<float>(max_iter);
        }
        q_out[calf] = (1.f - b) * ini + b * fin;
    }
}

bool recovery_fsm::legs_at_pose(
    const unitree_go::msg::dds_::LowState_& state,
    const std::array<float, 12>& target,
    const std::array<bool, 4>& leg_mask) const
{
    for (int leg = 0; leg < 4; ++leg) {
        if (!leg_mask[leg]) {
            continue;
        }
        for (int j = 0; j < 3; ++j) {
            const int idx = leg * 3 + j;
            if (std::abs(state.motor_state()[idx].q() - target[idx]) >
                recovery_.joint_reach_tol) {
                return false;
            }
        }
    }
    return true;
}

standup_fsm::standup_fsm(const standup_config& stand_up, float control_hz)
    : stand_up_(stand_up),
      control_hz_(control_hz),
      warmup_ticks_(seconds_to_ticks(stand_up.warmup_s, control_hz)),
      hold_ticks_(seconds_to_ticks(stand_up.hold_s, control_hz))
{
    for (int i = 0; i < stand_up.num_phases; ++i) {
        phase_ticks_[i] =
            seconds_to_ticks(stand_up.phase_duration_s[i], control_hz);
    }
}

void standup_fsm::reset(const unitree_go::msg::dds_::LowState_& state)
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

bool standup_fsm::near_stable_pose(
    const unitree_go::msg::dds_::LowState_& state) const
{
    if (!stand_up_.configured) {
        return true;
    }
    for (int i = 0; i < 12; ++i) {
        if (std::abs(state.motor_state()[i].q() - stand_up_.stable_pose[i]) >
            stand_up_.joint_tolerance) {
            return false;
        }
    }
    return true;
}

bool standup_fsm::update(bool state_received,
                         const unitree_go::msg::dds_::LowState_& state,
                         std::array<float, 12>& q_out)
{
    if (!state_received || !stand_up_.configured || stand_up_.num_phases <= 0) {
        q_out = stand_up_.stable_pose;
        return true;
    }

    if (stage_ == standup_stage::HOLD) {
        hold_percent_ =
            clamp01(hold_percent_ + 1.f / static_cast<float>(hold_ticks_));
        q_out = stand_up_.stable_pose;
        return hold_percent_ >= 1.f;
    }

    ++tick_;

    if (stage_ == standup_stage::WARMUP) {
        if (tick_ < warmup_ticks_) {
            for (int i = 0; i < 12; ++i) {
                q_out[i] = state.motor_state()[i].q();
            }
            return false;
        }
        stage_ = standup_stage::PHASES;
    }

    if (!captured_start_) {
        for (int i = 0; i < 12; ++i) {
            start_pos_[i] = state.motor_state()[i].q();
        }
        segment_start_ = start_pos_;
        captured_start_ = true;
    }

    if (phase_index_ < stand_up_.num_phases) {
        const int duration = phase_ticks_[phase_index_];
        phase_percent_ =
            clamp01(phase_percent_ + 1.f / static_cast<float>(duration));
        const auto& target = stand_up_.keyframes[phase_index_];
        for (int i = 0; i < 12; ++i) {
            q_out[i] = (1.f - phase_percent_) * segment_start_[i] +
                       phase_percent_ * target[i];
        }
        if (phase_percent_ >= 1.f) {
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
