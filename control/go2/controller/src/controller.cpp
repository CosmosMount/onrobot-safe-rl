#include "controller.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

#include "safety.hpp"

namespace control
{
    namespace
    {
        int64_t elapsed_ms(
            const std::chrono::steady_clock::time_point& since,
            const std::chrono::steady_clock::time_point& now)
        {
            return std::chrono::duration_cast<std::chrono::milliseconds>(
                now - since).count();
        }
    }

    controller::controller(int domain_id,
                       const std::string& network_interface,
                       const app_config& app,
                       const std::string& ipc_socket,
                       const std::string& state_socket,
                       float control_hz)
        :   config_(app.control),
            imu_thresholds_(app.imu),
            recovery_config_(app.recovery),
            stand_up_config_(app.stand_up),
            control_hz_(control_hz),
            control_dt_(1.0f / control_hz),
            scheduler_(control_hz, 20.0f),
            phase_(controller_phase::AWAIT_STATE),
            policy_receiver_(std::make_unique<policy_receiver>(ipc_socket)),
            state_publisher_(std::make_unique<state_publisher>(state_socket)),
            cmd_(config_),
            recovery_(app.recovery, control_hz),
            standup_(app.stand_up, control_hz)
    {
        unitree::robot::ChannelFactory::Instance()->Init(domain_id, network_interface);
        cmd_pub_.reset(new unitree::robot::ChannelPublisher<unitree_go::msg::dds_::LowCmd_>("rt/lowcmd"));
        cmd_pub_->InitChannel();

        state_sub_.reset(new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::LowState_>("rt/lowstate"));
        state_sub_->InitChannel([this](const void* data) {
            std::lock_guard<std::mutex> lock(state_mutex_);
            low_state_ = *static_cast<const unitree_go::msg::dds_::LowState_*>(data);
            state_received_ = true;
            ++state_generation_;
        });

        sport_state_sub_.reset(new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>("rt/sportmodestate"));
        sport_state_sub_->InitChannel([this](const void* data) {
            std::lock_guard<std::mutex> lock(state_mutex_);
            sport_state_ = *static_cast<const unitree_go::msg::dds_::SportModeState_*>(data);
            sport_state_received_ = true;
        });

        cmd_.init(low_cmd_);
        std::cout << "[controller] policy/controller joint mapping" << std::endl;
        for (size_t policy_index = 0; policy_index < 12; ++policy_index)
        {
            std::cout << "  policy[" << policy_index << "] -> motor["
                      << go2_layout::kPolicyToMotorIndex[policy_index] << "]"
                      << std::endl;
        }
    }

    void controller::start()
    {
        running_ = true;
        q_target = config_.init_qpos;
        policy_receiver_->start();

        const int period_us = static_cast<int>(control_dt_ * 1e6);
        control_thread_ = unitree::common::CreateRecurrentThreadEx(
            "go2_control", UT_CPU_ID_NONE, period_us, &controller::loop, this);
    }

    void controller::run()
    {
        while (running_) 
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    void controller::stop()
    {
        running_ = false;
        policy_receiver_->stop();
    }

    void controller::set_transition_event(transition_event event)
    {
        if (event == transition_event::NONE)
            return;
        current_event_ = event;
        event_action_id_ = applied_action_id_;
        const auto now = std::chrono::steady_clock::now();
        if (event == transition_event::FALLEN_STANDUP &&
            fallen_timer_active_)
            event_confirm_ms_ = static_cast<uint32_t>(
                std::max<int64_t>(0, elapsed_ms(fallen_since_, now)));
        else if (event == transition_event::UPSIDE_DOWN_RECOVERY &&
                 upside_down_timer_active_)
            event_confirm_ms_ = static_cast<uint32_t>(
                std::max<int64_t>(0, elapsed_ms(upside_down_since_, now)));
        else
            event_confirm_ms_ = 0;
        std::cout << "[safety] event=" << static_cast<int>(event)
                  << " action_id=" << event_action_id_
                  << " roll=" << current_roll_
                  << " pitch=" << current_pitch_
                  << " up_cos=" << current_up_cos_
                  << " acc_z=" << current_acc_z_
                  << " confirm_ms=" << event_confirm_ms_
                  << std::endl;
    }

    void controller::clear_transition_event()
    {
        current_event_ = transition_event::NONE;
        event_action_id_ = 0;
        event_confirm_ms_ = 0;
    }

    void controller::update_safety_state(
        const unitree_go::msg::dds_::LowState_& state,
        uint64_t state_generation)
    {
        if (state_generation == last_safety_generation_)
            return;
        last_safety_generation_ = state_generation;

        const auto sample = safety::measure(state);
        current_roll_ = sample.roll;
        current_pitch_ = sample.pitch;
        current_up_cos_ = sample.up_cos;
        current_acc_z_ = sample.acc_z;
        fallen_raw_ = safety::is_fallen(sample, imu_thresholds_);
        upside_down_raw_ = safety::is_upside_down(sample, imu_thresholds_);
        const auto now = std::chrono::steady_clock::now();

        if (fallen_raw_)
        {
            if (!fallen_timer_active_)
            {
                fallen_timer_active_ = true;
                fallen_since_ = now;
            }
            fallen_confirmed_ = elapsed_ms(fallen_since_, now) >=
                imu_thresholds_.fallen_confirm_ms;
        }
        else
        {
            fallen_timer_active_ = false;
            fallen_confirmed_ = false;
        }

        if (upside_down_raw_)
        {
            if (!upside_down_timer_active_)
            {
                upside_down_timer_active_ = true;
                upside_down_since_ = now;
            }
            upside_down_confirmed_ = elapsed_ms(upside_down_since_, now) >=
                imu_thresholds_.upside_down_confirm_ms;
        }
        else
        {
            upside_down_timer_active_ = false;
            upside_down_confirmed_ = false;
        }
    }

    void controller::enter_recover(
        const unitree_go::msg::dds_::LowState_& state,
        transition_event event)
    {
        set_transition_event(event);
        recovery_.reset(state);
        // Once recovery is running, a subsequent stand-up must not start a
        // second recovery fallback if that complete lifecycle also fails.
        standup_fallback_recovery_used_ = true;
        phase_ = controller_phase::RECOVER;
    }

    void controller::enter_stand_up(
        const unitree_go::msg::dds_::LowState_& state,
        transition_event event)
    {
        set_transition_event(event);
        standup_.reset(state);
        standup_motion_done_ = false;
        standup_stable_timer_active_ = false;
        if (event == transition_event::FALLEN_STANDUP)
            standup_fallback_recovery_used_ = false;
        phase_ = controller_phase::STAND_UP;
    }

    void controller::enter_safety_hold()
    {
        set_transition_event(transition_event::STANDUP_FAILED);
        q_target = config_.init_qpos;
        phase_ = controller_phase::SAFETY_HOLD;
        policy_receiver_->clear_latest_target();
        std::cerr << "[safety] stand-up verification timed out; "
                     "holding initial pose and rejecting policy actions."
                  << std::endl;
    }

    void controller::enter_return_home(
        const unitree_go::msg::dds_::LowState_& state)
    {
        for (size_t i = 0; i < shutdown_start_q_.size(); ++i)
            shutdown_start_q_[i] = state.motor_state()[i].q();
        shutdown_tick_ = 0;
        phase_ = controller_phase::RETURN_HOME;
    }

    void controller::loop()
    {
        if (!running_)
            return;
        
        unitree_go::msg::dds_::LowState_ state{};
        unitree_go::msg::dds_::SportModeState_ sport_state{};
        bool state_received = false;
        bool sport_state_received = false;
        uint64_t state_generation = 0;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            state = low_state_;
            sport_state = sport_state_;
            state_received = state_received_;
            sport_state_received = sport_state_received_;
            state_generation = state_generation_;
        }

        const bool new_state = state_received &&
            state_generation != last_safety_generation_;
        if (state_received)
            update_safety_state(state, state_generation);

        const bool policy_tick = scheduler_.tick();

        uint64_t stop_action_id = 0;
        if (policy_receiver_->consume_pending_stop(stop_action_id))
        {
            shutdown_requested_ = true;
            applied_action_id_ = std::max(applied_action_id_, stop_action_id);
            policy_receiver_->clear_latest_target();
            if (state_received &&
                phase_ == controller_phase::POLICY)
            {
                if (upside_down_confirmed_)
                    enter_recover(state);
                else if (fallen_confirmed_)
                    enter_stand_up(state);
                else
                    enter_return_home(state);
            }
        }

        // A new learner session explicitly requests STAND_UP from the
        // terminal hold state.  Re-arm only on that lifecycle request; the
        // shutdown path itself never loops back into stand-up.
        if (state_received && phase_ == controller_phase::SHUTDOWN)
        {
            const uint8_t motion_flags =
                policy_receiver_->consume_pending_motion_flags();
            if (motion_flags & policy_packet_t::FLAG_STAND_UP)
            {
                shutdown_requested_ = false;
                clear_transition_event();
                phase_ = controller_phase::AWAIT_STATE;
            }
        }

        switch (phase_)
        {
            case controller_phase::AWAIT_STATE:
            {
                if (!state_received) 
                {
                    cmd_.fill(low_cmd_, config_.init_qpos);
                    cmd_pub_->Write(low_cmd_);
                    return;
                }

                if (shutdown_requested_)
                {
                    if (upside_down_confirmed_)
                        enter_recover(state);
                    else if (fallen_confirmed_)
                        enter_stand_up(state);
                    else
                        enter_return_home(state);
                    break;
                }

                // Use the same recovery predicate at startup as in POLICY.
                // A robot that is merely away from the stable pose should
                // execute stand-up only; recovery is reserved for a fallen
                // or upside-down robot.
                if (upside_down_confirmed_)
                {
                    enter_recover(
                        state, transition_event::UPSIDE_DOWN_RECOVERY);
                }
                else if (fallen_confirmed_)
                {
                    enter_stand_up(
                        state, transition_event::FALLEN_STANDUP);
                }
                else if (upside_down_raw_ || fallen_raw_)
                {
                    // Wait for a sustained classification while holding the
                    // measured pose; never route a single IMU sample into a
                    // recovery motion.
                    for (size_t i = 0; i < q_target.size(); ++i)
                        q_target[i] = state.motor_state()[i].q();
                    cmd_.fill(low_cmd_, q_target);
                    cmd_pub_->Write(low_cmd_);
                }
                else if (!standup_.near_stable_pose(state))
                {
                    if (shutdown_requested_)
                        enter_return_home(state);
                    else
                        enter_stand_up(state);
                }
                else
                {
                    clear_transition_event();
                    phase_ = shutdown_requested_
                        ? controller_phase::SHUTDOWN
                        : controller_phase::POLICY;
                }
                break;
            }
                
            case controller_phase::RECOVER:
            {
                const bool done = recovery_.update(state_received, state, q_target);
                cmd_.fill(low_cmd_, q_target);
                cmd_pub_->Write(low_cmd_);
                if (done)
                {
                    std::cout << "recovery done, chaining stand-up." << std::endl;
                    enter_stand_up(state);
                }
                break;
            }
                
            case controller_phase::STAND_UP:
            {
                bool done = standup_motion_done_;
                if (!standup_motion_done_)
                {
                    done = standup_.update(state_received, state, q_target);
                    if (done)
                    {
                        standup_motion_done_ = true;
                        standup_verify_started_ =
                            std::chrono::steady_clock::now();
                        standup_stable_timer_active_ = false;
                        std::cout << "[standup] motion complete; verifying "
                                     "stable pose before POLICY."
                                  << std::endl;
                    }
                }
                else
                {
                    q_target = config_.init_qpos;
                }
                cmd_.fill(low_cmd_, q_target);
                cmd_pub_->Write(low_cmd_);
                if (done && state_received)
                {
                    const auto now = std::chrono::steady_clock::now();
                    if (new_state)
                    {
                        // A stand-up motion can finish while the robot is
                        // still belly-up (for example when the initial
                        // orientation was classified before the IMU
                        // confirmation window elapsed). Do not wait for the
                        // generic stand-up timeout in that case: the only
                        // valid next lifecycle action is belly-up recovery.
                        if (upside_down_confirmed_ &&
                            !standup_fallback_recovery_used_)
                        {
                            std::cout << "[safety] stand-up verification "
                                         "still upside-down; restarting "
                                         "recovery sequence." << std::endl;
                            enter_recover(
                                state,
                                transition_event::UPSIDE_DOWN_RECOVERY);
                            break;
                        }

                        // The ordinary stand-up trajectory is deliberately
                        // tried first for a side fall.  On hardware it can
                        // finish with the joints at the target while the
                        // body is still on its side, which used to spend the
                        // whole 5 s verification timeout before entering
                        // SAFETY_HOLD.  Use the stronger recovery trajectory
                        // exactly once as a bounded fallback.  This does not
                        // classify ordinary falls as upside-down and cannot
                        // create a recovery/stand-up loop.
                        if (fallen_confirmed_ &&
                            !standup_fallback_recovery_used_)
                        {
                            standup_fallback_recovery_used_ = true;
                            std::cout << "[safety] stand-up completed while "
                                         "still fallen; running one bounded "
                                         "recovery fallback." << std::endl;
                            enter_recover(state);
                            break;
                        }

                        const bool orientation_stable =
                            std::abs(current_roll_) <
                                imu_thresholds_.stable_roll_pitch_limit_rad &&
                            std::abs(current_pitch_) <
                                imu_thresholds_.stable_roll_pitch_limit_rad;
                        const bool stable = orientation_stable &&
                            standup_.near_stable_pose(state);
                        if (stable && !standup_stable_timer_active_)
                        {
                            standup_stable_timer_active_ = true;
                            standup_stable_since_ = now;
                        }
                        else if (!stable)
                        {
                            standup_stable_timer_active_ = false;
                        }
                    }

                    if (new_state && standup_stable_timer_active_ &&
                        elapsed_ms(standup_stable_since_, now) >=
                            imu_thresholds_.stable_confirm_ms)
                    {
                        std::cout << "stand-up stable, leaving stand-up phase."
                                  << std::endl;
                        policy_receiver_->clear_pending_motion_flags();
                        clear_transition_event();
                        if (shutdown_requested_)
                            enter_return_home(state);
                        else
                            phase_ = controller_phase::POLICY;
                    }
                    else if (elapsed_ms(standup_verify_started_, now) >=
                        imu_thresholds_.standup_verify_timeout_ms)
                    {
                        enter_safety_hold();
                    }
                }
                break;
            }
                
            case controller_phase::POLICY:
            {
                if (shutdown_requested_)
                {
                    if (state_received)
                        enter_return_home(state);
                    break;
                }
                // Safety recovery must not depend on the learner sending a
                // recovery flag.  Once the controller is in POLICY, detect
                // the same fallen/upside-down condition used during startup
                // and enter the autonomous recovery state immediately.
                if (state_received && upside_down_confirmed_)
                {
                    enter_recover(
                        state, transition_event::UPSIDE_DOWN_RECOVERY);
                    break;
                }
                if (state_received && fallen_confirmed_)
                {
                    enter_stand_up(
                        state, transition_event::FALLEN_STANDUP);
                    break;
                }

                double timestamp = 0.0;
                uint8_t flags = 0;
                uint64_t action_id = 0;

                const uint8_t motion_flags = state_received
                    ? policy_receiver_->consume_pending_motion_flags()
                    : 0u;
                if ((motion_flags & policy_packet_t::FLAG_RECOVERY) &&
                    upside_down_confirmed_)
                {
                    enter_recover(state);
                }
                else if (motion_flags & policy_packet_t::FLAG_RECOVERY)
                {
                    // A non-inverted fallen robot needs stand-up, not the
                    // aggressive belly-up recovery sequence.
                    if (fallen_confirmed_)
                        enter_stand_up(state);
                    else
                        std::cout << "[controller] ignored stale recovery request phase="
                                  << static_cast<int>(phase_)
                                  << " acc_z=" << current_acc_z_
                                  << " up_cos=" << current_up_cos_
                                  << std::endl;
                }
                else if (motion_flags & policy_packet_t::FLAG_STAND_UP)
                {
                    enter_stand_up(state);
                }

                if (phase_ != controller_phase::POLICY)
                    break;

                const bool has_target = policy_receiver_->get_latest_target(
                    policy_target, timestamp, flags, action_id);

                const bool fresh_target = has_target 
                                        && policy_receiver_->has_fresh_target(config_.policy_timeout_ms);

                if (fresh_target && action_id > applied_action_id_)
                {
                    q_target = policy_target;
                    applied_action_id_ = action_id;
                }
                else
                {
                    // A stale/unknown target is held exactly.  Do not
                    // synthesize a different target while retaining the old
                    // action_id: the learner must either see the same action
                    // being held or fail closed on the id mismatch. Recovery
                    // and stand-up own their targets in their phases.
                }

                cmd_.fill(low_cmd_, q_target);
                cmd_pub_->Write(low_cmd_);
                break;
            }

            case controller_phase::RETURN_HOME:
            {
                constexpr uint32_t kReturnHomeTicks = 1000;
                const float alpha = std::min(
                    1.0f, static_cast<float>(shutdown_tick_ + 1) /
                              static_cast<float>(kReturnHomeTicks));
                for (size_t i = 0; i < q_target.size(); ++i)
                {
                    q_target[i] = (1.0f - alpha) * shutdown_start_q_[i] +
                                  alpha * config_.init_qpos[i];
                }
                cmd_.fill(low_cmd_, q_target);
                cmd_pub_->Write(low_cmd_);
                ++shutdown_tick_;
                if (shutdown_tick_ >= kReturnHomeTicks)
                    phase_ = controller_phase::SHUTDOWN;
                break;
            }

            case controller_phase::SHUTDOWN:
            {
                // Terminal hold state: no stand-up sequence and no learner
                // target can be applied after this point.
                q_target = config_.init_qpos;
                cmd_.fill(low_cmd_, q_target);
                cmd_pub_->Write(low_cmd_);
                break;
            }

            case controller_phase::SAFETY_HOLD:
            {
                // Latched terminal safety state. Only a controller restart
                // can re-arm policy execution after a failed stand-up.
                q_target = config_.init_qpos;
                cmd_.fill(low_cmd_, q_target);
                cmd_pub_->Write(low_cmd_);
                break;
            }
        }

        // Publish after consuming the action for this policy tick.  This
        // makes applied_action_id an exact causal acknowledgement: the
        // state received by the learner describes the target applied during
        // the interval that follows the action it sent.
        if (policy_tick && state_received)
        {
            state_packet_t packet{};
            packet.SOF = state_packet_t::magicSOF;
            packet.phase = static_cast<uint8_t>(phase_);
            packet.event = static_cast<uint8_t>(current_event_);
            packet.policy_sequence = scheduler_.policy_sequence();
            packet.applied_action_id = applied_action_id_;
            packet.event_action_id = event_action_id_;
            packet.event_confirm_ms = event_confirm_ms_;
            packet.timestamp = static_cast<double>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count()) * 1e-9;
            packet.low_state_count = ++state_publish_count_;
            packet.sport_state_count = sport_state_received ? 1u : 0u;
            for (size_t i = 0; i < 12; ++i)
            {
                const int policy_index = go2_layout::kMotorToPolicyIndex[i];
                packet.joint_q[policy_index] = state.motor_state()[i].q();
                packet.joint_dq[policy_index] = state.motor_state()[i].dq();
                packet.q_target[policy_index] = q_target[policy_index];
            }
            for (size_t i = 0; i < 4; ++i)
            {
                packet.imu_quat[i] = state.imu_state().quaternion()[i];
            }
            for (size_t i = 0; i < 3; ++i)
            {
                packet.imu_gyro[i] = state.imu_state().gyroscope()[i];
                packet.imu_accel[i] = state.imu_state().accelerometer()[i];
                packet.sport_velocity[i] = sport_state_received ? sport_state.velocity()[i] : 0.0f;
                packet.world_position[i] = sport_state_received ? sport_state.position()[i] : 0.0f;
            }
            state_publisher_->publish(packet);
        }
        
    }
}
