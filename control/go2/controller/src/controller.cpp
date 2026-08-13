#include "controller.hpp"

#include <cmath>
#include <iostream>

namespace control
{
    namespace
    {
        void roll_pitch(
            const unitree_go::msg::dds_::LowState_& state,
            float& roll,
            float& pitch)
        {
            const auto& q = state.imu_state().quaternion();
            float w = q[0];
            float x = q[1];
            float y = q[2];
            float z = q[3];
            const float norm_sq = w * w + x * x + y * y + z * z;
            if (norm_sq < 1e-6f)
            {
                roll = 0.0f;
                pitch = 0.0f;
                return;
            }
            const float inv_norm = 1.0f / std::sqrt(norm_sq);
            w *= inv_norm;
            x *= inv_norm;
            y *= inv_norm;
            z *= inv_norm;

            roll = std::atan2(
                2.0f * (w * x + y * z),
                1.0f - 2.0f * (x * x + y * y));
            const float sinp = 2.0f * (w * y - z * x);
            if (std::abs(sinp) >= 1.0f)
            {
                constexpr float half_pi = 1.57079632679f;
                pitch = std::copysign(half_pi, sinp);
            }
            else
            {
                pitch = std::asin(sinp);
            }
        }

        float body_up_cos(const unitree_go::msg::dds_::LowState_& state)
        {
            const auto& q = state.imu_state().quaternion();
            const float w = q[0];
            const float x = q[1];
            const float y = q[2];
            const float z = q[3];
            const float norm_sq = w * w + x * x + y * y + z * z;
            if (norm_sq < 1e-6f)
            {
                return 1.0f;
            }
            return 1.0f - 2.0f * (x * x + y * y) / norm_sq;
        }

        bool is_fallen(
            const unitree_go::msg::dds_::LowState_& state,
            const motions::imu_thresholds& thresholds)
        {
            float roll = 0.0f;
            float pitch = 0.0f;
            roll_pitch(state, roll, pitch);
            return std::abs(roll) > thresholds.fallen_roll_pitch_limit_rad ||
                std::abs(pitch) > thresholds.fallen_roll_pitch_limit_rad;
        }

        bool is_upside_down(
            const unitree_go::msg::dds_::LowState_& state,
            const motions::imu_thresholds& thresholds)
        {
            const float up_cos = body_up_cos(state);
            const bool pose_inverted = up_cos < thresholds.upside_down_up_cos_on;
            const bool accel_inverted =
                state.imu_state().accelerometer()[2] < thresholds.upside_down_acc_z_on;
            return pose_inverted || (accel_inverted && is_fallen(state, thresholds));
        }

        bool accepts_recovery_motion(
            const unitree_go::msg::dds_::LowState_& state,
            const motions::imu_thresholds& thresholds)
        {
            return is_upside_down(state, thresholds) || is_fallen(state, thresholds);
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

    void controller::enter_recover(const unitree_go::msg::dds_::LowState_& state)
    {
        recovery_.reset(state);
        phase_ = controller_phase::RECOVER;
    }

    void controller::enter_stand_up(const unitree_go::msg::dds_::LowState_& state)
    {
        standup_.reset(state);
        phase_ = controller_phase::STAND_UP;
    }

    void controller::loop()
    {
        if (!running_)
            return;
        
        unitree_go::msg::dds_::LowState_ state{};
        unitree_go::msg::dds_::SportModeState_ sport_state{};
        bool state_received = false;
        bool sport_state_received = false;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            state = low_state_;
            sport_state = sport_state_;
            state_received = state_received_;
            sport_state_received = sport_state_received_;
        }

        const bool policy_tick = scheduler_.tick();

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

                if (is_upside_down(state, imu_thresholds_))
                {
                    enter_recover(state);
                }
                else if (!standup_.near_stable_pose(state))
                {
                    enter_stand_up(state);
                }
                else
                {
                    phase_ = controller_phase::POLICY;
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
                const bool done = standup_.update(state_received, state, q_target);
                cmd_.fill(low_cmd_, q_target);
                cmd_pub_->Write(low_cmd_);
                if (done)
                {
                    std::cout << "stand-up done, entering policy phase." << std::endl;
                    // Do not replay a reset-time stand-up request after the
                    // controller has reached the policy phase.
                    policy_receiver_->clear_pending_motion_flags();
                    phase_ = controller_phase::POLICY;
                }
                break;
            }
                
            case controller_phase::POLICY:
            {
                // Safety recovery must not depend on the learner sending a
                // recovery flag.  Once the controller is in POLICY, detect
                // the same fallen/upside-down condition used during startup
                // and enter the autonomous recovery state immediately.
                if (state_received && accepts_recovery_motion(state, imu_thresholds_))
                {
                    enter_recover(state);
                    break;
                }

                double timestamp = 0.0;
                uint8_t flags = 0;
                uint64_t action_id = 0;

                const uint8_t motion_flags = state_received
                    ? policy_receiver_->consume_pending_motion_flags()
                    : 0u;
                const bool recovery_needed = accepts_recovery_motion(state, imu_thresholds_);
                if ((motion_flags & policy_packet_t::FLAG_RECOVERY) && recovery_needed)
                {
                    enter_recover(state);
                }
                else if (motion_flags & policy_packet_t::FLAG_RECOVERY)
                {
                    std::cout << "[controller] ignored stale recovery request phase="
                            << static_cast<int>(phase_)
                            << " acc_z=" << state.imu_state().accelerometer()[2]
                            << " up_cos=" << body_up_cos(state)
                            << std::endl;
                }
                else if (motion_flags & policy_packet_t::FLAG_STAND_UP)
                {
                    enter_stand_up(state);
                }


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
            packet.policy_sequence = scheduler_.policy_sequence();
            packet.applied_action_id = applied_action_id_;
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
