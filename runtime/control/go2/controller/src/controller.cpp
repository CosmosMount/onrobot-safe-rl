#include "controller.hpp"

namespace control
{
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
            scheduler_(control_hz, 50.0f),
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
        if (policy_tick && state_received)
        {
            state_packet_t packet{};
            packet.SOF = state_packet_t::magicSOF;
            packet.phase = static_cast<uint8_t>(phase_);
            packet.timestamp = static_cast<double>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count()) * 1e-9;
            packet.low_state_count = ++state_publish_count_;
            packet.sport_state_count = sport_state_received ? 1u : 0u;
            for (size_t i = 0; i < 12; ++i)
            {
                packet.joint_q[i] = state.motor_state()[i].q();
                packet.joint_dq[i] = state.motor_state()[i].dq();
                packet.q_target[i] = q_target[i];
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

                const float acc_z = state.imu_state().accelerometer()[2];
                if (acc_z < imu_thresholds_.upside_down_acc_z_on)
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
                    phase_ = controller_phase::POLICY;
                }
                break;
            }
                
            case controller_phase::POLICY:
            {
                double timestamp = 0.0;
                uint8_t flags = 0;

                const uint8_t motion_flags = state_received
                    ? policy_receiver_->consume_pending_motion_flags()
                    : 0u;
                const bool upside_down = state.imu_state().accelerometer()[2] < imu_thresholds_.upside_down_acc_z_on;
                if ((motion_flags & policy_packet_t::FLAG_RECOVERY) && upside_down)
                {
                    enter_recover(state);
                }
                else if (motion_flags & policy_packet_t::FLAG_RECOVERY)
                {
                    std::cout << "[controller] ignored stale recovery request phase="
                            << static_cast<int>(phase_)
                            << " acc_z=" << state.imu_state().accelerometer()[2]
                            << std::endl;
                }
                else if (motion_flags & policy_packet_t::FLAG_STAND_UP)
                {
                    enter_stand_up(state);
                }


                const bool has_target = policy_receiver_->get_latest_target(policy_target, timestamp, flags);

                const bool fresh_target = has_target 
                                        && policy_receiver_->has_fresh_target(config_.policy_timeout_ms);

                if (fresh_target) 
                {
                    q_target = policy_target;
                } 
                else if (has_target) 
                {
                    constexpr float max_delta_per_tick = 0.01f;
                    for (size_t i = 0; i < q_target.size(); ++i) 
                    {
                        const float err = config_.init_qpos[i] - q_target[i];
                        if (err > max_delta_per_tick) 
                        {
                            q_target[i] += max_delta_per_tick;
                        } 
                        else if (err < -max_delta_per_tick) 
                        {
                            q_target[i] -= max_delta_per_tick;
                        } 
                        else 
                        {
                            q_target[i] = config_.init_qpos[i];
                        }
                    }
                } 
                else 
                {
                    q_target = config_.init_qpos;
                }

                cmd_.fill(low_cmd_, q_target);
                cmd_pub_->Write(low_cmd_);
                break;
            }
        }
        
    }
}
