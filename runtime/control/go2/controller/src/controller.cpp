#include "controller.hpp"

namespace control
{
    controller::controller(int domain_id,
                       const std::string& network_interface,
                       const app_config& app,
                       const std::string& ipc_socket,
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

    void controller::loop()
    {
        if (!running_)
            return;
        
        unitree_go::msg::dds_::LowState_ state{};
        bool state_received = false;
        std::lock_guard<std::mutex> lock(state_mutex_);
        state = low_state_;
        state_received = state_received_;

        const bool policy_tick = scheduler_.tick();
        (void)policy_tick;

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
                    phase_ = controller_phase::RECOVER;
                }
                else if (!standup_.near_stable_pose(state))
                {
                    phase_ = controller_phase::STAND_UP;
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
                    phase_ = controller_phase::STAND_UP;
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
                const bool has_target = policy_receiver_->get_latest_target(policy_target, timestamp, flags);
                const uint8_t motion_flags = policy_receiver_->consume_pending_motion_flags();

                if (motion_flags & policy_packet_t::FLAG_RECOVERY) 
                {
                    phase_ = controller_phase::RECOVER;
                    break;
                }

                if (motion_flags & policy_packet_t::FLAG_STAND_UP) 
                {
                    phase_ = controller_phase::STAND_UP;
                    break;
                }

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
