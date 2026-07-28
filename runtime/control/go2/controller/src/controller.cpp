#include "controller.hpp"

namespace control
{
    controller::controller(int domain_id,
                       const std::string& network_interface,
                       const app_config& app,
                       const std::string& ipc_socket,
                       float control_hz)
        :   config_(app.control),
            imu_config_(app.imu),
            recovery_config_(app.recovery),
            stand_up_config_(app.stand_up),
            control_hz_(control_hz),
            control_dt_(1.0f / control_hz),
            scheduler_(control_hz, 20.0f),
            phase_((app.recovery.configured || app.stand_up.configured)
                        ? controller_phase::AWAIT_STATE
                        : controller_phase::POLICY),
            policy_receiver_(std::make_unique<policy_receiver>(ipc_socket)),
            commander_(config_),
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

        commander_.init(low_cmd_);
    }

    void controller::start()
    {
        running_ = true;
        q_target = config_.init_qpos;
        policy_receiver_->start();

        const int period_us = static_cast<int>(control_dt_ * 1e6);
        control_thread_ = unitree::common::CreateRecurrentThreadEx(
            "go2_control", UT_CPU_ID_NONE, period_us, &controller::control_loop, this);
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
    }
}