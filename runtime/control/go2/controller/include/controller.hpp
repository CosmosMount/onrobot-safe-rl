#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>

#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/common/thread/thread.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include "policy.hpp"
#include "motions.hpp"
#include "recovery.hpp"
#include "standup.hpp"
#include "lowlevel.hpp"
#include "config.hpp"

namespace control
{
    enum class controller_phase
    {
        AWAIT_STATE,
        RECOVER,
        STAND_UP,
        POLICY,
    };

    class controller
    {
    public:

        controller(int domain_id,
                const std::string& network_interface,
                const control::app_config& app,
                const std::string& ipc_socket,
                float control_hz);

        void start();
        void run();
        void stop();

    private:

        void loop();

        lowlevel::config config_;
        motions::imu_thresholds imu_thresholds_;

        float control_hz_;
        float control_dt_;
        policy_scheduler scheduler_;
        controller_phase phase_{controller_phase::AWAIT_STATE};
        std::unique_ptr<policy_receiver> policy_receiver_;

        lowlevel::cmd cmd_;

        motions::recovery_config recovery_config_;
        motions::standup_config stand_up_config_;
        motions::recovery recovery_;
        motions::standup standup_;

        std::array<float, 12> q_target{};
        std::array<float, 12> policy_target{};
        bool state_received_{false};
        mutable std::mutex state_mutex_;

        unitree_go::msg::dds_::LowCmd_ low_cmd_{};
        unitree_go::msg::dds_::LowState_ low_state_{};

        unitree::robot::ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> cmd_pub_;
        unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> state_sub_;
        unitree::common::ThreadPtr control_thread_;

        std::atomic<bool> running_{false};
    };
}
