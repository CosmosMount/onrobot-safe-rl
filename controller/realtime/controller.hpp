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

#include "config_loader.hpp"
#include "imu_utils.hpp"
#include "lowlevel_commander.hpp"
#include "motion_service.hpp"
#include "policy_scheduler.hpp"
#include "policy_receiver.hpp"

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
               const app_config& app,
               const std::string& ipc_socket,
               float control_hz);

    void start();
    void run();
    void stop();

private:
    void control_loop();
    void enter_policy_phase(const unitree_go::msg::dds_::LowState_& state);
    void start_recover(const unitree_go::msg::dds_::LowState_& state,
                       bool state_received);
    void start_standup(const unitree_go::msg::dds_::LowState_& state,
                       bool state_received);
    void begin_motion_service_if_needed(bool should_deactivate);
    void copy_state_snapshot(unitree_go::msg::dds_::LowState_& state,
                             bool& state_received) const;
    void blend_target_to_nominal();

    control_config config_;
    imu_orientation_config imu_config_;
    recovery_config recovery_config_;
    standup_config stand_up_config_;
    float control_hz_;
    float control_dt_;
    policy_scheduler scheduler_;
    controller_phase phase_{controller_phase::AWAIT_STATE};
    std::unique_ptr<policy_receiver> policy_receiver_;
    lowlevel_commander commander_;
    recovery_fsm recovery_;
    standup_fsm standup_;

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
