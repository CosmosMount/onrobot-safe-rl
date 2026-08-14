#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>

#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/go2/SportModeState_.hpp>
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
        RETURN_HOME,
        SHUTDOWN,
        SAFETY_HOLD,
    };

    class controller
    {
    public:

        controller(int domain_id,
                const std::string& network_interface,
                const control::app_config& app,
                const std::string& ipc_socket,
                const std::string& state_socket,
                float control_hz);

        void start();
        void run();
        void stop();

    private:

        void loop();
        void enter_recover(
            const unitree_go::msg::dds_::LowState_& state,
            transition_event event = transition_event::NONE);
        void enter_stand_up(
            const unitree_go::msg::dds_::LowState_& state,
            transition_event event = transition_event::NONE);
        void enter_return_home(const unitree_go::msg::dds_::LowState_& state);
        void enter_safety_hold();
        void update_safety_state(
            const unitree_go::msg::dds_::LowState_& state,
            uint64_t state_generation);
        void set_transition_event(transition_event event);
        void clear_transition_event();

        lowlevel::config config_;
        motions::imu_thresholds imu_thresholds_;

        float control_hz_;
        float control_dt_;
        policy_scheduler scheduler_;
        controller_phase phase_{controller_phase::AWAIT_STATE};
        std::unique_ptr<policy_receiver> policy_receiver_;
        std::unique_ptr<state_publisher> state_publisher_;

        lowlevel::cmd cmd_;

        motions::recovery_config recovery_config_;
        motions::standup_config stand_up_config_;
        motions::recovery recovery_;
        motions::standup standup_;

        std::array<float, 12> q_target{};
        std::array<float, 12> policy_target{};
        uint64_t applied_action_id_{0};
        uint32_t state_publish_count_{0};
        bool state_received_{false};
        uint64_t state_generation_{0};
        mutable std::mutex state_mutex_;

        unitree_go::msg::dds_::LowCmd_ low_cmd_{};
        unitree_go::msg::dds_::LowState_ low_state_{};
        unitree_go::msg::dds_::SportModeState_ sport_state_{};
        bool sport_state_received_{false};
        bool shutdown_requested_{false};
        std::array<float, 12> shutdown_start_q_{};
        uint32_t shutdown_tick_{0};

        uint64_t last_safety_generation_{0};
        float current_roll_{0.f};
        float current_pitch_{0.f};
        float current_up_cos_{1.f};
        float current_acc_z_{9.8f};
        bool fallen_raw_{false};
        bool upside_down_raw_{false};
        bool fallen_confirmed_{false};
        bool upside_down_confirmed_{false};
        bool fallen_timer_active_{false};
        bool upside_down_timer_active_{false};
        std::chrono::steady_clock::time_point fallen_since_{};
        std::chrono::steady_clock::time_point upside_down_since_{};

        bool standup_motion_done_{false};
        // A normal side-fall gets one stand-up attempt first.  If the robot
        // is still confirmed fallen when that motion finishes, allow one
        // bounded recovery fallback; never loop recovery indefinitely.
        bool standup_fallback_recovery_used_{false};
        bool standup_stable_timer_active_{false};
        std::chrono::steady_clock::time_point standup_verify_started_{};
        std::chrono::steady_clock::time_point standup_stable_since_{};

        transition_event current_event_{transition_event::NONE};
        uint64_t event_action_id_{0};
        uint32_t event_confirm_ms_{0};

        unitree::robot::ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> cmd_pub_;
        unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> state_sub_;
        unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::SportModeState_> sport_state_sub_;
        unitree::common::ThreadPtr control_thread_;

        std::atomic<bool> running_{false};
    };
}
