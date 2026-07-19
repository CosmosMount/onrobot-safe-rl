#include "controller.hpp"

#include <iostream>
#include <thread>

#include "imu_utils.hpp"
#include "motion_service.hpp"

controller::controller(int domain_id,
                       const std::string& network_interface,
                       const app_config& app,
                       const std::string& ipc_socket,
                       float control_hz)
    : config_(app.control),
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
    while (running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void controller::stop()
{
    running_ = false;
    policy_receiver_->stop();
}

void controller::begin_motion_service_if_needed(bool should_deactivate)
{
    if (should_deactivate) {
        deactivate_motion_service();
    }
}

void controller::copy_state_snapshot(unitree_go::msg::dds_::LowState_& state,
                                     bool& state_received) const
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    state = low_state_;
    state_received = state_received_;
}

void controller::blend_target_to_nominal()
{
    constexpr float max_delta_per_tick = 0.01f;
    for (size_t i = 0; i < q_target.size(); ++i) {
        const float err = config_.init_qpos[i] - q_target[i];
        if (err > max_delta_per_tick) {
            q_target[i] += max_delta_per_tick;
        } else if (err < -max_delta_per_tick) {
            q_target[i] -= max_delta_per_tick;
        } else {
            q_target[i] = config_.init_qpos[i];
        }
    }
}

void controller::start_recover(
    const unitree_go::msg::dds_::LowState_& state,
    bool state_received)
{
    if (!recovery_config_.configured || !state_received) {
        start_standup(state, state_received);
        return;
    }
    const float acc_z = imu_utils::gravity_acc_z(state.imu_state());
    if (!imu_utils::is_clearly_belly_up(acc_z,
                                        imu_config_.upside_down_acc_z_on)) {
        std::cout << "Recovery skipped (acc_z=" << acc_z
                  << "), starting stand-up." << std::endl;
        start_standup(state, state_received);
        return;
    }
    begin_motion_service_if_needed(recovery_config_.deactivate_motion_service);
    recovery_.reset(state);
    phase_ = controller_phase::RECOVER;
    std::cout << "Starting recovery. acc_z=" << acc_z << std::endl;
}

void controller::start_standup(
    const unitree_go::msg::dds_::LowState_& state,
    bool state_received)
{
    if (!stand_up_config_.configured || !state_received) {
        return;
    }
    begin_motion_service_if_needed(stand_up_config_.deactivate_motion_service);
    standup_.reset(state);
    phase_ = controller_phase::STAND_UP;
    std::cout << "Starting stand-up." << std::endl;
}

void controller::enter_policy_phase(
    const unitree_go::msg::dds_::LowState_& state)
{
    policy_receiver_->clear_pending_motion_flags();
    phase_ = controller_phase::POLICY;
    double timestamp = 0.0;
    uint8_t flags = 0;
    if (policy_receiver_->get_latest_target(policy_target, timestamp, flags)) {
        q_target = policy_target;
    } else {
        q_target = config_.init_qpos;
    }
    const float acc_z = imu_utils::gravity_acc_z(state.imu_state());
    std::cout << "Entering policy phase. acc_z=" << acc_z << std::endl;
}

void controller::control_loop()
{
    if (!running_) {
        return;
    }

    unitree_go::msg::dds_::LowState_ state{};
    bool state_received = false;
    copy_state_snapshot(state, state_received);
    const bool policy_tick = scheduler_.tick();
    (void)policy_tick;

    if (phase_ == controller_phase::AWAIT_STATE) {
        if (!state_received) {
            commander_.fill_cmd(low_cmd_, config_.init_qpos);
            cmd_pub_->Write(low_cmd_);
            return;
        }

        const float acc_z = imu_utils::gravity_acc_z(state.imu_state());
        if (recovery_config_.configured &&
            imu_utils::is_clearly_belly_up(acc_z,
                                           imu_config_.upside_down_acc_z_on)) {
            start_recover(state, state_received);
            return;
        }

        if (stand_up_config_.configured &&
            !standup_.near_stable_pose(state)) {
            start_standup(state, state_received);
            return;
        }

        enter_policy_phase(state);
        return;
    }

    if (phase_ == controller_phase::RECOVER) {
        const bool done =
            recovery_.update(state_received, state, q_target);
        commander_.fill_cmd(low_cmd_, q_target,
                            recovery_config_.kp, recovery_config_.kd);
        cmd_pub_->Write(low_cmd_);
        if (done) {
            std::cout << "Recovery FSM done, chaining stand-up." << std::endl;
            start_standup(state, state_received);
        }
        return;
    }

    if (phase_ == controller_phase::STAND_UP) {
        const bool done = standup_.update(state_received, state, q_target);
        commander_.fill_cmd(low_cmd_, q_target,
                            stand_up_config_.kp, stand_up_config_.kd);
        cmd_pub_->Write(low_cmd_);
        if (done) {
            enter_policy_phase(state);
        }
        return;
    }

    double timestamp = 0.0;
    uint8_t flags = 0;
    const bool has_target =
        policy_receiver_->get_latest_target(policy_target, timestamp, flags);

    const uint8_t motion_flags =
        policy_receiver_->consume_pending_motion_flags();
    if (motion_flags & policy_packet_t::FLAG_RECOVERY) {
        start_recover(state, state_received);
        return;
    }
    if (motion_flags & policy_packet_t::FLAG_STAND_UP) {
        start_standup(state, state_received);
        return;
    }

    const bool fresh_target =
        has_target && policy_receiver_->has_fresh_target(
                          config_.policy_timeout_ms);

    if (fresh_target) {
        q_target = policy_target;
    } else if (has_target) {
        blend_target_to_nominal();
    } else {
        q_target = config_.init_qpos;
    }

    commander_.fill_cmd(low_cmd_, q_target);
    cmd_pub_->Write(low_cmd_);
}
