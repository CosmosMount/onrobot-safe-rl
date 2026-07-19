#pragma once

#include <array>

#include <unitree/idl/go2/LowState_.hpp>

bool deactivate_motion_service();

struct recovery_config
{
    bool configured = false;
    std::array<float, 12> fold_jpos{};
    std::array<float, 12> above_jpos{};
    std::array<float, 12> swing_down_jpos{};
    std::array<float, 12> push_jpos{};
    std::array<bool, 4> swing_legs{{true, true, true, true}};
    std::array<bool, 4> push_legs{{false, false, true, true}};
    float fold_ramp_s = 0.45f;
    float fold_settle_s = 0.50f;
    float above_ramp_s = 0.45f;
    float above_settle_s = 0.35f;
    float swing_down_ramp_s = 0.55f;
    float swing_down_settle_s = 0.45f;
    float push_ramp_s = 0.30f;
    float push_settle_s = 0.25f;
    float joint_reach_tol = 0.12f;
    float kp = 100.f;
    float kd = 8.f;
    bool deactivate_motion_service = false;
};

struct standup_config
{
    bool configured = false;
    static constexpr int kMaxPhases = 4;
    int num_phases = 0;
    std::array<std::array<float, 12>, kMaxPhases> keyframes{};
    std::array<float, kMaxPhases> phase_duration_s{};
    float warmup_s = 1.0f;
    float hold_s = 0.2f;
    std::array<float, 12> stable_pose{};
    float joint_tolerance = 0.15f;
    float kp = 60.f;
    float kd = 5.f;
    bool deactivate_motion_service = false;
};

enum class recovery_stage
{
    FOLD,
    ABOVE,
    SWING_DOWN,
    PUSH,
};

enum class standup_stage
{
    WARMUP,
    PHASES,
    HOLD,
};

class recovery_fsm
{
public:
    explicit recovery_fsm(const recovery_config& recovery, float control_hz);

    void reset(const unitree_go::msg::dds_::LowState_& state);
    bool update(bool state_received,
                const unitree_go::msg::dds_::LowState_& state,
                std::array<float, 12>& q_out);

private:
    void capture_jpos(const unitree_go::msg::dds_::LowState_& state);
    void begin_segment(const std::array<float, 12>& start_jpos);
    void interpolate_leg(int leg,
                         int curr_iter,
                         int max_iter,
                         const std::array<float, 3>& ini,
                         const std::array<float, 3>& fin,
                         std::array<float, 12>& q_out) const;
    void interpolate_all(int curr_iter,
                         int max_iter,
                         const std::array<float, 12>& fin,
                         std::array<float, 12>& q_out) const;
    void apply_leg_pose(const std::array<float, 12>& pose,
                        const std::array<bool, 4>& leg_mask,
                        std::array<float, 12>& q_out) const;
    void interpolate_active(int curr_iter,
                              int max_iter,
                              const std::array<float, 12>& active_target,
                              std::array<float, 12>& q_out) const;
    void interpolate_calf_push(int curr_iter,
                                 int max_iter,
                                 const std::array<float, 12>& push_pose,
                                 std::array<float, 12>& q_out) const;
    bool legs_at_pose(const unitree_go::msg::dds_::LowState_& state,
                      const std::array<float, 12>& target,
                      const std::array<bool, 4>& leg_mask) const;

    recovery_config recovery_;
    float control_hz_;
    int fold_ramp_ticks_;
    int fold_settle_ticks_;
    int above_ramp_ticks_;
    int above_settle_ticks_;
    int swing_down_ramp_ticks_;
    int swing_down_settle_ticks_;
    int push_ramp_ticks_;
    int push_settle_ticks_;
    recovery_stage stage_{recovery_stage::FOLD};
    int tick_{0};
    int segment_start_tick_{0};
    std::array<std::array<float, 3>, 4> initial_jpos_{};
};

class standup_fsm
{
public:
    standup_fsm(const standup_config& stand_up, float control_hz);

    void reset(const unitree_go::msg::dds_::LowState_& state);
    bool update(bool state_received,
                const unitree_go::msg::dds_::LowState_& state,
                std::array<float, 12>& q_out);
    bool near_stable_pose(const unitree_go::msg::dds_::LowState_& state) const;

private:
    standup_config stand_up_;
    float control_hz_;
    int warmup_ticks_;
    int hold_ticks_;
    std::array<int, standup_config::kMaxPhases> phase_ticks_{};
    standup_stage stage_{standup_stage::WARMUP};
    int tick_{0};
    int phase_index_{0};
    float phase_percent_{0.f};
    float hold_percent_{0.f};
    bool captured_start_{false};
    std::array<float, 12> start_pos_{};
    std::array<float, 12> segment_start_{};
};
