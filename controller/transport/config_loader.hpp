#pragma once

#include <array>
#include <stdexcept>
#include <string>

#include <yaml-cpp/yaml.h>

#include "lowlevel_commander.hpp"
#include "imu_utils.hpp"
#include "motion_service.hpp"

struct app_config
{
    control_config control;
    imu_orientation_config imu;
    recovery_config recovery;
    standup_config stand_up;
};

inline void load_joint_array(const YAML::Node& node, std::array<float, 12>& out)
{
    if (!node || !node.IsSequence() || node.size() != 12) {
        throw std::runtime_error("Expected YAML sequence of length 12");
    }
    for (size_t i = 0; i < 12; i++) {
        out[i] = node[i].as<float>();
    }
}

inline standup_config load_standup_config(
    const YAML::Node& node,
    const std::array<float, 12>& stable_pose,
    int num_phases)
{
    standup_config cfg;
    if (!node) {
        return cfg;
    }

    cfg.configured = true;
    cfg.stable_pose = stable_pose;
    cfg.num_phases = num_phases;
    cfg.warmup_s = node["warmup_s"].as<float>(1.0f);
    cfg.hold_s = node["hold_s"].as<float>(0.2f);
    cfg.joint_tolerance = node["joint_tolerance"].as<float>(0.15f);
    cfg.deactivate_motion_service =
        node["deactivate_motion_service"].as<bool>(false);

    for (int i = 0; i < num_phases; ++i) {
        const std::string pose_key = "pose_" + std::to_string(i + 1);
        load_joint_array(node[pose_key], cfg.keyframes[i]);
        const std::string phase_key = "phase_" + std::to_string(i + 1) + "_s";
        cfg.phase_duration_s[i] = node[phase_key].as<float>(1.0f);
    }

    return cfg;
}

inline recovery_config load_recovery_config(const YAML::Node& node)
{
    recovery_config cfg;
    if (!node) {
        return cfg;
    }

    cfg.configured = true;
    cfg.deactivate_motion_service =
        node["deactivate_motion_service"].as<bool>(false);
    cfg.fold_ramp_s = node["fold_ramp_s"].as<float>(0.45f);
    cfg.fold_settle_s = node["fold_settle_s"].as<float>(0.50f);
    cfg.above_ramp_s = node["above_ramp_s"].as<float>(
        node["extend_ramp_s"].as<float>(0.45f));
    cfg.above_settle_s = node["above_settle_s"].as<float>(
        node["extend_settle_s"].as<float>(0.35f));
    cfg.swing_down_ramp_s = node["swing_down_ramp_s"].as<float>(0.55f);
    cfg.swing_down_settle_s = node["swing_down_settle_s"].as<float>(0.45f);
    cfg.push_ramp_s = node["push_ramp_s"].as<float>(0.30f);
    cfg.push_settle_s = node["push_settle_s"].as<float>(0.25f);
    cfg.joint_reach_tol = node["joint_reach_tol"].as<float>(0.12f);
    cfg.kp = node["kp"].as<float>(100.f);
    cfg.kd = node["kd"].as<float>(8.f);

    load_joint_array(node["fold_jpos"], cfg.fold_jpos);
    if (node["above_jpos"]) {
        load_joint_array(node["above_jpos"], cfg.above_jpos);
    } else if (node["extend_jpos"]) {
        load_joint_array(node["extend_jpos"], cfg.above_jpos);
    }
    load_joint_array(node["swing_down_jpos"], cfg.swing_down_jpos);
    load_joint_array(node["push_jpos"], cfg.push_jpos);

    if (node["swing_legs"] && node["swing_legs"].IsSequence() &&
        node["swing_legs"].size() == 4) {
        for (size_t i = 0; i < 4; ++i) {
            cfg.swing_legs[i] = node["swing_legs"][i].as<bool>();
        }
    }
    if (node["push_legs"] && node["push_legs"].IsSequence() &&
        node["push_legs"].size() == 4) {
        for (size_t i = 0; i < 4; ++i) {
            cfg.push_legs[i] = node["push_legs"][i].as<bool>();
        }
    }
    return cfg;
}

inline app_config load_app_config(const YAML::Node& root)
{
    app_config app;
    control_config& cfg = app.control;

    cfg.kp = root["kp"].as<float>(60.f);
    cfg.kd = root["kd"].as<float>(10.f);
    cfg.policy_timeout_ms = root["policy_timeout_ms"].as<int>(200);
    cfg.policy_delay_ms = root["policy_delay_ms"].as<int>(0);
    load_joint_array(root["init_qpos"], cfg.init_qpos);
    load_joint_array(root["joint_min"], cfg.joint_min);
    load_joint_array(root["joint_max"], cfg.joint_max);

    const YAML::Node stand_node = root["stand_up"];
    if (stand_node) {
        app.stand_up = load_standup_config(stand_node, cfg.init_qpos, 2);
    }

    app.recovery = load_recovery_config(root["recovery"]);

    const YAML::Node imu_node = root["imu"];
    if (imu_node) {
        app.imu.upside_down_acc_z_on =
            imu_node["upside_down_acc_z_on"].as<float>(
                imu_node["upside_down_acc_z"].as<float>(-3.f));
        app.imu.upside_down_acc_z_off =
            imu_node["upside_down_acc_z_off"].as<float>(-1.f);
        app.imu.fallen_acc_z_off =
            imu_node["fallen_acc_z_off"].as<float>(7.f);
    }

    return app;
}
