#pragma once

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

#include <yaml-cpp/yaml.h>

#include "lowlevel.hpp"
#include "recovery.hpp"
#include "standup.hpp"
#include "motions.hpp"

namespace control
{
    struct app_config
    {
        lowlevel::config control;
        motions::imu_thresholds imu;
        motions::recovery_config recovery;
        motions::standup_config stand_up;
    };

    inline void load_joint_array(const YAML::Node& node, std::array<float, 12>& out)
    {
        if (!node || !node.IsSequence() || node.size() != 12) {
            throw std::runtime_error("Expected YAML sequence of length 12");
        }
        for (size_t i = 0; i < 12; i++) {
            out[i] = node[i].as<float>();
            if (!std::isfinite(out[i])) {
                throw std::runtime_error("Joint configuration contains a non-finite value");
            }
        }
    }

    inline void validate_joint_bounds(const std::array<float, 12>& init,
                                      const std::array<float, 12>& min,
                                      const std::array<float, 12>& max)
    {
        for (size_t i = 0; i < 12; ++i) {
            if (!(min[i] < max[i])) {
                throw std::runtime_error("joint_min must be less than joint_max");
            }
            if (init[i] < min[i] || init[i] > max[i]) {
                throw std::runtime_error("init_qpos is outside joint limits");
            }
        }
    }

    inline motions::standup_config load_standup_config(
        const YAML::Node& node,
        const std::array<float, 12>& stable_pose,
        int num_phases)
    {
        motions::standup_config cfg;
        if (!node) {
            return cfg;
        }

        cfg.stable_pose = stable_pose;
        cfg.num_phases = num_phases;
        cfg.warmup_s = node["warmup_s"].as<float>(1.0f);
        cfg.hold_s = node["hold_s"].as<float>(0.2f);
        cfg.joint_tolerance = node["joint_tolerance"].as<float>(0.15f);
        cfg.kp = node["kp"].as<float>(60.f);
        cfg.kd = node["kd"].as<float>(5.f);

        for (int i = 0; i < num_phases; ++i) {
            const std::string pose_key = "pose_" + std::to_string(i + 1);
            load_joint_array(node[pose_key], cfg.keyframes[i]);
            const std::string phase_key = "phase_" + std::to_string(i + 1) + "_s";
            cfg.phase_duration_s[i] = node[phase_key].as<float>(1.0f);
        }

        return cfg;
    }

    inline motions::recovery_config load_recovery_config(const YAML::Node& node)
    {
        motions::recovery_config cfg;
        if (!node) 
        {
            return cfg;
        }

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
        lowlevel::config& cfg = app.control;

        cfg.kp = root["kp"].as<float>(60.f);
        cfg.kd = root["kd"].as<float>(10.f);
        cfg.policy_timeout_ms = root["policy_timeout_ms"].as<int>(200);
        cfg.policy_delay_ms = root["policy_delay_ms"].as<int>(0);
        load_joint_array(root["init_qpos"], cfg.init_qpos);
        load_joint_array(root["joint_min"], cfg.joint_min);
        load_joint_array(root["joint_max"], cfg.joint_max);
        validate_joint_bounds(cfg.init_qpos, cfg.joint_min, cfg.joint_max);
        if (!std::isfinite(cfg.kp) || !std::isfinite(cfg.kd) || cfg.kp < 0.f || cfg.kd < 0.f) {
            throw std::runtime_error("kp and kd must be finite and non-negative");
        }

        const YAML::Node stand_node = root["stand_up"];
        if (stand_node) {
            app.stand_up = load_standup_config(stand_node, cfg.init_qpos, 2);
        }

        app.recovery = load_recovery_config(root["recovery"]);

        const auto validate_target = [&](const std::array<float, 12>& target,
                                         const char* name) {
            for (size_t i = 0; i < 12; ++i) {
                if (target[i] < cfg.joint_min[i] || target[i] > cfg.joint_max[i]) {
                    throw std::runtime_error(std::string(name) + " is outside joint limits");
                }
            }
        };
        if (stand_node) {
            for (int i = 0; i < app.stand_up.num_phases; ++i) {
                validate_target(app.stand_up.keyframes[i], "stand_up target");
            }
            validate_target(app.stand_up.stable_pose, "stand_up stable target");
        }
        if (root["recovery"]) {
            validate_target(app.recovery.fold_jpos, "recovery fold target");
            validate_target(app.recovery.above_jpos, "recovery above target");
            validate_target(app.recovery.swing_down_jpos, "recovery swing_down target");
            validate_target(app.recovery.push_jpos, "recovery push target");
        }

        const YAML::Node imu_node = root["imu"];
        if (imu_node) {
            app.imu.upside_down_acc_z_on =
                imu_node["upside_down_acc_z_on"].as<float>(
                    imu_node["upside_down_acc_z"].as<float>(-3.f));
            app.imu.upside_down_acc_z_off =
                imu_node["upside_down_acc_z_off"].as<float>(-1.f);
            app.imu.fallen_acc_z_off =
                imu_node["fallen_acc_z_off"].as<float>(7.f);
            app.imu.upside_down_up_cos_on =
                imu_node["upside_down_up_cos_on"].as<float>(
                    imu_node["upside_down_up_cos"].as<float>(-0.7f));
            app.imu.fallen_roll_pitch_limit_rad =
                imu_node["fallen_roll_pitch_limit_rad"].as<float>(
                    imu_node["fallen_orientation_rad"].as<float>(0.523599f));
        }

        return app;
    }

}
