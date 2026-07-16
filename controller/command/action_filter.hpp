#pragma once

#include <array>
#include <cstddef>

class command_lowpass_filter
{
public:
    explicit command_lowpass_filter(float alpha = 1.0f) : alpha_(alpha) {}

    void reset(const std::array<float, 12>& value)
    {
        state_ = value;
        initialized_ = true;
    }

    std::array<float, 12> filter(const std::array<float, 12>& target)
    {
        if (!initialized_) {
            reset(target);
            return target;
        }
        for (size_t i = 0; i < state_.size(); ++i) {
            state_[i] = alpha_ * target[i] + (1.0f - alpha_) * state_[i];
        }
        return state_;
    }

private:
    float alpha_{1.0f};
    bool initialized_{false};
    std::array<float, 12> state_{};
};
