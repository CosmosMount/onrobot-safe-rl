#include "policy.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <iostream>
#include <iterator>
#include <utility>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace control
{
    policy_receiver::policy_receiver(std::string socket_path)
        : socket_path_(std::move(socket_path))
    {
    }

    policy_receiver::~policy_receiver()
    {
        stop();
    }

    void policy_receiver::start()
    {
        if (running_.exchange(true))
        {
            return;
        }
        server_thread_ = std::thread(&policy_receiver::loop, this);
    }

    void policy_receiver::stop()
    {
        if (!running_.exchange(false))
        {
            return;
        }

        const int fd = ::socket(AF_UNIX, SOCK_DGRAM, 0);
        if (fd >= 0)
        {
            sockaddr_un addr{};
            addr.sun_family = AF_UNIX;
            std::strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);
            ::sendto(fd, "", 1, 0, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
            ::close(fd);
        }

        if (server_thread_.joinable())
        {
            server_thread_.join();
        }
        ::unlink(socket_path_.c_str());
    }

    bool policy_receiver::get_latest_target(std::array<float, 12>& out,
                                            double& timestamp,
                                            uint8_t& flags,
                                            uint64_t& action_id) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (last_update_ns_.load() == 0)
        {
            return false;
        }
        out = latest_target_;
        timestamp = latest_timestamp_;
        flags = latest_flags_;
        action_id = latest_action_id_;
        return true;
    }

    bool policy_receiver::has_fresh_target(int timeout_ms) const
    {
        const int64_t last_update = last_update_ns_.load();
        if (last_update == 0)
        {
            return false;
        }
        const int64_t age_ns = get_now_ns() - last_update;
        return age_ns <= static_cast<int64_t>(timeout_ms) * 1000000;
    }

    uint8_t policy_receiver::consume_pending_motion_flags()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const uint8_t flags = pending_motion_flags_;
        pending_motion_flags_ = 0;
        return flags;
    }

    bool policy_receiver::consume_pending_stop(uint64_t& action_id)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!pending_stop_)
        {
            return false;
        }
        pending_stop_ = false;
        action_id = pending_stop_action_id_;
        return true;
    }

    void policy_receiver::clear_pending_motion_flags()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_motion_flags_ = 0;
    }

    void policy_receiver::clear_latest_target()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        latest_target_.fill(0.0f);
        latest_timestamp_ = 0.0;
        latest_action_id_ = 0;
        latest_flags_ = 0;
        pending_motion_flags_ = 0;
        pending_stop_ = false;
        pending_stop_action_id_ = 0;
        last_update_ns_ = 0;
    }

    void policy_receiver::loop()
    {
        ::unlink(socket_path_.c_str());

        const int fd = ::socket(AF_UNIX, SOCK_DGRAM, 0);
        if (fd < 0)
        {
            std::cerr << "policy_receiver socket failed: " << std::strerror(errno) << std::endl;
            running_ = false;
            return;
        }

        sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        std::strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);
        if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0)
        {
            std::cerr << "policy_receiver bind failed: " << std::strerror(errno) << std::endl;
            ::close(fd);
            running_ = false;
            return;
        }

        while (running_)
        {
            policy_packet_t packet{};
            const ssize_t n = ::recv(fd, &packet, sizeof(packet), 0);
            if (n != static_cast<ssize_t>(sizeof(packet)))
            {
                continue;
            }
            if (packet.SOF != policy_packet_t::magicSOF)
            {
                continue;
            }

            std::lock_guard<std::mutex> lock(mutex_);
            std::copy(std::begin(packet.q_target), std::end(packet.q_target),
                      latest_target_.begin());
            latest_timestamp_ = packet.timestamp;
            latest_action_id_ = packet.action_id;
            latest_flags_ = packet.flags;
            pending_motion_flags_ |=
                packet.flags & (policy_packet_t::FLAG_STAND_UP |
                                policy_packet_t::FLAG_RECOVERY);
            if (packet.flags & policy_packet_t::FLAG_STOP)
            {
                pending_stop_ = true;
                pending_stop_action_id_ = packet.action_id;
            }
            last_update_ns_ = get_now_ns();
        }

        ::close(fd);
    }

    state_publisher::state_publisher(std::string socket_path)
        : socket_path_(std::move(socket_path))
    {
        fd_ = ::socket(AF_UNIX, SOCK_DGRAM, 0);
        if (fd_ < 0)
        {
            std::cerr << "state_publisher socket failed: " << std::strerror(errno) << std::endl;
        }
    }

    void state_publisher::publish(const state_packet_t& packet)
    {
        if (fd_ < 0)
        {
            return;
        }
        sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        std::strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);
        ::sendto(fd_, &packet, sizeof(packet), 0,
                 reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    }
}
