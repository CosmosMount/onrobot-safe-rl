#include "policy_receiver.hpp"

#include <chrono>
#include <cstring>
#include <iostream>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>


policy_receiver::policy_receiver(std::string socket_path)
    : socket_path_(std::move(socket_path)) {}

policy_receiver::~policy_receiver() { stop(); }

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
    if (server_thread_.joinable()) 
    {
        server_thread_.join();
    }
    unlink(socket_path_.c_str());
}

bool policy_receiver::get_latest_target(std::array<float, 12>& out,
                                        double& timestamp,
                                        uint8_t& flags) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    out = latest_target_;
    timestamp = latest_timestamp_;
    flags = latest_flags_;
    return last_update_ns_.load() > 0;
}

bool policy_receiver::has_fresh_target(int timeout_ms) const 
{
    const int64_t age_ns =
        get_now_ns() - last_update_ns_.load(std::memory_order_relaxed);
    return age_ns >= 0 &&
           age_ns <= static_cast<int64_t>(timeout_ms) * 1000000LL;
}

uint8_t policy_receiver::consume_pending_motion_flags()
{
    std::lock_guard<std::mutex> lock(mutex_);
    const uint8_t flags = pending_motion_flags_;
    pending_motion_flags_ = 0;
    return flags;
}

void policy_receiver::clear_pending_motion_flags()
{
    std::lock_guard<std::mutex> lock(mutex_);
    pending_motion_flags_ = 0;
}

void policy_receiver::loop() 
{
    unlink(socket_path_.c_str());

    const int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd < 0) 
    {
        std::cerr << "policy_receiver: socket() failed\n";
        running_ = false;
        return;
    }

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

    if (bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) 
    {
        std::cerr << "policy_receiver: bind(" << socket_path_ << ") failed\n";
        close(server_fd);
        running_ = false;
        return;
    }

    if (listen(server_fd, 1) < 0) 
    {
        std::cerr << "policy_receiver: listen() failed\n";
        close(server_fd);
        running_ = false;
        return;
    }

    std::cout << "policy_receiver listening on " << socket_path_ << std::endl;

    while (running_) 
    {
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(server_fd, &readfds);
        timeval tv{0, 200000};
        if (select(server_fd + 1, &readfds, nullptr, nullptr, &tv) <= 0) 
        {
            continue;
        }

        const int client_fd = accept(server_fd, nullptr, nullptr);
        if (client_fd < 0) 
        {
            continue;
        }

        while (running_) 
        {
            policy_packet_t packet{};
            const ssize_t n =
                recv(client_fd, &packet, sizeof(packet), MSG_WAITALL);
            if (n != static_cast<ssize_t>(sizeof(packet))) 
            {
                break;
            }
            if (packet.SOF != policy_packet_t::magicSOF) 
            {
                continue;
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                std::memcpy(latest_target_.data(), packet.q_target,
                            sizeof(packet.q_target));
                latest_timestamp_ = packet.timestamp;
                latest_flags_ = packet.flags;
                if (packet.flags & policy_packet_t::FLAG_RECOVERY) {
                    pending_motion_flags_ = policy_packet_t::FLAG_RECOVERY;
                } else if (packet.flags & policy_packet_t::FLAG_STAND_UP) {
                    pending_motion_flags_ = policy_packet_t::FLAG_STAND_UP;
                }
            }
            last_update_ns_.store(get_now_ns(), std::memory_order_relaxed);
        }
        close(client_fd);
    }

    close(server_fd);
    unlink(socket_path_.c_str());
}
