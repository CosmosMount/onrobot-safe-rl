#include <iostream>
#include <string>

#include <yaml-cpp/yaml.h>

#include "config_loader.hpp"
#include "controller.hpp"

int main(int argc, char** argv)
{
    const std::string config_path =
        argc > 1 ? std::string(argv[1]) : "../../config/go2.yaml";

    try
    {
        const YAML::Node root = YAML::LoadFile(config_path);
        const int domain_id = root["domain_id"].as<int>(1);
        const std::string network_interface = root["interface"].as<std::string>("lo");
        const std::string ipc_socket = root["ipc_socket"].as<std::string>("/tmp/go2_policy.sock");
        const float control_hz = root["control_hz"].as<float>(500.f);

        const app_config app = load_app_config(root);
        controller controller_(domain_id, network_interface, app, ipc_socket, control_hz);
        std::cout << "go2_control started config=" << config_path
                  << " domain_id=" << domain_id
                  << " interface=" << network_interface
                  << " hz=" << control_hz
                  << " recovery=" << app.recovery.configured
                  << " stand_up=" << app.stand_up.configured
                  << std::endl;
        controller_.start();
        controller_.run();
        controller_.stop();
    }
    catch (const std::exception& ex)
    {
        std::cerr << "go2_control error: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
