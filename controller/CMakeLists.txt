cmake_minimum_required(VERSION 3.16)
project(go2_control LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

list(APPEND CMAKE_PREFIX_PATH "/opt/unitree_robotics/lib/cmake")
find_package(unitree_sdk2 REQUIRED)
find_package(yaml-cpp REQUIRED)

add_executable(go2_control
    main.cpp
    realtime/controller.cpp
    transport/ipc_server.cpp
    command/lowlevel_commander.cpp
    safety/motion_service.cpp
)

target_include_directories(go2_control PRIVATE
    realtime
    transport
    command
    safety
)
target_link_libraries(go2_control PRIVATE unitree_sdk2 yaml-cpp)
