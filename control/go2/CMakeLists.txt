cmake_minimum_required(VERSION 3.16)

project(go2_controller LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_PREFIX_PATH "/opt/unitree_robotics")

find_package(Threads REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(unitree_sdk2 REQUIRED CONFIG)

add_library(go2_lowlevel STATIC
    lowlevel/src/lowlevel.cpp
)
target_include_directories(go2_lowlevel
    PUBLIC
        lowlevel/include
)
target_link_libraries(go2_lowlevel
    PUBLIC
        unitree_sdk2
)

add_library(go2_motions STATIC
    motions/src/recovery.cpp
    motions/src/standup.cpp
)
target_include_directories(go2_motions
    PUBLIC
        motions/include
)
target_link_libraries(go2_motions
    PUBLIC
        unitree_sdk2
)

add_library(go2_config INTERFACE)
target_include_directories(go2_config
    INTERFACE
        utils/include
)
target_link_libraries(go2_config
    INTERFACE
        go2_lowlevel
        go2_motions
        yaml-cpp
)

add_library(go2_controller STATIC
    controller/src/controller.cpp
    controller/src/policy.cpp
)
target_include_directories(go2_controller
    PUBLIC
        controller/include
)
target_link_libraries(go2_controller
    PUBLIC
        go2_lowlevel
        go2_motions
        go2_config
        unitree_sdk2
        Threads::Threads
)

add_executable(go2_control
    main.cpp
)
target_link_libraries(go2_control
    PRIVATE
        go2_controller
        yaml-cpp
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/go2.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/go2.yaml
    COPYONLY
)
