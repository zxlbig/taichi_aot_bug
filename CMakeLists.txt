cmake_minimum_required(VERSION 3.24)

project(Meshy3DNative)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(Taichi REQUIRED)
# file(COPY ${Taichi_REDIST_LIBRARIES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_executable(
    Meshy3DAotTest ${CMAKE_CURRENT_SOURCE_DIR}/App.cpp
)

target_link_libraries(
    Meshy3DAotTest PUBLIC
    Taichi::Runtime
)
