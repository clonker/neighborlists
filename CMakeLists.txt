cmake_minimum_required(VERSION 3.11)
project(neighborlists LANGUAGES CXX VERSION 0.0.0)

set(CMAKE_CXX_STANDARD 20)

set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(FMT_TEST OFF CACHE BOOL "" FORCE)
set(FMT_DOC OFF CACHE BOOL "" FORCE)

add_subdirectory(contrib/fmt)
add_definitions(-DSPDLOG_FMT_EXTERNAL)
add_subdirectory(contrib/spdlog)

add_subdirectory(neighborlists)

add_subdirectory(contrib/Catch2)
add_subdirectory(test)

