project(pso)
cmake_minimum_required(VERSION 2.8)

aux_source_directory(. SRC_LIST)
include_directories(${CMAKE_SOURCE_DIR})
add_library(${PROJECT_NAME} SHARED ${SRC_LIST})
