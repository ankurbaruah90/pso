project(pso)
cmake_minimum_required(VERSION 2.8)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")

aux_source_directory(. SRC_LIST)
include_directories(${CMAKE_SOURCE_DIR})
#add_library(${PROJECT_NAME} SHARED ${SRC_LIST})

add_executable(${PROJECT_NAME} ${SRC_LIST})
target_link_libraries(${PROJECT_NAME} stdc++ m gsl gslcblas)


