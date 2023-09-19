cmake_minimum_required(VERSION 3.22)
project(c_net C)

set(CMAKE_C_STANDARD 11)

add_executable(c_net main.c matrix.c image.c)
target_link_libraries(c_net m)
