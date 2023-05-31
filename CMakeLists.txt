cmake_minimum_required(VERSION 3.21)
project(makemore C)

set(CMAKE_C_STANDARD 99)

add_executable(makemore main.c makemore.c makemore.h)

if(CMAKE_C_COMPILER_ID MATCHES "AppleClang|Clang|GNU")
  target_link_libraries(makemore m)
endif()
