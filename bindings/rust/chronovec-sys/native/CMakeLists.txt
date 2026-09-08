cmake_minimum_required(VERSION 3.18)
project(chronovec_sys LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(chronovec STATIC src/chronovec.cpp src/distance.cpp)
target_include_directories(chronovec PUBLIC include)
if(MSVC)
  target_compile_options(chronovec PRIVATE /O2 /W4)
else()
  target_compile_options(chronovec PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

install(TARGETS chronovec
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
