cmake_minimum_required(VERSION 3.0)
project(Histomatching)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(src main.cpp compute.cpp csv_util.cpp filter.cpp)
target_link_libraries(src ${OpenCV_LIBS})
