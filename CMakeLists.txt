cmake_minimum_required(VERSION 3.10)
set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)
set(CMAKE_CXX_STANDARD 11)
set(OpenCV_DIR "C:\\lib\\build_opencv")
include_directories("C:\\lib\\install\\opencv\\include")
link_directories("C:\\lib\\build_opencv\\lib")

find_package(OpenCV)

# set the project name
project(TwoViews)

# add the executable
add_executable(TwoViews demo.cpp)
target_link_libraries(TwoViews debug opencv_world453d optimized opencv_world453)
