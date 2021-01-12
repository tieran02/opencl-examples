cmake_minimum_required(VERSION 3.17)
project(opencl)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCL REQUIRED)

add_executable(opencl main.cpp "stopwatch.h")
target_link_libraries(opencl OpenCL::OpenCL)

add_custom_target(build-time-make-directory ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${PROJECT_NAME}>/kernals)
add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/kernals/vectorAdd.cl
        $<TARGET_FILE_DIR:${PROJECT_NAME}>/kernals/vectorAdd.cl)