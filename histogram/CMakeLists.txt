add_executable(HistogramExample histogramexample.cpp)
target_link_libraries(HistogramExample OpenCL::OpenCL)
target_include_directories(HistogramExample PRIVATE ${PROJECT_SOURCE_DIR}/shared)

add_custom_command(
        TARGET HistogramExample POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/resources
        $<TARGET_FILE_DIR:HistogramExample>/resources)