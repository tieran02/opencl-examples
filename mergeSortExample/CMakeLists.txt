add_executable(MergeSortExample mergesortexample.cpp)
target_link_libraries(MergeSortExample OpenCL::OpenCL)
target_include_directories(MergeSortExample PRIVATE ${PROJECT_SOURCE_DIR}/shared)

add_custom_command(
        TARGET MergeSortExample POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels
        $<TARGET_FILE_DIR:MergeSortExample>/kernels)