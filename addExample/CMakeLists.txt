add_executable(AddExample addexample.cpp)
target_link_libraries(AddExample OpenCL::OpenCL)
target_include_directories(AddExample PUBLIC ${PROJECT_SOURCE_DIR}/shared)

add_custom_command(
        TARGET AddExample POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/kernals/vectorAdd.cl
        $<TARGET_FILE_DIR:AddExample>/kernals/vectorAdd.cl)