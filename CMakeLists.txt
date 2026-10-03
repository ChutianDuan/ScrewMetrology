cmake_minimum_required(VERSION 3.16)
project(ScrewMetrology VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 导出 compile_commands.json,供 C/C++ IntelliSense 使用
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
if(OpenCV_VERSION_MAJOR GREATER_EQUAL 5)
    find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs geometry)
endif()
add_library(screw_core
    src/config.cpp src/segmentation.cpp src/locator.cpp
    src/metrology.cpp src/calibration.cpp src/io.cpp
    src/evaluation.cpp src/visualization.cpp)
target_include_directories(screw_core PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(screw_core PUBLIC ${OpenCV_LIBS})
target_compile_options(screw_core PRIVATE -Wall -Wextra -Wpedantic)
add_executable(screw_metrology src/main.cpp)
target_link_libraries(screw_metrology PRIVATE screw_core)

# 手动学习入口：复用算法库；仅此程序额外链接窗口显示模块。
find_package(OpenCV REQUIRED COMPONENTS highgui)
add_executable(main_test main_test.cpp)
target_link_libraries(main_test PRIVATE screw_core opencv_highgui)
# 将项目根目录作为编译期字符串，手动实验不依赖启动时的工作目录。
target_compile_definitions(main_test PRIVATE SCREW_TEST_PROJECT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

include(CTest)
if(BUILD_TESTING)
    add_executable(screw_tests tests/test_core.cpp)
    target_link_libraries(screw_tests PRIVATE screw_core)
    add_test(NAME screw_core COMMAND screw_tests ${CMAKE_CURRENT_BINARY_DIR}/test_artifacts)
    add_executable(generate_samples tests/generate_samples.cpp)
    target_link_libraries(generate_samples PRIVATE screw_core)
    add_test(NAME cli_help COMMAND screw_metrology --help)
    add_test(NAME cli_bad_input COMMAND screw_metrology measure --input /nonexistent/screw.png)
    set_tests_properties(cli_bad_input PROPERTIES WILL_FAIL TRUE)
    add_test(NAME cli_bad_decode COMMAND screw_metrology measure --input ${CMAKE_CURRENT_BINARY_DIR}/test_artifacts/bad.png --output ${CMAKE_CURRENT_BINARY_DIR}/test_artifacts/bad_decode)
    add_test(NAME cli_invalid_config COMMAND screw_metrology measure --input ${CMAKE_CURRENT_BINARY_DIR}/test_artifacts/clean.png --config ${CMAKE_CURRENT_BINARY_DIR}/test_artifacts/bad_config.yaml)
    add_test(NAME cli_no_objects COMMAND screw_metrology measure --input ${CMAKE_CURRENT_BINARY_DIR}/test_artifacts/blank.png --segmentation otsu --output ${CMAKE_CURRENT_BINARY_DIR}/test_artifacts/no_objects)
    add_test(NAME cli_scaled_measure COMMAND screw_metrology measure --input ${CMAKE_CURRENT_BINARY_DIR}/test_artifacts/clean.png --segmentation otsu --scale ${CMAKE_CURRENT_BINARY_DIR}/test_artifacts/scale.yaml --recipe ${CMAKE_CURRENT_BINARY_DIR}/test_artifacts/good_recipe.yaml --mode single --output ${CMAKE_CURRENT_BINARY_DIR}/test_artifacts/scaled)
    set_tests_properties(cli_bad_decode cli_invalid_config PROPERTIES WILL_FAIL TRUE)
    set_tests_properties(cli_bad_decode cli_invalid_config cli_no_objects cli_scaled_measure PROPERTIES DEPENDS screw_core)
endif()
