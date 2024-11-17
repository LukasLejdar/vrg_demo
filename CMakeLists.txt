

cmake_minimum_required(VERSION 3.15...3.30)
project( vrg_demo VERSION 1.0 LANGUAGES CXX )

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../bin")

find_package(Eigen3 3 REQUIRED)
add_executable(vrg_demo src/main.cpp)
target_link_libraries(vrg_demo Eigen3::Eigen)

add_executable(hit_moving_target src/other_examples/hit_moving_target.cpp)
target_link_libraries(hit_moving_target Eigen3::Eigen)

find_package(Catch2 3 REQUIRED)
add_executable(tests test/test_main.cpp)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain Eigen3::Eigen)




