
cmake_minimum_required(VERSION 3.15...3.30)

project( vrg_demo VERSION 1.0 LANGUAGES CXX )

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Catch2)
find_package(Eigen3)

include(FetchContent)
if(NOT Catch2_FOUND)
    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG        v3.7.0
    )
    FetchContent_MakeAvailable(Catch2)
endif()

FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
)

set(BUILD_TESTING OFF)
set(EIGEN_BUILD_TESTING OFF)
set(EIGEN_MPL2_ONLY ON)
set(EIGEN_BUILD_PKGCONFIG OFF)
set(EIGEN_BUILD_DOC OFF)

FetchContent_MakeAvailable(Eigen)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../bin")

add_executable(vrg_demo src/main.cpp)
target_link_libraries(vrg_demo Eigen3::Eigen)

add_executable(hit_moving_target src/other_examples/hit_moving_target.cpp)
target_link_libraries(hit_moving_target Eigen3::Eigen)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)
include(CTest)
include(Catch)


