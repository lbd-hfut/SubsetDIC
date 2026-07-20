cmake_minimum_required(VERSION 3.21)
project(subsetdic VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core
    src/cpp/bindings.cpp
    src/cpp/interpolation.cpp
    src/cpp/solver.cpp
    src/cpp/region.cpp
    src/cpp/seeds.cpp
    src/cpp/rgdic.cpp
    src/cpp/strain.cpp
    src/cpp/extrapolate.cpp
)

target_include_directories(_core PRIVATE src/cpp)
target_compile_definitions(_core PRIVATE VERSION_INFO="0.1.0")

install(TARGETS _core DESTINATION subsetdic)
