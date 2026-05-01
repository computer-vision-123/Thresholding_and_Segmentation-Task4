cmake_minimum_required(VERSION 3.15...3.27)
project(cv_task4 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Find pybind11 ─────────────────────────────────────────────────────────────
find_package(pybind11 CONFIG REQUIRED)

# ── Source files ──────────────────────────────────────────────────────────────
set(CPP_SOURCES
    cpp/src/thresholding.cpp
    cpp/src/segmentation.cpp
    cpp/bindings/bindings.cpp
)

# ── Build the extension module ────────────────────────────────────────────────
pybind11_add_module(cv_backend ${CPP_SOURCES})

target_include_directories(cv_backend PRIVATE cpp/include)

# Optimisation flags for release builds
target_compile_options(cv_backend PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Debug>:-O0 -g>
)

# ── Install ───────────────────────────────────────────────────────────────────
# scikit-build-core will handle the install destination automatically.
install(TARGETS cv_backend LIBRARY DESTINATION .)
