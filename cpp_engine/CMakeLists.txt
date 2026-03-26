# ==============================================================================
# CMakeLists.txt — C++ Limit Order Book Engine
# ==============================================================================
#
# Build targets:
#   - order_book_cpp  : Python extension module (pybind11)
#   - order_book_lib  : Static library for C++ consumers
#   - test_order_book : Unit tests (Catch2)
#
# Quick start:
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   cmake --build . --config Release -j$(nproc)
#
# Options:
#   -DBUILD_TESTS=ON       Build unit tests (default: ON)
#   -DBUILD_PYTHON=ON      Build Python bindings (default: ON)
#
# ==============================================================================

cmake_minimum_required(VERSION 3.15)

project(OrderBookEngine
    VERSION 1.0.0
    DESCRIPTION "High-performance C++ Limit Order Book with pybind11 bindings"
    LANGUAGES CXX
)

# ------------------------------------------------------------------------------
# C++ standard and compiler settings
# ------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization flags for Release builds
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fsanitize=address,undefined -fno-omit-frame-pointer")
    add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wnon-virtual-dtor)
elseif(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
    add_compile_options(/W4 /permissive-)
endif()

# ------------------------------------------------------------------------------
# Options
# ------------------------------------------------------------------------------

option(BUILD_TESTS  "Build unit tests"      ON)
option(BUILD_PYTHON "Build Python bindings"  ON)

# ------------------------------------------------------------------------------
# Core static library
# ------------------------------------------------------------------------------

add_library(order_book_lib STATIC
    src/order_book.cpp
)

target_include_directories(order_book_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(order_book_lib PUBLIC cxx_std_17)

# Enable link-time optimization in Release
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set_target_properties(order_book_lib PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION TRUE
    )
endif()

# ------------------------------------------------------------------------------
# Python bindings (pybind11)
# ------------------------------------------------------------------------------

if(BUILD_PYTHON)
    # Try to find pybind11 via multiple methods:
    # 1. Installed system-wide or via pip (pybind11 --cmakedir)
    # 2. As a subdirectory (git submodule)
    # 3. Via FetchContent (automatic download)

    find_package(pybind11 QUIET)

    if(NOT pybind11_FOUND)
        message(STATUS "pybind11 not found via find_package, trying FetchContent...")
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG        v2.12.0
        )
        FetchContent_MakeAvailable(pybind11)
    endif()

    pybind11_add_module(order_book_cpp
        src/bindings.cpp
    )

    target_link_libraries(order_book_cpp
        PRIVATE order_book_lib
    )

    target_compile_definitions(order_book_cpp
        PRIVATE VERSION_INFO="${PROJECT_VERSION}"
    )

    # Enable LTO for the Python module in Release
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        set_target_properties(order_book_cpp PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION TRUE
        )
    endif()

    message(STATUS "Python bindings will be built")
endif()

# ------------------------------------------------------------------------------
# Unit tests (Catch2)
# ------------------------------------------------------------------------------

if(BUILD_TESTS)
    enable_testing()

    # Fetch Catch2 if not installed
    find_package(Catch2 3 QUIET)

    if(NOT Catch2_FOUND)
        message(STATUS "Catch2 not found, fetching via FetchContent...")
        include(FetchContent)
        FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG        v3.5.2
        )
        FetchContent_MakeAvailable(Catch2)
        list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
    endif()

    add_executable(test_order_book
        tests/test_order_book.cpp
    )

    target_link_libraries(test_order_book
        PRIVATE
            order_book_lib
            Catch2::Catch2WithMain
    )

    include(CTest)
    include(Catch)
    catch_discover_tests(test_order_book)

    message(STATUS "Unit tests will be built")
endif()

# ------------------------------------------------------------------------------
# Install rules
# ------------------------------------------------------------------------------

install(TARGETS order_book_lib
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
)

if(BUILD_PYTHON)
    install(TARGETS order_book_cpp
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python
    )
endif()

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------

message(STATUS "")
message(STATUS "=== OrderBook Engine Configuration ===")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  C++ Standard:   C++${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  Python Bindings: ${BUILD_PYTHON}")
message(STATUS "  Unit Tests:     ${BUILD_TESTS}")
message(STATUS "  Compiler:       ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "======================================")
message(STATUS "")
