cmake_minimum_required(VERSION 3.14)

project(laya-ggml C)

set(CMAKE_C_STANDARD 11)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------- options
# Build ggml from source instead of linking someone else's prebuilt .so. That
# makes this project self-contained: one configure, one build, no chasing a
# matching llama.cpp build.
option(LAYA_CUDA "Enable the CUDA backend (requires CUDA toolkit)" ON)
option(LAYA_STATIC "Link ggml statically instead of as shared libraries" OFF)

# Where the ggml sources live. Point this at a ggml checkout, a llama.cpp
# checkout (its ggml/ subdirectory is used), or leave it unset to have CMake
# fetch the pinned version below.
set(GGML_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../ggml"
    CACHE PATH "Path to ggml source (or to llama.cpp, whose ggml/ is used)")

# ggml's API moves, and this engine is written against a specific one. Pin it.
# This is a commit in ggml-org/ggml (0.25.3), not in llama.cpp.
set(GGML_PINNED_TAG "353b63b439f27ab2cc19dac97ab1681ba6d2d084"
    CACHE STRING "ggml commit to fetch")

# ---------------------------------------------------------------- locate ggml
# Try the configured path, then the usual sibling checkouts, then fetch.
set(_ggml_dir "")
foreach(_candidate "${GGML_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../ggml"
                   "${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/ggml")
    if(EXISTS "${_candidate}/CMakeLists.txt" AND EXISTS "${_candidate}/src")
        set(_ggml_dir "${_candidate}")
        message(STATUS "ggml sources: ${_ggml_dir} (local)")
        break()
    endif()
endforeach()

if(_ggml_dir STREQUAL "")
    # Nothing local: fetch ggml itself. It has its own repository -- llama.cpp
    # only vendors a copy of it -- so this pulls ~36 MB instead of a 500 MB
    # llama.cpp checkout. Shallow, and pinned for reproducible builds.
    message(STATUS "ggml not found locally, fetching ggml ${GGML_PINNED_TAG}...")
    include(FetchContent)
    set(FETCHCONTENT_QUIET OFF)
    FetchContent_Declare(ggml_src
        GIT_REPOSITORY https://github.com/ggml-org/ggml
        GIT_TAG        ${GGML_PINNED_TAG}
        GIT_SHALLOW    TRUE
        GIT_PROGRESS   TRUE
    )
    FetchContent_GetProperties(ggml_src)
    if(NOT ggml_src_POPULATED)
        FetchContent_Populate(ggml_src)
    endif()
    set(_ggml_dir "${ggml_src_SOURCE_DIR}")
    if(NOT EXISTS "${_ggml_dir}/CMakeLists.txt")
        message(FATAL_ERROR "fetched ggml has no CMakeLists.txt at ${_ggml_dir}")
    endif()
    message(STATUS "ggml sources: ${_ggml_dir} (fetched)")
endif()

# ---------------------------------------------------------------- build ggml
# ggml is a CMake project of its own, so just add it. These are cache vars and
# need FORCE, otherwise an earlier value in the cache wins silently.
set(GGML_CUDA          ${LAYA_CUDA} CACHE BOOL "ggml: use CUDA" FORCE)
set(GGML_BUILD_TESTS   OFF CACHE BOOL "ggml: build tests" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "ggml: build examples" FORCE)
set(GGML_STATIC        ${LAYA_STATIC} CACHE BOOL "ggml: static libs" FORCE)

if(LAYA_CUDA)
    # Nothing here: ggml probes for the toolkit itself and turns GGML_CUDA off
    # when it is missing. We report what happened below.
endif()

if(NOT LAYA_STATIC)
    set(BUILD_SHARED_LIBS ON CACHE BOOL "build shared libraries" FORCE)
endif()

add_subdirectory(${_ggml_dir} ${CMAKE_CURRENT_BINARY_DIR}/ggml)

# ---------------------------------------------------------------- our binary
add_executable(laya-infer laya-infer.c)

target_link_libraries(laya-infer PRIVATE
    ggml
    ggml-base
    ggml-cpu
    m
)

if(GGML_CUDA)
    target_compile_definitions(laya-infer PRIVATE GGML_USE_CUDA)
    # ggml names the CUDA backend target ggml-cuda; not every build exposes it.
    if(TARGET ggml-cuda)
        target_link_libraries(laya-infer PRIVATE ggml-cuda)
    endif()
    message(STATUS "CUDA backend: enabled")
else()
    message(STATUS "CUDA backend: disabled (no toolkit found, or -DLAYA_CUDA=OFF)")
endif()

# Find the ggml shared libraries next to our own binary at runtime.
set_target_properties(laya-infer PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)

install(TARGETS laya-infer RUNTIME DESTINATION bin)
