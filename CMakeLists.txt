cmake_minimum_required(VERSION 3.21)
project(hip_image_filters LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Dependencies ───────────────────────────────────────────────────────────────
find_package(hip REQUIRED)

include(FetchContent)
FetchContent_Declare(stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(stb)

# ── Filter library (HIP kernels + CPU reference) ───────────────────────────────
add_library(filters STATIC
    src/filters.hip
    src/filters_cpu.cpp
    src/stb_impl.cpp
)
set_source_files_properties(src/filters.hip PROPERTIES LANGUAGE HIP)
target_include_directories(filters PUBLIC include ${stb_SOURCE_DIR})
target_link_libraries(filters PUBLIC hip::device)

# ── CLI executable ─────────────────────────────────────────────────────────────
add_executable(hip_filters src/main.cpp)
target_include_directories(hip_filters PRIVATE include ${stb_SOURCE_DIR})
target_link_libraries(hip_filters PRIVATE filters hip::host)

# ── Test-image generator (no GPU required) ────────────────────────────────────
add_executable(gen_test_image tools/gen_test_image.cpp)
target_include_directories(gen_test_image PRIVATE ${stb_SOURCE_DIR})
target_link_libraries(gen_test_image PRIVATE filters)

install(TARGETS hip_filters gen_test_image RUNTIME DESTINATION bin)
