cmake_minimum_required(VERSION 3.15)
project(vocoder_tools LANGUAGES C CXX ASM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# aarch64 JIT lives in WDL's glue_aarch64.h. build.sh keeps the x64 asm unit
# off this architecture so YSFX_PORTABLE can stay off.
set(YSFX_PORTABLE OFF CACHE BOOL "" FORCE)
set(YSFX_PLUGIN OFF CACHE BOOL "" FORCE)
set(YSFX_GFX OFF CACHE BOOL "" FORCE)
set(YSFX_TESTS OFF CACHE BOOL "" FORCE)
set(YSFX_TOOLS OFF CACHE BOOL "" FORCE)
set(YSFX_SKIP_CHECKSUM ON CACHE BOOL "" FORCE)

add_subdirectory(ysfx)

add_executable(jsfx_render src/jsfx_render.cpp)
target_link_libraries(jsfx_render PRIVATE ysfx::ysfx)
target_compile_features(jsfx_render PRIVATE cxx_std_17)
if(NOT MSVC)
  target_compile_options(jsfx_render PRIVATE "-Wall" "-Wextra" "-Wno-unused-parameter")
endif()
set_target_properties(jsfx_render PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
