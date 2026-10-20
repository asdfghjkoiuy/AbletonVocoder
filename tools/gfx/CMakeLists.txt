cmake_minimum_required(VERSION 3.15)
project(vocoder_gfx LANGUAGES C CXX OBJCXX ASM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Separate from tools/CMakeLists.txt so the audio renderer keeps YSFX_GFX off.
set(YSFX_PORTABLE OFF CACHE BOOL "" FORCE)
set(YSFX_PLUGIN OFF CACHE BOOL "" FORCE)
set(YSFX_GFX ON CACHE BOOL "" FORCE)
set(YSFX_TESTS OFF CACHE BOOL "" FORCE)
set(YSFX_TOOLS OFF CACHE BOOL "" FORCE)
set(YSFX_SKIP_CHECKSUM ON CACHE BOOL "" FORCE)

add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../ysfx" "${CMAKE_CURRENT_BINARY_DIR}/ysfx")

add_executable(jsfx_gfx_snap "${CMAKE_CURRENT_LIST_DIR}/../src/jsfx_gfx_snap.cpp")
target_link_libraries(jsfx_gfx_snap PRIVATE ysfx::ysfx)
target_include_directories(jsfx_gfx_snap PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../ysfx/thirdparty/stb")
target_compile_features(jsfx_gfx_snap PRIVATE cxx_std_17)
if(NOT MSVC)
  target_compile_options(jsfx_gfx_snap PRIVATE "-Wall" "-Wextra" "-Wno-unused-parameter")
endif()
if(APPLE)
  target_link_libraries(jsfx_gfx_snap PRIVATE
    "-framework Cocoa"
    "-framework Carbon"
    "-framework Foundation"
    "-framework Metal"
    "-framework QuartzCore"
    "-framework CoreVideo")
endif()
set_target_properties(jsfx_gfx_snap PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../build")
