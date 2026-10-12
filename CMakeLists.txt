# Orderit - a minimal HIP inference engine for Qwen3-Embedding-4B and
# Qwen3-Reranker-4B on AMD Strix Halo (gfx1151).
#
# One platform, two models, one process. Nothing here is general purpose.
cmake_minimum_required(VERSION 3.22)
project(orderit LANGUAGES CXX HIP)

if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES gfx1151)
endif()
if(NOT CMAKE_HIP_ARCHITECTURES STREQUAL "gfx1151")
  message(FATAL_ERROR "Orderit kernels require CMAKE_HIP_ARCHITECTURES=gfx1151")
endif()
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  message(FATAL_ERROR "Orderit targets Linux x86-64 with an AMD gfx1151 GPU only")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

find_package(hip REQUIRED)

# The checkpoint the tests run against. Not fetched, not vendored: these two
# models are 3.7 GiB each and the engine refuses to guess.
set(ORDERIT_MODEL "" CACHE FILEPATH "Checkpoint used by the model tests")
if(NOT ORDERIT_MODEL AND DEFINED ENV{ORDERIT_MODEL})
  set(ORDERIT_MODEL $ENV{ORDERIT_MODEL})
endif()

# Standalone kernel microbenchmarks, kept out of the production binary.
add_executable(orderit-bench tools/bench.hip)
target_link_libraries(orderit-bench PRIVATE hip::host)
target_compile_options(orderit-bench PRIVATE -O3)

enable_testing()
add_subdirectory(tests)