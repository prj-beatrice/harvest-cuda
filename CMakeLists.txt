cmake_minimum_required(VERSION 3.24)
project(harvest_cuda LANGUAGES CXX)

option(HARVEST_CUDA_ENABLE_CUDA "Build the CUDA backend" ON)
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core STABLE_ABI src/bindings.cpp src/engine.cpp
  third_party/world/src/common.cpp
  third_party/world/src/fft.cpp
  third_party/world/src/matlabfunctions.cpp)
target_include_directories(_core PRIVATE src third_party/world/src)
target_compile_features(_core PRIVATE cxx_std_17)

if(HARVEST_CUDA_ENABLE_CUDA)
  # Distribution builds must not inherit the build host's GPU architecture.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90 100 120)
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(_core PRIVATE src/cuda_backend.cu)
  target_link_libraries(_core PRIVATE CUDA::cufft)
  set_target_properties(_core PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
    # CUDA 13.2+ on Windows replaces shared cudart with a driver-side loader.
    # Use the static runtime there to handle missing/older drivers normally.
    CUDA_RUNTIME_LIBRARY "$<IF:$<PLATFORM_ID:Windows>,Static,Shared>"
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  if(UNIX AND NOT APPLE)
    # Resolve PyPI CUDA runtime packages in the same site-packages.
    # Wheel repair may add its own relative paths for bundled libraries.
    set_target_properties(_core PROPERTIES INSTALL_RPATH "$ORIGIN;$ORIGIN/../nvidia/cu13/lib")
  endif()
  # Preserve threshold-sensitive arithmetic; do not enable --use_fast_math.
  target_compile_options(_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
else()
  target_sources(_core PRIVATE src/cuda_stub.cpp)
endif()

if(MSVC)
  target_compile_options(_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/fp:precise> $<$<COMPILE_LANGUAGE:CXX>:/O2>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
else()
  target_compile_options(_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off> $<$<COMPILE_LANGUAGE:CXX>:-O3>)
endif()
install(TARGETS _core LIBRARY DESTINATION harvest_cuda RUNTIME DESTINATION harvest_cuda)
