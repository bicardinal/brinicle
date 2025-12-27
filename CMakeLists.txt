cmake_minimum_required(VERSION 3.18)
project(brinicle LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP)

# C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build the Python extension module as brinicle/_brinicle*.so|pyd
pybind11_add_module(_brinicle brinicle/src/bindings.cpp)

target_include_directories(_brinicle PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/brinicle/src
)

target_compile_options(_brinicle PRIVATE -msse3)

set_target_properties(_brinicle PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

if (MSVC)
  target_compile_options(_brinicle PRIVATE /O2 /DNDEBUG)
else()
  target_compile_options(_brinicle PRIVATE
    -O3
    -DNDEBUG
    -fPIC
    -funroll-loops
  )

  option(BRINICLE_NATIVE "Enable -march=native/-mfma (non-portable)" OFF)
  if (BRINICLE_NATIVE)
    target_compile_options(_brinicle PRIVATE -march=native -mfma)
  endif()

  include(CheckCXXCompilerFlag)

  include(CheckIPOSupported)
  check_ipo_supported(RESULT ipo_ok OUTPUT ipo_err)
  if (ipo_ok)
    set_property(TARGET _brinicle PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  endif()

  check_cxx_compiler_flag("-fvisibility=hidden" HAS_VIS_HIDDEN)
  if (HAS_VIS_HIDDEN)
    target_compile_options(_brinicle PRIVATE -fvisibility=hidden)
  endif()
endif()

if (OpenMP_CXX_FOUND)
  target_link_libraries(_brinicle PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS _brinicle
  LIBRARY DESTINATION brinicle
  RUNTIME DESTINATION brinicle
  ARCHIVE DESTINATION brinicle
)