cmake_minimum_required(VERSION 3.20)
project(sparkmobility_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
find_package(h3 CONFIG REQUIRED)
find_package(Snappy CONFIG)
if(NOT Snappy_FOUND)
  find_library(SNAPPY_LIBRARY NAMES snappy REQUIRED)
  find_path(SNAPPY_INCLUDE_DIR snappy.h REQUIRED)
endif()
# Apple Clang doesn't ship OpenMP and FindOpenMP can't locate brew's keg-only
# libomp without hints. Supply the flags + library path so find_package(OpenMP)
# succeeds with the libomp installed by `brew install libomp`.
if(APPLE)
  execute_process(
    COMMAND brew --prefix libomp
    OUTPUT_VARIABLE LIBOMP_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
  set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
  set(OpenMP_C_LIB_NAMES "omp")
  set(OpenMP_CXX_LIB_NAMES "omp")
  set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
endif()
find_package(OpenMP REQUIRED)

add_executable(module_2_3_1
  sparkmobility/models/timegeo/_native/module_2_3_1.cpp
)

target_link_libraries(module_2_3_1 PRIVATE
  Arrow::arrow_shared
  Parquet::parquet_shared
  h3::h3
  OpenMP::OpenMP_CXX
)

if(Snappy_FOUND)
  target_link_libraries(module_2_3_1 PRIVATE Snappy::snappy)
else()
  target_include_directories(module_2_3_1 PRIVATE ${SNAPPY_INCLUDE_DIR})
  target_link_libraries(module_2_3_1 PRIVATE ${SNAPPY_LIBRARY})
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_libraries(module_2_3_1 PRIVATE stdc++fs)
endif()

install(
  TARGETS module_2_3_1
  RUNTIME DESTINATION sparkmobility/models/timegeo/_native
)
