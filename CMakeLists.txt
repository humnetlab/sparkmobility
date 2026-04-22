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
find_package(OpenMP REQUIRED)
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

add_executable(module_2_3_1
  sparkmobility/models/timegeo/_native/module_2_3_1.cpp
)

target_link_libraries(module_2_3_1 PRIVATE
  Arrow::arrow_shared
  Parquet::parquet_shared
  h3::h3
  OpenMP::OpenMP_CXX
  pybind11::embed
  Python::Python
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
