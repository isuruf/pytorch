# Defines torch::symengine, the SymEngine library behind torch/csrc/symbolic.
# With USE_SYSTEM_SYMENGINE it is an installed SymEngine, else a static build
# of a pinned commit with Boost.Multiprecision integers: from an installed
# Boost with USE_SYSTEM_BOOSTMP, else from headers SymEngine downloads.
if(TARGET torch::symengine)
  return()
endif()

add_library(torch::symengine INTERFACE IMPORTED)

if(USE_SYSTEM_SYMENGINE)
  find_package(SymEngine 0.14 REQUIRED CONFIG)
  message(STATUS "Using system SymEngine ${SymEngine_VERSION} from ${SymEngine_DIR}")
  target_include_directories(torch::symengine SYSTEM INTERFACE ${SYMENGINE_INCLUDE_DIRS})
  target_link_libraries(torch::symengine INTERFACE ${SYMENGINE_LIBRARIES})
  return()
endif()

set(_symengine_boost_args -DUSE_SYSTEM_BOOSTMP=${USE_SYSTEM_BOOSTMP})
if(USE_SYSTEM_BOOSTMP)
  find_package(Boost REQUIRED)
  list(APPEND _symengine_boost_args -DBoost_INCLUDE_DIR=${Boost_INCLUDE_DIRS})
endif()

include(ExternalProject)
set(_symengine_prefix "${CMAKE_BINARY_DIR}/symengine")
set(_symengine_lib
  "${_symengine_prefix}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}symengine${CMAKE_STATIC_LIBRARY_SUFFIX}")
ExternalProject_Add(symengine_external
  URL https://github.com/symengine/symengine/archive/1f83971879e23dd016438d5bc7ced410edde3ae5.tar.gz
  URL_HASH SHA256=c77c306684fd96e3903c696d63e45b4dde9e45a0657d3730d48447d05b3fb0ad
  PREFIX "${_symengine_prefix}"
  INSTALL_DIR "${_symengine_prefix}"
  CMAKE_ARGS
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    -DCMAKE_INSTALL_LIBDIR=lib
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DBUILD_SHARED_LIBS=OFF
    -DBUILD_TESTS=OFF
    -DBUILD_BENCHMARKS=OFF
    -DINTEGER_CLASS=boostmp
    -DWITH_SYMENGINE_RCP=ON
    -DWITH_SYMENGINE_THREAD_SAFE=ON
    ${_symengine_boost_args}
  BUILD_BYPRODUCTS "${_symengine_lib}")

# The include directories must exist when the target is configured.
set(_symengine_includes "${_symengine_prefix}/include")
if(USE_SYSTEM_BOOSTMP)
  list(APPEND _symengine_includes ${Boost_INCLUDE_DIRS})
else()
  list(APPEND _symengine_includes
    "${_symengine_prefix}/include/symengine/utilities/boost_multiprecision/include")
endif()
file(MAKE_DIRECTORY ${_symengine_includes})
add_dependencies(torch::symengine symengine_external)
target_include_directories(torch::symengine SYSTEM INTERFACE ${_symengine_includes})
target_link_libraries(torch::symengine INTERFACE "${_symengine_lib}")
