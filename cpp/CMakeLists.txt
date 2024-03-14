cmake_minimum_required(VERSION 3.0)
project(klocal_sparse)
find_package(Boost REQUIRED)

set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_VERBOSE_MAKEFILE ON)
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Building linear.so
add_library(linear SHARED linear.cpp)
set_target_properties(linear PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Building stab.so
add_library(stab SHARED stab.cpp)
set_target_properties(stab PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(stab linear)

# Building state.so
add_library(state SHARED state.cpp)
set_target_properties(state PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_compile_options(state PRIVATE -fopenmp)
target_link_libraries(state PRIVATE stab linear -fopenmp)

# Building test_linear
add_executable(test_linear test_linear.cpp)
target_link_libraries(test_linear linear)
set_target_properties(test_linear PROPERTIES
    INSTALL_RPATH "$ORIGIN"
)

# Building test_stab
add_executable(test_stab test_stab.cpp)
target_link_libraries(test_stab stab linear)
set_target_properties(test_stab PROPERTIES
    INSTALL_RPATH "$ORIGIN"
)

# Building klocal_sparse
add_executable(klocal_sparse klocal_sparse.cpp)
target_link_libraries(klocal_sparse PRIVATE state stab linear -fopenmp -lboost_program_options)
set_target_properties(klocal_sparse PROPERTIES
    INSTALL_RPATH "$ORIGIN"
)

# Building sparse
add_executable(sparse sparse.cpp)
target_link_libraries(sparse PRIVATE state stab linear -fopenmp)
set_target_properties(sparse PROPERTIES
    INSTALL_RPATH "$ORIGIN"
)
