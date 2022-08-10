find_package(OpenMP REQUIRED)

add_executable(bench bench.cpp)
target_link_libraries(bench PRIVATE neighborlists::neighborlists OpenMP::OpenMP_CXX)
