find_package(Threads REQUIRED)

set(SOURCES CellLinkedList.h CellLinkedList.cpp)
add_library(neighborlists SHARED ${SOURCES})
target_link_libraries(neighborlists PRIVATE fmt spdlog Threads::Threads)

