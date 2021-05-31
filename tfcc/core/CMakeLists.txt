cmake_minimum_required(VERSION 3.14)

search_library(DEP_LIBRARYS "z")

# check boost
check_module(OFF Boost)
if (BOOST_HOME)
    include_directories(${BOOST_HOME}/include)
    if (CMAKE_SYSTEM_NAME MATCHES "Linux")
        add_definitions(-DBOOST_STACKTRACE_USE_ADDR2LINE)
        search_library(DEP_LIBRARYS "dl")
    endif()
else()
    add_definitions(-DWITHOUT_BOOST)
endif()
# ===========

set(DIRECTORIES "allocators" "dataloaders" "exceptions" "framework" "interfaces" "layers" "operations" "rnn" "utils")
search_srcs(SRC_FILES ${DIRECTORIES})
add_library(tfcc_core ${SRC_FILES})

target_include_directories(
    tfcc_core
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>
)
target_link_libraries(tfcc_core ${DEP_LIBRARYS})

# install
file(GLOB ROOT_HEADERS *.h)
set(HEADER_DIRECTORIES "allocators" "dataloaders" "exceptions" "framework" "interfaces" "layers" "operations" "rnn" "utils")

install(TARGETS tfcc_core EXPORT ${PROJECT_NAME})
install(DIRECTORY ${HEADER_DIRECTORIES} DESTINATION ${CMAKE_INSTALL_PREFIX}/include PATTERN *.cpp EXCLUDE PATTERN *.cu EXCLUDE)
install(FILES ${ROOT_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
