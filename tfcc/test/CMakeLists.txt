cmake_minimum_required(VERSION 3.14)
set(CMAKE_CUDA_DEVICE_LINK_EXECUTABLE "<CMAKE_CUDA_COMPILER> <FLAGS> <CMAKE_CUDA_LINK_FLAGS> ${CMAKE_CUDA_COMPILE_OPTIONS_PIC} -shared -dlink <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")

if (NOT DEFINED GTEST_HOME AND DEFINED ENV{GTEST_HOME})
    set(GTEST_HOME $ENV{GTEST_HOME})
endif()

#check gtest
if (NOT DEFINED GTEST_HOME OR "${GTEST_HOME}" STREQUAL "" OR NOT EXISTS ${GTEST_HOME})
    message(SEND_ERROR "`GTEST_HOME` not defined or invalid. You can specify `GTEST_HOME` environment variable or cmake variable to the corresponding path.")
else()
    include_directories(${GTEST_HOME}/include)
endif()

if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    add_definitions(-DBOOST_STACKTRACE_USE_ADDR2LINE)
endif()

if (${TFCC_WITH_MKL} STREQUAL "ON")
    add_definitions(-DTFCC_WITH_MKL)
endif()

if (${TFCC_WITH_CUDA} STREQUAL "ON")
    add_definitions(-DTFCC_WITH_CUDA)
endif()
#== == == == == =


set(DIRECTORIES "./")

set(SRC_FILES "")

foreach(DIR ${DIRECTORIES})
    aux_source_directory(${DIR} SRC_FILES)
endforeach()

include_directories(./)
add_executable(tfcc_test ${SRC_FILES})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

if (${TFCC_WITH_MKL} STREQUAL "ON")
    target_link_libraries(tfcc_test tfcc_mkl)
endif()
if (${TFCC_WITH_CUDA} STREQUAL "ON")
    target_link_libraries(tfcc_test tfcc_cuda)
endif()

target_link_libraries(tfcc_test tfcc_core)
if (NOT EXISTS ${GTEST_HOME}/make/gtest-all.o)
    target_link_libraries(tfcc_test "-L${GTEST_HOME}/lib64 -lgtest")
else()
    target_link_libraries(tfcc_test ${GTEST_HOME}/make/gtest-all.o)
endif()

add_custom_target(tfcc_test_data python ${CMAKE_CURRENT_SOURCE_DIR}/python/createtestdata.py --out=${CMAKE_CURRENT_BINARY_DIR}/data/tfcc_test_data --force_update=False)

if (${TFCC_OPEN_COVERAGE})
    set(COVERAGE_LCOV_EXCLUDES samples/* /usr/local/*)
    setup_target_for_coverage_lcov(
        NAME check_mkl
        EXECUTABLE tfcc_test mkl ${CMAKE_CURRENT_BINARY_DIR}/data --gtest_color=yes
        DEPENDENCIES tfcc_test tfcc_test_data
    )
    setup_target_for_coverage_lcov(
        NAME check_cuda
        EXECUTABLE tfcc_test cuda ${CMAKE_CURRENT_BINARY_DIR}/data --gtest_color=yes
        DEPENDENCIES tfcc_test tfcc_test_data
    )
    setup_target_for_coverage_lcov(
        NAME check
        EXECUTABLE tfcc_test all ${CMAKE_CURRENT_BINARY_DIR}/data --gtest_color=yes
        DEPENDENCIES tfcc_test tfcc_test_data
    )

else()
    add_custom_target(check_mkl ./tfcc_test mkl ${CMAKE_CURRENT_BINARY_DIR}/data --gtest_color=yes)
    add_dependencies(check_mkl tfcc_test tfcc_test_data)
    add_custom_target(check_cuda ./tfcc_test cuda ${CMAKE_CURRENT_BINARY_DIR}/data --gtest_color=yes)
    add_dependencies(check_cuda tfcc_test tfcc_test_data)
    add_custom_target(check)
    if (${TFCC_WITH_MKL} STREQUAL "ON")
        add_dependencies(check check_mkl)
    endif()

    if (${TFCC_WITH_CUDA} STREQUAL "ON")
        add_dependencies(check check_cuda)
    endif()
endif()

