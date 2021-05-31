cmake_minimum_required(VERSION 3.14)

include(CheckCompilerFeatures)

option(TFCC_MKL_USE_AVX2 "Build tfcc_mkl with avx2 support" ON)
option(TFCC_MKL_USE_AVX512 "Build tfcc_mkl with avx512 support" ON)

if (MSVC)
    add_compile_options(/openmp)
else()
    add_compile_options(-fopenmp -masm=intel -fabi-version=0)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
        add_compile_options(-Wno-noexcept-type)
    endif()
endif()

if (${TFCC_MKL_USE_AVX2})
    add_definitions(-DTFCC_USE_AVX2)
    search_srcs_with_filter(".*\.avx256\..*" AVX256_SRC_FILES "kernel")
    set(SRC_FILES ${SRC_FILES} ${AVX256_SRC_FILES})
    set_source_files_properties(${AVX256_SRC_FILES} PROPERTIES COMPILE_FLAGS "-mfma -mavx2 -Wno-unused-variable -Wno-error=uninitialized")
endif()

if (${TFCC_MKL_USE_AVX512})
    add_definitions(-DTFCC_USE_AVX512)
    search_srcs_with_filter(".*\.avx512\..*" AVX512_SRC_FILES "kernel")
    set(SRC_FILES ${SRC_FILES} ${AVX512_SRC_FILES})
    set_source_files_properties(${AVX512_SRC_FILES} PROPERTIES COMPILE_FLAGS "-mfma -mavx512f -mavx512dq -mavx512cd -mavx512bw -mavx512vl")
endif()

search_library(DEP_LIBRARYS "m")

# check dnnl
check_module(ON dnnl)
include_directories(${DNNL_HOME}/include)
search_library(DEP_LIBRARYS "dnnl" ${DNNL_HOME}/lib64 ${DNNL_HOME}/lib)

# check vectorclass
check_module(ON vectorclass)
include_directories(${VECTORCLASS_HOME})

# svml
check_module(OFF svml)
if (SVML_HOME)
    add_definitions(-DTFCC_USE_SVML)
    search_library(DEP_LIBRARYS "svml" ${SVML_HOME})
    search_library(DEP_LIBRARYS "irc" ${SVML_HOME})
endif()

# mkl, for fft
check_module(OFF mkl)
if (MKL_HOME)
    add_definitions(-DTFCC_USE_MKL)
    include_directories(${MKL_HOME}/include)
    search_library(MKL_DEP_LIBRARYS "mkl_gf_lp64" ${MKL_HOME}/lib/intel64)
    search_library(MKL_DEP_LIBRARYS "mkl_gnu_thread" ${MKL_HOME}/lib/intel64)
    search_library(MKL_DEP_LIBRARYS "mkl_core" ${MKL_HOME}/lib/intel64)
endif()

# check compiler features
is_support_parameter_pack_lambda_capture(IS_SUPPORT_PPLC)

if (IS_SUPPORT_PPLC)
    message("compiler is support parameter pack lambda capture")
    add_definitions(-DTFCC_COMPILER_IS_SUPPORT_PPLC)
endif()
# target
include_directories("../../3rd")
set(DIRECTORIES "framework" "interfaces" "operations" "utils" "fusionop" "fusionop/fusionoperations")
search_srcs(SRC_FILES ${DIRECTORIES})

add_library(tfcc_mkl ${SRC_FILES})
target_link_libraries(tfcc_mkl tfcc_core)
target_link_libraries(tfcc_mkl ${DEP_LIBRARYS})
target_link_libraries(tfcc_mkl -lpthread)
if (MKL_HOME)
    target_link_libraries(tfcc_mkl -Wl,--start-group ${MKL_DEP_LIBRARYS} -Wl,--end-group)
endif()

target_include_directories(
    tfcc_mkl
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>
)

# openmp
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(tfcc_mkl OpenMP::OpenMP_CXX)
endif()

# install
file(GLOB ROOT_HEADERS *.h)
set(HEADER_DIRECTORIES "framework" "operations" "utils")

install(TARGETS tfcc_mkl EXPORT ${PROJECT_NAME})
install(DIRECTORY ${HEADER_DIRECTORIES} DESTINATION ${CMAKE_INSTALL_PREFIX}/include PATTERN *.cpp EXCLUDE PATTERN *.cu EXCLUDE)
install(FILES ${ROOT_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
