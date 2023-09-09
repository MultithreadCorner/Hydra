### CMakeList for Hydra examples and documentation

cmake_minimum_required(VERSION 3.24)

# project name
project(Hydra_examples_and_documentation LANGUAGES CXX)

# warn user if system is not UNIX
if(NOT UNIX)
  message(FATAL_ERROR "This is an unsupported system.")
endif()

#cmake path dir
SET(Hydra_CMAKE_DIR "${PROJECT_SOURCE_DIR}/cmake")
SET(CMAKE_MODULE_PATH "${Hydra_CMAKE_DIR}" ${CMAKE_MODULE_PATH})
SET(CMAKE_VERBOSE_MAKEFILE  ON)

#check if compilers are C++17 compliant
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("--std=c++17" COMPILER_SUPPORTS_CXX17)
if(NOT COMPILER_SUPPORTS_CXX17)
 message(FATAL "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
endif()

#compiler flags
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")

 MESSAGE(STATUS "Setting Clang flags")
 set(CMAKE_CXX_FLAGS " --std=c++17 -W -march=native -fPIC -O4 -ldl" CACHE STRING "compile flags" FORCE)

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")

 MESSAGE(STATUS "Setting GCC flags")
 set(CMAKE_CXX_FLAGS " --std=c++17 -W -march=native -fPIC -O4 -ldl" CACHE STRING "compile flags" FORCE)

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")

 MESSAGE(STATUS "Setting ICC flags")
 set(CMAKE_CXX_FLAGS " --std=c++17 -W -march=native -fPIC -O4 -ldl" CACHE STRING "compile flags" FORCE)
endif()

#-----------------------
# Handling CUDA

include(CheckLanguage)

check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  message(STATUS "CUDA compiler found. Enabling CUDA support")
else(CMAKE_CUDA_COMPILER)
  message(STATUS "CUDA compiler not found. Disabling CUDA support")
endif(CMAKE_CUDA_COMPILER)


if(CMAKE_CUDA_COMPILER)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


find_package(CUDAToolkit)

if(CUDAToolkit_FOUND)

set( CUDA_FOUND TRUE)
list(APPEND HYDRA_CUDA_FLAGS "--std=c++17")
list(APPEND HYDRA_CUDA_FLAGS "--gpu-architecture=native")
list(APPEND HYDRA_CUDA_FLAGS "-extended-lambda")
list(APPEND HYDRA_CUDA_FLAGS "-split-compile 0")
list(APPEND HYDRA_CUDA_FLAGS "-O4")
#list(APPEND HYDRA_CUDA_FLAGS "--debug")
#list(APPEND HYDRA_CUDA_FLAGS "--device-debug")
#list(APPEND HYDRA_CUDA_FLAGS "-lineinfo")
list(APPEND HYDRA_CUDA_FLAGS "-m64")
list(APPEND HYDRA_CUDA_FLAGS "--threads=0")
list(APPEND HYDRA_CUDA_FLAGS "--verbose")
list(APPEND HYDRA_CUDA_FLAGS "--expt-relaxed-constexpr")
#list(APPEND HYDRA_CUDA_FLAGS "")
list(JOIN   HYDRA_CUDA_FLAGS " " HYDRA_CUDA_FLAGS)
set( CMAKE_CUDA_FLAGS "${HYDRA_CUDA_FLAGS}" CACHE STRING "CUDA compile flags" FORCE)
get_target_property(HYDRA_CUDA_RT CUDA::cudart ALIASED_TARGET)
get_target_property(HYDRA_CUDA_FFT CUDA::cufft ALIASED_TARGET)

endif(CUDAToolkit_FOUND)
endif(CMAKE_CUDA_COMPILER)

#-----------------------
# get Hydra
find_package(Hydra REQUIRED)
include_directories(${Hydra_INCLUDE_DIR})

#-----------------------
# get TCLAP
find_package(TCLAP REQUIRED)
if(TCLAP_FOUND)
include_directories(${TCLAP_INCLUDE_DIR})
endif(TCLAP_FOUND)

#-----------------------
# get ROOT
find_package(ROOT COMPONENTS Minuit2)
if(ROOT_FOUND)
include_directories(${ROOT_INCLUDE_DIRS})
link_directories(${ROOT_LIBRARIES})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_ROOT_AVAILABLE_")
if(CUDA_FOUND)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_ROOT_AVAILABLE_")
endif(CUDA_FOUND)
if(${ROOT_Minuit2_LIBRARY} MATCHES "libMinuit2.so")
set(Minuit2_FOUND ON)
else(Minuit2_FOUND OFF)
endif()
endif(ROOT_FOUND)

#-----------------------
# get FFTW for convolution examples
find_package(FFTW)
if(FFTW_FOUND)
include_directories(${FFTW_INCLUDE_DIRS})
endif(FFTW_FOUND)


#-----------------------
#get TBB
find_package(TBB )
if(TBB_FOUND)
include_directories(${TBB_INCLUDE_DIRS})
link_directories(${TBB_LIBRARY})
endif(TBB_FOUND)

#-----------------------
#get OpenMP
find_package(OpenMP)
if(OPENMP_CXX_FOUND OR OPENMP_FOUND)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OPENMP_CXX_FOUND OR OPENMP_FOUND)

#-----------------------
#get TestU01
find_package(TestU01)
if(TESTU01_FOUND)
include_directories(${TESTU01_INCLUDE_DIRS})
endif(TESTU01_FOUND)

#generate API documentation with Doxygen
#find_package(Doxygen)
option(BUILD_DOCUMENTATION "Create and install the HTML based API documentation (requires Doxygen)" NO)

# examples etc
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#always on
SET(BUILD_CPP_TARGETS  TRUE)

#crucial for clang build, where openmp support is not embeded in the compiler
if( OPENMP_CXX_FOUND OR OPENMP_FOUND)

        SET(BUILD_OMP_TARGETS  TRUE)
else()
        SET(BUILD_OMP_TARGETS  FALSE)
endif()

#not every computer has a GPU and/or cuda installed or
if( CUDA_FOUND )
	SET(BUILD_CUDA_TARGETS  TRUE)
else()
	SET(BUILD_CUDA_TARGETS  FALSE)
endif()

# to enable tbb builds
if(TBB_FOUND)
    SET(BUILD_TBB_TARGETS  TRUE)
else()
    SET(BUILD_TBB_TARGETS  FALSE)
endif(TBB_FOUND)

# messages
MESSAGE(STATUS "TCLAP Include path: ${TCLAP_INCLUDE_DIR}")
MESSAGE(STATUS "ROOT include path: ${ROOT_INCLUDE_DIRS}")
MESSAGE(STATUS "ROOT library path: ${ROOT_LIBRARY_DIR}" )
MESSAGE(STATUS "ROOT libraries: ${ROOT_LIBRARIES}")
MESSAGE(STATUS "ROOT::Minuit2: ${ROOT_Minuit2_LIBRARY}")
MESSAGE(STATUS "Build CUDA/NVCC-based targets: ${BUILD_CUDA_TARGETS}")
MESSAGE(STATUS "CUDAToolkit include: ${CUDAToolkit_INCLUDE_DIRS}" )
MESSAGE(STATUS "CUDA RT libraries: ${HYDRA_CUDA_RT}"  )
MESSAGE(STATUS "CUDA CUFFT libraries: CUDA::cufft: ${HYDRA_CUDA_FFT}" )
MESSAGE(STATUS "NVCC flags: ${CMAKE_CUDA_FLAGS}" )
MESSAGE(STATUS "Build TBB targets: ${BUILD_TBB_TARGETS}")
MESSAGE(STATUS "TBB include: ${TBB_INCLUDE_DIRS}" )
MESSAGE(STATUS "TBB libraries: ${TBB_LIBRARIES}" )
MESSAGE(STATUS "Build OpenMP targets: ${BUILD_OMP_TARGETS}")
MESSAGE(STATUS "OpenMP flags: ${OpenMP_CXX_FLAGS}" )
MESSAGE(STATUS "OpenMP libraries:  ${OpenMP_CXX_LIBRARIES}" )
MESSAGE(STATUS "Host compiler flags: ${CMAKE_CXX_FLAGS}" )
MESSAGE(STATUS "Install diretory: ${CMAKE_INSTALL_PREFIX}"  )
MESSAGE(STATUS "Project source dir: ${PROJECT_SOURCE_DIR}" )
MESSAGE(STATUS "Project build dir: ${PROJECT_BINARY_DIR}" )

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/plots)


#+++++++++++++++++++++++++++
#        EXAMPLES          +
#+++++++++++++++++++++++++++
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_custom_target(examples)
add_custom_target(examples_cpp)
add_custom_target(examples_tbb)
add_custom_target(examples_omp)
add_custom_target(examples_cuda)

include(${Hydra_CMAKE_DIR}/AddHydraExample.cmake)

add_subdirectory(examples/phase_space)
add_subdirectory(examples/numerical_integration)
add_subdirectory(examples/random)
add_subdirectory(examples/histograming)
add_subdirectory(examples/async)
add_subdirectory(examples/misc)
if(Minuit2_FOUND)
add_subdirectory(examples/convolution)
add_subdirectory(examples/phys)
add_subdirectory(examples/fit)
endif(Minuit2_FOUND)


#+++++++++++++++++++++++++++
#        TESTING           +
#+++++++++++++++++++++++++++
include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.4.0
)
FetchContent_MakeAvailable(Catch2)

include(CTest)

add_custom_target(practrand_streamers)

if(BUILD_TESTING)
add_subdirectory(testing)
endif()


#+++++++++++++++++++++++++++
#       DOXYGEN            +
#+++++++++++++++++++++++++++

if(BUILD_DOXYGEN_DOCUMENTATION)
find_package(Doxygen)
    if(NOT DOXYGEN_FOUND)
        message(FALTAL_ERROR "Doxygen documentation generation requested ( BUILD_DOXYGEN_DOCUMENTATION=ON) but Doxygen is not installed.")
    endif()

    set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(doxyfile    ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)

    configure_file(${doxyfile_in} ${doxyfile} @ONLY)

    add_custom_target(doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)

   install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc/html DESTINATION hydra/doc/doxygen)

endif(BUILD_DOXYGEN_DOCUMENTATION)

#+++++++++++++++++++++++++++
#         INSTALL          +
#+++++++++++++++++++++++++++

#install headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/hydra
  DESTINATION include)

#install docs
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/docs
  DESTINATION hydra/docs)

#install examples executables
if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/examples/")
install(DIRECTORY  ${CMAKE_CURRENT_BINARY_DIR}/examples
  DESTINATION hydra/bin/examples)
endif(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/examples/")

#install performance executables
if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/performance/")
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/performance
  DESTINATION hydra/bin/performance)
endif(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/performance/")

#install test executables
if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/tests/")
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tests
  DESTINATION hydra/bin/tests)
endif(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/tests/")

#install source
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DESTINATION hydra/dist)
