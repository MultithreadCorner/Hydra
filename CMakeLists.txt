### CMakeList for Hydra examples and documentation

cmake_minimum_required(VERSION 3.2)

# project name
project(Hydra_examples_and_documentation)

# warn user if system is not UNIX
if(NOT UNIX)
  message(FATAL_ERROR "This is an unsupported system.")
endif()

#cmake path dir
SET(Hydra_CMAKE_DIR "${PROJECT_SOURCE_DIR}/cmake")
SET(CMAKE_MODULE_PATH "${Hydra_CMAKE_DIR}" ${CMAKE_MODULE_PATH})
SET(CMAKE_VERBOSE_MAKEFILE  ON)

#check if compiler is C++11 compliant
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("--std=c++11" COMPILER_SUPPORTS_CXX11)
if(NOT COMPILER_SUPPORTS_CXX11)
 message(FATAL "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

#compiler flags
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")

 MESSAGE(STATUS "Setting Clang flags")
 set(CMAKE_CXX_FLAGS " --std=c++11 -W -march=native -fPIC -O3 -ldl" CACHE STRING "compile flags" FORCE)

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")

 MESSAGE(STATUS "Setting GCC flags")
 set(CMAKE_CXX_FLAGS " --std=c++11 -W -march=native -fPIC -O4 -ldl" CACHE STRING "compile flags" FORCE)

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")

 MESSAGE(STATUS "Setting ICC flags")
 set(CMAKE_CXX_FLAGS " --std=c++11 -W -march=native -fPIC -O4 -ldl" CACHE STRING "compile flags" FORCE)

#elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
 #MESSAGE(STATUS "===> Compiling with MSVC")
endif()

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
include_directories(${ROOT_INCLUDE_DIR})
link_directories(${ROOT_LIBRARY_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_ROOT_AVAILABLE_")
if(${ROOT_Minuit2_LIBRARY} MATCHES "libMinuit2.so")
set(Minuit2_FOUND ON)
else(Minuit2_FOUND OFF)
endif()
endif(ROOT_FOUND)

#-----------------------
# get GSL
find_package(GSL)
if(GSL_FOUND)
include_directories(${GSL_INCLUDE_DIRS})
endif(GSL_FOUND)

#-----------------------
# get FFTW
find_package(FFTW)
if(FFTW_FOUND)
include_directories(${FFTW_INCLUDE_DIRS})
endif(FFTW_FOUND)

#-----------------------
#get CUDA
find_package(CUDA 8.0)
if(CUDA_FOUND)
link_directories( ${CUDA_TOOLKIT_ROOT_DIR}/lib64/)

#set cuda flags
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  -ftemplate-backtrace-limit=0; --cudart;  static; -O4; --expt-relaxed-constexpr; -fmad=true; --expt-extended-lambda;--relocatable-device-code=false ;
-Xptxas -dlcm=ca; -Xptxas --opt-level=4 )

SET(CUDA_PROPAGATE_HOST_FLAGS ON)
SET(CUDA_SEPARABLE_COMPILATION OFF)
SET(CUDA_VERBOSE_BUILD OFF)

if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.8)
  # LIST(APPEND CUDA_NVCC_FLAGS " -Xcompiler -D_MWAITXINTRIN_H_INCLUDED ")
endif()

if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.4)
    LIST(APPEND CUDA_NVCC_FLAGS " -Xcompiler -D__CORRECT_ISO_CPP11_MATH_H_PROTO ")
endif()
# Detect CUDA architecture and get best NVCC flags

INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindCudaArch.cmake)

SELECT_NVCC_ARCH_FLAGS(NVCC_FLAGS_EXTRA)

LIST(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})

endif(CUDA_FOUND)

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
MESSAGE(STATUS "ROOT include path: ${ROOT_INCLUDE_DIR}")
MESSAGE(STATUS "ROOT library path: ${ROOT_LIBRARY_DIR}" )
MESSAGE(STATUS "ROOT libraries: ${ROOT_LIBRARIES}")
MESSAGE(STATUS "ROOT::Minuit2: ${ROOT_Minuit2_LIBRARY}")
MESSAGE(STATUS "Build CUDA/NVCC-based targets: ${BUILD_CUDA_TARGETS}")
MESSAGE(STATUS "CUDA include: ${CUDA_INCLUDE_DIRS}" )
MESSAGE(STATUS "CUDA RT libraries: ${CUDA_LIBRARIES}"  )
MESSAGE(STATUS "CUDA CUFFT libraries: ${CUDA_CUFFT_LIBRARIES}")
MESSAGE(STATUS "NVCC flags: ${CUDA_NVCC_FLAGS}" )
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
enable_testing()

add_custom_target(tests)
add_subdirectory(testing)

#add_dependencies(tests
#Hydra_Test_Host_OMP_Device_CUDA
#Hydra_Test_Host_CPP_Device_CUDA
#Hydra_Test_Host_OMP_Device_OMP
#Hydra_Test_Host_CPP_Device_OMP
#)

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
