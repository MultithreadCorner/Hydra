### CMakeList for MC Hydra

cmake_minimum_required(VERSION 2.8)

# project name
project(Hydra)

# The version number.
set (Hydra_VERSION_MAJOR 1)
set (Hydra_VERSION_MINOR 0)
set (Hydra_VERSION_PATCH 0)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
 MESSAGE(STATUS "===> Compiling with Clang")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
 MESSAGE(STATUS "===> Compiling with GCC")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
 MESSAGE(STATUS "===> Compiling with Intel")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
 MESSAGE(STATUS "===> Compiling with MSVC")
endif()

# warn user if system is not UNIX
if(NOT UNIX)
  message(FATAL_ERROR "This is an unsupported system.")
endif()

SET(Hydra_CMAKE_DIR "${PROJECT_SOURCE_DIR}/cmake")
SET(CMAKE_MODULE_PATH "${Hydra_CMAKE_DIR}" ${CMAKE_MODULE_PATH})
SET(CMAKE_VERBOSE_MAKEFILE  ON)

#options to gcc
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("--std=c++11" COMPILER_SUPPORTS_CXX11)
if(NOT COMPILER_SUPPORTS_CXX11)
 message(FATAL "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

#options to gcc-DTHRUST_VARIADIC_TUPLE -D_GLIBCXX_USE_CXX11_ABI=0 


set(CMAKE_CXX_FLAGS "-ftemplate-backtrace-limit=0 --std=c++11 -W -march=native -fPIC -O4" CACHE STRING "compile flags" FORCE)

# get TCLAP
find_package(TCLAP REQUIRED)
include_directories(${TCLAP_INCLUDE_DIR})


# get THRUST
#find_package(Thrust 1.8.3 REQUIRED)
#include_directories(./thrust)

# get ROOT
find_package(ROOT)
if(ROOT_FOUND)
include_directories(${ROOT_INCLUDE_DIR})
link_directories(${ROOT_LIBRARY_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_ROOT_AVAILABLE_")
endif(ROOT_FOUND)

SET(CUDA_SEPARABLE_COMPILATION ON)
#get CUDA
find_package(CUDA 8.0) 
if(CUDA_FOUND)
link_directories( ${CUDA_TOOLKIT_ROOT_DIR}/lib64/)
endif(CUDA_FOUND)
#get TBB
find_package(TBB )
if(TBB_FOUND)
include_directories(${TBB_INCLUDE_DIRS})
link_directories(${TBB_LIBRARY})
endif(TBB_FOUND)

#get OpenMP
find_package(OpenMP)
if(OPENMP_FOUND)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OPENMP_FOUND)


#generate API documentation with Doxygen
find_package(Doxygen)
option(BUILD_DOCUMENTATION "Create and install the HTML based API documentation (requires Doxygen)" NO)


#including directories

include_directories(include;  ${PROJECT_SOURCE_DIR})



if(CUDA_FOUND)

	#set cuda flags
	
	SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  -ftemplate-backtrace-limit=0; --cudart ;  static ; -O4 ;--expt-relaxed-constexpr; -fmad=true; --expt-extended-lambda;--relocatable-device-code=false ;
	-Xptxas -dlcm=ca; -Xptxas --opt-level=4 )

        SET(CUDA_PROPAGATE_HOST_FLAGS ON)
	SET(CUDA_SEPARABLE_COMPILATION OFF)
	SET(CUDA_VERBOSE_BUILD OFF)
	
	if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.8)
	   LIST(APPEND CUDA_NVCC_FLAGS " -Xcompiler -D_MWAITXINTRIN_H_INCLUDED ")
    	endif()
    	
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.4)
	    LIST(APPEND CUDA_NVCC_FLAGS " -Xcompiler -D__CORRECT_ISO_CPP11_MATH_H_PROTO ")
    	endif()
	# Detect CUDA architecture and get best NVCC flags
	
	INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindCudaArch.cmake)
	
	SELECT_NVCC_ARCH_FLAGS(NVCC_FLAGS_EXTRA)
	
	LIST(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})

	
    		
endif(CUDA_FOUND)

SET(BUILD_CPP_TARGETS  ON)

if( OPENMP_FOUND )
        SET(BUILD_OMP_TARGETS  ON)
else()
        SET(BUILD_OMP_TARGETS  OFF)
endif()

if( CUDA_FOUND )
	SET(BUILD_CUDA_TARGETS  ON)
else()
	SET(BUILD_CUDA_TARGETS  OFF)
endif()

if(TBB_FOUND)
    SET(BUILD_TBB_TARGETS  ON)
else()
    SET(BUILD_TBB_TARGETS  OFF)
endif(TBB_FOUND)

# messages
MESSAGE(STATUS "Build CUDA/NVCC-based targets: ${BUILD_CUDA_TARGETS}")
MESSAGE(STATUS "TCLAP Include path: ${TCLAP_INCLUDE_DIR}")
MESSAGE(STATUS "Thrust Include path: ${THRUST_INCLUDE_DIR}")
MESSAGE(STATUS "ROOT Include path: ${ROOT_INCLUDE_DIR}")
MESSAGE(STATUS "ROOT Library path: ${ROOT_LIBRARY_DIR}" )
MESSAGE(STATUS "ROOT Libraries: ${ROOT_LIBRARIES}")
MESSAGE(STATUS "project source dir: " ${PROJECT_SOURCE_DIR} )
MESSAGE(STATUS "project build dir: " ${PROJECT_BINARY_DIR} )
MESSAGE(STATUS "CUDA include: " ${CUDA_INCLUDE_DIRS})
MESSAGE(STATUS "CUDA libraries: " ${CUDA_LIBRARIES}  )
MESSAGE(STATUS "NVCC cflags: " ${CUDA_NVCC_FLAGS} )
MESSAGE(STATUS "GCC cflags: " ${CMAKE_CXX_FLAGS} )
MESSAGE(STATUS "Include directories: " ${INCLUDE_DIRECTORIES} )
MESSAGE(STATUS "Install diretory: " ${CMAKE_INSTALL_PREFIX}  )

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/plots)


#+++++++++++++++++++++++++++
#        EXAMPLES          +
#+++++++++++++++++++++++++++
add_custom_target(examples)

add_subdirectory(examples/phase_space)
add_subdirectory(examples/numerical_integration)
add_subdirectory(examples/random)
add_subdirectory(examples/fit)
add_subdirectory(examples/histograming)
add_subdirectory(examples/async)
add_subdirectory(examples/misc)



install(DIRECTORY Hydra
  DESTINATION include)

install(DIRECTORY 
	src
  DESTINATION src/hydra/samples)
  
install(DIRECTORY 
	cmake
  DESTINATION src/hydra/cmake)

install(FILES
  CMakeLists.txt
  COPYING
  THANKS.md
  README.md
  CHANGELOG.md
  DESTINATION src/Hydra)
  


if(BUILD_DOCUMENTATION)

    if(NOT DOXYGEN_FOUND)
        message(FALTAL_ERROR "Doxygen is needed to build the documentation.")
    endif()

    set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(doxyfile    ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)

    configure_file(${doxyfile_in} ${doxyfile} @ONLY)

    add_custom_target(doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)

    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc DESTINATION src/hydra/doc)
    
endif()

# build a CPack driven installer package
include (InstallRequiredSystemLibraries)
set (CPACK_SOURCE_IGNORE_FILES  build/*)
set (CPACK_RESOURCE_FILE_LICENSE  
     "${CMAKE_CURRENT_SOURCE_DIR}/COPYING")
set (CPACK_RESOURCE_FILE_README  
     "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
set (CPACK_PACKAGE_VERSION_MAJOR "${Hydra_VERSION_MAJOR}")
set (CPACK_PACKAGE_VERSION_MINOR "${Hydra_VERSION_MINOR}")
set (CPACK_PACKAGE_VERSION_PATCH "${Hydra_VERSION_PATCH}")
include (CPack)



