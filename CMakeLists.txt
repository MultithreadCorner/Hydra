### CMakeList for MC Hydra

cmake_minimum_required(VERSION 3.1)

# Project name and version
project(Hydra LANGUAGES CXX
              VERSION 1.0.0)

# Warn user if system is not UNIX
if(NOT UNIX)
  message(WARNING "This is an unsupported system.")
endif()

# Adding the helper files in the cmake subdirectory
set(Hydra_CMAKE_DIR "${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_MODULE_PATH "${Hydra_CMAKE_DIR}" ${CMAKE_MODULE_PATH})

# Options to gcc
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("--std=c++11" COMPILER_SUPPORTS_CXX11)
if(NOT COMPILER_SUPPORTS_CXX11)
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

set(CMAKE_CXX_FLAGS "-ftemplate-backtrace-limit=0 --std=c++11 -W -march=native -fPIC -O4 -ldl" CACHE STRING "compile flags" FORCE)

# Get TCLAP
find_package(TCLAP REQUIRED)
include_directories(${TCLAP_INCLUDE_DIR})

# Get ROOT
find_package(ROOT COMPONENTS Minuit2)
if(ROOT_FOUND)
    include_directories(${ROOT_INCLUDE_DIR})
    link_directories(${ROOT_LIBRARY_DIR})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_ROOT_AVAILABLE_")
endif(ROOT_FOUND)

# Get TBB
find_package(TBB )
if(TBB_FOUND)
    include_directories(${TBB_INCLUDE_DIRS})
    link_directories(${TBB_LIBRARY})
endif(TBB_FOUND)

# Get OpenMP
find_package(OpenMP)
if(OPENMP_CXX_FOUND OR OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OPENMP_CXX_FOUND OR OPENMP_FOUND)

# Generate API documentation with Doxygen
find_package(Doxygen)
option(BUILD_DOCUMENTATION "Create and install the HTML based API documentation (requires Doxygen)" NO)

# Include directories
include_directories(${PROJECT_SOURCE_DIR})

# Get CUDA
find_package(CUDA 8.0) 
if(CUDA_FOUND)
    link_directories( ${CUDA_TOOLKIT_ROOT_DIR}/lib64/)

    # Set cuda flags
	
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  -ftemplate-backtrace-limit=0; --cudart ;  static ; -O4 ;--expt-relaxed-constexpr; -fmad=true; --expt-extended-lambda;--relocatable-device-code=false ;
        -Xptxas -dlcm=ca; -Xptxas --opt-level=4 )

    set(CUDA_PROPAGATE_HOST_FLAGS ON)
    set(CUDA_SEPARABLE_COMPILATION OFF)
    set(CUDA_VERBOSE_BUILD OFF)
	
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.8)
        list(APPEND CUDA_NVCC_FLAGS " -Xcompiler -D_MWAITXINTRIN_H_INCLUDED ")
    endif()
    	
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.4)
        list(APPEND CUDA_NVCC_FLAGS " -Xcompiler -D__CORRECT_ISO_CPP11_MATH_H_PROTO ")
    endif()
	# Detect CUDA architecture and get best NVCC flags
	
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindCudaArch.cmake)
	
    select_nvcc_arch_flags(NVCC_FLAGS_EXTRA)
	
    list(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})
    		
endif(CUDA_FOUND)


set(BUILD_CPP_TARGETS ON)

if(OPENMP_CXX_FOUND OR OPENMP_FOUND)
    set(BUILD_OMP_TARGETS ON)
else()
    set(BUILD_OMP_TARGETS OFF)
endif()

if( CUDA_FOUND )
    set(BUILD_CUDA_TARGETS ON)
else()
    set(BUILD_CUDA_TARGETS OFF)
endif()

if(TBB_FOUND)
    set(BUILD_TBB_TARGETS ON)
else()
    set(BUILD_TBB_TARGETS OFF)
endif(TBB_FOUND)

# If this is not included as a subproject
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

    # Messages
    message(STATUS "TCLAP Include path: ${TCLAP_INCLUDE_DIR}")
    message(STATUS "ROOT include path: ${ROOT_INCLUDE_DIR}")
    message(STATUS "ROOT library path: ${ROOT_LIBRARY_DIR}" )
    message(STATUS "ROOT libraries: ${ROOT_LIBRARIES}")
    message(STATUS "Build CUDA/NVCC-based targets: ${BUILD_CUDA_TARGETS}")
    message(STATUS "CUDA include: ${CUDA_INCLUDE_DIRS}" )
    message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}"  )
    message(STATUS "NVCC flags: ${CUDA_NVCC_FLAGS}" )
    message(STATUS "Build TBB targets: ${BUILD_TBB_TARGETS}")
    message(STATUS "TBB include: ${TBB_INCLUDE_DIRS}" )
    message(STATUS "TBB libraries: ${TBB_LIBRARIES}" )
    message(STATUS "Build OpenMP targets: ${BUILD_OMP_TARGETS}")
    message(STATUS "OpenMP flags: ${OpenMP_CXX_FLAGS}" )
    message(STATUS "OpenMP libraries:  ${OpenMP_CXX_LIBRARIES}" )
    message(STATUS "Host compiler flags: ${CMAKE_CXX_FLAGS}" )
    message(STATUS "Install diretory: ${CMAKE_INSTALL_PREFIX}"  )
    message(STATUS "Project source dir: ${PROJECT_SOURCE_DIR}" )
    message(STATUS "Project build dir: ${PROJECT_BINARY_DIR}" )

    # This is just the default setting, it can be turned
    # on and off by the user using -DHYDRA_EXAMPLES=ON, etc.
    option(HYDRA_EXAMPLES "Build the Hydra examples" ON)
# If this is a subproject of another project	
else()
    option(HYDRA_EXAMPLES "Build the Hydra examples" OFF)
endif()

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/plots)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/plots/dalitz)


#+++++++++++++++++++++++++++
#        EXAMPLES          +
#+++++++++++++++++++++++++++

add_custom_target(examples)
include(${Hydra_CMAKE_DIR}/AddHydraExample.cmake)

if(HYDRA_EXAMPLES)

    if(NOT TCLAP_FOUND)
        message(FATAL_ERROR "TCLAP is needed to build the examples.")
    endif()

    add_subdirectory(examples/phase_space)
    add_subdirectory(examples/numerical_integration)
    add_subdirectory(examples/random)
    add_subdirectory(examples/fit)
    add_subdirectory(examples/histograming)
    add_subdirectory(examples/async)
    add_subdirectory(examples/misc)
    add_subdirectory(examples/phys)
endif()

install(DIRECTORY Hydra
  DESTINATION include)

install(DIRECTORY src
        DESTINATION src/hydra/samples)
  
install(DIRECTORY cmake
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
        message(FATAL_ERROR "Doxygen is needed to build the documentation.")
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
set(CPACK_SOURCE_IGNORE_FILES  build/*)
set(CPACK_RESOURCE_FILE_LICENSE  
     "${CMAKE_CURRENT_SOURCE_DIR}/COPYING")
set(CPACK_RESOURCE_FILE_README  
     "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
set(CPACK_PACKAGE_VERSION_MAJOR "${Hydra_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${Hydra_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${Hydra_VERSION_PATCH}")
include(CPack)
