### CMakeList for MC Hydra

cmake_minimum_required(VERSION 2.8)

# project name
project(Hydra)

# The version number.
set (Hydra_VERSION_MAJOR 1)
set (Hydra_VERSION_MINOR 0)
set (Hydra_VERSION_PATCH 0)

# warn user if system is not UNIX
if(NOT UNIX)
  message(FATAL_ERROR "This is an unsupported system.")
endif()

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/hydra/Hydra.h.in"
  "${PROJECT_SOURCE_DIR}/hydra/Hydra.h"
 )
mark_as_advanced(Hydra_VERSION_MAJOR Hydra_VERSION_MINOR Hydra_VERSION_PATCH)

SET(Hydra_CMAKE_DIR "${PROJECT_SOURCE_DIR}/cmake")
SET(CMAKE_MODULE_PATH "${Hydra_CMAKE_DIR}" ${CMAKE_MODULE_PATH})
SET(CMAKE_VERBOSE_MAKEFILE  ON)

# get TCLAP
find_package(TCLAP REQUIRED)
include_directories(${TCLAP_INCLUDE_DIR})

# get THRUST
find_package(Thrust 1.8 REQUIRED)
include_directories(./thrust)

# get ROOT
find_package(ROOT REQUIRED)
include_directories(${ROOT_INCLUDE_DIR})
link_directories(${ROOT_LIBRARY_DIR})

#cuda
find_package(CUDA 6.5) 

#generate API documentation with Doxygen
find_package(Doxygen)
option(BUILD_DOCUMENTATION "Create and install the HTML based API documentation (requires Doxygen)" NO)

#options to gcc
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("--std=c++11" COMPILER_SUPPORTS_CXX11)
if(NOT COMPILER_SUPPORTS_CXX11)
 message(FATAL "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

#options to gcc
set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 -DTHRUST_VARIADIC_TUPLE --std=c++11  -march=native -fPIC -O4 -fopenmp -lgomp" CACHE STRING "compile flags" FORCE)

#including directories

include_directories(include;  ${PROJECT_SOURCE_DIR})



if(CUDA_FOUND)

	#set cuda flags
	
	SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --cudart ;  static ; -O4 ;--expt-relaxed-constexpr; -ftemplate-backtrace-limit=0; --expt-extended-lambda;--relocatable-device-code=false ;
	 -Xptxas -fmad=true ;-Xptxas -dlcm=cg; -Xptxas --opt-level=4   ) #) 
# 
	SET(CUDA_SEPARABLE_COMPILATION OFF)
	SET(CUDA_VERBOSE_BUILD ON)
	
	# Detect CUDA architecture and get best NVCC flags
	
	INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindCudaArch.cmake)
	
	SELECT_NVCC_ARCH_FLAGS(NVCC_FLAGS_EXTRA)
	
	LIST(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})
	#hack for gcc 5.x.x
	if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9)
	    LIST(APPEND CUDA_NVCC_FLAGS " -D_MWAITXINTRIN_H_INCLUDED ")
    	endif()
endif(CUDA_FOUND)

if( CUDA_FOUND )

	SET(BUILD_CUDA_TARGETS  ON)

else()

	SET(BUILD_CUDA_TARGETS  OFF)

endif()

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
MESSAGE(STATUS "nvcc cflags: " ${CUDA_NVCC_FLAGS} )
MESSAGE(STATUS "gcc cflags: " ${CMAKE_CXX_FLAGS} )
MESSAGE(STATUS "Include directories: " ${INCLUDE_DIRECTORIES} )
MESSAGE(STATUS "Install diretory: " ${CMAKE_INSTALL_PREFIX}  )

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/plots)

if(BUILD_CUDA_TARGETS)

#make cuda targets
#--------------------
#phsp generation
 
    cuda_add_executable(Hydra_Example_NVCC_DEVICE_CUDA_HOST_OMP_PHSP
    ${CMAKE_CURRENT_SOURCE_DIR}/src/HydraPhaseSpaceExample.cu
    OPTIONS -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA  -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP -lgomp
    )
    
    target_link_libraries(Hydra_Example_NVCC_DEVICE_CUDA_HOST_OMP_PHSP
    ${ROOT_LIBRARIES}
    rt    
    )

#random number generation
 
    cuda_add_executable(Hydra_Example_NVCC_DEVICE_CUDA_HOST_OMP_Random
    ${CMAKE_CURRENT_SOURCE_DIR}/src/HydraRandomExample.cu
    OPTIONS -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA  -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP -lgomp
    )
    
    target_link_libraries(Hydra_Example_NVCC_DEVICE_CUDA_HOST_OMP_Random
    ${ROOT_LIBRARIES}
    rt    
    )
    
#--------------
#Data fitting

    cuda_add_executable(Hydra_Example_NVCC_DEVICE_CUDA_HOST_OMP_Fit
    ${CMAKE_CURRENT_SOURCE_DIR}/src/HydraFitExample.cu
    OPTIONS -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA  -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP -lgomp
    )
    
    target_link_libraries(Hydra_Example_NVCC_DEVICE_CUDA_HOST_OMP_Fit
    ${ROOT_LIBRARIES}
    rt
    )

#--------------
#Lambda function evaluation
	
    cuda_add_executable(Hydra_Example_NVCC_DEVICE_CUDA_HOST_OMP_Eval
    ${CMAKE_CURRENT_SOURCE_DIR}/src/HydraEvaluateExample.cu
    OPTIONS -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP  -lgomp
    )
    
    target_link_libraries(Hydra_Example_NVCC_DEVICE_CUDA_HOST_OMP_Eval
    ${ROOT_LIBRARIES}
    rt
    )
    
	
#--------------
#testing
	
    cuda_add_executable(Hydra_Multivector_Test_CUDA
    ${CMAKE_CURRENT_SOURCE_DIR}/src/multivector_testing.cu
    OPTIONS -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP  -lgomp
    )
    
    target_link_libraries(Hydra_Multivector_Test_CUDA
  rt
    )    
	
#--------------
#performance
	
    cuda_add_executable(Hydra_Multivector_Perf_CUDA
    ${CMAKE_CURRENT_SOURCE_DIR}/src/multivector_performance.cu
    OPTIONS -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP  -lgomp
    )
    
    target_link_libraries(Hydra_Multivector_Perf_CUDA
    rt
    )    
	
#--------------
#performance
	
    cuda_add_executable(Hydra_Multivector_CUDA
    ${CMAKE_CURRENT_SOURCE_DIR}/src/multivector.cu
    OPTIONS -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP  -lgomp
    )
    
    target_link_libraries(Hydra_Multivector_CUDA
    rt
    )    
	
	
endif(BUILD_CUDA_TARGETS)


#--------------
#phsp  generation
add_executable(Hydra_Example_GCC_DEVICE_OMP_HOST_CPP_PHSP
${CMAKE_CURRENT_SOURCE_DIR}/src/HydraPhaseSpaceExample.cpp
)

set_target_properties(Hydra_Example_GCC_DEVICE_OMP_HOST_CPP_PHSP PROPERTIES 
COMPILE_FLAGS "-fopenmp -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp")

target_link_libraries(Hydra_Example_GCC_DEVICE_OMP_HOST_CPP_PHSP
${ROOT_LIBRARIES}
rt
)


#--------------
#random number generation
add_executable(Hydra_Example_GCC_DEVICE_OMP_HOST_CPP_Random
${CMAKE_CURRENT_SOURCE_DIR}/src/HydraRandomExample.cpp
)

set_target_properties(Hydra_Example_GCC_DEVICE_OMP_HOST_CPP_Random PROPERTIES 
COMPILE_FLAGS "-fopenmp -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp")

target_link_libraries(Hydra_Example_GCC_DEVICE_OMP_HOST_CPP_Random
${ROOT_LIBRARIES}
rt
)


#--------------
#Fit
add_executable(Hydra_Example_GCC_DEVICE_OMP_HOST_CPP_Fit
${CMAKE_CURRENT_SOURCE_DIR}/src/HydraFitExample.cpp
)

set_target_properties(Hydra_Example_GCC_DEVICE_OMP_HOST_CPP_Fit PROPERTIES 
COMPILE_FLAGS "-pg -fopenmp -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp")

target_link_libraries(Hydra_Example_GCC_DEVICE_OMP_HOST_CPP_Fit
${ROOT_LIBRARIES}
rt
)


#--------------
#Lambda function evaluation

add_executable(Hydra_Example_GCC_DEVICE_OMP_HOST_CPP_Eval
${CMAKE_CURRENT_SOURCE_DIR}/src/HydraEvaluateExample.cpp
)

set_target_properties(Hydra_Example_GCC_DEVICE_OMP_HOST_CPP_Eval PROPERTIES 
COMPILE_FLAGS  "-fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP -lgomp" )

target_link_libraries(Hydra_Example_GCC_DEVICE_OMP_HOST_CPP_Eval
${ROOT_LIBRARIES}
rt
)

add_executable(Hydra_Multivector
${CMAKE_CURRENT_SOURCE_DIR}/src/multivector.cpp
)

set_target_properties(Hydra_Multivector PROPERTIES 
COMPILE_FLAGS  "-fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP -lgomp" )

target_link_libraries(Hydra_Multivector
rt
)

add_executable(Hydra_Multivector_Test
${CMAKE_CURRENT_SOURCE_DIR}/src/multivector_testing.cpp
)

set_target_properties(Hydra_Multivector_Test PROPERTIES 
COMPILE_FLAGS  "-fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP -lgomp" )

target_link_libraries(Hydra_Multivector_Test
${ROOT_LIBRARIES}
rt
)

add_executable(Hydra_Multivector_Perf
${CMAKE_CURRENT_SOURCE_DIR}/src/multivector_performance.cpp
)

set_target_properties(Hydra_Multivector_Perf PROPERTIES 
COMPILE_FLAGS  "-pthread -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP -lgomp" )

target_link_libraries(Hydra_Multivector_Perf)


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



