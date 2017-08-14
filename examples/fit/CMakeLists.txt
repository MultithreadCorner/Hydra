project(examples)

#+++++++++++++++++++++++++++++++++
# Hydra random number generation |
#+++++++++++++++++++++++++++++++++
                                         
#+++++++++++++++++++++++++
# CUDA TARGETS           |
#+++++++++++++++++++++++++
if(BUILD_CUDA_TARGETS)

         #+++++++++++++++++++++++++++++++++
         cuda_add_executable( basic_fit_cuda EXCLUDE_FROM_ALL  basic_fit.cu    
            OPTIONS -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA  -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -lgomp
            )
         target_link_libraries(basic_fit_cuda  ${ROOT_LIBRARIES} )
                        
endif(BUILD_CUDA_TARGETS)

#+++++++++++++++++++++++++
# TBB TARGETS           |
#+++++++++++++++++++++++++
if(BUILD_TBB_TARGETS)

    #+++++++++++++++++++++++++++++++
    add_executable( basic_fit_tbb EXCLUDE_FROM_ALL
     basic_fit.cpp
    )
    
    set_target_properties( basic_fit_tbb PROPERTIES 
    COMPILE_FLAGS "-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB")
    
    target_link_libraries( basic_fit_tbb tbb  ${ROOT_LIBRARIES})
   
    
endif(BUILD_TBB_TARGETS)


#+++++++++++++++++++++++++
# OMP TARGETS            |
#+++++++++++++++++++++++++
IF(OPENMP_FOUND)

    #+++++++++++++++++++++++++++++++
    add_executable(basic_fit_omp EXCLUDE_FROM_ALL
   basic_fit.cpp
    )
    
    set_target_properties( basic_fit_omp PROPERTIES 
    COMPILE_FLAGS "-fopenmp -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp")
    
    target_link_libraries( basic_fit_omp   ${ROOT_LIBRARIES})
 
    
    
endif(OPENMP_FOUND)

