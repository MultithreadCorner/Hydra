project(examples)

#+++++++++++++++++++++++++++++++++
# Hydra random number generation |
#+++++++++++++++++++++++++++++++++
                                         
#+++++++++++++++++++++++++
# CUDA TARGETS           |
#+++++++++++++++++++++++++
if(BUILD_CUDA_TARGETS)

         #+++++++++++++++++++++++++++++++++
         cuda_add_executable( basic_distributions_cuda EXCLUDE_FROM_ALL  basic_distributions.cu    
            OPTIONS -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA  -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -lgomp
            )
         target_link_libraries(basic_distributions_cuda  ${ROOT_LIBRARIES} )
        
         add_dependencies(examples basic_distributions_cuda)
        
         #+++++++++++++++++++++++++++++++++
         cuda_add_executable( sample_distribution_cuda EXCLUDE_FROM_ALL  sample_distribution.cu    
            OPTIONS -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA  -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -lgomp
            )
         target_link_libraries(sample_distribution_cuda  ${ROOT_LIBRARIES} )
        
         add_dependencies(examples sample_distribution_cuda)
        
                        
endif(BUILD_CUDA_TARGETS)

#+++++++++++++++++++++++++
# TBB TARGETS           |
#+++++++++++++++++++++++++
if(BUILD_TBB_TARGETS)

    #+++++++++++++++++++++++++++++++
    add_executable( basic_distributions_tbb EXCLUDE_FROM_ALL
     basic_distributions.cpp
    )
    
    set_target_properties( basic_distributions_tbb PROPERTIES 
    COMPILE_FLAGS "-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB")
    
    target_link_libraries( basic_distributions_tbb tbb  ${ROOT_LIBRARIES})
   
         add_dependencies(examples basic_distributions_tbb)
        
    #+++++++++++++++++++++++++++++++
    add_executable( sample_distribution_tbb EXCLUDE_FROM_ALL
     sample_distribution.cpp
    )
    
    set_target_properties( sample_distribution_tbb PROPERTIES 
    COMPILE_FLAGS "-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB")
    
    target_link_libraries( sample_distribution_tbb tbb  ${ROOT_LIBRARIES})
   
         add_dependencies(examples sample_distribution_tbb)
        
    
endif(BUILD_TBB_TARGETS)


#+++++++++++++++++++++++++
# OMP TARGETS            |
#+++++++++++++++++++++++++
IF(OPENMP_FOUND)

    #+++++++++++++++++++++++++++++++
    add_executable(basic_distributions_omp EXCLUDE_FROM_ALL
   basic_distributions.cpp
    )
    
    set_target_properties( basic_distributions_omp PROPERTIES 
    COMPILE_FLAGS "-fopenmp -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp")
    
    target_link_libraries( basic_distributions_omp   ${ROOT_LIBRARIES})

         add_dependencies(examples basic_distributions_omp)
        
    
    #+++++++++++++++++++++++++++++++
    add_executable(sample_distribution_omp EXCLUDE_FROM_ALL
        sample_distribution.cpp
    )
    
    set_target_properties( sample_distribution_omp PROPERTIES 
    COMPILE_FLAGS "-fopenmp -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp")
    
    target_link_libraries( sample_distribution_omp   ${ROOT_LIBRARIES})

    
    
         add_dependencies(examples sample_distribution_omp)
        
    
endif(OPENMP_FOUND)

