project(integration)

#+++++++++++++++++++++++++++++++
# Hydra numerical integration  |
#+++++++++++++++++++++++++++++++
                                         
#+++++++++++++++++++++++++
# CUDA TARGETS           |
#+++++++++++++++++++++++++
if(BUILD_CUDA_TARGETS)

         #+++++++++++++++++++++++++++++++++
         cuda_add_executable( vegas_cuda EXCLUDE_FROM_ALL  vegas.cu    
            OPTIONS -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA  -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -lgomp
            )
         target_link_libraries(vegas_cuda  )
         
          add_dependencies(examples vegas_cuda)
         
         #+++++++++++++++++++++++++++++++++
         cuda_add_executable( plain_mc_cuda EXCLUDE_FROM_ALL  plain_mc.cu    
            OPTIONS -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA  -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -lgomp
            )
         target_link_libraries(plain_mc_cuda  )
         
          add_dependencies(examples plain_mc_cuda)
         
         #+++++++++++++++++++++++++++++++++
         cuda_add_executable( gauss_kronrod_cuda EXCLUDE_FROM_ALL  gauss_kronrod.cu    
            OPTIONS -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA  -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -lgomp
            )
         target_link_libraries( gauss_kronrod_cuda  )
         
          add_dependencies(examples gauss_kronrod_cuda)
         
         #+++++++++++++++++++++++++++++++++
         cuda_add_executable( adaptive_gauss_kronrod_cuda EXCLUDE_FROM_ALL  adaptive_gauss_kronrod.cu    
            OPTIONS -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA  -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -lgomp
            )
         target_link_libraries( adaptive_gauss_kronrod_cuda  )
         
          add_dependencies(examples adaptive_gauss_kronrod_cuda)
         
         
                        
endif(BUILD_CUDA_TARGETS)

#+++++++++++++++++++++++++
# TBB TARGETS           |
#+++++++++++++++++++++++++
if(BUILD_TBB_TARGETS)

    #+++++++++++++++++++++++++++++++
    add_executable( vegas_tbb EXCLUDE_FROM_ALL
     vegas.cpp
    )
    
    set_target_properties( vegas_tbb PROPERTIES 
    COMPILE_FLAGS "-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB")
    
    target_link_libraries( vegas_tbb tbb )
   
          add_dependencies(examples  vegas_tbb tbb)
         
    #+++++++++++++++++++++++++++++++
    add_executable( plain_mc_tbb EXCLUDE_FROM_ALL
     plain_mc.cpp
    )
    
    set_target_properties( plain_mc_tbb PROPERTIES 
    COMPILE_FLAGS "-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB")
    
    target_link_libraries( plain_mc_tbb tbb )
   
          add_dependencies(examples  plain_mc_tbb tbb)
         
    #+++++++++++++++++++++++++++++++
    add_executable( gauss_kronrod_tbb EXCLUDE_FROM_ALL
     gauss_kronrod.cpp
    )
    
    set_target_properties( gauss_kronrod_tbb PROPERTIES 
    COMPILE_FLAGS "-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB")
    
    target_link_libraries( gauss_kronrod_tbb tbb )
     
          add_dependencies(examples  gauss_kronrod_tbb)
         
    #+++++++++++++++++++++++++++++++
    add_executable( adaptive_gauss_kronrod_tbb EXCLUDE_FROM_ALL
     adaptive_gauss_kronrod.cpp
    )
    
    set_target_properties( adaptive_gauss_kronrod_tbb PROPERTIES 
    COMPILE_FLAGS "-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB")
    
    target_link_libraries( adaptive_gauss_kronrod_tbb tbb )
     
          add_dependencies(examples  adaptive_gauss_kronrod_tbb)
           
endif(BUILD_TBB_TARGETS)


#+++++++++++++++++++++++++
# OMP TARGETS            |
#+++++++++++++++++++++++++
IF(OPENMP_FOUND)

    #+++++++++++++++++++++++++++++++
    add_executable(vegas_omp EXCLUDE_FROM_ALL
    vegas.cpp
    )
    
    set_target_properties( vegas_omp PROPERTIES 
    COMPILE_FLAGS "-fopenmp -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp")
    
    target_link_libraries( vegas_omp  )

          add_dependencies(examples vegas_omp)
         
    #+++++++++++++++++++++++++++++++
    add_executable(plain_mc_omp EXCLUDE_FROM_ALL
    plain_mc.cpp
    )
    
    set_target_properties( plain_mc_omp PROPERTIES 
    COMPILE_FLAGS "-fopenmp -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp")
    
    target_link_libraries( plain_mc_omp  )
    
          add_dependencies(examples  plain_mc_omp)
         
    #+++++++++++++++++++++++++++++++
    add_executable(gauss_kronrod_omp EXCLUDE_FROM_ALL
    gauss_kronrod.cpp
    )
    
    set_target_properties( gauss_kronrod_omp PROPERTIES 
    COMPILE_FLAGS "-fopenmp -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp")
    
    target_link_libraries( gauss_kronrod_omp  )
    
    
          add_dependencies(examples  gauss_kronrod_omp)
         
    #+++++++++++++++++++++++++++++++
    add_executable(adaptive_gauss_kronrod_omp EXCLUDE_FROM_ALL
    adaptive_gauss_kronrod.cpp
    )
    
    set_target_properties( adaptive_gauss_kronrod_omp PROPERTIES 
    COMPILE_FLAGS "-fopenmp -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp")
    
    target_link_libraries( adaptive_gauss_kronrod_omp  )
    
          add_dependencies(examples adaptive_gauss_kronrod_omp)
         
endif(OPENMP_FOUND)

