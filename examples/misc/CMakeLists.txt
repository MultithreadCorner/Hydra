project(examples)

#+++++++++++++++++++++++++++++++++
# Hydra variant                  |
#+++++++++++++++++++++++++++++++++
                                         
#+++++++++++++++++++++++++
# CUDA TARGETS           |
#+++++++++++++++++++++++++
if(BUILD_CUDA_TARGETS)

         #+++++++++++++++++++++++++++++++++
         cuda_add_executable( multivector_container_cuda EXCLUDE_FROM_ALL  multivector_container.cu    
            OPTIONS -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA  -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -lgomp
            )
         target_link_libraries(multivector_container_cuda  ${ROOT_LIBRARIES} )
        
         add_dependencies(examples multivector_container_cuda)
        
         #+++++++++++++++++++++++++++++++++
         cuda_add_executable( multiarray_container_cuda EXCLUDE_FROM_ALL  multiarray_container.cu    
            OPTIONS -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA  -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -lgomp
            )
         target_link_libraries(multiarray_container_cuda  ${ROOT_LIBRARIES} )
        
         add_dependencies(examples multiarray_container_cuda)
        
                        
endif(BUILD_CUDA_TARGETS)

#+++++++++++++++++++++++++
# TBB TARGETS           |
#+++++++++++++++++++++++++
if(BUILD_TBB_TARGETS)

       
    #+++++++++++++++++++++++++++++++
    add_executable( variant_types_tbb EXCLUDE_FROM_ALL
     variant_types.cpp
    )
    
    set_target_properties( variant_types_tbb PROPERTIES 
    COMPILE_FLAGS "-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB")
    
    target_link_libraries( variant_types_tbb tbb  ${ROOT_LIBRARIES})
   
         add_dependencies(examples variant_types_tbb)
        
    #+++++++++++++++++++++++++++++++
    add_executable( multivector_container_tbb EXCLUDE_FROM_ALL
     multivector_container.cpp
    )
    
    set_target_properties( multivector_container_tbb PROPERTIES 
    COMPILE_FLAGS "-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB")
    
    target_link_libraries( multivector_container_tbb tbb  ${ROOT_LIBRARIES})
   
    add_dependencies(examples multivector_container_tbb) 
             
    #+++++++++++++++++++++++++++++++
    add_executable( multiarray_container_tbb EXCLUDE_FROM_ALL
     multiarray_container.cpp
    )
    
    set_target_properties( multiarray_container_tbb PROPERTIES 
    COMPILE_FLAGS "-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB")
    
    target_link_libraries( multiarray_container_tbb tbb  ${ROOT_LIBRARIES})
   
    add_dependencies(examples multiarray_container_tbb) 
          
endif(BUILD_TBB_TARGETS)


#+++++++++++++++++++++++++
# OMP TARGETS            |
#+++++++++++++++++++++++++
IF(BUILD_OMP_TARGETS)

    #+++++++++++++++++++++++++++++++
    add_executable(variant_types_omp EXCLUDE_FROM_ALL
        variant_types.cpp
    )
    
    set_target_properties( variant_types_omp PROPERTIES 
    COMPILE_FLAGS "-fopenmp -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp")
    
    target_link_libraries( variant_types_omp   ${ROOT_LIBRARIES})

    add_dependencies(examples variant_types_omp)
        
 
    #+++++++++++++++++++++++++++++++
    add_executable(multivector_container_omp EXCLUDE_FROM_ALL
        multivector_container.cpp
    )
    
    set_target_properties( multivector_container_omp PROPERTIES 
    COMPILE_FLAGS "-fopenmp -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp")
    
    target_link_libraries( multivector_container_omp   ${ROOT_LIBRARIES})
    
    add_dependencies(examples multivector_container_omp)
           
    #+++++++++++++++++++++++++++++++
    add_executable(multiarray_container_omp EXCLUDE_FROM_ALL
        multiarray_container.cpp
    )
    
    set_target_properties( multiarray_container_omp PROPERTIES 
    COMPILE_FLAGS "-fopenmp -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp")
    
    target_link_libraries( multiarray_container_omp   ${ROOT_LIBRARIES})
    
    add_dependencies(examples multiarray_container_omp)
           
endif(BUILD_OMP_TARGETS)



#+++++++++++++++++++++++++
# CPP TARGETS            |
#+++++++++++++++++++++++++
IF(BUILD_CPP_TARGETS)

    #+++++++++++++++++++++++++++++++
    add_executable(variant_types_cpp EXCLUDE_FROM_ALL
        variant_types.cpp
    )
    
    set_target_properties( variant_types_cpp PROPERTIES 
    COMPILE_FLAGS " -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP ")
    
    target_link_libraries( variant_types_cpp   ${ROOT_LIBRARIES})

    add_dependencies(examples variant_types_cpp)
        
 
    #+++++++++++++++++++++++++++++++
    add_executable(multivector_container_cpp EXCLUDE_FROM_ALL
        multivector_container.cpp
    )
    
    set_target_properties( multivector_container_cpp PROPERTIES 
    COMPILE_FLAGS " -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP ")
    
    target_link_libraries( multivector_container_cpp   ${ROOT_LIBRARIES})
    
    add_dependencies(examples multivector_container_cpp)
           
    #+++++++++++++++++++++++++++++++
    add_executable(multiarray_container_cpp EXCLUDE_FROM_ALL
        multiarray_container.cpp
    )
    
    set_target_properties( multiarray_container_cpp PROPERTIES 
    COMPILE_FLAGS " -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP")
    
    target_link_libraries( multiarray_container_cpp   ${ROOT_LIBRARIES})
    
    add_dependencies(examples multiarray_container_cpp)
           
endif(BUILD_CPP_TARGETS)

