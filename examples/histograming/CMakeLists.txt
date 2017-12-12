project(examples)

#+++++++++++++++++++++++++++++++
# Hydra histograming           |
#+++++++++++++++++++++++++++++++
                                         
#+++++++++++++++++++++++++
# CUDA TARGETS           |
#+++++++++++++++++++++++++
if(BUILD_CUDA_TARGETS)

         #+++++++++++++++++++++++++++++++++
         cuda_add_executable( 3d_dense_histogram_cuda EXCLUDE_FROM_ALL  dense_histogram.cu    
            OPTIONS -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA  -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -lgomp
            )
         target_link_libraries(3d_dense_histogram_cuda   ${ROOT_LIBRARIES} )
         
          add_dependencies(examples 3d_dense_histogram_cuda)
        
         
         #+++++++++++++++++++++++++++++++++
         cuda_add_executable( 3d_sparse_histogram_cuda EXCLUDE_FROM_ALL  sparse_histogram.cu    
            OPTIONS -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA  -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -lgomp
            )
         target_link_libraries(3d_sparse_histogram_cuda   ${ROOT_LIBRARIES} )
         
          add_dependencies(examples 3d_sparse_histogram_cuda)
        
         
                        
endif(BUILD_CUDA_TARGETS)

#+++++++++++++++++++++++++
# TBB TARGETS           |
#+++++++++++++++++++++++++
if(BUILD_TBB_TARGETS)

    #+++++++++++++++++++++++++++++++
    add_executable( 3d_dense_histogram_tbb EXCLUDE_FROM_ALL dense_histogram.cpp )
    
    set_target_properties( 3d_dense_histogram_tbb PROPERTIES 
    COMPILE_FLAGS "-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB")
    
    target_link_libraries( 3d_dense_histogram_tbb tbb   ${ROOT_LIBRARIES})
   
    add_dependencies(examples  3d_dense_histogram_tbb tbb)
         
   
    #+++++++++++++++++++++++++++++++
    add_executable( 3d_sparse_histogram_tbb EXCLUDE_FROM_ALL sparse_histogram.cpp  )
    
    set_target_properties( 3d_sparse_histogram_tbb PROPERTIES 
    COMPILE_FLAGS "-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB")
    
    target_link_libraries( 3d_sparse_histogram_tbb tbb   ${ROOT_LIBRARIES})
   
    add_dependencies(examples  3d_sparse_histogram_tbb tbb)
         
   
           
endif(BUILD_TBB_TARGETS)


#+++++++++++++++++++++++++
# OMP TARGETS            |
#+++++++++++++++++++++++++
IF(BUILD_OMP_TARGETS)

    #+++++++++++++++++++++++++++++++
    add_executable(3d_dense_histogram_omp EXCLUDE_FROM_ALL
    dense_histogram.cpp
    )
    
    set_target_properties( 3d_dense_histogram_omp PROPERTIES 
    COMPILE_FLAGS "-fopenmp -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp")
    
    target_link_libraries( 3d_dense_histogram_omp    ${ROOT_LIBRARIES})

    add_dependencies(examples 3d_dense_histogram_omp)
    
    #+++++++++++++++++++++++++++++++
    add_executable(3d_sparse_histogram_omp EXCLUDE_FROM_ALL sparse_histogram.cpp )
    
    set_target_properties( 3d_sparse_histogram_omp PROPERTIES 
    COMPILE_FLAGS "-fopenmp -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp")
    
    target_link_libraries( 3d_sparse_histogram_omp    ${ROOT_LIBRARIES})

    add_dependencies(examples 3d_sparse_histogram_omp)
         
    
endif(BUILD_OMP_TARGETS)



#+++++++++++++++++++++++++
# CPP TARGETS            |
#+++++++++++++++++++++++++
IF(BUILD_CPP_TARGETS)

    #+++++++++++++++++++++++++++++++
    add_executable(3d_dense_histogram_cpp EXCLUDE_FROM_ALL
    dense_histogram.cpp
    )
    
    set_target_properties( 3d_dense_histogram_cpp PROPERTIES 
    COMPILE_FLAGS "-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP ")
    
    target_link_libraries( 3d_dense_histogram_cpp    ${ROOT_LIBRARIES})

    add_dependencies(examples 3d_dense_histogram_cpp)
    
    #+++++++++++++++++++++++++++++++
    add_executable(3d_sparse_histogram_cpp EXCLUDE_FROM_ALL sparse_histogram.cpp )
    
    set_target_properties( 3d_sparse_histogram_cpp PROPERTIES 
    COMPILE_FLAGS "-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP ")
    
    target_link_libraries( 3d_sparse_histogram_cpp    ${ROOT_LIBRARIES})

    add_dependencies(examples 3d_sparse_histogram_cpp)
         
    
endif(BUILD_CPP_TARGETS)
