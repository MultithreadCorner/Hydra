project(examples)

#+++++++++++++++++++++++++++++++++
# Hydra random number generation |
#+++++++++++++++++++++++++++++++++
                                         
#+++++++++++++++++++++++++
# CUDA TARGETS           |
#+++++++++++++++++++++++++
if(BUILD_CUDA_TARGETS)

         #+++++++++++++++++++++++++++++++++
         cuda_add_executable( gaussian_plus_argus_cuda EXCLUDE_FROM_ALL  gaussian_plus_argus.cu    
            OPTIONS -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA  -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -lgomp
            )
         target_link_libraries(gaussian_plus_argus_cuda  ${ROOT_LIBRARIES} )
        
         add_dependencies(examples gaussian_plus_argus_cuda)
        
        
         #+++++++++++++++++++++++++++++++++
         cuda_add_executable( crystal_ball_plus_exponential_cuda EXCLUDE_FROM_ALL  crystal_ball_plus_exponential.cu    
            OPTIONS -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA  -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -lgomp
            )
         target_link_libraries(crystal_ball_plus_exponential_cuda  ${ROOT_LIBRARIES} )
        
         add_dependencies(examples crystal_ball_plus_exponential_cuda)
        
           
         #+++++++++++++++++++++++++++++++++
         cuda_add_executable( double_gaussian_plus_exponential_cuda EXCLUDE_FROM_ALL  double_gaussian_plus_exponential.cu    
            OPTIONS -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA  -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -lgomp
            )
         target_link_libraries(double_gaussian_plus_exponential_cuda  ${ROOT_LIBRARIES} )
        
         add_dependencies(examples double_gaussian_plus_exponential_cuda)
        
        
           
         #+++++++++++++++++++++++++++++++++
         cuda_add_executable( breit_wigner_plus_polynomial_cuda EXCLUDE_FROM_ALL breit_wigner_plus_polynomial.cu    
            OPTIONS -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA  -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -lgomp
            )
         target_link_libraries(breit_wigner_plus_polynomial_cuda  ${ROOT_LIBRARIES} )
        
         add_dependencies(examples breit_wigner_plus_polynomial_cuda)
        
        
          #+++++++++++++++++++++++++++++++++
         cuda_add_executable( dalitz_plot_cuda EXCLUDE_FROM_ALL dalitz_plot.cu    
            OPTIONS -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA  -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP -lgomp
            )
         target_link_libraries(dalitz_plot_cuda  ${ROOT_LIBRARIES} )
        
         add_dependencies(examples dalitz_plot_cuda)
        
        
                        
endif(BUILD_CUDA_TARGETS)

#+++++++++++++++++++++++++
# TBB TARGETS           |
#+++++++++++++++++++++++++
if(BUILD_TBB_TARGETS)

    #+++++++++++++++++++++++++++++++
    add_executable( gaussian_plus_argus_tbb EXCLUDE_FROM_ALL
     gaussian_plus_argus.cpp
    )
    
    set_target_properties( gaussian_plus_argus_tbb PROPERTIES 
    COMPILE_FLAGS "-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB")
    
    target_link_libraries( gaussian_plus_argus_tbb tbb  ${ROOT_LIBRARIES})
   
         add_dependencies(examples gaussian_plus_argus_tbb)
        
   
    #+++++++++++++++++++++++++++++++
    add_executable( crystal_ball_plus_exponential_tbb EXCLUDE_FROM_ALL
     crystal_ball_plus_exponential.cpp
    )
    
    set_target_properties( crystal_ball_plus_exponential_tbb PROPERTIES 
    COMPILE_FLAGS "-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB")
    
    target_link_libraries( crystal_ball_plus_exponential_tbb tbb  ${ROOT_LIBRARIES})
   
         add_dependencies(examples crystal_ball_plus_exponential_tbb)
        
        
   
    #+++++++++++++++++++++++++++++++
    add_executable( double_gaussian_plus_exponential_tbb EXCLUDE_FROM_ALL
     double_gaussian_plus_exponential.cpp
    )
    
    set_target_properties( double_gaussian_plus_exponential_tbb PROPERTIES 
    COMPILE_FLAGS "-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB")
    
    target_link_libraries( double_gaussian_plus_exponential_tbb tbb  ${ROOT_LIBRARIES})
   
         add_dependencies(examples double_gaussian_plus_exponential_tbb)
             
   
    #+++++++++++++++++++++++++++++++
    add_executable( breit_wigner_plus_polynomial_tbb EXCLUDE_FROM_ALL
     breit_wigner_plus_polynomial.cpp
    )
    
    set_target_properties( breit_wigner_plus_polynomial_tbb PROPERTIES 
    COMPILE_FLAGS "-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB")
    
    target_link_libraries( breit_wigner_plus_polynomial_tbb tbb  ${ROOT_LIBRARIES})
        add_dependencies(examples breit_wigner_plus_polynomial_tbb)
        
           
    
    #+++++++++++++++++++++++++++++++
    add_executable(dalitz_plot_tbb EXCLUDE_FROM_ALL
    dalitz_plot.cpp
    )
    
    set_target_properties( dalitz_plot_tbb PROPERTIES 
    COMPILE_FLAGS "-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB" )
    
    target_link_libraries(dalitz_plot_tbb tbb ${ROOT_LIBRARIES})

    add_dependencies(examples dalitz_plot_tbb)
     
    
endif(BUILD_TBB_TARGETS)


#+++++++++++++++++++++++++
# OMP TARGETS            |
#+++++++++++++++++++++++++
IF(BUILD_OMP_TARGETS)

    #+++++++++++++++++++++++++++++++
    add_executable(gaussian_plus_argus_omp EXCLUDE_FROM_ALL
   gaussian_plus_argus.cpp
    )
    
    set_target_properties( gaussian_plus_argus_omp PROPERTIES 
    COMPILE_FLAGS "-fopenmp -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp")
    
    target_link_libraries( gaussian_plus_argus_omp   ${ROOT_LIBRARIES})

         add_dependencies(examples gaussian_plus_argus_omp)
        
    
    #+++++++++++++++++++++++++++++++
    add_executable(crystal_ball_plus_exponential_omp EXCLUDE_FROM_ALL
   crystal_ball_plus_exponential.cpp
    )
    
    set_target_properties( crystal_ball_plus_exponential_omp PROPERTIES 
    COMPILE_FLAGS "-fopenmp -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp")
    
    target_link_libraries( crystal_ball_plus_exponential_omp   ${ROOT_LIBRARIES})

         add_dependencies(examples crystal_ball_plus_exponential_omp)
        
         
    
    #+++++++++++++++++++++++++++++++
    add_executable(double_gaussian_plus_exponential_omp EXCLUDE_FROM_ALL
   double_gaussian_plus_exponential.cpp
    )
    
    set_target_properties( double_gaussian_plus_exponential_omp PROPERTIES 
    COMPILE_FLAGS "-fopenmp -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp")
    
    target_link_libraries( double_gaussian_plus_exponential_omp   ${ROOT_LIBRARIES})

         add_dependencies(examples double_gaussian_plus_exponential_omp)
        
    
          
    
    #+++++++++++++++++++++++++++++++
    add_executable(breit_wigner_plus_polynomial_omp EXCLUDE_FROM_ALL
   breit_wigner_plus_polynomial.cpp
    )
    
    set_target_properties( breit_wigner_plus_polynomial_omp PROPERTIES 
    COMPILE_FLAGS "-fopenmp -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp")
    
    target_link_libraries(breit_wigner_plus_polynomial_omp   ${ROOT_LIBRARIES})

     add_dependencies(examples breit_wigner_plus_polynomial_omp)
        
            
    
    #+++++++++++++++++++++++++++++++
    add_executable(dalitz_plot_omp EXCLUDE_FROM_ALL
    dalitz_plot.cpp
    )
    
    set_target_properties( dalitz_plot_omp PROPERTIES 
    COMPILE_FLAGS "-fopenmp -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp")
    
    target_link_libraries(dalitz_plot_omp   ${ROOT_LIBRARIES})

     add_dependencies(examples dalitz_plot_omp)
        
    
    
endif(BUILD_OMP_TARGETS)

#+++++++++++++++++++++++++
# CPP TARGETS            |
#+++++++++++++++++++++++++
IF(BUILD_CPP_TARGETS)

    #+++++++++++++++++++++++++++++++
    add_executable(gaussian_plus_argus_cpp EXCLUDE_FROM_ALL
   gaussian_plus_argus.cpp
    )
    
    set_target_properties( gaussian_plus_argus_cpp PROPERTIES 
    COMPILE_FLAGS "-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP")
    
    target_link_libraries( gaussian_plus_argus_cpp   ${ROOT_LIBRARIES})

         add_dependencies(examples gaussian_plus_argus_cpp)
        
    
    #+++++++++++++++++++++++++++++++
    add_executable(crystal_ball_plus_exponential_cpp EXCLUDE_FROM_ALL
   crystal_ball_plus_exponential.cpp
    )
    
    set_target_properties( crystal_ball_plus_exponential_cpp PROPERTIES 
    COMPILE_FLAGS "-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP")
    
    target_link_libraries( crystal_ball_plus_exponential_cpp   ${ROOT_LIBRARIES})

         add_dependencies(examples crystal_ball_plus_exponential_cpp)
        
    
         
    
    #+++++++++++++++++++++++++++++++
    add_executable(double_gaussian_plus_exponential_cpp EXCLUDE_FROM_ALL
   double_gaussian_plus_exponential.cpp
    )
    
    set_target_properties( double_gaussian_plus_exponential_cpp PROPERTIES 
    COMPILE_FLAGS "-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP")
    
    target_link_libraries( double_gaussian_plus_exponential_cpp   ${ROOT_LIBRARIES})

         add_dependencies(examples double_gaussian_plus_exponential_cpp)
        
          
    
    #+++++++++++++++++++++++++++++++
    add_executable(breit_wigner_plus_polynomial_cpp EXCLUDE_FROM_ALL
    breit_wigner_plus_polynomial.cpp
    )
    
    set_target_properties( breit_wigner_plus_polynomial_cpp PROPERTIES 
    COMPILE_FLAGS "-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP")
    
    target_link_libraries(breit_wigner_plus_polynomial_cpp ${ROOT_LIBRARIES})

    add_dependencies(examples breit_wigner_plus_polynomial_cpp)
        
            
    
    #+++++++++++++++++++++++++++++++
    add_executable(dalitz_plot_cpp EXCLUDE_FROM_ALL
    dalitz_plot.cpp
    )
    
    set_target_properties( dalitz_plot_cpp PROPERTIES 
    COMPILE_FLAGS "-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP")
    
    target_link_libraries(dalitz_plot_cpp   ${ROOT_LIBRARIES})

    add_dependencies(examples dalitz_plot_cpp)
        
    
endif(BUILD_CPP_TARGETS)