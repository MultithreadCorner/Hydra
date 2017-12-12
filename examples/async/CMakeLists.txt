project(examples)

#+++++++++++++++++++++++++++++++
# Hydra histograming           |
#+++++++++++++++++++++++++++++++
                                         
#+++++++++++++++++++++++++++
# CUDA + TBB + OMP TARGETS |
#+++++++++++++++++++++++++++
if(BUILD_CUDA_TARGETS AND BUILD_TBB_TARGETS AND BUILD_OMP_TARGETS )

         #+++++++++++++++++++++++++++++++++
         cuda_add_executable( asynchronous_monte_carlo EXCLUDE_FROM_ALL  async_mc.cu    
            OPTIONS -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA  -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -lgomp )
            
         target_link_libraries( asynchronous_monte_carlo  ${ROOT_LIBRARIES} tbb )
         
         add_dependencies(examples asynchronous_monte_carlo )
                           
endif(BUILD_CUDA_TARGETS AND BUILD_TBB_TARGETS AND BUILD_OMP_TARGETS )
