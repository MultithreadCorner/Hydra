project(examples)

#+++++++++++++++++++++++++++++++
# Hydra async                  |
#+++++++++++++++++++++++++++++++

#+++++++++++++++++++++++++++
# CUDA + TBB + OMP TARGETS |
#+++++++++++++++++++++++++++
if(BUILD_CUDA_TARGETS AND BUILD_TBB_TARGETS AND BUILD_OMP_TARGETS )

         #+++++++++++++++++++++++++++++++++
         add_executable( asynchronous_monte_carlo EXCLUDE_FROM_ALL  async_mc.cu
            OPTIONS -Xcompiler -fopenmp -DHYDRA_DEVICE_SYSTEM=CUDA  -DHYDRA_HOST_SYSTEM=CPP -lgomp )

         target_link_libraries( asynchronous_monte_carlo  ${ROOT_LIBRARIES} tbb )

         add_dependencies(examples asynchronous_monte_carlo )

endif(BUILD_CUDA_TARGETS AND BUILD_TBB_TARGETS AND BUILD_OMP_TARGETS )
