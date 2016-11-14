project(examples)


#+++++++++++++++++++++++++
# CUDA TARGETS           |
#+++++++++++++++++++++++++

if(BUILD_CUDA_TARGETS)
 
    #+++++++++++++++++++++++++++++++
    # PHSP generation              |
    #+++++++++++++++++++++++++++++++
    cuda_add_executable(Hydra_Example_PHSP_Host_OMP_Device_CUDA EXCLUDE_FROM_ALL
    HydraPhaseSpaceExample.cu
    OPTIONS -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA  -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP -lgomp
    )
    
    target_link_libraries(Hydra_Example_PHSP_Host_OMP_Device_CUDA
    ${ROOT_LIBRARIES}
    rt    
    )
 
    #+++++++++++++++++++++++++++++++
    # Phsp generation              |
    #+++++++++++++++++++++++++++++++
    cuda_add_executable(Hydra_Example_Experimental_PHSP_CUDA EXCLUDE_FROM_ALL
    HydraPhaseSpaceExperimental.cu
    OPTIONS -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA  -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP -lgomp
    )
    
    target_link_libraries(Hydra_Example_Experimental_PHSP_CUDA
    ${ROOT_LIBRARIES}
    rt    
    )
 
    #+++++++++++++++++++++++++++++++
    # PDF sampling                 |
    #+++++++++++++++++++++++++++++++ 
    cuda_add_executable(Hydra_Example_Random_Host_OMP_Device_CUDA EXCLUDE_FROM_ALL
    HydraRandomExample.cu
    OPTIONS -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA  -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP -lgomp
    )
    
    target_link_libraries(Hydra_Example_Random_Host_OMP_Device_CUDA
    ${ROOT_LIBRARIES}
    rt    
    )
    
    #+++++++++++++++++++++++++++++++
    # Data fitting                 |
    #+++++++++++++++++++++++++++++++
    cuda_add_executable(Hydra_Example_Fit_Host_OMP_Device_CUDA EXCLUDE_FROM_ALL
    HydraFitExample.cu
    OPTIONS -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA  -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP -lgomp
    )
    
    target_link_libraries(Hydra_Example_Fit_Host_OMP_Device_CUDA
    ${ROOT_LIBRARIES}
    rt
    )

    #+++++++++++++++++++++++++++++++
    # Lambda function evaluation   |
    #+++++++++++++++++++++++++++++++	
    cuda_add_executable(Hydra_Example_Eval_Host_OMP_Device_CUDA EXCLUDE_FROM_ALL
    HydraEvaluateExample.cu
    OPTIONS -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP  -lgomp
    )
    
    target_link_libraries(Hydra_Example_Eval_Host_OMP_Device_CUDA
    ${ROOT_LIBRARIES}
    rt
    )
       
endif(BUILD_CUDA_TARGETS)

#+++++++++++++++++++++++++
# TBB TARGETS           |
#+++++++++++++++++++++++++
if(BUILD_TBB_TARGETS)

    #+++++++++++++++++++++++++++++++
    # PHSP generation              |
    #+++++++++++++++++++++++++++++++
    add_executable(Hydra_Example_Experimental_TBB_PHSP EXCLUDE_FROM_ALL
    HydraPhaseSpaceExperimental.cpp
    )
    
    set_target_properties(Hydra_Example_Experimental_TBB_PHSP PROPERTIES 
    COMPILE_FLAGS "-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB ")
    
    target_link_libraries(Hydra_Example_Experimental_TBB_PHSP
    ${ROOT_LIBRARIES}
    tbb
    )
    
            
    #+++++++++++++++++++++++++++++++
    # PHSP generation              |
    #+++++++++++++++++++++++++++++++
    add_executable(Hydra_Example_PHSP_Host_CPP_Device_TBB EXCLUDE_FROM_ALL
    HydraPhaseSpaceExample.cpp
    )
    
    set_target_properties(Hydra_Example_PHSP_Host_CPP_Device_TBB PROPERTIES 
    COMPILE_FLAGS "-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB")
    
    target_link_libraries(Hydra_Example_PHSP_Host_CPP_Device_TBB
    ${ROOT_LIBRARIES}
    rt
    tbb
    )
    
    
    #+++++++++++++++++++++++++++++++
    # PDF sampling                 |
    #+++++++++++++++++++++++++++++++     
    add_executable(Hydra_Example_Random_Host_CPP_Device_TBB EXCLUDE_FROM_ALL
    HydraRandomExample.cpp
    )
    
    set_target_properties(Hydra_Example_Random_Host_CPP_Device_TBB PROPERTIES 
    COMPILE_FLAGS "-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB")
    
    target_link_libraries(Hydra_Example_Random_Host_CPP_Device_TBB
    ${ROOT_LIBRARIES}
    rt
    tbb
    )
    
    
    #+++++++++++++++++++++++++++++++
    # Data fitting                 |
    #+++++++++++++++++++++++++++++++
    add_executable(Hydra_Example_Fit_Host_CPP_Device_TBB EXCLUDE_FROM_ALL
    HydraFitExample.cpp
    )
    
    set_target_properties(Hydra_Example_Fit_Host_CPP_Device_TBB PROPERTIES 
    COMPILE_FLAGS "-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB")
    
    target_link_libraries(Hydra_Example_Fit_Host_CPP_Device_TBB
    ${ROOT_LIBRARIES}
    rt
    tbb
    )
    
    
    
    #+++++++++++++++++++++++++++++++
    # Lambda function evaluation   |
    #+++++++++++++++++++++++++++++++	
    add_executable(Hydra_Example_Eval_Host_CPP_Device_TBB EXCLUDE_FROM_ALL
    HydraEvaluateExample.cpp
    )
    
    set_target_properties(Hydra_Example_Eval_Host_CPP_Device_TBB PROPERTIES 
    COMPILE_FLAGS  "-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_TBB" )
    
    target_link_libraries(Hydra_Example_Eval_Host_CPP_Device_TBB
    ${ROOT_LIBRARIES}
    rt
    tbb
    )
    
        
endif(BUILD_TBB_TARGETS)


#+++++++++++++++++++++++++
# OMP TARGETS           |
#+++++++++++++++++++++++++

#+++++++++++++++++++++++++++++++
# PHSP generation              |
#+++++++++++++++++++++++++++++++
add_executable(Hydra_Example_Experimental_PHSP_OMP EXCLUDE_FROM_ALL
HydraPhaseSpaceExperimental.cpp
)

set_target_properties(Hydra_Example_Experimental_PHSP_OMP PROPERTIES 
COMPILE_FLAGS "-fopenmp -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp")

target_link_libraries(Hydra_Example_Experimental_PHSP_OMP
${ROOT_LIBRARIES}
rt
)

#+++++++++++++++++++++++++++++++
# PHSP generation              |
#+++++++++++++++++++++++++++++++
add_executable(Hydra_Example_PHSP_Host_CPP_Device_OMP EXCLUDE_FROM_ALL
HydraPhaseSpaceExample.cpp
)

set_target_properties(Hydra_Example_PHSP_Host_CPP_Device_OMP PROPERTIES 
COMPILE_FLAGS "-fopenmp -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp")

target_link_libraries(Hydra_Example_PHSP_Host_CPP_Device_OMP
${ROOT_LIBRARIES}
rt
)


#+++++++++++++++++++++++++++++++
# PDF sampling                 |
#+++++++++++++++++++++++++++++++     
add_executable(Hydra_Example_Random_Host_CPP_Device_OMP EXCLUDE_FROM_ALL
HydraRandomExample.cpp
)

set_target_properties(Hydra_Example_Random_Host_CPP_Device_OMP PROPERTIES 
COMPILE_FLAGS "-fopenmp -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp")

target_link_libraries(Hydra_Example_Random_Host_CPP_Device_OMP
${ROOT_LIBRARIES}
rt
)


#+++++++++++++++++++++++++++++++
# Data fitting                 |
#+++++++++++++++++++++++++++++++
add_executable(Hydra_Example_Fit_Host_CPP_Device_OMP EXCLUDE_FROM_ALL
HydraFitExample.cpp
)

set_target_properties(Hydra_Example_Fit_Host_CPP_Device_OMP PROPERTIES 
COMPILE_FLAGS "-pg -fopenmp -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp")

target_link_libraries(Hydra_Example_Fit_Host_CPP_Device_OMP
${ROOT_LIBRARIES}
rt
)



#+++++++++++++++++++++++++++++++
# Lambda function evaluation   |
#+++++++++++++++++++++++++++++++	
add_executable(Hydra_Example_Eval_Host_CPP_Device_OMP EXCLUDE_FROM_ALL
HydraEvaluateExample.cpp
)

set_target_properties(Hydra_Example_Eval_Host_CPP_Device_OMP PROPERTIES 
COMPILE_FLAGS  "-fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP -lgomp" )

target_link_libraries(Hydra_Example_Eval_Host_CPP_Device_OMP
${ROOT_LIBRARIES}
rt
)







