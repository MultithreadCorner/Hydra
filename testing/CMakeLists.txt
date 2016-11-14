project(testing)

if(BUILD_CUDA_TARGETS)

cuda_add_executable(Hydra_Test_Host_OMP_Device_CUDA EXCLUDE_FROM_ALL
    unit_tests.cu
    multivector.cu
    Events.cu
    Chain.cu
    OPTIONS  -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP  -lgomp
    )
    add_test(Hydra_Test_Host_OMP_Device_CUDA Hydra_Test_Host_OMP_Device_CUDA)
    
cuda_add_executable(Hydra_Test_Host_CPP_Device_CUDA EXCLUDE_FROM_ALL
    unit_tests.cu
    multivector.cu
    Events.cu
    Chain.cu
    OPTIONS  -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP  -lgomp
    )
    add_test(Hydra_Test_Host_CPP_Device_CUDA Hydra_Test_Host_CPP_Device_CUDA)
    
endif(BUILD_CUDA_TARGETS)

add_executable(Hydra_Test_Host_OMP_Device_OMP EXCLUDE_FROM_ALL
    unit_tests.cpp
    multivector.cpp
    Events.cpp
    Chain.cpp
)
set_target_properties(Hydra_Test_Host_OMP_Device_OMP PROPERTIES 
COMPILE_FLAGS  "-fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP -lgomp" LINKER_LANGUAGE CXX )
add_test(Hydra_Test_Host_OMP_Device_OMP Hydra_Test_Host_OMP_Device_OMP)

add_executable(Hydra_Test_Host_CPP_Device_OMP EXCLUDE_FROM_ALL
    unit_tests.cpp
    multivector.cpp
    Events.cpp
    Chain.cpp
)
set_target_properties(Hydra_Test_Host_CPP_Device_OMP PROPERTIES 
COMPILE_FLAGS  "-x c++ -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP -lgomp" LINKER_LANGUAGE CXX )
add_test(Hydra_Test_Host_CPP_Device_OMP Hydra_Test_Host_CPP_Device_OMP)