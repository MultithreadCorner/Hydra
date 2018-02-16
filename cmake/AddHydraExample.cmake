function(ADD_HYDRA_EXAMPLE target_name)
        message(STATUS "-----------")
        #+++++++++++++++++++++++++
        # CUDA TARGETS           |
        #+++++++++++++++++++++++++
        if(BUILD_CUDA_TARGETS)
                  message(STATUS "Adding target ${target_name} to CUDA backend. Executable file name: ${target_name}_cuda")
                  
                  cuda_add_executable("${target_name}_cuda"
                   #EXCLUDE_FROM_ALL 
                   "${target_name}.cu"    
                   OPTIONS -Xcompiler -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA  -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP)
                    
                  target_link_libraries("${target_name}_cuda" ${ROOT_LIBRARIES} )
                 
                  add_dependencies(examples      "${target_name}_cuda")
                
        endif(BUILD_CUDA_TARGETS)
    
        #+++++++++++++++++++++++++
        # TBB TARGETS            |
        #+++++++++++++++++++++++++
        if(BUILD_TBB_TARGETS)
                 message(STATUS "Adding target ${target_name} to TBB backend. Executable file name: ${target_name}_tbb")
                 add_executable("${target_name}_tbb"
                 # EXCLUDE_FROM_ALL
                 "${target_name}.cpp" )
                    
                 set_target_properties( "${target_name}_tbb" 
                  PROPERTIES COMPILE_FLAGS "-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB")
                    
                 target_link_libraries( "${target_name}_tbb" ${ROOT_LIBRARIES} ${TBB_LIBRARIES} )
                   
                 add_dependencies(examples "${target_name}_tbb")
                       
         endif(BUILD_TBB_TARGETS)
         
        #+++++++++++++++++++++++++
        # CPP TARGETS            |
        #+++++++++++++++++++++++++
        if(BUILD_CPP_TARGETS)
                 message(STATUS "Adding target ${target_name} to CPP backend. Executable file name: ${target_name}_cpp")
                 add_executable("${target_name}_cpp"
                 # EXCLUDE_FROM_ALL 
                 "${target_name}.cpp" )
                    
                 set_target_properties( "${target_name}_cpp" 
                  PROPERTIES COMPILE_FLAGS "-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP")
                    
                 target_link_libraries( "${target_name}_cpp" ${ROOT_LIBRARIES} ${TBB_LIBRARIES} )
                   
                 add_dependencies(examples "${target_name}_cpp")
                       
         endif(BUILD_CPP_TARGETS)
         
          
        #+++++++++++++++++++++++++
        # OMP TARGETS            |
        #+++++++++++++++++++++++++
        if(BUILD_OMP_TARGETS)
                 message(STATUS "Adding target ${target_name} to OMP backend. Executable file name: ${target_name}_omp")
                 add_executable("${target_name}_omp" 
                 #EXCLUDE_FROM_ALL
                 "${target_name}.cpp" )
                    
                 set_target_properties( "${target_name}_omp" 
                  PROPERTIES COMPILE_FLAGS "-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP ${OpenMP_CXX_FLAGS}")
                    
                 target_link_libraries( "${target_name}_omp" ${ROOT_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
                   
                 add_dependencies(examples "${target_name}_omp")
                       
         endif(BUILD_OMP_TARGETS)
         
endfunction(ADD_HYDRA_EXAMPLE)  