function(ADD_HYDRA_EXAMPLE target_name build_cuda build_tbb build_omp build_cpp )
        message(STATUS "-----------")
        #+++++++++++++++++++++++++
        # CUDA TARGETS           |
        #+++++++++++++++++++++++++
        if(${${build_cuda}} )
                  message(STATUS "Adding target ${target_name} to CUDA backend. Executable file name: ${target_name}_cuda")
                  
                  cuda_add_executable("${target_name}_cuda"
                   #EXCLUDE_FROM_ALL 
                   "${target_name}.cu"    
                    OPTIONS -Xcompiler -DHYDRA_DEVICE_SYSTEM=CUDA -DHYDRA_HOST_SYSTEM=CPP)
                    
                  target_link_libraries("${target_name}_cuda" ${ROOT_LIBRARIES} ${TBB_LIBRARIES}  ${GSL_LIBRARIES}  ${CUDA_CUFFT_LIBRARIES} -lm)
                 
                  add_dependencies(examples      "${target_name}_cuda")
                
        endif(${${build_cuda}})
    
        #+++++++++++++++++++++++++
        # TBB TARGETS            |
        #+++++++++++++++++++++++++
        if( ${${build_tbb}} )
                 message(STATUS "Adding target ${target_name} to TBB backend. Executable file name: ${target_name}_tbb")
                 add_executable("${target_name}_tbb"
                 # EXCLUDE_FROM_ALL
                 "${target_name}.cpp" )
                    
                 set_target_properties( "${target_name}_tbb" 
                  PROPERTIES COMPILE_FLAGS "-DHYDRA_HOST_SYSTEM=CPP -DHYDRA_DEVICE_SYSTEM=TBB")
                    
                 target_link_libraries( "${target_name}_tbb" ${ROOT_LIBRARIES} ${TBB_LIBRARIES}  ${GSL_LIBRARIES} ${FFTW_LIBRARIES} -lm)
                   
                 add_dependencies(examples "${target_name}_tbb")
                       
         endif(${${build_tbb}})
         
        #+++++++++++++++++++++++++
        # CPP TARGETS            |
        #+++++++++++++++++++++++++
        if(${${build_cpp}})
                 message(STATUS "Adding target ${target_name} to CPP backend. Executable file name: ${target_name}_cpp")
                 add_executable("${target_name}_cpp"
                 # EXCLUDE_FROM_ALL 
                 "${target_name}.cpp" )
                    
                 set_target_properties( "${target_name}_cpp" 
                  PROPERTIES COMPILE_FLAGS "-DHYDRA_HOST_SYSTEM=CPP -DHYDRA_DEVICE_SYSTEM=CPP")
                    
                 target_link_libraries( "${target_name}_cpp" ${ROOT_LIBRARIES} ${TBB_LIBRARIES} ${GSL_LIBRARIES} ${FFTW_LIBRARIES} -lm)
                   
                 add_dependencies(examples "${target_name}_cpp")
                       
         endif(${${build_cpp}})
         
          
        #+++++++++++++++++++++++++
        # OMP TARGETS            |
        #+++++++++++++++++++++++++
        if(${${build_omp}})
                 message(STATUS "Adding target ${target_name} to OMP backend. Executable file name: ${target_name}_omp")
                 add_executable("${target_name}_omp" 
                 #EXCLUDE_FROM_ALL
                 "${target_name}.cpp" )
                    
                 set_target_properties( "${target_name}_omp" 
                  PROPERTIES COMPILE_FLAGS "-DHYDRA_HOST_SYSTEM=CPP -DHYDRA_DEVICE_SYSTEM=OMP ${OpenMP_CXX_FLAGS}")
                    
                 target_link_libraries( "${target_name}_omp" ${ROOT_LIBRARIES} ${OpenMP_CXX_LIBRARIES} ${GSL_LIBRARIES} ${FFTW_LIBRARIES} -lm)
                   
                 add_dependencies(examples "${target_name}_omp")
                       
         endif(${${build_omp}})
         
endfunction(ADD_HYDRA_EXAMPLE)  
