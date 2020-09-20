project(testing)

message(STATUS "-----------")
#+++++++++++++++++++++++++
# CUDA TARGETS           |
#+++++++++++++++++++++++++
if(BUILD_CUDA_TARGETS)
          message(STATUS "Adding target tests to CUDA backend. Executable file name: tests_cuda")
          
          cuda_add_executable(tests_cuda main.cu  OPTIONS -Xcompiler -DHYDRA_DEVICE_SYSTEM=CUDA -DHYDRA_HOST_SYSTEM=CPP)
          
          target_link_libraries("tests_cuda" ${ROOT_LIBRARIES} )
         
          add_dependencies(tests tests_cuda)
        
endif(BUILD_CUDA_TARGETS)
    
#+++++++++++++++++++++++++
# TBB TARGETS            |
#+++++++++++++++++++++++++
if(BUILD_TBB_TARGETS)
         message(STATUS "Adding target tests to TBB backend. Executable file name: tests_tbb")
         add_executable(tests_tbb main.cpp )
            
         set_target_properties( tests_tbb PROPERTIES COMPILE_FLAGS "-DHYDRA_HOST_SYSTEM=CPP -DHYDRA_DEVICE_SYSTEM=TBB")
            
         target_link_libraries( tests_tbb ${ROOT_LIBRARIES} ${TBB_LIBRARIES} )
           
         add_dependencies(tests tests_tbb)
               
 endif(BUILD_TBB_TARGETS)
 
#+++++++++++++++++++++++++
# CPP TARGETS            |
#+++++++++++++++++++++++++
if(BUILD_CPP_TARGETS)

         message(STATUS "Adding target tests to CPP backend. Executable file name: tests_cpp")
         add_executable(tests_cpp main.cpp )
            
         set_target_properties( tests_cpp  PROPERTIES COMPILE_FLAGS "-DHYDRA_HOST_SYSTEM=CPP -DHYDRA_DEVICE_SYSTEM=CPP")
            
         target_link_libraries( tests_cpp ${ROOT_LIBRARIES} ${TBB_LIBRARIES} )
           
         add_dependencies(tests tests_cpp)
 
 #----------------------------
 # BigCrush for hydra::Random   
 #----------------------------
 
         message(STATUS "Adding target BigCrush CPP backend. Executable file name: bigcrush")
         
         add_executable(hydra_ars_bigcrush ars_bigcrush.cpp )
            
         set_target_properties( hydra_ars_bigcrush  PROPERTIES COMPILE_FLAGS "-DHYDRA_HOST_SYSTEM=CPP -DHYDRA_DEVICE_SYSTEM=CPP")
            
         target_link_libraries( hydra_ars_bigcrush -L/usr/lib64 ${TESTU01_LIBRARIES}  )
         
         add_dependencies(tests  hydra_ars_bigcrush)
         
 endif(BUILD_CPP_TARGETS)
 
  
#+++++++++++++++++++++++++
# OMP TARGETS            |
#+++++++++++++++++++++++++
if(BUILD_OMP_TARGETS)
         message(STATUS "Adding target tests to OMP backend. Executable file name: tests_omp")
         add_executable(tests_omp main.cpp )
            
         set_target_properties( tests_omp PROPERTIES COMPILE_FLAGS "-DHYDRA_HOST_SYSTEM=CPP -DHYDRA_DEVICE_SYSTEM=OMP ${OpenMP_CXX_FLAGS}")
            
         target_link_libraries( tests_omp ${ROOT_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
           
         add_dependencies(tests tests_omp)
               
 endif(BUILD_OMP_TARGETS)