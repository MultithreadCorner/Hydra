project(testing)

message(STATUS "-----------")
#+++++++++++++++++++++++++
# CUDA TARGETS           |
#+++++++++++++++++++++++++
if(BUILD_CUDA_TARGETS)
          message(STATUS "Adding target tests to CUDA backend. Executable file name: tests_cuda")

          add_executable(tests_cuda main.cu  OPTIONS -Xcompiler -DHYDRA_DEVICE_SYSTEM=CUDA -DHYDRA_HOST_SYSTEM=CPP)

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

	 if(TESTU01_FOUND)
	 #----------------------------
	 # BigCrush for hydra::Random
	 #----------------------------

	         message(STATUS "Adding target to CPP backend. Executable file name: timing_baseline")

	         add_executable(timing_baseline timing_baseline.cpp )

	         set_target_properties( timing_baseline  PROPERTIES COMPILE_FLAGS "-DHYDRA_HOST_SYSTEM=CPP -DHYDRA_DEVICE_SYSTEM=CPP")

	         target_link_libraries( timing_baseline -L/usr/lib64 ${TESTU01_LIBRARIES}  )

	         add_dependencies(tests  timing_baseline)

	  #----------------------------

	         message(STATUS "Adding target to CPP backend. Executable file name: hydra_squares4_bigcrush")

	         add_executable(hydra_squares4_bigcrush squares4_bigcrush.cpp )

	         set_target_properties( hydra_squares4_bigcrush  PROPERTIES COMPILE_FLAGS "-DHYDRA_HOST_SYSTEM=CPP -DHYDRA_DEVICE_SYSTEM=CPP")

	         target_link_libraries( hydra_squares4_bigcrush -L/usr/lib64 ${TESTU01_LIBRARIES}  )

	         add_dependencies(tests  hydra_squares4_bigcrush)

	  #----------------------------

	         message(STATUS "Adding target to CPP backend. Executable file name: hydra_squares3_bigcrush")

	         add_executable(hydra_squares3_bigcrush squares3_bigcrush.cpp )

	         set_target_properties( hydra_squares3_bigcrush  PROPERTIES COMPILE_FLAGS "-DHYDRA_HOST_SYSTEM=CPP -DHYDRA_DEVICE_SYSTEM=CPP")

	         target_link_libraries( hydra_squares3_bigcrush -L/usr/lib64 ${TESTU01_LIBRARIES}  )

	         add_dependencies(tests  hydra_squares3_bigcrush)

	  #----------------------------

	         message(STATUS "Adding target to CPP backend. Executable file name: hydra_ars_bigcrush")

	         add_executable(hydra_ars_bigcrush ars_bigcrush.cpp )

	         set_target_properties( hydra_ars_bigcrush  PROPERTIES COMPILE_FLAGS "-DHYDRA_HOST_SYSTEM=CPP -DHYDRA_DEVICE_SYSTEM=CPP")

	         target_link_libraries( hydra_ars_bigcrush -L/usr/lib64 ${TESTU01_LIBRARIES}  )

	         add_dependencies(tests  hydra_ars_bigcrush)

	  #----------------------------

	         message(STATUS "Adding target to CPP backend. Executable file name: hydra_philox_bigcrush")

	         add_executable(hydra_philox_bigcrush philox_bigcrush.cpp )

	         set_target_properties( hydra_philox_bigcrush  PROPERTIES COMPILE_FLAGS "-DHYDRA_HOST_SYSTEM=CPP -DHYDRA_DEVICE_SYSTEM=CPP")

	         target_link_libraries( hydra_philox_bigcrush -L/usr/lib64 ${TESTU01_LIBRARIES}  )

	         add_dependencies(tests  hydra_philox_bigcrush)

	  #----------------------------

	         message(STATUS "Adding target to CPP backend. Executable file name: hydra_philox_long_bigcrush")

	         add_executable(hydra_philox_long_bigcrush philox_long_bigcrush.cpp )

	         set_target_properties( hydra_philox_long_bigcrush  PROPERTIES COMPILE_FLAGS "-DHYDRA_HOST_SYSTEM=CPP -DHYDRA_DEVICE_SYSTEM=CPP")

	         target_link_libraries( hydra_philox_long_bigcrush -L/usr/lib64 ${TESTU01_LIBRARIES}  )

	         add_dependencies(tests  hydra_philox_long_bigcrush)

	  #----------------------------

	         message(STATUS "Adding target to CPP backend. Executable file name: hydra_threefry_bigcrush")

	         add_executable(hydra_threefry_bigcrush threefry_bigcrush.cpp )

	         set_target_properties( hydra_threefry_bigcrush  PROPERTIES COMPILE_FLAGS "-DHYDRA_HOST_SYSTEM=CPP -DHYDRA_DEVICE_SYSTEM=CPP")

	         target_link_libraries( hydra_threefry_bigcrush -L/usr/lib64 ${TESTU01_LIBRARIES}  )

	         add_dependencies(tests  hydra_threefry_bigcrush)

	      #----------------------------

	         message(STATUS "Adding target to CPP backend. Executable file name: hydra_threefry_long_bigcrush")

	         add_executable(hydra_threefry_long_bigcrush threefry_long_bigcrush.cpp )

	         set_target_properties( hydra_threefry_long_bigcrush  PROPERTIES COMPILE_FLAGS "-DHYDRA_HOST_SYSTEM=CPP -DHYDRA_DEVICE_SYSTEM=CPP")

	         target_link_libraries( hydra_threefry_long_bigcrush -L/usr/lib64 ${TESTU01_LIBRARIES}  )

	         add_dependencies(tests  hydra_threefry_long_bigcrush)

	         #----------------------------

	         message(STATUS "Adding target to CPP backend. Executable file name: batch_test_generator")

	         add_executable(batch_test_generator batch_test_generator.cpp )

	         set_target_properties( batch_test_generator  PROPERTIES COMPILE_FLAGS "-DHYDRA_HOST_SYSTEM=CPP -DHYDRA_DEVICE_SYSTEM=CPP")

	         target_link_libraries( batch_test_generator -L/usr/lib64  )

	         add_dependencies(tests  batch_test_generator)

	         #----------------------------

	         message(STATUS "Adding target to CPP backend. Executable file name: hydra_squares3_practrand")

	         add_executable(hydra_squares3_practrand squares3_practrand.cpp )

	         set_target_properties( hydra_squares3_practrand  PROPERTIES COMPILE_FLAGS "-DHYDRA_HOST_SYSTEM=CPP -DHYDRA_DEVICE_SYSTEM=CPP")

	         target_link_libraries( hydra_squares3_practrand -L/usr/lib64 ${TESTU01_LIBRARIES}  )

	         add_dependencies(tests hydra_squares3_practrand)

	         #----------------------------

	         message(STATUS "Adding target to CPP backend. Executable file name: hydra_squares4_practrand")

	         add_executable(hydra_squares4_practrand squares4_practrand.cpp )

	         set_target_properties( hydra_squares4_practrand  PROPERTIES COMPILE_FLAGS "-DHYDRA_HOST_SYSTEM=CPP -DHYDRA_DEVICE_SYSTEM=CPP")

	         target_link_libraries( hydra_squares4_practrand -L/usr/lib64 ${TESTU01_LIBRARIES}  )

	         add_dependencies(tests hydra_squares4_practrand)
	         #----------------------------

	         message(STATUS "Adding target to CPP backend. Executable file name: hydra_ars_practrand")

	         add_executable(hydra_ars_practrand ars_practrand.cpp )

	         set_target_properties( hydra_ars_practrand  PROPERTIES COMPILE_FLAGS "-DHYDRA_HOST_SYSTEM=CPP -DHYDRA_DEVICE_SYSTEM=CPP")

	         target_link_libraries( hydra_ars_practrand -L/usr/lib64 ${TESTU01_LIBRARIES}  )

	         add_dependencies(tests hydra_ars_practrand)

	         #----------------------------

	         message(STATUS "Adding target to CPP backend. Executable file name: hydra_philox_practrand")

	         add_executable(hydra_philox_practrand philox_practrand.cpp )

	         set_target_properties( hydra_philox_practrand  PROPERTIES COMPILE_FLAGS "-DHYDRA_HOST_SYSTEM=CPP -DHYDRA_DEVICE_SYSTEM=CPP")

	         target_link_libraries( hydra_philox_practrand -L/usr/lib64 ${TESTU01_LIBRARIES}  )

	         add_dependencies(tests hydra_philox_practrand)


	         #----------------------------

	         message(STATUS "Adding target to CPP backend. Executable file name: hydra_threefry_practrand")

	         add_executable(hydra_threefry_practrand threefry_practrand.cpp )

	         set_target_properties( hydra_threefry_practrand  PROPERTIES COMPILE_FLAGS "-DHYDRA_HOST_SYSTEM=CPP -DHYDRA_DEVICE_SYSTEM=CPP")

	         target_link_libraries( hydra_threefry_practrand -L/usr/lib64 ${TESTU01_LIBRARIES}  )

	         add_dependencies(tests hydra_threefry_practrand)

	         #----------------------------

	         message(STATUS "Adding target to CPP backend. Executable file name: hydra_philox_long_practrand")

	         add_executable(hydra_philox_long_practrand philox_long_practrand.cpp )

	         set_target_properties( hydra_philox_long_practrand  PROPERTIES COMPILE_FLAGS "-DHYDRA_HOST_SYSTEM=CPP -DHYDRA_DEVICE_SYSTEM=CPP")

	         target_link_libraries( hydra_philox_long_practrand -L/usr/lib64 ${TESTU01_LIBRARIES}  )

	         add_dependencies(tests hydra_philox_long_practrand)


	         #----------------------------

	         message(STATUS "Adding target to CPP backend. Executable file name: hydra_threefry_long_practrand")

	         add_executable(hydra_threefry_long_practrand threefry_long_practrand.cpp )

	         set_target_properties( hydra_threefry_long_practrand  PROPERTIES COMPILE_FLAGS "-DHYDRA_HOST_SYSTEM=CPP -DHYDRA_DEVICE_SYSTEM=CPP")

	         target_link_libraries( hydra_threefry_long_practrand -L/usr/lib64 ${TESTU01_LIBRARIES}  )

	         add_dependencies(tests hydra_threefry_long_practrand)

	 endif(TESTU01_FOUND)

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
