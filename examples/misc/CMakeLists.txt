project(examples)

#+++++++++++++++++++++++++++++++++
# Hydra variant                  |
#+++++++++++++++++++++++++++++++++
ADD_HYDRA_EXAMPLE(multivector_container BUILD_CUDA_TARGETS BUILD_TBB_TARGETS BUILD_OMP_TARGETS BUILD_CPP_TARGETS)
ADD_HYDRA_EXAMPLE(multiarray_container BUILD_CUDA_TARGETS BUILD_TBB_TARGETS BUILD_OMP_TARGETS BUILD_CPP_TARGETS)
ADD_HYDRA_EXAMPLE(range_semantics BUILD_CUDA_TARGETS BUILD_TBB_TARGETS BUILD_OMP_TARGETS BUILD_CPP_TARGETS)
ADD_HYDRA_EXAMPLE(zipping_iterables BUILD_CUDA_TARGETS BUILD_TBB_TARGETS BUILD_OMP_TARGETS BUILD_CPP_TARGETS)
ADD_HYDRA_EXAMPLE(spline_interpolation BUILD_CUDA_TARGETS BUILD_TBB_TARGETS BUILD_OMP_TARGETS BUILD_CPP_TARGETS)
ADD_HYDRA_EXAMPLE(spline2D_interpolation BUILD_CUDA_TARGETS BUILD_TBB_TARGETS BUILD_OMP_TARGETS BUILD_CPP_TARGETS)
ADD_HYDRA_EXAMPLE(spline3D_interpolation BUILD_CUDA_TARGETS BUILD_TBB_TARGETS BUILD_OMP_TARGETS BUILD_CPP_TARGETS)
ADD_HYDRA_EXAMPLE(spline4D_interpolation BUILD_CUDA_TARGETS BUILD_TBB_TARGETS BUILD_OMP_TARGETS BUILD_CPP_TARGETS)
ADD_HYDRA_EXAMPLE(quick_test BUILD_CUDA_TARGETS BUILD_TBB_TARGETS BUILD_OMP_TARGETS BUILD_CPP_TARGETS)