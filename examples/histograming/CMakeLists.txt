project(examples)

#+++++++++++++++++++++++++++++++
# Hydra histograming           |
#+++++++++++++++++++++++++++++++
ADD_HYDRA_EXAMPLE(dense_histogram BUILD_CUDA_TARGETS BUILD_TBB_TARGETS BUILD_OMP_TARGETS BUILD_CPP_TARGETS)
ADD_HYDRA_EXAMPLE(sparse_histogram BUILD_CUDA_TARGETS BUILD_TBB_TARGETS BUILD_OMP_TARGETS BUILD_CPP_TARGETS)
