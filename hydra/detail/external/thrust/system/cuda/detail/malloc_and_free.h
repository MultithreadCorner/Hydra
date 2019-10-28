/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <hydra/detail/external/thrust/system/cuda/detail/guarded_cuda_runtime_api.h>

#include <hydra/detail/external/thrust/detail/config.h>
#include <hydra/detail/external/thrust/detail/seq.h>
#include <hydra/detail/external/thrust/memory.h>
#include <hydra/detail/external/thrust/system/cuda/config.h>
#ifdef HYDRA_THRUST_CACHING_DEVICE_MALLOC
#include <hydra/detail/external/thrust/system/cuda/detail/cub/util_allocator.cuh>
#endif
#include <hydra/detail/external/thrust/system/cuda/detail/util.h>
#include <hydra/detail/external/thrust/system/detail/bad_alloc.h>

HYDRA_EXTERNAL_NAMESPACE_BEGIN
HYDRA_THRUST_BEGIN_NS
namespace cuda_cub {

#ifdef HYDRA_THRUST_CACHING_DEVICE_MALLOC
#define __CUB_CACHING_MALLOC
#ifndef __CUDA_ARCH__
inline cub::CachingDeviceAllocator &get_allocator()
{
  static cub::CachingDeviceAllocator g_allocator(true);
  return g_allocator;
}
#endif
#endif


// note that malloc returns a raw pointer to avoid
// depending on the heavyweight thrust/system/cuda/memory.h header
template<typename DerivedPolicy>
__hydra_host__ __hydra_device__
void *malloc(execution_policy<DerivedPolicy> &, std::size_t n)
{
  void *result = 0;

#ifndef __CUDA_ARCH__
#ifdef __CUB_CACHING_MALLOC
  cub::CachingDeviceAllocator &alloc = get_allocator();
  cudaError_t status = alloc.DeviceAllocate(&result, n);
#else
  cudaError_t status = cudaMalloc(&result, n);
#endif

  if(status != cudaSuccess)
  {
  //  cuda_cub::throw_on_error(status, "device malloc failed");
    HYDRA_EXTERNAL_NS::thrust::system::detail::bad_alloc(HYDRA_EXTERNAL_NS::thrust::cuda_category().message(status).c_str());
  } 
#else
  result = HYDRA_EXTERNAL_NS::thrust::raw_pointer_cast(HYDRA_EXTERNAL_NS::thrust::malloc(HYDRA_EXTERNAL_NS::thrust::seq, n));
#endif

  return result;
} // end malloc()


template<typename DerivedPolicy, typename Pointer>
__hydra_host__ __hydra_device__
void free(execution_policy<DerivedPolicy> &, Pointer ptr)
{
#ifndef __CUDA_ARCH__
#ifdef __CUB_CACHING_MALLOC
  cub::CachingDeviceAllocator &alloc = get_allocator();
  cudaError_t status = alloc.DeviceFree(HYDRA_EXTERNAL_NS::thrust::raw_pointer_cast(ptr));
#else
  cudaError_t status = cudaFree(HYDRA_EXTERNAL_NS::thrust::raw_pointer_cast(ptr));
#endif
  cuda_cub::throw_on_error(status, "device free failed");
#else
  HYDRA_EXTERNAL_NS::thrust::free(HYDRA_EXTERNAL_NS::thrust::seq, ptr);
#endif
} // end free()

}    // namespace cuda_cub
HYDRA_THRUST_END_NS
HYDRA_EXTERNAL_NAMESPACE_END
