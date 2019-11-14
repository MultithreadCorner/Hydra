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

#if HYDRA_THRUST_DEVICE_COMPILER == HYDRA_THRUST_DEVICE_COMPILER_NVCC
#include <hydra/detail/external/thrust/detail/config.h>
#include <hydra/detail/external/thrust/system/cuda/config.h>

#include <hydra/detail/external/thrust/detail/raw_pointer_cast.h>
#include <hydra/detail/external/thrust/system/cuda/detail/execution_policy.h>
#include <hydra/detail/external/thrust/swap.h>

HYDRA_EXTERNAL_NAMESPACE_BEGIN

HYDRA_THRUST_BEGIN_NS
namespace cuda_cub {


template<typename DerivedPolicy, typename Pointer1, typename Pointer2>
inline __hydra_host__ __hydra_device__
void iter_swap(HYDRA_EXTERNAL_NS::thrust::cuda::execution_policy<DerivedPolicy> &, Pointer1 a, Pointer2 b)
{
  // XXX war nvbugs/881631
  struct war_nvbugs_881631
  {
    __hydra_host__ inline static void host_path(Pointer1 a, Pointer2 b)
    {
      HYDRA_EXTERNAL_NS::thrust::swap_ranges(a, a + 1, b);
    }

    __device__ inline static void device_path(Pointer1 a, Pointer2 b)
    {
      using HYDRA_EXTERNAL_NS::thrust::swap;
      swap(*HYDRA_EXTERNAL_NS::thrust::raw_pointer_cast(a),
           *HYDRA_EXTERNAL_NS::thrust::raw_pointer_cast(b));
    }
  };

#ifndef __CUDA_ARCH__
  return war_nvbugs_881631::host_path(a, b);
#else
  return war_nvbugs_881631::device_path(a, b);
#endif // __CUDA_ARCH__
} // end iter_swap()


} // end cuda_cub
HYDRA_THRUST_END_NS

HYDRA_EXTERNAL_NAMESPACE_END
#endif
