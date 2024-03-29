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

#include <hydra/detail/external/hydra_thrust/detail/config.h>

#if HYDRA_THRUST_DEVICE_COMPILER == HYDRA_THRUST_DEVICE_COMPILER_NVCC

#include <hydra/detail/external/hydra_thrust/system/cuda/config.h>

#include <hydra/detail/external/hydra_thrust/detail/raw_pointer_cast.h>
#include <hydra/detail/external/hydra_thrust/system/cuda/detail/execution_policy.h>
#include <hydra/detail/external/hydra_thrust/swap.h>

#include <hydra/detail/external/hydra_libcudacxx/nv/target>

HYDRA_THRUST_NAMESPACE_BEGIN
namespace cuda_cub {


template<typename DerivedPolicy, typename Pointer1, typename Pointer2>
inline __host__ __device__
void iter_swap(hydra_thrust::cuda::execution_policy<DerivedPolicy> &, Pointer1 a, Pointer2 b)
{
  // XXX war nvbugs/881631
  struct war_nvbugs_881631
  {
    __host__ inline static void host_path(Pointer1 a, Pointer2 b)
    {
      hydra_thrust::swap_ranges(a, a + 1, b);
    }

    __device__ inline static void device_path(Pointer1 a, Pointer2 b)
    {
      using hydra_thrust::swap;
      swap(*hydra_thrust::raw_pointer_cast(a),
           *hydra_thrust::raw_pointer_cast(b));
    }
  };

  NV_IF_TARGET(NV_IS_HOST, (
    war_nvbugs_881631::host_path(a, b);
  ), (
    war_nvbugs_881631::device_path(a, b);
  ));

} // end iter_swap()


} // end cuda_cub
HYDRA_THRUST_NAMESPACE_END
#endif
