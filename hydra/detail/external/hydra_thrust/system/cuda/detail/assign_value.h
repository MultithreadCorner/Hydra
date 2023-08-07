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
#include <hydra/detail/external/hydra_thrust/system/cuda/detail/execution_policy.h>
#include <hydra/detail/external/hydra_thrust/detail/raw_pointer_cast.h>
#include <hydra/detail/external/hydra_thrust/system/cuda/detail/copy.h>

#include <hydra/detail/external/hydra_libcudacxx/nv/target>

HYDRA_THRUST_NAMESPACE_BEGIN
namespace cuda_cub {


template<typename DerivedPolicy, typename Pointer1, typename Pointer2>
inline __host__ __device__
  void assign_value(hydra_thrust::cuda::execution_policy<DerivedPolicy> &exec, Pointer1 dst, Pointer2 src)
{
  // XXX war nvbugs/881631
  struct war_nvbugs_881631
  {
    __host__ inline static void host_path(hydra_thrust::cuda::execution_policy<DerivedPolicy> &exec, Pointer1 dst, Pointer2 src)
    {
      cuda_cub::copy(exec, src, src + 1, dst);
    }

    __device__ inline static void device_path(hydra_thrust::cuda::execution_policy<DerivedPolicy> &, Pointer1 dst, Pointer2 src)
    {
      *hydra_thrust::raw_pointer_cast(dst) = *hydra_thrust::raw_pointer_cast(src);
    }
  };

  NV_IF_TARGET(NV_IS_HOST, (
    war_nvbugs_881631::host_path(exec,dst,src);
  ), (
    war_nvbugs_881631::device_path(exec,dst,src);
  ));

} // end assign_value()


template<typename System1, typename System2, typename Pointer1, typename Pointer2>
inline __host__ __device__
  void assign_value(cross_system<System1,System2> &systems, Pointer1 dst, Pointer2 src)
{
  // XXX war nvbugs/881631
  struct war_nvbugs_881631
  {
    __host__ inline static void host_path(cross_system<System1,System2> &systems, Pointer1 dst, Pointer2 src)
    {
      // rotate the systems so that they are ordered the same as (src, dst)
      // for the call to hydra_thrust::copy
      cross_system<System2,System1> rotated_systems = systems.rotate();
      cuda_cub::copy(rotated_systems, src, src + 1, dst);
    }

    __device__ inline static void device_path(cross_system<System1,System2> &, Pointer1 dst, Pointer2 src)
    {
      // XXX forward the true cuda::execution_policy inside systems here
      //     instead of materializing a tag
      hydra_thrust::cuda::tag cuda_tag;
      hydra_thrust::cuda_cub::assign_value(cuda_tag, dst, src);
    }
  };

  NV_IF_TARGET(NV_IS_HOST, (
    war_nvbugs_881631::host_path(systems,dst,src);
  ), (
    war_nvbugs_881631::device_path(systems,dst,src);
  ));
} // end assign_value()


} // end cuda_cub
HYDRA_THRUST_NAMESPACE_END
#endif
