/*
 *  Copyright 2008-2018 NVIDIA Corporation
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
#include <hydra/detail/external/hydra_thrust/detail/allocator_aware_execution_policy.h>
#include <hydra/detail/external/hydra_thrust/system/detail/sequential/execution_policy.h>

HYDRA_THRUST_NAMESPACE_BEGIN
namespace detail
{


struct seq_t : hydra_thrust::system::detail::sequential::execution_policy<seq_t>,
  hydra_thrust::detail::allocator_aware_execution_policy<
    hydra_thrust::system::detail::sequential::execution_policy>
{
  __host__ __device__
  constexpr seq_t() : hydra_thrust::system::detail::sequential::execution_policy<seq_t>() {}

  // allow any execution_policy to convert to seq_t
  template<typename DerivedPolicy>
  __host__ __device__
  seq_t(const hydra_thrust::execution_policy<DerivedPolicy> &)
    : hydra_thrust::system::detail::sequential::execution_policy<seq_t>()
  {}
};


} // end detail


HYDRA_THRUST_INLINE_CONSTANT detail::seq_t seq;


HYDRA_THRUST_NAMESPACE_END


