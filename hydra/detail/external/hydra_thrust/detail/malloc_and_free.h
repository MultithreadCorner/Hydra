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
#include <hydra/detail/external/hydra_thrust/detail/execution_policy.h>
#include <hydra/detail/external/hydra_thrust/detail/pointer.h>
#include <hydra/detail/external/hydra_thrust/detail/raw_pointer_cast.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/memory.h>
#include <hydra/detail/external/hydra_thrust/system/detail/adl/malloc_and_free.h>

HYDRA_THRUST_NAMESPACE_BEGIN

__hydra_thrust_exec_check_disable__
template<typename DerivedPolicy>
__host__ __device__
pointer<void,DerivedPolicy> malloc(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec, std::size_t n)
{
  using hydra_thrust::system::detail::generic::malloc;

  // XXX should use a hypothetical hydra_thrust::static_pointer_cast here
  void *raw_ptr = static_cast<void*>(hydra_thrust::raw_pointer_cast(malloc(hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)), n)));

  return pointer<void,DerivedPolicy>(raw_ptr);
}

__hydra_thrust_exec_check_disable__
template<typename T, typename DerivedPolicy>
__host__ __device__
pointer<T,DerivedPolicy> malloc(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec, std::size_t n)
{
  using hydra_thrust::system::detail::generic::malloc;

  T *raw_ptr = static_cast<T*>(hydra_thrust::raw_pointer_cast(malloc<T>(hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)), n)));

  return pointer<T,DerivedPolicy>(raw_ptr);
}


// XXX WAR nvbug 992955
#if HYDRA_THRUST_DEVICE_COMPILER == HYDRA_THRUST_DEVICE_COMPILER_NVCC
#if CUDART_VERSION < 5000

// cudafe generates unqualified calls to free(int *volatile)
// which get confused with hydra_thrust::free
// spoof a hydra_thrust::free which simply maps to ::free
inline __host__ __device__
void free(int *volatile ptr)
{
  ::free(ptr);
}

#endif // CUDART_VERSION
#endif // HYDRA_THRUST_DEVICE_COMPILER

__hydra_thrust_exec_check_disable__
template<typename DerivedPolicy, typename Pointer>
__host__ __device__
void free(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec, Pointer ptr)
{
  using hydra_thrust::system::detail::generic::free;

  free(hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)), ptr);
}

// XXX consider another form of free which does not take a system argument and
// instead infers the system from the pointer

HYDRA_THRUST_NAMESPACE_END
