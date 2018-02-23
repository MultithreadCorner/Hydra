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

#include <hydra/detail/external/thrust/detail/config.h>
#include <hydra/detail/external/thrust/detail/execution_policy.h>
#include <hydra/detail/external/thrust/detail/pointer.h>
#include <hydra/detail/external/thrust/detail/raw_pointer_cast.h>
#include <hydra/detail/external/thrust/system/detail/generic/memory.h>
#include <hydra/detail/external/thrust/system/detail/adl/malloc_and_free.h>

HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
{

__thrust_exec_check_disable__
template<typename DerivedPolicy>
__hydra_host__ __hydra_device__
pointer<void,DerivedPolicy> malloc(const thrust::detail::execution_policy_base<DerivedPolicy> &exec, std::size_t n)
{
  using thrust::system::detail::generic::malloc;

  // XXX should use a hypothetical thrust::static_pointer_cast here
  void *raw_ptr = static_cast<void*>(thrust::raw_pointer_cast(malloc(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), n)));

  return pointer<void,DerivedPolicy>(raw_ptr);
}

__thrust_exec_check_disable__
template<typename T, typename DerivedPolicy>
__hydra_host__ __hydra_device__
pointer<T,DerivedPolicy> malloc(const thrust::detail::execution_policy_base<DerivedPolicy> &exec, std::size_t n)
{
  using thrust::system::detail::generic::malloc;

  T *raw_ptr = static_cast<T*>(thrust::raw_pointer_cast(malloc<T>(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), n)));

  return pointer<T,DerivedPolicy>(raw_ptr);
}


// XXX WAR nvbug 992955
#if HYDRA_THRUST_DEVICE_COMPILER == HYDRA_THRUST_DEVICE_COMPILER_NVCC
#if CUDA_VERSION < 5000

// cudafe generates unqualified calls to free(int *volatile)
// which get confused with thrust::free
// spoof a thrust::free which simply maps to ::free
inline __hydra_host__ __hydra_device__
void free(int *volatile ptr)
{
  ::free(ptr);
}

#endif // CUDA_VERSION
#endif // HYDRA_THRUST_DEVICE_COMPILER

__thrust_exec_check_disable__
template<typename DerivedPolicy, typename Pointer>
__hydra_host__ __hydra_device__
void free(const thrust::detail::execution_policy_base<DerivedPolicy> &exec, Pointer ptr)
{
  using thrust::system::detail::generic::free;

  free(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), ptr);
}

// XXX consider another form of free which does not take a system argument and
// instead infers the system from the pointer

} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust

HYDRA_EXTERNAL_NAMESPACE_END
