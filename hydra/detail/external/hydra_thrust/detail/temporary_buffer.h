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
#include <hydra/detail/external/hydra_thrust/pair.h>
#include <hydra/detail/external/hydra_thrust/detail/pointer.h>
#include <hydra/detail/external/hydra_thrust/detail/raw_pointer_cast.h>
#include <hydra/detail/external/hydra_thrust/detail/execute_with_allocator.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/temporary_buffer.h>
#include <hydra/detail/external/hydra_thrust/system/detail/adl/temporary_buffer.h>

HYDRA_THRUST_NAMESPACE_BEGIN
namespace detail
{


template<typename T, typename DerivedPolicy, typename Pair>
__host__ __device__
  hydra_thrust::pair<hydra_thrust::pointer<T,DerivedPolicy>, typename hydra_thrust::pointer<T,DerivedPolicy>::difference_type>
    down_cast_pair(Pair p)
{
  // XXX should use a hypothetical hydra_thrust::static_pointer_cast here
  hydra_thrust::pointer<T,DerivedPolicy> ptr = hydra_thrust::pointer<T,DerivedPolicy>(static_cast<T*>(hydra_thrust::raw_pointer_cast(p.first)));

  typedef hydra_thrust::pair<hydra_thrust::pointer<T,DerivedPolicy>, typename hydra_thrust::pointer<T,DerivedPolicy>::difference_type> result_type;
  return result_type(ptr, p.second);
} // end down_cast_pair()


} // end detail


__hydra_thrust_exec_check_disable__
template<typename T, typename DerivedPolicy>
__host__ __device__
  hydra_thrust::pair<hydra_thrust::pointer<T,DerivedPolicy>, typename hydra_thrust::pointer<T,DerivedPolicy>::difference_type>
    get_temporary_buffer(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec, typename hydra_thrust::pointer<T,DerivedPolicy>::difference_type n)
{
  using hydra_thrust::detail::get_temporary_buffer; // execute_with_allocator
  using hydra_thrust::system::detail::generic::get_temporary_buffer;

  return hydra_thrust::detail::down_cast_pair<T,DerivedPolicy>(get_temporary_buffer<T>(hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)), n));
} // end get_temporary_buffer()


__hydra_thrust_exec_check_disable__
template<typename DerivedPolicy, typename Pointer>
__host__ __device__
  void return_temporary_buffer(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec, Pointer p, std::ptrdiff_t n)
{
  using hydra_thrust::detail::return_temporary_buffer; // execute_with_allocator
  using hydra_thrust::system::detail::generic::return_temporary_buffer;

  return return_temporary_buffer(hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)), p, n);
} // end return_temporary_buffer()


HYDRA_THRUST_NAMESPACE_END

