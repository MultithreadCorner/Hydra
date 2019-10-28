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
#include <hydra/detail/external/thrust/pair.h>
#include <hydra/detail/external/thrust/detail/pointer.h>
#include <hydra/detail/external/thrust/detail/raw_pointer_cast.h>
#include <hydra/detail/external/thrust/detail/execute_with_allocator.h>
#include <hydra/detail/external/thrust/system/detail/generic/temporary_buffer.h>
#include <hydra/detail/external/thrust/system/detail/adl/temporary_buffer.h>

HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
{
namespace detail
{
namespace get_temporary_buffer_detail
{


template<typename T, typename DerivedPolicy, typename Pair>
__hydra_host__ __hydra_device__
  HYDRA_EXTERNAL_NS::thrust::pair<HYDRA_EXTERNAL_NS::thrust::pointer<T,DerivedPolicy>, typename HYDRA_EXTERNAL_NS::thrust::pointer<T,DerivedPolicy>::difference_type>
    down_cast_pair(Pair p)
{
  // XXX should use a hypothetical HYDRA_EXTERNAL_NS::thrust::static_pointer_cast here
  HYDRA_EXTERNAL_NS::thrust::pointer<T,DerivedPolicy> ptr = HYDRA_EXTERNAL_NS::thrust::pointer<T,DerivedPolicy>(static_cast<T*>(HYDRA_EXTERNAL_NS::thrust::raw_pointer_cast(p.first)));

  typedef HYDRA_EXTERNAL_NS::thrust::pair<HYDRA_EXTERNAL_NS::thrust::pointer<T,DerivedPolicy>, typename HYDRA_EXTERNAL_NS::thrust::pointer<T,DerivedPolicy>::difference_type> result_type;
  return result_type(ptr, p.second);
} // end down_cast_pair()


} // end get_temporary_buffer_detail
} // end detail


__thrust_exec_check_disable__
template<typename T, typename DerivedPolicy>
__hydra_host__ __hydra_device__
  HYDRA_EXTERNAL_NS::thrust::pair<HYDRA_EXTERNAL_NS::thrust::pointer<T,DerivedPolicy>, typename HYDRA_EXTERNAL_NS::thrust::pointer<T,DerivedPolicy>::difference_type>
    get_temporary_buffer(const HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> &exec, typename HYDRA_EXTERNAL_NS::thrust::pointer<T,DerivedPolicy>::difference_type n)
{
  using HYDRA_EXTERNAL_NS::thrust::detail::get_temporary_buffer; // execute_with_allocator
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::get_temporary_buffer;

  return HYDRA_EXTERNAL_NS::thrust::detail::get_temporary_buffer_detail::down_cast_pair<T,DerivedPolicy>(get_temporary_buffer<T>(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec)), n));
} // end get_temporary_buffer()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename Pointer>
__hydra_host__ __hydra_device__
  void return_temporary_buffer(const HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> &exec, Pointer p)
{
  using HYDRA_EXTERNAL_NS::thrust::detail::return_temporary_buffer; // execute_with_allocator
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::return_temporary_buffer;

  return return_temporary_buffer(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec)), p);
} // end return_temporary_buffer()


} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust


HYDRA_EXTERNAL_NAMESPACE_END
