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

#include <hydra/detail/external/hydra_thrust/swap.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/select_system.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/swap_ranges.h>
#include <hydra/detail/external/hydra_thrust/system/detail/adl/swap_ranges.h>

HYDRA_THRUST_NAMESPACE_BEGIN


__hydra_thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename ForwardIterator1,
         typename ForwardIterator2>
__host__ __device__
  ForwardIterator2 swap_ranges(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                               ForwardIterator1 first1,
                               ForwardIterator1 last1,
                               ForwardIterator2 first2)
{
  using hydra_thrust::system::detail::generic::swap_ranges;
  return swap_ranges(hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)), first1, last1, first2);
} // end swap_ranges()


template<typename ForwardIterator1,
         typename ForwardIterator2>
  ForwardIterator2 swap_ranges(ForwardIterator1 first1,
                               ForwardIterator1 last1,
                               ForwardIterator2 first2)
{
  using hydra_thrust::system::detail::generic::select_system;

  typedef typename hydra_thrust::iterator_system<ForwardIterator1>::type System1;
  typedef typename hydra_thrust::iterator_system<ForwardIterator2>::type System2;

  System1 system1;
  System2 system2;

  return hydra_thrust::swap_ranges(select_system(system1,system2), first1, last1, first2);
} // end swap_ranges()


HYDRA_THRUST_NAMESPACE_END

