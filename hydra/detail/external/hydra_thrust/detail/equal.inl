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
#include <hydra/detail/external/hydra_thrust/equal.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/select_system.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/equal.h>
#include <hydra/detail/external/hydra_thrust/system/detail/adl/equal.h>

HYDRA_THRUST_NAMESPACE_BEGIN

__hydra_thrust_exec_check_disable__
template<typename System, typename InputIterator1, typename InputIterator2>
__host__ __device__
bool equal(const hydra_thrust::detail::execution_policy_base<System> &system, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2)
{
  using hydra_thrust::system::detail::generic::equal;
  return equal(hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(system)), first1, last1, first2);
} // end equal()


__hydra_thrust_exec_check_disable__
template<typename System, typename InputIterator1, typename InputIterator2, typename BinaryPredicate>
__host__ __device__
bool equal(const hydra_thrust::detail::execution_policy_base<System> &system, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, BinaryPredicate binary_pred)
{
  using hydra_thrust::system::detail::generic::equal;
  return equal(hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(system)), first1, last1, first2, binary_pred);
} // end equal()


template <typename InputIterator1, typename InputIterator2>
bool equal(InputIterator1 first1, InputIterator1 last1,
           InputIterator2 first2)
{
  using hydra_thrust::system::detail::generic::select_system;

  typedef typename hydra_thrust::iterator_system<InputIterator1>::type System1;
  typedef typename hydra_thrust::iterator_system<InputIterator2>::type System2;

  System1 system1;
  System2 system2;

  return hydra_thrust::equal(select_system(system1,system2), first1, last1, first2);
}


template <typename InputIterator1, typename InputIterator2,
          typename BinaryPredicate>
bool equal(InputIterator1 first1, InputIterator1 last1,
           InputIterator2 first2, BinaryPredicate binary_pred)
{
  using hydra_thrust::system::detail::generic::select_system;

  typedef typename hydra_thrust::iterator_system<InputIterator1>::type System1;
  typedef typename hydra_thrust::iterator_system<InputIterator2>::type System2;

  System1 system1;
  System2 system2;

  return hydra_thrust::equal(select_system(system1,system2), first1, last1, first2, binary_pred);
}

HYDRA_THRUST_NAMESPACE_END
