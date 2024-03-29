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
#include <hydra/detail/external/hydra_thrust/extrema.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/select_system.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/extrema.h>
#include <hydra/detail/external/hydra_thrust/system/detail/adl/extrema.h>

HYDRA_THRUST_NAMESPACE_BEGIN

__hydra_thrust_exec_check_disable__
template<typename DerivedPolicy, typename ForwardIterator>
__host__ __device__
ForwardIterator min_element(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec, ForwardIterator first, ForwardIterator last)
{
  using hydra_thrust::system::detail::generic::min_element;
  return min_element(hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)), first, last);
} // end min_element()


__hydra_thrust_exec_check_disable__
template<typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
__host__ __device__
ForwardIterator min_element(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec, ForwardIterator first, ForwardIterator last, BinaryPredicate comp)
{
  using hydra_thrust::system::detail::generic::min_element;
  return min_element(hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)), first, last, comp);
} // end min_element()


__hydra_thrust_exec_check_disable__
template<typename DerivedPolicy, typename ForwardIterator>
__host__ __device__
ForwardIterator max_element(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec, ForwardIterator first, ForwardIterator last)
{
  using hydra_thrust::system::detail::generic::max_element;
  return max_element(hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)), first, last);
} // end max_element()


__hydra_thrust_exec_check_disable__
template<typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
__host__ __device__
ForwardIterator max_element(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec, ForwardIterator first, ForwardIterator last, BinaryPredicate comp)
{
  using hydra_thrust::system::detail::generic::max_element;
  return max_element(hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)), first, last, comp);
} // end max_element()


__hydra_thrust_exec_check_disable__
template<typename DerivedPolicy, typename ForwardIterator>
__host__ __device__
hydra_thrust::pair<ForwardIterator,ForwardIterator> minmax_element(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec, ForwardIterator first, ForwardIterator last)
{
  using hydra_thrust::system::detail::generic::minmax_element;
  return minmax_element(hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)), first, last);
} // end minmax_element()


__hydra_thrust_exec_check_disable__
template<typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
__host__ __device__
hydra_thrust::pair<ForwardIterator,ForwardIterator> minmax_element(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec, ForwardIterator first, ForwardIterator last, BinaryPredicate comp)
{
  using hydra_thrust::system::detail::generic::minmax_element;
  return minmax_element(hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)), first, last, comp);
} // end minmax_element()


template <typename ForwardIterator>
ForwardIterator min_element(ForwardIterator first, ForwardIterator last)
{
  using hydra_thrust::system::detail::generic::select_system;

  typedef typename hydra_thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return hydra_thrust::min_element(select_system(system), first, last);
} // end min_element()


template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator min_element(ForwardIterator first, ForwardIterator last,
                            BinaryPredicate comp)
{
  using hydra_thrust::system::detail::generic::select_system;

  typedef typename hydra_thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return hydra_thrust::min_element(select_system(system), first, last, comp);
} // end min_element()


template <typename ForwardIterator>
ForwardIterator max_element(ForwardIterator first, ForwardIterator last)
{
  using hydra_thrust::system::detail::generic::select_system;

  typedef typename hydra_thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return hydra_thrust::max_element(select_system(system), first, last);
} // end max_element()


template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator max_element(ForwardIterator first, ForwardIterator last,
                            BinaryPredicate comp)
{
  using hydra_thrust::system::detail::generic::select_system;

  typedef typename hydra_thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return hydra_thrust::max_element(select_system(system), first, last, comp);
} // end max_element()


template <typename ForwardIterator>
hydra_thrust::pair<ForwardIterator,ForwardIterator>
minmax_element(ForwardIterator first, ForwardIterator last)
{
  using hydra_thrust::system::detail::generic::select_system;

  typedef typename hydra_thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return hydra_thrust::minmax_element(select_system(system), first, last);
} // end minmax_element()


template <typename ForwardIterator, typename BinaryPredicate>
hydra_thrust::pair<ForwardIterator,ForwardIterator>
minmax_element(ForwardIterator first, ForwardIterator last, BinaryPredicate comp)
{
  using hydra_thrust::system::detail::generic::select_system;

  typedef typename hydra_thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return hydra_thrust::minmax_element(select_system(system), first, last, comp);
} // end minmax_element()

HYDRA_THRUST_NAMESPACE_END
