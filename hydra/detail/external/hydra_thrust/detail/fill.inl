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

#include <hydra/detail/external/hydra_thrust/fill.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/select_system.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/fill.h>
#include <hydra/detail/external/hydra_thrust/system/detail/adl/fill.h>

HYDRA_THRUST_NAMESPACE_BEGIN

__hydra_thrust_exec_check_disable__
template<typename DerivedPolicy, typename ForwardIterator, typename T>
__host__ __device__
  void fill(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec,
            ForwardIterator first,
            ForwardIterator last,
            const T &value)
{
  using hydra_thrust::system::detail::generic::fill;
  return fill(hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)), first, last, value);
} // end fill()


__hydra_thrust_exec_check_disable__
template<typename DerivedPolicy, typename OutputIterator, typename Size, typename T>
__host__ __device__
  OutputIterator fill_n(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                        OutputIterator first,
                        Size n,
                        const T &value)
{
  using hydra_thrust::system::detail::generic::fill_n;
  return fill_n(hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)), first, n, value);
} // end fill_n()


template<typename ForwardIterator, typename T>
__host__ __device__
  void fill(ForwardIterator first,
            ForwardIterator last,
            const T &value)
{
  using hydra_thrust::system::detail::generic::select_system;

  typedef typename hydra_thrust::iterator_system<ForwardIterator>::type System;

  System system;

  hydra_thrust::fill(select_system(system), first, last, value);
} // end fill()


template<typename OutputIterator, typename Size, typename T>
__host__ __device__
  OutputIterator fill_n(OutputIterator first,
                        Size n,
                        const T &value)
{
  using hydra_thrust::system::detail::generic::select_system;

  typedef typename hydra_thrust::iterator_system<OutputIterator>::type System;

  System system;

  return hydra_thrust::fill_n(select_system(system), first, n, value);
} // end fill()

HYDRA_THRUST_NAMESPACE_END
