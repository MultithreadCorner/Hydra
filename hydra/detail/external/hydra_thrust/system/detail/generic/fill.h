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

#include <hydra/detail/external/hydra_thrust/detail/internal_functional.h>
#include <hydra/detail/external/hydra_thrust/generate.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/tag.h>

HYDRA_THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{


template<typename DerivedPolicy, typename OutputIterator, typename Size, typename T>
__host__ __device__
  OutputIterator fill_n(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                        OutputIterator first,
                        Size n,
                        const T &value)
{
  // XXX consider using the placeholder expression _1 = value
  return hydra_thrust::generate_n(exec, first, n, hydra_thrust::detail::fill_functor<T>(value));
}

template<typename DerivedPolicy, typename ForwardIterator, typename T>
__host__ __device__
  void fill(hydra_thrust::execution_policy<DerivedPolicy> &exec,
            ForwardIterator first,
            ForwardIterator last,
            const T &value)
{
  // XXX consider using the placeholder expression _1 = value
  hydra_thrust::generate(exec, first, last, hydra_thrust::detail::fill_functor<T>(value));
}


} // end namespace generic
} // end namespace detail
} // end namespace system
HYDRA_THRUST_NAMESPACE_END

