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

HYDRA_THRUST_NAMESPACE_BEGIN

template<typename DerivedPolicy,
         typename InputIterator,
         typename EqualityComparable>
__host__ __device__
  typename hydra_thrust::iterator_traits<InputIterator>::difference_type
    count(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          InputIterator first,
          InputIterator last,
          const EqualityComparable& value);

template<typename DerivedPolicy,
         typename InputIterator,
         typename Predicate>
__host__ __device__
  typename hydra_thrust::iterator_traits<InputIterator>::difference_type
    count_if(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec,
             InputIterator first,
             InputIterator last,
             Predicate pred);

template <typename InputIterator,
          typename EqualityComparable>
  typename hydra_thrust::iterator_traits<InputIterator>::difference_type
    count(InputIterator first,
          InputIterator last,
          const EqualityComparable& value);

template <typename InputIterator,
          typename Predicate>
  typename hydra_thrust::iterator_traits<InputIterator>::difference_type
    count_if(InputIterator first,
             InputIterator last,
             Predicate pred);

HYDRA_THRUST_NAMESPACE_END

#include <hydra/detail/external/hydra_thrust/detail/count.inl>
