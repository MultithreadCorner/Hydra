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
#include <hydra/detail/external/hydra_thrust/system/detail/generic/tag.h>

HYDRA_THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{


template<typename DerivedPolicy,
         typename ForwardIterator>
__host__ __device__
ForwardIterator unique(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                       ForwardIterator first,
                       ForwardIterator last);


template<typename DerivedPolicy,
         typename ForwardIterator,
         typename BinaryPredicate>
__host__ __device__
ForwardIterator unique(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                       ForwardIterator first,
                       ForwardIterator last,
                       BinaryPredicate binary_pred);


template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator>
__host__ __device__
OutputIterator unique_copy(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                           InputIterator first,
                           InputIterator last,
                           OutputIterator output);


template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename BinaryPredicate>
__host__ __device__
OutputIterator unique_copy(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                           InputIterator first,
                           InputIterator last,
                           OutputIterator output,
                           BinaryPredicate binary_pred);


template<typename DerivedPolicy,
         typename ForwardIterator>
__host__ __device__
typename hydra_thrust::iterator_traits<ForwardIterator>::difference_type
    unique_count(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                 ForwardIterator first,
                 ForwardIterator last);


template<typename DerivedPolicy,
         typename ForwardIterator,
         typename BinaryPredicate>
__host__ __device__
typename hydra_thrust::iterator_traits<ForwardIterator>::difference_type
    unique_count(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                 ForwardIterator first,
                 ForwardIterator last,
                 BinaryPredicate binary_pred);


} // end namespace generic
} // end namespace detail
} // end namespace system
HYDRA_THRUST_NAMESPACE_END

#include <hydra/detail/external/hydra_thrust/system/detail/generic/unique.inl>

