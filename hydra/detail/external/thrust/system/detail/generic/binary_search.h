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


/*! \file binary_search.h
 *  \brief Generic implementations of binary search functions.
 */

#pragma once

#include <hydra/detail/external/thrust/detail/config.h>
#include <hydra/detail/external/thrust/system/detail/generic/tag.h>

HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
{
namespace system
{
namespace detail
{
namespace generic
{


template <typename DerivedPolicy, typename ForwardIterator, typename T>
__hydra_host__ __hydra_device__
ForwardIterator lower_bound(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec, 
                            ForwardIterator begin,
                            ForwardIterator end,
                            const T& value);

template <typename DerivedPolicy, typename ForwardIterator, typename T, typename StrictWeakOrdering>
__hydra_host__ __hydra_device__
ForwardIterator lower_bound(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                            ForwardIterator begin,
                            ForwardIterator end,
                            const T& value, 
                            StrictWeakOrdering comp);


template <typename DerivedPolicy, typename ForwardIterator, typename T>
__hydra_host__ __hydra_device__
ForwardIterator upper_bound(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                            ForwardIterator begin,
                            ForwardIterator end,
                            const T& value);

template <typename DerivedPolicy, typename ForwardIterator, typename T, typename StrictWeakOrdering>
__hydra_host__ __hydra_device__
ForwardIterator upper_bound(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec, 
                            ForwardIterator begin,
                            ForwardIterator end,
                            const T& value, 
                            StrictWeakOrdering comp);


template <typename DerivedPolicy, typename ForwardIterator, typename T>
__hydra_host__ __hydra_device__
bool binary_search(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                   ForwardIterator begin,
                   ForwardIterator end,
                   const T& value);

template <typename DerivedPolicy, typename ForwardIterator, typename T, typename StrictWeakOrdering>
__hydra_host__ __hydra_device__
bool binary_search(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                   ForwardIterator begin,
                   ForwardIterator end,
                   const T& value, 
                   StrictWeakOrdering comp);


template <typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename OutputIterator>
__hydra_host__ __hydra_device__
OutputIterator lower_bound(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                           ForwardIterator begin, 
                           ForwardIterator end,
                           InputIterator values_begin, 
                           InputIterator values_end,
                           OutputIterator output);


template <typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename OutputIterator, typename StrictWeakOrdering>
__hydra_host__ __hydra_device__
OutputIterator lower_bound(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                           ForwardIterator begin, 
                           ForwardIterator end,
                           InputIterator values_begin, 
                           InputIterator values_end,
                           OutputIterator output,
                           StrictWeakOrdering comp);


template <typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename OutputIterator>
__hydra_host__ __hydra_device__
OutputIterator upper_bound(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                           ForwardIterator begin, 
                           ForwardIterator end,
                           InputIterator values_begin, 
                           InputIterator values_end,
                           OutputIterator output);


template <typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename OutputIterator, typename StrictWeakOrdering>
__hydra_host__ __hydra_device__
OutputIterator upper_bound(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                           ForwardIterator begin, 
                           ForwardIterator end,
                           InputIterator values_begin, 
                           InputIterator values_end,
                           OutputIterator output,
                           StrictWeakOrdering comp);


template <typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename OutputIterator>
__hydra_host__ __hydra_device__
OutputIterator binary_search(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                             ForwardIterator begin, 
                             ForwardIterator end,
                             InputIterator values_begin, 
                             InputIterator values_end,
                             OutputIterator output);


template <typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename OutputIterator, typename StrictWeakOrdering>
__hydra_host__ __hydra_device__
OutputIterator binary_search(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                             ForwardIterator begin, 
                             ForwardIterator end,
                             InputIterator values_begin, 
                             InputIterator values_end,
                             OutputIterator output,
                             StrictWeakOrdering comp);


template <typename DerivedPolicy, typename ForwardIterator, typename LessThanComparable>
__hydra_host__ __hydra_device__
HYDRA_EXTERNAL_NS::thrust::pair<ForwardIterator,ForwardIterator>
equal_range(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
            ForwardIterator first,
            ForwardIterator last,
            const LessThanComparable &value);


template <typename DerivedPolicy, typename ForwardIterator, typename LessThanComparable, typename StrictWeakOrdering>
__hydra_host__ __hydra_device__
HYDRA_EXTERNAL_NS::thrust::pair<ForwardIterator,ForwardIterator>
equal_range(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
            ForwardIterator first,
            ForwardIterator last,
            const LessThanComparable &value,
            StrictWeakOrdering comp);



} // end namespace generic
} // end namespace detail
} // end namespace system
} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
HYDRA_EXTERNAL_NAMESPACE_END
#include <hydra/detail/external/thrust/system/detail/generic/binary_search.inl>

