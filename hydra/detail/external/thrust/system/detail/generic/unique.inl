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


/*! \file unique.inl
 *  \brief Inline file for unique.h.
 */

#pragma once

#include <hydra/detail/external/thrust/detail/config.h>
#include <hydra/detail/external/thrust/system/detail/generic/unique.h>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/thrust/transform.h>
#include <hydra/detail/external/thrust/unique.h>
#include <hydra/detail/external/thrust/detail/temporary_array.h>
#include <hydra/detail/external/thrust/detail/internal_functional.h>
#include <hydra/detail/external/thrust/detail/copy_if.h>
#include <hydra/detail/external/thrust/distance.h>
#include <hydra/detail/external/thrust/functional.h>
#include <hydra/detail/external/thrust/detail/range/head_flags.h>

HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
{
namespace system
{
namespace detail
{
namespace generic
{


template<typename DerivedPolicy,
         typename ForwardIterator>
__hydra_host__ __hydra_device__
  ForwardIterator unique(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                         ForwardIterator first,
                         ForwardIterator last)
{
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<ForwardIterator>::value_type InputType;

  return HYDRA_EXTERNAL_NS::thrust::unique(exec, first, last, HYDRA_EXTERNAL_NS::thrust::equal_to<InputType>());
} // end unique()


template<typename DerivedPolicy,
         typename ForwardIterator,
         typename BinaryPredicate>
__hydra_host__ __hydra_device__
  ForwardIterator unique(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                         ForwardIterator first,
                         ForwardIterator last,
                         BinaryPredicate binary_pred)
{
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<ForwardIterator>::value_type InputType;
  
  HYDRA_EXTERNAL_NS::thrust::detail::temporary_array<InputType,DerivedPolicy> input(exec, first, last);
  
  return HYDRA_EXTERNAL_NS::thrust::unique_copy(exec, input.begin(), input.end(), first, binary_pred);
} // end unique()


template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator>
__hydra_host__ __hydra_device__
  OutputIterator unique_copy(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                             InputIterator first,
                             InputIterator last,
                             OutputIterator output)
{
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_value<InputIterator>::type value_type;
  return HYDRA_EXTERNAL_NS::thrust::unique_copy(exec, first,last,output,HYDRA_EXTERNAL_NS::thrust::equal_to<value_type>());
} // end unique_copy()


template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename BinaryPredicate>
__hydra_host__ __hydra_device__
  OutputIterator unique_copy(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                             InputIterator first,
                             InputIterator last,
                             OutputIterator output,
                             BinaryPredicate binary_pred)
{
  HYDRA_EXTERNAL_NS::thrust::detail::head_flags<InputIterator, BinaryPredicate> stencil(first, last, binary_pred);
  
  using  namespace HYDRA_EXTERNAL_NS::thrust::placeholders;
  
  return HYDRA_EXTERNAL_NS::thrust::copy_if(exec, first, last, stencil.begin(), output, _1);
} // end unique_copy()


} // end namespace generic
} // end namespace detail
} // end namespace system
} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust

HYDRA_EXTERNAL_NAMESPACE_END
