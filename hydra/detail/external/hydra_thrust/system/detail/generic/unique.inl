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
#include <hydra/detail/external/hydra_thrust/system/detail/generic/unique.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/transform.h>
#include <hydra/detail/external/hydra_thrust/unique.h>
#include <hydra/detail/external/hydra_thrust/detail/temporary_array.h>
#include <hydra/detail/external/hydra_thrust/detail/internal_functional.h>
#include <hydra/detail/external/hydra_thrust/detail/copy_if.h>
#include <hydra/detail/external/hydra_thrust/detail/count.h>
#include <hydra/detail/external/hydra_thrust/distance.h>
#include <hydra/detail/external/hydra_thrust/functional.h>
#include <hydra/detail/external/hydra_thrust/detail/range/head_flags.h>

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
                         ForwardIterator last)
{
  typedef typename hydra_thrust::iterator_traits<ForwardIterator>::value_type InputType;

  return hydra_thrust::unique(exec, first, last, hydra_thrust::equal_to<InputType>());
} // end unique()


template<typename DerivedPolicy,
         typename ForwardIterator,
         typename BinaryPredicate>
__host__ __device__
  ForwardIterator unique(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                         ForwardIterator first,
                         ForwardIterator last,
                         BinaryPredicate binary_pred)
{
  typedef typename hydra_thrust::iterator_traits<ForwardIterator>::value_type InputType;

  hydra_thrust::detail::temporary_array<InputType,DerivedPolicy> input(exec, first, last);

  return hydra_thrust::unique_copy(exec, input.begin(), input.end(), first, binary_pred);
} // end unique()


template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator>
__host__ __device__
  OutputIterator unique_copy(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                             InputIterator first,
                             InputIterator last,
                             OutputIterator output)
{
  typedef typename hydra_thrust::iterator_value<InputIterator>::type value_type;
  return hydra_thrust::unique_copy(exec, first,last,output,hydra_thrust::equal_to<value_type>());
} // end unique_copy()


template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename BinaryPredicate>
__host__ __device__
  OutputIterator unique_copy(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                             InputIterator first,
                             InputIterator last,
                             OutputIterator output,
                             BinaryPredicate binary_pred)
{
  hydra_thrust::detail::head_flags<InputIterator, BinaryPredicate> stencil(first, last, binary_pred);

  using namespace hydra_thrust::placeholders;

  return hydra_thrust::copy_if(exec, first, last, stencil.begin(), output, _1);
} // end unique_copy()


template<typename DerivedPolicy,
         typename ForwardIterator,
         typename BinaryPredicate>
__host__ __device__
  typename hydra_thrust::iterator_traits<ForwardIterator>::difference_type
    unique_count(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                 ForwardIterator first,
                 ForwardIterator last,
                 BinaryPredicate binary_pred)
{
  hydra_thrust::detail::head_flags<ForwardIterator, BinaryPredicate> stencil(first, last, binary_pred);
  
  using namespace hydra_thrust::placeholders;
  
  return hydra_thrust::count_if(exec, stencil.begin(), stencil.end(), _1);
} // end unique_copy()


template<typename DerivedPolicy,
         typename ForwardIterator>
__host__ __device__
  typename hydra_thrust::iterator_traits<ForwardIterator>::difference_type
    unique_count(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                 ForwardIterator first,
                 ForwardIterator last)
{
  typedef typename hydra_thrust::iterator_value<ForwardIterator>::type value_type;
  return hydra_thrust::unique_count(exec, first, last, hydra_thrust::equal_to<value_type>());
} // end unique_copy()


} // end namespace generic
} // end namespace detail
} // end namespace system
HYDRA_THRUST_NAMESPACE_END

