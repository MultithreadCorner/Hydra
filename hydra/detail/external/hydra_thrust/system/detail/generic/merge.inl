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
#include <hydra/detail/external/hydra_thrust/detail/static_assert.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/merge.h>
#include <hydra/detail/external/hydra_thrust/merge.h>
#include <hydra/detail/external/hydra_thrust/functional.h>
#include <hydra/detail/external/hydra_thrust/iterator/zip_iterator.h>
#include <hydra/detail/external/hydra_thrust/detail/internal_functional.h>

namespace hydra_thrust
{
namespace system
{
namespace detail
{
namespace generic
{


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakOrdering>
__host__ __device__
  OutputIterator merge(hydra_thrust::execution_policy<DerivedPolicy> &,
                       InputIterator1,
                       InputIterator1,
                       InputIterator2,
                       InputIterator2,
                       OutputIterator result,
                       StrictWeakOrdering)
{
  HYDRA_THRUST_STATIC_ASSERT_MSG(
    (hydra_thrust::detail::depend_on_instantiation<InputIterator1, false>::value)
  , "unimplemented for this system"
  );
  return result;
} // end merge()


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
__host__ __device__
  OutputIterator merge(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                       InputIterator1 first1,
                       InputIterator1 last1,
                       InputIterator2 first2,
                       InputIterator2 last2,
                       OutputIterator result)
{
  typedef typename hydra_thrust::iterator_value<InputIterator1>::type value_type;
  return hydra_thrust::merge(exec,first1,last1,first2,last2,result,hydra_thrust::less<value_type>());
} // end merge()


template<typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename InputIterator3, typename InputIterator4, typename OutputIterator1, typename OutputIterator2, typename Compare>
__host__ __device__
  hydra_thrust::pair<OutputIterator1,OutputIterator2>
    merge_by_key(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                 InputIterator1 keys_first1, InputIterator1 keys_last1,
                 InputIterator2 keys_first2, InputIterator2 keys_last2,
                 InputIterator3 values_first1, InputIterator4 values_first2,
                 OutputIterator1 keys_result,
                 OutputIterator2 values_result,
                 Compare comp)
{
  typedef hydra_thrust::tuple<InputIterator1, InputIterator3>   iterator_tuple1;
  typedef hydra_thrust::tuple<InputIterator2, InputIterator4>   iterator_tuple2;
  typedef hydra_thrust::tuple<OutputIterator1, OutputIterator2> iterator_tuple3;

  typedef hydra_thrust::zip_iterator<iterator_tuple1> zip_iterator1;
  typedef hydra_thrust::zip_iterator<iterator_tuple2> zip_iterator2;
  typedef hydra_thrust::zip_iterator<iterator_tuple3> zip_iterator3;

  zip_iterator1 zipped_first1 = hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(keys_first1, values_first1));
  zip_iterator1 zipped_last1  = hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(keys_last1, values_first1));

  zip_iterator2 zipped_first2 = hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(keys_first2, values_first2));
  zip_iterator2 zipped_last2  = hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(keys_last2, values_first2));

  zip_iterator3 zipped_result = hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(keys_result, values_result));

  hydra_thrust::detail::compare_first<Compare> comp_first(comp);

  iterator_tuple3 result = hydra_thrust::merge(exec, zipped_first1, zipped_last1, zipped_first2, zipped_last2, zipped_result, comp_first).get_iterator_tuple();

  return hydra_thrust::make_pair(hydra_thrust::get<0>(result), hydra_thrust::get<1>(result));
} // end merge_by_key()


template<typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename InputIterator3, typename InputIterator4, typename OutputIterator1, typename OutputIterator2>
__host__ __device__
  hydra_thrust::pair<OutputIterator1,OutputIterator2>
    merge_by_key(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                 InputIterator1 keys_first1, InputIterator1 keys_last1,
                 InputIterator2 keys_first2, InputIterator2 keys_last2,
                 InputIterator3 values_first1, InputIterator4 values_first2,
                 OutputIterator1 keys_result,
                 OutputIterator2 values_result)
{
  typedef typename hydra_thrust::iterator_value<InputIterator1>::type value_type;
  return hydra_thrust::merge_by_key(exec, keys_first1, keys_last1, keys_first2, keys_last2, values_first1, values_first2, keys_result, values_result, hydra_thrust::less<value_type>());
} // end merge_by_key()


} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace hydra_thrust

