/*
 *  Copyright 2008-2021 NVIDIA Corporation
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

#include <hydra/detail/external/hydra_thrust/reverse.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/system/detail/sequential/stable_merge_sort.h>
#include <hydra/detail/external/hydra_thrust/system/detail/sequential/stable_primitive_sort.h>

#include <hydra/detail/external/hydra_libcudacxx/nv/target>

HYDRA_THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace sequential
{
namespace sort_detail
{


////////////////////
// Primitive Sort //
////////////////////


template<typename KeyType, typename Compare>
struct needs_reverse
  : hydra_thrust::detail::integral_constant<
      bool,
      hydra_thrust::detail::is_same<Compare, typename hydra_thrust::greater<KeyType> >::value
    >
{};


template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
__host__ __device__
void stable_sort(sequential::execution_policy<DerivedPolicy> &exec,
                 RandomAccessIterator first,
                 RandomAccessIterator last,
                 StrictWeakOrdering,
                 hydra_thrust::detail::true_type)
{
  hydra_thrust::system::detail::sequential::stable_primitive_sort(exec, first, last);

  // if comp is greater<T> then reverse the keys
  typedef typename hydra_thrust::iterator_traits<RandomAccessIterator>::value_type KeyType;

  if(needs_reverse<KeyType,StrictWeakOrdering>::value)
  {
    hydra_thrust::reverse(exec, first, last);
  }
}


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
__host__ __device__
void stable_sort_by_key(sequential::execution_policy<DerivedPolicy> &exec,
                        RandomAccessIterator1 first1,
                        RandomAccessIterator1 last1,
                        RandomAccessIterator2 first2,
                        StrictWeakOrdering,
                        hydra_thrust::detail::true_type)
{
  // if comp is greater<T> then reverse the keys and values
  typedef typename hydra_thrust::iterator_traits<RandomAccessIterator1>::value_type KeyType;

  // note, we also have to reverse the (unordered) input to preserve stability
  if(needs_reverse<KeyType,StrictWeakOrdering>::value)
  {
    hydra_thrust::reverse(exec, first1,  last1);
    hydra_thrust::reverse(exec, first2, first2 + (last1 - first1));
  }

  hydra_thrust::system::detail::sequential::stable_primitive_sort_by_key(exec, first1, last1, first2);

  if(needs_reverse<KeyType,StrictWeakOrdering>::value)
  {
    hydra_thrust::reverse(exec, first1,  last1);
    hydra_thrust::reverse(exec, first2, first2 + (last1 - first1));
  }
}


////////////////
// Merge Sort //
////////////////


template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
__host__ __device__
void stable_sort(sequential::execution_policy<DerivedPolicy> &exec,
                 RandomAccessIterator first,
                 RandomAccessIterator last,
                 StrictWeakOrdering comp,
                 hydra_thrust::detail::false_type)
{
  hydra_thrust::system::detail::sequential::stable_merge_sort(exec, first, last, comp);
}


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
__host__ __device__
void stable_sort_by_key(sequential::execution_policy<DerivedPolicy> &exec,
                        RandomAccessIterator1 first1,
                        RandomAccessIterator1 last1,
                        RandomAccessIterator2 first2,
                        StrictWeakOrdering comp,
                        hydra_thrust::detail::false_type)
{
  hydra_thrust::system::detail::sequential::stable_merge_sort_by_key(exec, first1, last1, first2, comp);
}


template<typename KeyType, typename Compare>
struct use_primitive_sort
  : hydra_thrust::detail::and_<
      hydra_thrust::detail::is_arithmetic<KeyType>,
      hydra_thrust::detail::or_<
        hydra_thrust::detail::is_same<Compare, hydra_thrust::less<KeyType> >,
        hydra_thrust::detail::is_same<Compare, hydra_thrust::greater<KeyType> >
      >
    >
{};


} // end namespace sort_detail


template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
__host__ __device__
void stable_sort(sequential::execution_policy<DerivedPolicy> &exec,
                 RandomAccessIterator first,
                 RandomAccessIterator last,
                 StrictWeakOrdering comp)
{

  // the compilation time of stable_primitive_sort is too expensive to use within a single CUDA thread
  NV_IF_TARGET(NV_IS_HOST, (
    using KeyType = hydra_thrust::iterator_value_t<RandomAccessIterator>;
    sort_detail::use_primitive_sort<KeyType, StrictWeakOrdering> use_primitive_sort;
    sort_detail::stable_sort(exec, first, last, comp, use_primitive_sort);
  ), ( // NV_IS_DEVICE:
    hydra_thrust::detail::false_type use_primitive_sort;
    sort_detail::stable_sort(exec, first, last, comp, use_primitive_sort);
  ));
}


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
__host__ __device__
void stable_sort_by_key(sequential::execution_policy<DerivedPolicy> &exec,
                        RandomAccessIterator1 first1,
                        RandomAccessIterator1 last1,
                        RandomAccessIterator2 first2,
                        StrictWeakOrdering comp)
{

  // the compilation time of stable_primitive_sort_by_key is too expensive to use within a single CUDA thread
  NV_IF_TARGET(NV_IS_HOST, (
    using KeyType = hydra_thrust::iterator_value_t<RandomAccessIterator1>;
    sort_detail::use_primitive_sort<KeyType, StrictWeakOrdering> use_primitive_sort;
    sort_detail::stable_sort_by_key(exec, first1, last1, first2, comp, use_primitive_sort);
  ), ( // NV_IS_DEVICE:
    hydra_thrust::detail::false_type use_primitive_sort;
    sort_detail::stable_sort_by_key(exec, first1, last1, first2, comp, use_primitive_sort);
  ));
}


} // end namespace sequential
} // end namespace detail
} // end namespace system
HYDRA_THRUST_NAMESPACE_END

