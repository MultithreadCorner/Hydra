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
#include <hydra/detail/external/hydra_thrust/pair.h>
#include <hydra/detail/external/hydra_thrust/detail/function.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>

namespace hydra_thrust
{

namespace system
{

namespace detail
{

namespace generic
{

namespace scalar
{

template<typename RandomAccessIterator, typename Size, typename T, typename BinaryPredicate>
__host__ __device__
RandomAccessIterator lower_bound_n(RandomAccessIterator first,
                                   Size n,
                                   const T &val,
                                   BinaryPredicate comp)
{
  // wrap comp
  hydra_thrust::detail::wrapped_function<
    BinaryPredicate,
    bool
  > wrapped_comp(comp);

  Size start = 0, i;
  while(start < n)
  {
    i = (start + n) / 2;
    if(wrapped_comp(first[i], val))
    {
      start = i + 1;
    }
    else
    {
      n = i;
    }
  } // end while
  
  return first + start;
}

// XXX generalize these upon implementation of scalar::distance & scalar::advance

template<typename RandomAccessIterator, typename T, typename BinaryPredicate>
__host__ __device__
RandomAccessIterator lower_bound(RandomAccessIterator first, RandomAccessIterator last,
                                 const T &val,
                                 BinaryPredicate comp)
{
  typename hydra_thrust::iterator_difference<RandomAccessIterator>::type n = last - first;
  return lower_bound_n(first, n, val, comp);
}

template<typename RandomAccessIterator, typename Size, typename T, typename BinaryPredicate>
__host__ __device__
RandomAccessIterator upper_bound_n(RandomAccessIterator first,
                                   Size n,
                                   const T &val,
                                   BinaryPredicate comp)
{
  // wrap comp
  hydra_thrust::detail::wrapped_function<
    BinaryPredicate,
    bool
  > wrapped_comp(comp);

  Size start = 0, i;
  while(start < n)
  {
    i = (start + n) / 2;
    if(wrapped_comp(val, first[i]))
    {
      n = i;
    }
    else
    {
      start = i + 1;
    }
  } // end while
  
  return first + start;
}

template<typename RandomAccessIterator, typename T, typename BinaryPredicate>
__host__ __device__
RandomAccessIterator upper_bound(RandomAccessIterator first, RandomAccessIterator last,
                                 const T &val,
                                 BinaryPredicate comp)
{
  typename hydra_thrust::iterator_difference<RandomAccessIterator>::type n = last - first;
  return upper_bound_n(first, n, val, comp);
}

template<typename RandomAccessIterator, typename T, typename BinaryPredicate>
__host__ __device__
  pair<RandomAccessIterator,RandomAccessIterator>
    equal_range(RandomAccessIterator first, RandomAccessIterator last,
                const T &val,
                BinaryPredicate comp)
{
  RandomAccessIterator lb = hydra_thrust::system::detail::generic::scalar::lower_bound(first, last, val, comp);
  return hydra_thrust::make_pair(lb, hydra_thrust::system::detail::generic::scalar::upper_bound(lb, last, val, comp));
}


template<typename RandomAccessIterator, typename T, typename Compare>
__host__ __device__
bool binary_search(RandomAccessIterator first, RandomAccessIterator last, const T &value, Compare comp)
{
  RandomAccessIterator iter = hydra_thrust::system::detail::generic::scalar::lower_bound(first, last, value, comp);

  // wrap comp
  hydra_thrust::detail::wrapped_function<
    Compare,
    bool
  > wrapped_comp(comp);

  return iter != last && !wrapped_comp(value,*iter);
}

} // end scalar

} // end generic

} // end detail

} // end system

} // end hydra_thrust

#include <hydra/detail/external/hydra_thrust/system/detail/generic/scalar/binary_search.inl>

