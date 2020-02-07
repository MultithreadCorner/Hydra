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
#include <hydra/detail/external/hydra_thrust/iterator/transform_iterator.h>
#include <hydra/detail/external/hydra_thrust/iterator/zip_iterator.h>
#include <hydra/detail/external/hydra_thrust/iterator/counting_iterator.h>
#include <hydra/detail/external/hydra_thrust/tuple.h>
#include <hydra/detail/external/hydra_thrust/functional.h>

namespace hydra_thrust
{
namespace detail
{


template<typename RandomAccessIterator,
         typename BinaryPredicate = hydra_thrust::equal_to<typename hydra_thrust::iterator_value<RandomAccessIterator>::type>,
         typename ValueType = bool,
         typename IndexType = typename hydra_thrust::iterator_difference<RandomAccessIterator>::type>
  class tail_flags
{
  // XXX WAR cudafe bug
  //private:
  public:
    struct tail_flag_functor
    {
      BinaryPredicate binary_pred; // this must be the first member for performance reasons
      RandomAccessIterator iter;
      IndexType n;

      typedef ValueType result_type;

      __host__ __device__
      tail_flag_functor(RandomAccessIterator first, RandomAccessIterator last)
        : binary_pred(), iter(first), n(last - first)
      {}

      __host__ __device__
      tail_flag_functor(RandomAccessIterator first, RandomAccessIterator last, BinaryPredicate binary_pred)
        : binary_pred(binary_pred), iter(first), n(last - first)
      {}

      __host__ __device__ __hydra_thrust_forceinline__
      result_type operator()(const IndexType &i)
      {
        return (i == (n - 1) || !binary_pred(iter[i], iter[i+1]));
      }
    };

    typedef hydra_thrust::counting_iterator<IndexType> counting_iterator;

  public:
    typedef hydra_thrust::transform_iterator<
      tail_flag_functor,
      counting_iterator
    > iterator;

    __hydra_thrust_exec_check_disable__
    __host__ __device__
    tail_flags(RandomAccessIterator first, RandomAccessIterator last)
      : m_begin(hydra_thrust::make_transform_iterator(hydra_thrust::counting_iterator<IndexType>(0),
                                                tail_flag_functor(first, last))),
        m_end(m_begin + (last - first))
    {}

    __hydra_thrust_exec_check_disable__
    __host__ __device__
    tail_flags(RandomAccessIterator first, RandomAccessIterator last, BinaryPredicate binary_pred)
      : m_begin(hydra_thrust::make_transform_iterator(hydra_thrust::counting_iterator<IndexType>(0),
                                                tail_flag_functor(first, last, binary_pred))),
        m_end(m_begin + (last - first))
    {}

    __host__ __device__
    iterator begin() const
    {
      return m_begin;
    }

    __host__ __device__
    iterator end() const
    {
      return m_end;
    }

    template<typename OtherIndex>
    __host__ __device__
    typename iterator::reference operator[](OtherIndex i)
    {
      return *(begin() + i);
    }

  private:
    iterator m_begin, m_end;
};


template<typename RandomAccessIterator, typename BinaryPredicate>
__host__ __device__
tail_flags<RandomAccessIterator, BinaryPredicate>
  make_tail_flags(RandomAccessIterator first, RandomAccessIterator last, BinaryPredicate binary_pred)
{
  return tail_flags<RandomAccessIterator, BinaryPredicate>(first, last, binary_pred);
}


template<typename RandomAccessIterator>
__host__ __device__
tail_flags<RandomAccessIterator>
  make_tail_flags(RandomAccessIterator first, RandomAccessIterator last)
{
  return tail_flags<RandomAccessIterator>(first, last);
}


} // end detail
} // end hydra_thrust

