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

#include <hydra/detail/external/thrust/detail/config.h>
#include <hydra/detail/external/thrust/iterator/transform_iterator.h>
#include <hydra/detail/external/thrust/iterator/zip_iterator.h>
#include <hydra/detail/external/thrust/iterator/counting_iterator.h>
#include <hydra/detail/external/thrust/tuple.h>
#include <hydra/detail/external/thrust/functional.h>


HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
{
namespace detail
{


template<typename RandomAccessIterator,
         typename BinaryPredicate = HYDRA_EXTERNAL_NS::thrust::equal_to<typename HYDRA_EXTERNAL_NS::thrust::iterator_value<RandomAccessIterator>::type>,
         typename ValueType = bool,
         typename IndexType = typename HYDRA_EXTERNAL_NS::thrust::iterator_difference<RandomAccessIterator>::type>
  class head_flags_with_init
{
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_value<RandomAccessIterator>::type init_type;

  // XXX WAR cudafe issue
  //private:
  public:
    struct head_flag_functor
    {
      BinaryPredicate binary_pred; // this must be the first member for performance reasons
      init_type init;
      IndexType n;

      typedef ValueType result_type;

      __hydra_host__ __hydra_device__
      head_flag_functor(init_type init, IndexType n)
        : binary_pred(), init(init), n(n)
      {}

      __hydra_host__ __hydra_device__
      head_flag_functor(init_type init, IndexType n, BinaryPredicate binary_pred)
        : binary_pred(binary_pred), init(init), n(n)
      {}

      template<typename Tuple>
      __hydra_host__ __hydra_device__ __thrust_forceinline__
      result_type operator()(const Tuple &t)
      {
        const IndexType i = HYDRA_EXTERNAL_NS::thrust::get<0>(t);

        if(i == 0)
        {
          return !binary_pred(init, HYDRA_EXTERNAL_NS::thrust::get<1>(t));
        }

        return !binary_pred(HYDRA_EXTERNAL_NS::thrust::get<1>(t), HYDRA_EXTERNAL_NS::thrust::get<2>(t));
      }
    };

    typedef HYDRA_EXTERNAL_NS::thrust::counting_iterator<IndexType> counting_iterator;

  public:
    typedef HYDRA_EXTERNAL_NS::thrust::transform_iterator<
      head_flag_functor,
      HYDRA_EXTERNAL_NS::thrust::zip_iterator<HYDRA_EXTERNAL_NS::thrust::tuple<counting_iterator,RandomAccessIterator,RandomAccessIterator> >
    > iterator;

    __thrust_exec_check_disable__
    __hydra_host__ __hydra_device__
    head_flags_with_init(RandomAccessIterator first, RandomAccessIterator last, init_type init)
      : m_begin(HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(HYDRA_EXTERNAL_NS::thrust::counting_iterator<IndexType>(0), first, first - 1)),
                                                head_flag_functor(init, last - first))),
        m_end(m_begin + (last - first))
    {}

    __thrust_exec_check_disable__
    __hydra_host__ __hydra_device__
    head_flags_with_init(RandomAccessIterator first, RandomAccessIterator last, init_type init, BinaryPredicate binary_pred)
      : m_begin(HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(HYDRA_EXTERNAL_NS::thrust::counting_iterator<IndexType>(0), first, first - 1)),
                                                head_flag_functor(init, last - first, binary_pred))),
        m_end(m_begin + (last - first))
    {}

    __hydra_host__ __hydra_device__
    iterator begin() const
    {
      return m_begin;
    }

    __hydra_host__ __hydra_device__
    iterator end() const
    {
      return m_end;
    }

    template<typename OtherIndex>
    __hydra_host__ __hydra_device__
    typename iterator::reference operator[](OtherIndex i)
    {
      return *(begin() + i);
    }

  private:
    iterator m_begin, m_end;
};



template<typename RandomAccessIterator,
         typename BinaryPredicate = HYDRA_EXTERNAL_NS::thrust::equal_to<typename HYDRA_EXTERNAL_NS::thrust::iterator_value<RandomAccessIterator>::type>,
         typename ValueType = bool,
         typename IndexType = typename HYDRA_EXTERNAL_NS::thrust::iterator_difference<RandomAccessIterator>::type>
  class head_flags
{
  // XXX WAR cudafe issue
  //private:
  public:
    struct head_flag_functor
    {
      BinaryPredicate binary_pred; // this must be the first member for performance reasons
      IndexType n;

      typedef ValueType result_type;

      __hydra_host__ __hydra_device__
      head_flag_functor(IndexType n)
        : binary_pred(), n(n)
      {}

      __hydra_host__ __hydra_device__
      head_flag_functor(IndexType n, BinaryPredicate binary_pred)
        : binary_pred(binary_pred), n(n)
      {}

      template<typename Tuple>
      __hydra_host__ __hydra_device__ __thrust_forceinline__
      result_type operator()(const Tuple &t)
      {
        const IndexType i = HYDRA_EXTERNAL_NS::thrust::get<0>(t);

        // note that we do not dereference the tuple's 2nd element when i <= 0
        // and therefore do not dereference a bad location at the boundary
        return (i == 0 || !binary_pred(HYDRA_EXTERNAL_NS::thrust::get<1>(t), HYDRA_EXTERNAL_NS::thrust::get<2>(t)));
      }
    };

    typedef HYDRA_EXTERNAL_NS::thrust::counting_iterator<IndexType> counting_iterator;

  public:
    typedef HYDRA_EXTERNAL_NS::thrust::transform_iterator<
      head_flag_functor,
      HYDRA_EXTERNAL_NS::thrust::zip_iterator<HYDRA_EXTERNAL_NS::thrust::tuple<counting_iterator,RandomAccessIterator,RandomAccessIterator> >
    > iterator;

    __hydra_host__ __hydra_device__
    head_flags(RandomAccessIterator first, RandomAccessIterator last)
      : m_begin(HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(HYDRA_EXTERNAL_NS::thrust::counting_iterator<IndexType>(0), first, first - 1)),
                                                head_flag_functor(last - first))),
        m_end(m_begin + (last - first))
    {}

    __hydra_host__ __hydra_device__
    head_flags(RandomAccessIterator first, RandomAccessIterator last, BinaryPredicate binary_pred)
      : m_begin(HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(HYDRA_EXTERNAL_NS::thrust::counting_iterator<IndexType>(0), first, first - 1)),
                                                head_flag_functor(last - first, binary_pred))),
        m_end(m_begin + (last - first))
    {}

    __hydra_host__ __hydra_device__
    iterator begin() const
    {
      return m_begin;
    }

    __hydra_host__ __hydra_device__
    iterator end() const
    {
      return m_end;
    }

    template<typename OtherIndex>
    __hydra_host__ __hydra_device__
    typename iterator::reference operator[](OtherIndex i)
    {
      return *(begin() + i);
    }

  private:
    iterator m_begin, m_end;
};


template<typename RandomAccessIterator, typename BinaryPredicate>
__hydra_host__ __hydra_device__
head_flags<RandomAccessIterator, BinaryPredicate>
  make_head_flags(RandomAccessIterator first, RandomAccessIterator last, BinaryPredicate binary_pred)
{
  return head_flags<RandomAccessIterator, BinaryPredicate>(first, last, binary_pred);
}


template<typename RandomAccessIterator>
__hydra_host__ __hydra_device__
head_flags<RandomAccessIterator>
  make_head_flags(RandomAccessIterator first, RandomAccessIterator last)
{
  return head_flags<RandomAccessIterator>(first, last);
}


} // end detail
} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust


HYDRA_EXTERNAL_NAMESPACE_END

