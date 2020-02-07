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


/*! \file distance.h
 *  \brief Device implementations for distance.
 */

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/get_iterator_value.h>
#include <hydra/detail/external/hydra_thrust/extrema.h>
#include <hydra/detail/external/hydra_thrust/functional.h>
#include <hydra/detail/external/hydra_thrust/pair.h>
#include <hydra/detail/external/hydra_thrust/reduce.h>
#include <hydra/detail/external/hydra_thrust/transform_reduce.h>

#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/iterator/counting_iterator.h>
#include <hydra/detail/external/hydra_thrust/iterator/zip_iterator.h>

namespace hydra_thrust
{
namespace system
{
namespace detail
{
namespace generic
{
namespace detail
{


//////////////
// Functors //
//////////////
//

// return the smaller/larger element making sure to prefer the 
// first occurance of the minimum/maximum element
template <typename InputType, typename IndexType, typename BinaryPredicate>
struct min_element_reduction
{
  BinaryPredicate comp;

  __host__ __device__ 
  min_element_reduction(BinaryPredicate comp) : comp(comp){}

  __host__ __device__ 
  hydra_thrust::tuple<InputType, IndexType>
  operator()(const hydra_thrust::tuple<InputType, IndexType>& lhs, 
             const hydra_thrust::tuple<InputType, IndexType>& rhs )
  {
    if(comp(hydra_thrust::get<0>(lhs), hydra_thrust::get<0>(rhs)))
      return lhs;
    if(comp(hydra_thrust::get<0>(rhs), hydra_thrust::get<0>(lhs)))
      return rhs;

    // values are equivalent, prefer value with smaller index
    if(hydra_thrust::get<1>(lhs) < hydra_thrust::get<1>(rhs))
      return lhs;
    else
      return rhs;
  } // end operator()()
}; // end min_element_reduction


template <typename InputType, typename IndexType, typename BinaryPredicate>
struct max_element_reduction
{
  BinaryPredicate comp;

  __host__ __device__ 
  max_element_reduction(BinaryPredicate comp) : comp(comp){}

  __host__ __device__ 
  hydra_thrust::tuple<InputType, IndexType>
  operator()(const hydra_thrust::tuple<InputType, IndexType>& lhs, 
             const hydra_thrust::tuple<InputType, IndexType>& rhs )
  {
    if(comp(hydra_thrust::get<0>(lhs), hydra_thrust::get<0>(rhs)))
      return rhs;
    if(comp(hydra_thrust::get<0>(rhs), hydra_thrust::get<0>(lhs)))
      return lhs;

    // values are equivalent, prefer value with smaller index
    if(hydra_thrust::get<1>(lhs) < hydra_thrust::get<1>(rhs))
      return lhs;
    else
      return rhs;
  } // end operator()()
}; // end max_element_reduction


// return the smaller & larger element making sure to prefer the 
// first occurance of the minimum/maximum element
template <typename InputType, typename IndexType, typename BinaryPredicate>
struct minmax_element_reduction
{
  BinaryPredicate comp;

  __host__ __device__
  minmax_element_reduction(BinaryPredicate comp) : comp(comp){}

  __host__ __device__ 
  hydra_thrust::tuple< hydra_thrust::tuple<InputType,IndexType>, hydra_thrust::tuple<InputType,IndexType> >
  operator()(const hydra_thrust::tuple< hydra_thrust::tuple<InputType,IndexType>, hydra_thrust::tuple<InputType,IndexType> >& lhs, 
             const hydra_thrust::tuple< hydra_thrust::tuple<InputType,IndexType>, hydra_thrust::tuple<InputType,IndexType> >& rhs )
  {

    return hydra_thrust::make_tuple(min_element_reduction<InputType, IndexType, BinaryPredicate>(comp)(hydra_thrust::get<0>(lhs), hydra_thrust::get<0>(rhs)),
                              max_element_reduction<InputType, IndexType, BinaryPredicate>(comp)(hydra_thrust::get<1>(lhs), hydra_thrust::get<1>(rhs)));
  } // end operator()()
}; // end minmax_element_reduction


template <typename InputType, typename IndexType>
struct duplicate_tuple
{
  __host__ __device__ 
  hydra_thrust::tuple< hydra_thrust::tuple<InputType,IndexType>, hydra_thrust::tuple<InputType,IndexType> >
  operator()(const hydra_thrust::tuple<InputType,IndexType>& t)
  {
    return hydra_thrust::make_tuple(t, t);
  }
}; // end duplicate_tuple


} // end namespace detail


template <typename DerivedPolicy, typename ForwardIterator>
__host__ __device__
ForwardIterator min_element(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                            ForwardIterator first,
                            ForwardIterator last)
{
  typedef typename hydra_thrust::iterator_value<ForwardIterator>::type value_type;

  return hydra_thrust::min_element(exec, first, last, hydra_thrust::less<value_type>());
} // end min_element()


template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
__host__ __device__
ForwardIterator min_element(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                            ForwardIterator first,
                            ForwardIterator last,
                            BinaryPredicate comp)
{
  if (first == last)
    return last;

  typedef typename hydra_thrust::iterator_traits<ForwardIterator>::value_type      InputType;
  typedef typename hydra_thrust::iterator_traits<ForwardIterator>::difference_type IndexType;

  hydra_thrust::tuple<InputType, IndexType> result =
    hydra_thrust::reduce
      (exec,
       hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(first, hydra_thrust::counting_iterator<IndexType>(0))),
       hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(first, hydra_thrust::counting_iterator<IndexType>(0))) + (last - first),
       hydra_thrust::tuple<InputType, IndexType>(hydra_thrust::detail::get_iterator_value(derived_cast(exec), first), 0),
       detail::min_element_reduction<InputType, IndexType, BinaryPredicate>(comp));

  return first + hydra_thrust::get<1>(result);
} // end min_element()


template <typename DerivedPolicy, typename ForwardIterator>
__host__ __device__
ForwardIterator max_element(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                            ForwardIterator first,
                            ForwardIterator last)
{
  typedef typename hydra_thrust::iterator_value<ForwardIterator>::type value_type;

  return hydra_thrust::max_element(exec, first, last, hydra_thrust::less<value_type>());
} // end max_element()


template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
__host__ __device__
ForwardIterator max_element(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                            ForwardIterator first,
                            ForwardIterator last,
                            BinaryPredicate comp)
{
  if (first == last)
    return last;

  typedef typename hydra_thrust::iterator_traits<ForwardIterator>::value_type      InputType;
  typedef typename hydra_thrust::iterator_traits<ForwardIterator>::difference_type IndexType;

  hydra_thrust::tuple<InputType, IndexType> result =
    hydra_thrust::reduce
      (exec,
       hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(first, hydra_thrust::counting_iterator<IndexType>(0))),
       hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(first, hydra_thrust::counting_iterator<IndexType>(0))) + (last - first),
       hydra_thrust::tuple<InputType, IndexType>(hydra_thrust::detail::get_iterator_value(derived_cast(exec),first), 0),
       detail::max_element_reduction<InputType, IndexType, BinaryPredicate>(comp));

  return first + hydra_thrust::get<1>(result);
} // end max_element()


template <typename DerivedPolicy, typename ForwardIterator>
__host__ __device__
hydra_thrust::pair<ForwardIterator,ForwardIterator> minmax_element(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                                                             ForwardIterator first, 
                                                             ForwardIterator last)
{
  typedef typename hydra_thrust::iterator_value<ForwardIterator>::type value_type;

  return hydra_thrust::minmax_element(exec, first, last, hydra_thrust::less<value_type>());
} // end minmax_element()


template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
__host__ __device__
hydra_thrust::pair<ForwardIterator,ForwardIterator> minmax_element(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                                                             ForwardIterator first, 
                                                             ForwardIterator last,
                                                             BinaryPredicate comp)
{
  if (first == last)
    return hydra_thrust::make_pair(last, last);

  typedef typename hydra_thrust::iterator_traits<ForwardIterator>::value_type      InputType;
  typedef typename hydra_thrust::iterator_traits<ForwardIterator>::difference_type IndexType;

  hydra_thrust::tuple< hydra_thrust::tuple<InputType,IndexType>, hydra_thrust::tuple<InputType,IndexType> > result = 
    hydra_thrust::transform_reduce
      (exec,
       hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(first, hydra_thrust::counting_iterator<IndexType>(0))),
       hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(first, hydra_thrust::counting_iterator<IndexType>(0))) + (last - first),
       detail::duplicate_tuple<InputType, IndexType>(),
       detail::duplicate_tuple<InputType, IndexType>()(
         hydra_thrust::tuple<InputType, IndexType>(hydra_thrust::detail::get_iterator_value(derived_cast(exec),first), 0)),
       detail::minmax_element_reduction<InputType, IndexType, BinaryPredicate>(comp));

  return hydra_thrust::make_pair(first + hydra_thrust::get<1>(hydra_thrust::get<0>(result)), first + hydra_thrust::get<1>(hydra_thrust::get<1>(result)));
} // end minmax_element()


} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace hydra_thrust

