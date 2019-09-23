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

#include <hydra/detail/external/thrust/detail/config.h>
#include <hydra/detail/external/thrust/detail/get_iterator_value.h>
#include <hydra/detail/external/thrust/extrema.h>
#include <hydra/detail/external/thrust/functional.h>
#include <hydra/detail/external/thrust/pair.h>
#include <hydra/detail/external/thrust/reduce.h>
#include <hydra/detail/external/thrust/transform_reduce.h>

#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/thrust/iterator/counting_iterator.h>
#include <hydra/detail/external/thrust/iterator/zip_iterator.h>

HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
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

  __hydra_host__ __hydra_device__ 
  min_element_reduction(BinaryPredicate comp) : comp(comp){}

  __hydra_host__ __hydra_device__ 
  HYDRA_EXTERNAL_NS::thrust::tuple<InputType, IndexType>
  operator()(const HYDRA_EXTERNAL_NS::thrust::tuple<InputType, IndexType>& lhs, 
             const HYDRA_EXTERNAL_NS::thrust::tuple<InputType, IndexType>& rhs )
  {
    if(comp(HYDRA_EXTERNAL_NS::thrust::get<0>(lhs), HYDRA_EXTERNAL_NS::thrust::get<0>(rhs)))
      return lhs;
    if(comp(HYDRA_EXTERNAL_NS::thrust::get<0>(rhs), HYDRA_EXTERNAL_NS::thrust::get<0>(lhs)))
      return rhs;

    // values are equivalent, prefer value with smaller index
    if(HYDRA_EXTERNAL_NS::thrust::get<1>(lhs) < HYDRA_EXTERNAL_NS::thrust::get<1>(rhs))
      return lhs;
    else
      return rhs;
  } // end operator()()
}; // end min_element_reduction


template <typename InputType, typename IndexType, typename BinaryPredicate>
struct max_element_reduction
{
  BinaryPredicate comp;

  __hydra_host__ __hydra_device__ 
  max_element_reduction(BinaryPredicate comp) : comp(comp){}

  __hydra_host__ __hydra_device__ 
  HYDRA_EXTERNAL_NS::thrust::tuple<InputType, IndexType>
  operator()(const HYDRA_EXTERNAL_NS::thrust::tuple<InputType, IndexType>& lhs, 
             const HYDRA_EXTERNAL_NS::thrust::tuple<InputType, IndexType>& rhs )
  {
    if(comp(HYDRA_EXTERNAL_NS::thrust::get<0>(lhs), HYDRA_EXTERNAL_NS::thrust::get<0>(rhs)))
      return rhs;
    if(comp(HYDRA_EXTERNAL_NS::thrust::get<0>(rhs), HYDRA_EXTERNAL_NS::thrust::get<0>(lhs)))
      return lhs;

    // values are equivalent, prefer value with smaller index
    if(HYDRA_EXTERNAL_NS::thrust::get<1>(lhs) < HYDRA_EXTERNAL_NS::thrust::get<1>(rhs))
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

  __hydra_host__ __hydra_device__
  minmax_element_reduction(BinaryPredicate comp) : comp(comp){}

  __hydra_host__ __hydra_device__ 
  HYDRA_EXTERNAL_NS::thrust::tuple< HYDRA_EXTERNAL_NS::thrust::tuple<InputType,IndexType>, HYDRA_EXTERNAL_NS::thrust::tuple<InputType,IndexType> >
  operator()(const HYDRA_EXTERNAL_NS::thrust::tuple< HYDRA_EXTERNAL_NS::thrust::tuple<InputType,IndexType>, HYDRA_EXTERNAL_NS::thrust::tuple<InputType,IndexType> >& lhs, 
             const HYDRA_EXTERNAL_NS::thrust::tuple< HYDRA_EXTERNAL_NS::thrust::tuple<InputType,IndexType>, HYDRA_EXTERNAL_NS::thrust::tuple<InputType,IndexType> >& rhs )
  {

    return HYDRA_EXTERNAL_NS::thrust::make_tuple(min_element_reduction<InputType, IndexType, BinaryPredicate>(comp)(HYDRA_EXTERNAL_NS::thrust::get<0>(lhs), HYDRA_EXTERNAL_NS::thrust::get<0>(rhs)),
                              max_element_reduction<InputType, IndexType, BinaryPredicate>(comp)(HYDRA_EXTERNAL_NS::thrust::get<1>(lhs), HYDRA_EXTERNAL_NS::thrust::get<1>(rhs)));
  } // end operator()()
}; // end minmax_element_reduction


template <typename InputType, typename IndexType>
struct duplicate_tuple
{
  __hydra_host__ __hydra_device__ 
  HYDRA_EXTERNAL_NS::thrust::tuple< HYDRA_EXTERNAL_NS::thrust::tuple<InputType,IndexType>, HYDRA_EXTERNAL_NS::thrust::tuple<InputType,IndexType> >
  operator()(const HYDRA_EXTERNAL_NS::thrust::tuple<InputType,IndexType>& t)
  {
    return HYDRA_EXTERNAL_NS::thrust::make_tuple(t, t);
  }
}; // end duplicate_tuple


} // end namespace detail


template <typename DerivedPolicy, typename ForwardIterator>
__hydra_host__ __hydra_device__
ForwardIterator min_element(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                            ForwardIterator first,
                            ForwardIterator last)
{
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_value<ForwardIterator>::type value_type;

  return HYDRA_EXTERNAL_NS::thrust::min_element(exec, first, last, HYDRA_EXTERNAL_NS::thrust::less<value_type>());
} // end min_element()


template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
__hydra_host__ __hydra_device__
ForwardIterator min_element(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                            ForwardIterator first,
                            ForwardIterator last,
                            BinaryPredicate comp)
{
  if (first == last)
    return last;

  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<ForwardIterator>::value_type      InputType;
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<ForwardIterator>::difference_type IndexType;

  HYDRA_EXTERNAL_NS::thrust::tuple<InputType, IndexType> result =
    HYDRA_EXTERNAL_NS::thrust::reduce
      (exec,
       HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(first, HYDRA_EXTERNAL_NS::thrust::counting_iterator<IndexType>(0))),
       HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(first, HYDRA_EXTERNAL_NS::thrust::counting_iterator<IndexType>(0))) + (last - first),
       HYDRA_EXTERNAL_NS::thrust::tuple<InputType, IndexType>(HYDRA_EXTERNAL_NS::thrust::detail::get_iterator_value(derived_cast(exec), first), 0),
       detail::min_element_reduction<InputType, IndexType, BinaryPredicate>(comp));

  return first + HYDRA_EXTERNAL_NS::thrust::get<1>(result);
} // end min_element()


template <typename DerivedPolicy, typename ForwardIterator>
__hydra_host__ __hydra_device__
ForwardIterator max_element(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                            ForwardIterator first,
                            ForwardIterator last)
{
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_value<ForwardIterator>::type value_type;

  return HYDRA_EXTERNAL_NS::thrust::max_element(exec, first, last, HYDRA_EXTERNAL_NS::thrust::less<value_type>());
} // end max_element()


template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
__hydra_host__ __hydra_device__
ForwardIterator max_element(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                            ForwardIterator first,
                            ForwardIterator last,
                            BinaryPredicate comp)
{
  if (first == last)
    return last;

  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<ForwardIterator>::value_type      InputType;
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<ForwardIterator>::difference_type IndexType;

  HYDRA_EXTERNAL_NS::thrust::tuple<InputType, IndexType> result =
    HYDRA_EXTERNAL_NS::thrust::reduce
      (exec,
       HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(first, HYDRA_EXTERNAL_NS::thrust::counting_iterator<IndexType>(0))),
       HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(first, HYDRA_EXTERNAL_NS::thrust::counting_iterator<IndexType>(0))) + (last - first),
       HYDRA_EXTERNAL_NS::thrust::tuple<InputType, IndexType>(HYDRA_EXTERNAL_NS::thrust::detail::get_iterator_value(derived_cast(exec),first), 0),
       detail::max_element_reduction<InputType, IndexType, BinaryPredicate>(comp));

  return first + HYDRA_EXTERNAL_NS::thrust::get<1>(result);
} // end max_element()


template <typename DerivedPolicy, typename ForwardIterator>
__hydra_host__ __hydra_device__
HYDRA_EXTERNAL_NS::thrust::pair<ForwardIterator,ForwardIterator> minmax_element(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                                                             ForwardIterator first, 
                                                             ForwardIterator last)
{
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_value<ForwardIterator>::type value_type;

  return HYDRA_EXTERNAL_NS::thrust::minmax_element(exec, first, last, HYDRA_EXTERNAL_NS::thrust::less<value_type>());
} // end minmax_element()


template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
__hydra_host__ __hydra_device__
HYDRA_EXTERNAL_NS::thrust::pair<ForwardIterator,ForwardIterator> minmax_element(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                                                             ForwardIterator first, 
                                                             ForwardIterator last,
                                                             BinaryPredicate comp)
{
  if (first == last)
    return HYDRA_EXTERNAL_NS::thrust::make_pair(last, last);

  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<ForwardIterator>::value_type      InputType;
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<ForwardIterator>::difference_type IndexType;

  HYDRA_EXTERNAL_NS::thrust::tuple< HYDRA_EXTERNAL_NS::thrust::tuple<InputType,IndexType>, HYDRA_EXTERNAL_NS::thrust::tuple<InputType,IndexType> > result = 
    HYDRA_EXTERNAL_NS::thrust::transform_reduce
      (exec,
       HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(first, HYDRA_EXTERNAL_NS::thrust::counting_iterator<IndexType>(0))),
       HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(first, HYDRA_EXTERNAL_NS::thrust::counting_iterator<IndexType>(0))) + (last - first),
       detail::duplicate_tuple<InputType, IndexType>(),
       detail::duplicate_tuple<InputType, IndexType>()(
         HYDRA_EXTERNAL_NS::thrust::tuple<InputType, IndexType>(HYDRA_EXTERNAL_NS::thrust::detail::get_iterator_value(derived_cast(exec),first), 0)),
       detail::minmax_element_reduction<InputType, IndexType, BinaryPredicate>(comp));

  return HYDRA_EXTERNAL_NS::thrust::make_pair(first + HYDRA_EXTERNAL_NS::thrust::get<1>(HYDRA_EXTERNAL_NS::thrust::get<0>(result)), first + HYDRA_EXTERNAL_NS::thrust::get<1>(HYDRA_EXTERNAL_NS::thrust::get<1>(result)));
} // end minmax_element()


} // end namespace generic
} // end namespace detail
} // end namespace system
} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust

HYDRA_EXTERNAL_NAMESPACE_END
