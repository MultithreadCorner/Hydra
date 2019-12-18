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

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/uninitialized_copy.h>
#include <hydra/detail/external/hydra_thrust/copy.h>
#include <hydra/detail/external/hydra_thrust/for_each.h>
#include <hydra/detail/external/hydra_thrust/detail/internal_functional.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>

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

template<typename InputType,
         typename OutputType>
  struct uninitialized_copy_functor
{
  template<typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    const InputType &in = hydra_thrust::get<0>(t);
    OutputType &out = hydra_thrust::get<1>(t);

    ::new(static_cast<void*>(&out)) OutputType(in);
  } // end operator()()
}; // end uninitialized_copy_functor


// non-trivial copy constructor path
template<typename ExecutionPolicy,
         typename InputIterator,
         typename ForwardIterator>
__host__ __device__
  ForwardIterator uninitialized_copy(hydra_thrust::execution_policy<ExecutionPolicy> &exec,
                                     InputIterator first,
                                     InputIterator last,
                                     ForwardIterator result,
                                     hydra_thrust::detail::false_type) // has_trivial_copy_constructor
{
  // zip up the iterators
  typedef hydra_thrust::tuple<InputIterator,ForwardIterator> IteratorTuple;
  typedef hydra_thrust::zip_iterator<IteratorTuple> ZipIterator;

  ZipIterator begin = hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(first,result));
  ZipIterator end = begin;

  // get a zip_iterator pointing to the end
  const typename hydra_thrust::iterator_difference<InputIterator>::type n = hydra_thrust::distance(first,last);
  hydra_thrust::advance(end, n);

  // create a functor
  typedef typename iterator_traits<InputIterator>::value_type InputType;
  typedef typename iterator_traits<ForwardIterator>::value_type OutputType;

  detail::uninitialized_copy_functor<InputType, OutputType> f;

  // do the for_each
  hydra_thrust::for_each(exec, begin, end, f);

  // return the end of the output range
  return hydra_thrust::get<1>(end.get_iterator_tuple());
} // end uninitialized_copy()


// trivial copy constructor path
template<typename ExecutionPolicy,
         typename InputIterator,
         typename ForwardIterator>
__host__ __device__
  ForwardIterator uninitialized_copy(hydra_thrust::execution_policy<ExecutionPolicy> &exec,
                                     InputIterator first,
                                     InputIterator last,
                                     ForwardIterator result,
                                     hydra_thrust::detail::true_type) // has_trivial_copy_constructor
{
  return hydra_thrust::copy(exec, first, last, result);
} // end uninitialized_copy()


// non-trivial copy constructor path
template<typename ExecutionPolicy,
         typename InputIterator,
         typename Size,
         typename ForwardIterator>
__host__ __device__
  ForwardIterator uninitialized_copy_n(hydra_thrust::execution_policy<ExecutionPolicy> &exec,
                                       InputIterator first,
                                       Size n,
                                       ForwardIterator result,
                                       hydra_thrust::detail::false_type) // has_trivial_copy_constructor
{
  // zip up the iterators
  typedef hydra_thrust::tuple<InputIterator,ForwardIterator> IteratorTuple;
  typedef hydra_thrust::zip_iterator<IteratorTuple> ZipIterator;

  ZipIterator zipped_first = hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(first,result));

  // create a functor
  typedef typename iterator_traits<InputIterator>::value_type   InputType;
  typedef typename iterator_traits<ForwardIterator>::value_type OutputType;

  detail::uninitialized_copy_functor<InputType, OutputType> f;

  // do the for_each_n
  ZipIterator zipped_last = hydra_thrust::for_each_n(exec, zipped_first, n, f);

  // return the end of the output range
  return hydra_thrust::get<1>(zipped_last.get_iterator_tuple());
} // end uninitialized_copy_n()


// trivial copy constructor path
template<typename ExecutionPolicy,
         typename InputIterator,
         typename Size,
         typename ForwardIterator>
__host__ __device__
  ForwardIterator uninitialized_copy_n(hydra_thrust::execution_policy<ExecutionPolicy> &exec,
                                       InputIterator first,
                                       Size n,
                                       ForwardIterator result,
                                       hydra_thrust::detail::true_type) // has_trivial_copy_constructor
{
  return hydra_thrust::copy_n(exec, first, n, result);
} // end uninitialized_copy_n()


} // end detail


template<typename ExecutionPolicy,
         typename InputIterator,
         typename ForwardIterator>
__host__ __device__
  ForwardIterator uninitialized_copy(hydra_thrust::execution_policy<ExecutionPolicy> &exec,
                                     InputIterator first,
                                     InputIterator last,
                                     ForwardIterator result)
{
  typedef typename iterator_traits<ForwardIterator>::value_type ResultType;

  typedef typename hydra_thrust::detail::has_trivial_copy_constructor<ResultType>::type ResultTypeHasTrivialCopyConstructor;

  return hydra_thrust::system::detail::generic::detail::uninitialized_copy(exec, first, last, result, ResultTypeHasTrivialCopyConstructor());
} // end uninitialized_copy()


template<typename ExecutionPolicy,
         typename InputIterator,
         typename Size,
         typename ForwardIterator>
__host__ __device__
  ForwardIterator uninitialized_copy_n(hydra_thrust::execution_policy<ExecutionPolicy> &exec,
                                       InputIterator first,
                                       Size n,
                                       ForwardIterator result)
{
  typedef typename iterator_traits<ForwardIterator>::value_type ResultType;

  typedef typename hydra_thrust::detail::has_trivial_copy_constructor<ResultType>::type ResultTypeHasTrivialCopyConstructor;

  return hydra_thrust::system::detail::generic::detail::uninitialized_copy_n(exec, first, n, result, ResultTypeHasTrivialCopyConstructor());
} // end uninitialized_copy_n()


} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace hydra_thrust

