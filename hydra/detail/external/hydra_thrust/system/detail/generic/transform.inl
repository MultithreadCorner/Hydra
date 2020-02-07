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
#include <hydra/detail/external/hydra_thrust/system/detail/generic/transform.h>
#include <hydra/detail/external/hydra_thrust/for_each.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/minimum_system.h>
#include <hydra/detail/external/hydra_thrust/tuple.h>
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
         typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction>
__host__ __device__
  OutputIterator transform(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                           InputIterator first,
                           InputIterator last,
                           OutputIterator result,
                           UnaryFunction op)
{
  typedef hydra_thrust::detail::unary_transform_functor<UnaryFunction> UnaryTransformFunctor;

  // make an iterator tuple
  typedef hydra_thrust::tuple<InputIterator,OutputIterator> IteratorTuple;
  typedef hydra_thrust::zip_iterator<IteratorTuple> ZipIterator;

  ZipIterator zipped_result =
    hydra_thrust::for_each(exec,
                     hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(first,result)),
                     hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(last,result)),
                     UnaryTransformFunctor(op));

  return hydra_thrust::get<1>(zipped_result.get_iterator_tuple());
} // end transform()


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryFunction>
__host__ __device__
  OutputIterator transform(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                           InputIterator1 first1,
                           InputIterator1 last1,
                           InputIterator2 first2,
                           OutputIterator result,
                           BinaryFunction op)
{
  // given the minimal system, determine the binary transform functor we need
  typedef hydra_thrust::detail::binary_transform_functor<BinaryFunction> BinaryTransformFunctor;

  // make an iterator tuple
  typedef hydra_thrust::tuple<InputIterator1,InputIterator2,OutputIterator> IteratorTuple;
  typedef hydra_thrust::zip_iterator<IteratorTuple> ZipIterator;

  ZipIterator zipped_result =
    hydra_thrust::for_each(exec,
                     hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(first1,first2,result)),
                     hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(last1,first2,result)),
                     BinaryTransformFunctor(op));

  return hydra_thrust::get<2>(zipped_result.get_iterator_tuple());
} // end transform()


template<typename DerivedPolicy,
         typename InputIterator,
         typename ForwardIterator,
         typename UnaryFunction,
         typename Predicate>
__host__ __device__
  ForwardIterator transform_if(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                               InputIterator first,
                               InputIterator last,
                               ForwardIterator result,
                               UnaryFunction unary_op,
                               Predicate pred)
{
  typedef hydra_thrust::detail::unary_transform_if_functor<UnaryFunction,Predicate> UnaryTransformIfFunctor;

  // make an iterator tuple
  typedef hydra_thrust::tuple<InputIterator,ForwardIterator> IteratorTuple;
  typedef hydra_thrust::zip_iterator<IteratorTuple> ZipIterator;

  ZipIterator zipped_result =
    hydra_thrust::for_each(exec,
                     hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(first,result)),
                     hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(last,result)),
                     UnaryTransformIfFunctor(unary_op,pred));

  return hydra_thrust::get<1>(zipped_result.get_iterator_tuple());
} // end transform_if()


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename ForwardIterator,
         typename UnaryFunction,
         typename Predicate>
__host__ __device__
  ForwardIterator transform_if(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                               InputIterator1 first,
                               InputIterator1 last,
                               InputIterator2 stencil,
                               ForwardIterator result,
                               UnaryFunction unary_op,
                               Predicate pred)
{
  typedef hydra_thrust::detail::unary_transform_if_with_stencil_functor<UnaryFunction,Predicate> UnaryTransformIfFunctor;

  // make an iterator tuple
  typedef hydra_thrust::tuple<InputIterator1,InputIterator2,ForwardIterator> IteratorTuple;
  typedef hydra_thrust::zip_iterator<IteratorTuple> ZipIterator;

  ZipIterator zipped_result =
    hydra_thrust::for_each(exec,
                     hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(first,stencil,result)),
                     hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(last,stencil,result)),
                     UnaryTransformIfFunctor(unary_op,pred));

  return hydra_thrust::get<2>(zipped_result.get_iterator_tuple());
} // end transform_if()


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename ForwardIterator,
         typename BinaryFunction,
         typename Predicate>
__host__ __device__
  ForwardIterator transform_if(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                               InputIterator1 first1,
                               InputIterator1 last1,
                               InputIterator2 first2,
                               InputIterator3 stencil,
                               ForwardIterator result,
                               BinaryFunction binary_op,
                               Predicate pred)
{
  typedef hydra_thrust::detail::binary_transform_if_functor<BinaryFunction,Predicate> BinaryTransformIfFunctor;

  // make an iterator tuple
  typedef hydra_thrust::tuple<InputIterator1,InputIterator2,InputIterator3,ForwardIterator> IteratorTuple;
  typedef hydra_thrust::zip_iterator<IteratorTuple> ZipIterator;

  ZipIterator zipped_result =
    hydra_thrust::for_each(exec,
                     hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(first1,first2,stencil,result)),
                     hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(last1,first2,stencil,result)),
                     BinaryTransformIfFunctor(binary_op,pred));

  return hydra_thrust::get<3>(zipped_result.get_iterator_tuple());
} // end transform_if()


} // end generic
} // end detail
} // end system
} // end hydra_thrust

