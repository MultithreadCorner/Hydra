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

#include <hydra/detail/external/thrust/detail/config.h>
#include <hydra/detail/external/thrust/system/detail/generic/transform.h>
#include <hydra/detail/external/thrust/for_each.h>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/thrust/iterator/detail/minimum_system.h>
#include <hydra/detail/external/thrust/tuple.h>
#include <hydra/detail/external/thrust/iterator/zip_iterator.h>
#include <hydra/detail/external/thrust/detail/internal_functional.h>

HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
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
__hydra_host__ __hydra_device__
  OutputIterator transform(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                           InputIterator first,
                           InputIterator last,
                           OutputIterator result,
                           UnaryFunction op)
{
  typedef HYDRA_EXTERNAL_NS::thrust::detail::unary_transform_functor<UnaryFunction> UnaryTransformFunctor;

  // make an iterator tuple
  typedef HYDRA_EXTERNAL_NS::thrust::tuple<InputIterator,OutputIterator> IteratorTuple;
  typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<IteratorTuple> ZipIterator;

  ZipIterator zipped_result =
    HYDRA_EXTERNAL_NS::thrust::for_each(exec,
                     HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(first,result)),
                     HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(last,result)),
                     UnaryTransformFunctor(op));

  return HYDRA_EXTERNAL_NS::thrust::get<1>(zipped_result.get_iterator_tuple());
} // end transform()


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryFunction>
__hydra_host__ __hydra_device__
  OutputIterator transform(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                           InputIterator1 first1,
                           InputIterator1 last1,
                           InputIterator2 first2,
                           OutputIterator result,
                           BinaryFunction op)
{
  // given the minimal system, determine the binary transform functor we need
  typedef HYDRA_EXTERNAL_NS::thrust::detail::binary_transform_functor<BinaryFunction> BinaryTransformFunctor;

  // make an iterator tuple
  typedef HYDRA_EXTERNAL_NS::thrust::tuple<InputIterator1,InputIterator2,OutputIterator> IteratorTuple;
  typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<IteratorTuple> ZipIterator;

  ZipIterator zipped_result =
    HYDRA_EXTERNAL_NS::thrust::for_each(exec,
                     HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(first1,first2,result)),
                     HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(last1,first2,result)),
                     BinaryTransformFunctor(op));

  return HYDRA_EXTERNAL_NS::thrust::get<2>(zipped_result.get_iterator_tuple());
} // end transform()


template<typename DerivedPolicy,
         typename InputIterator,
         typename ForwardIterator,
         typename UnaryFunction,
         typename Predicate>
__hydra_host__ __hydra_device__
  ForwardIterator transform_if(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                               InputIterator first,
                               InputIterator last,
                               ForwardIterator result,
                               UnaryFunction unary_op,
                               Predicate pred)
{
  typedef HYDRA_EXTERNAL_NS::thrust::detail::unary_transform_if_functor<UnaryFunction,Predicate> UnaryTransformIfFunctor;

  // make an iterator tuple
  typedef HYDRA_EXTERNAL_NS::thrust::tuple<InputIterator,ForwardIterator> IteratorTuple;
  typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<IteratorTuple> ZipIterator;

  ZipIterator zipped_result =
    HYDRA_EXTERNAL_NS::thrust::for_each(exec,
                     HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(first,result)),
                     HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(last,result)),
                     UnaryTransformIfFunctor(unary_op,pred));

  return HYDRA_EXTERNAL_NS::thrust::get<1>(zipped_result.get_iterator_tuple());
} // end transform_if()


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename ForwardIterator,
         typename UnaryFunction,
         typename Predicate>
__hydra_host__ __hydra_device__
  ForwardIterator transform_if(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                               InputIterator1 first,
                               InputIterator1 last,
                               InputIterator2 stencil,
                               ForwardIterator result,
                               UnaryFunction unary_op,
                               Predicate pred)
{
  typedef HYDRA_EXTERNAL_NS::thrust::detail::unary_transform_if_with_stencil_functor<UnaryFunction,Predicate> UnaryTransformIfFunctor;

  // make an iterator tuple
  typedef HYDRA_EXTERNAL_NS::thrust::tuple<InputIterator1,InputIterator2,ForwardIterator> IteratorTuple;
  typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<IteratorTuple> ZipIterator;

  ZipIterator zipped_result =
    HYDRA_EXTERNAL_NS::thrust::for_each(exec,
                     HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(first,stencil,result)),
                     HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(last,stencil,result)),
                     UnaryTransformIfFunctor(unary_op,pred));

  return HYDRA_EXTERNAL_NS::thrust::get<2>(zipped_result.get_iterator_tuple());
} // end transform_if()


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename ForwardIterator,
         typename BinaryFunction,
         typename Predicate>
__hydra_host__ __hydra_device__
  ForwardIterator transform_if(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                               InputIterator1 first1,
                               InputIterator1 last1,
                               InputIterator2 first2,
                               InputIterator3 stencil,
                               ForwardIterator result,
                               BinaryFunction binary_op,
                               Predicate pred)
{
  typedef HYDRA_EXTERNAL_NS::thrust::detail::binary_transform_if_functor<BinaryFunction,Predicate> BinaryTransformIfFunctor;

  // make an iterator tuple
  typedef HYDRA_EXTERNAL_NS::thrust::tuple<InputIterator1,InputIterator2,InputIterator3,ForwardIterator> IteratorTuple;
  typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<IteratorTuple> ZipIterator;

  ZipIterator zipped_result =
    HYDRA_EXTERNAL_NS::thrust::for_each(exec,
                     HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(first1,first2,stencil,result)),
                     HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(last1,first2,stencil,result)),
                     BinaryTransformIfFunctor(binary_op,pred));

  return HYDRA_EXTERNAL_NS::thrust::get<3>(zipped_result.get_iterator_tuple());
} // end transform_if()


} // end generic
} // end detail
} // end system
} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
HYDRA_EXTERNAL_NAMESPACE_END
