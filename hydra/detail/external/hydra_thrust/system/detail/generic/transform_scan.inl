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
#include <hydra/detail/external/hydra_thrust/system/detail/generic/transform_scan.h>
#include <hydra/detail/external/hydra_thrust/scan.h>
#include <hydra/detail/external/hydra_thrust/iterator/transform_iterator.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits/function_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits/iterator/is_output_iterator.h>

namespace hydra_thrust
{
namespace system
{
namespace detail
{
namespace generic
{


template<typename ExecutionPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction,
         typename BinaryFunction>
__host__ __device__
  OutputIterator transform_inclusive_scan(hydra_thrust::execution_policy<ExecutionPolicy> &exec,
                                          InputIterator first,
                                          InputIterator last,
                                          OutputIterator result,
                                          UnaryFunction unary_op,
                                          BinaryFunction binary_op)
{
  // the pseudocode for deducing the type of the temporary used below:
  // 
  // if UnaryFunction is AdaptableUnaryFunction
  //   TemporaryType = AdaptableUnaryFunction::result_type
  // else if OutputIterator is a "pure" output iterator
  //   TemporaryType = InputIterator::value_type
  // else
  //   TemporaryType = OutputIterator::value_type
  //
  // XXX upon c++0x, TemporaryType needs to be:
  // result_of_adaptable_function<UnaryFunction>::type

  typedef typename hydra_thrust::detail::eval_if<
    hydra_thrust::detail::has_result_type<UnaryFunction>::value,
    hydra_thrust::detail::result_type<UnaryFunction>,
    hydra_thrust::detail::eval_if<
      hydra_thrust::detail::is_output_iterator<OutputIterator>::value,
      hydra_thrust::iterator_value<InputIterator>,
      hydra_thrust::iterator_value<OutputIterator>
    >
  >::type ValueType;

  hydra_thrust::transform_iterator<UnaryFunction, InputIterator, ValueType> _first(first, unary_op);
  hydra_thrust::transform_iterator<UnaryFunction, InputIterator, ValueType> _last(last, unary_op);

  return hydra_thrust::inclusive_scan(exec, _first, _last, result, binary_op);
} // end transform_inclusive_scan()


template<typename ExecutionPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction,
         typename T,
         typename AssociativeOperator>
__host__ __device__
  OutputIterator transform_exclusive_scan(hydra_thrust::execution_policy<ExecutionPolicy> &exec,
                                          InputIterator first,
                                          InputIterator last,
                                          OutputIterator result,
                                          UnaryFunction unary_op,
                                          T init,
                                          AssociativeOperator binary_op)
{
  // the pseudocode for deducing the type of the temporary used below:
  // 
  // if UnaryFunction is AdaptableUnaryFunction
  //   TemporaryType = AdaptableUnaryFunction::result_type
  // else if OutputIterator is a "pure" output iterator
  //   TemporaryType = InputIterator::value_type
  // else
  //   TemporaryType = OutputIterator::value_type
  //
  // XXX upon c++0x, TemporaryType needs to be:
  // result_of_adaptable_function<UnaryFunction>::type

  typedef typename hydra_thrust::detail::eval_if<
    hydra_thrust::detail::has_result_type<UnaryFunction>::value,
    hydra_thrust::detail::result_type<UnaryFunction>,
    hydra_thrust::detail::eval_if<
      hydra_thrust::detail::is_output_iterator<OutputIterator>::value,
      hydra_thrust::iterator_value<InputIterator>,
      hydra_thrust::iterator_value<OutputIterator>
    >
  >::type ValueType;

  hydra_thrust::transform_iterator<UnaryFunction, InputIterator, ValueType> _first(first, unary_op);
  hydra_thrust::transform_iterator<UnaryFunction, InputIterator, ValueType> _last(last, unary_op);

  return hydra_thrust::exclusive_scan(exec, _first, _last, result, init, binary_op);
} // end transform_exclusive_scan()


} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace hydra_thrust

