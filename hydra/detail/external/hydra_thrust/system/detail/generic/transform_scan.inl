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
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits/function_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits/iterator/is_output_iterator.h>
#include <hydra/detail/external/hydra_thrust/iterator/transform_iterator.h>
#include <hydra/detail/external/hydra_thrust/scan.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/transform_scan.h>
#include <hydra/detail/external/hydra_thrust/type_traits/remove_cvref.h>

HYDRA_THRUST_NAMESPACE_BEGIN
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
  // Use the input iterator's value type per https://wg21.link/P0571
  using InputType = typename hydra_thrust::iterator_value<InputIterator>::type;
  using ResultType = hydra_thrust::detail::invoke_result_t<UnaryFunction, InputType>;
  using ValueType = hydra_thrust::remove_cvref_t<ResultType>;

  hydra_thrust::transform_iterator<UnaryFunction, InputIterator, ValueType> _first(first, unary_op);
  hydra_thrust::transform_iterator<UnaryFunction, InputIterator, ValueType> _last(last, unary_op);

  return hydra_thrust::inclusive_scan(exec, _first, _last, result, binary_op);
} // end transform_inclusive_scan()


template<typename ExecutionPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction,
         typename InitialValueType,
         typename AssociativeOperator>
__host__ __device__
  OutputIterator transform_exclusive_scan(hydra_thrust::execution_policy<ExecutionPolicy> &exec,
                                          InputIterator first,
                                          InputIterator last,
                                          OutputIterator result,
                                          UnaryFunction unary_op,
                                          InitialValueType init,
                                          AssociativeOperator binary_op)
{
  // Use the initial value type per https://wg21.link/P0571
  using ValueType = hydra_thrust::remove_cvref_t<InitialValueType>;

  hydra_thrust::transform_iterator<UnaryFunction, InputIterator, ValueType> _first(first, unary_op);
  hydra_thrust::transform_iterator<UnaryFunction, InputIterator, ValueType> _last(last, unary_op);

  return hydra_thrust::exclusive_scan(exec, _first, _last, result, init, binary_op);
} // end transform_exclusive_scan()


} // end namespace generic
} // end namespace detail
} // end namespace system
HYDRA_THRUST_NAMESPACE_END

