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

#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/minimum_system.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits/iterator/is_output_iterator.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits/function_traits.h>
#include <hydra/detail/external/hydra_thrust/transform.h>
#include <hydra/detail/external/hydra_thrust/scatter.h>
#include <hydra/detail/external/hydra_thrust/iterator/zip_iterator.h>
#include <limits>

#include <hydra/detail/external/hydra_thrust/detail/internal_functional.h>
#include <hydra/detail/external/hydra_thrust/scan.h>
#include <hydra/detail/external/hydra_thrust/detail/temporary_array.h>

HYDRA_THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{
namespace detail
{


template <typename ValueType, typename TailFlagType, typename AssociativeOperator>
struct reduce_by_key_functor
{
  AssociativeOperator binary_op;

  typedef typename hydra_thrust::tuple<ValueType, TailFlagType> result_type;

  __host__ __device__
  reduce_by_key_functor(AssociativeOperator _binary_op) : binary_op(_binary_op) {}

  __host__ __device__
  result_type operator()(result_type a, result_type b)
  {
    return result_type(hydra_thrust::get<1>(b) ? hydra_thrust::get<0>(b) : binary_op(hydra_thrust::get<0>(a), hydra_thrust::get<0>(b)),
                       hydra_thrust::get<1>(a) | hydra_thrust::get<1>(b));
  }
};


} // end namespace detail


template<typename ExecutionPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate,
         typename BinaryFunction>
__host__ __device__
  hydra_thrust::pair<OutputIterator1,OutputIterator2>
    reduce_by_key(hydra_thrust::execution_policy<ExecutionPolicy> &exec,
                  InputIterator1 keys_first,
                  InputIterator1 keys_last,
                  InputIterator2 values_first,
                  OutputIterator1 keys_output,
                  OutputIterator2 values_output,
                  BinaryPredicate binary_pred,
                  BinaryFunction binary_op)
{
    typedef typename hydra_thrust::iterator_traits<InputIterator1>::difference_type difference_type;

    typedef unsigned int FlagType;  // TODO use difference_type

    // Use the input iterator's value type per https://wg21.link/P0571
    using ValueType = typename hydra_thrust::iterator_value<InputIterator2>::type;

    if (keys_first == keys_last)
        return hydra_thrust::make_pair(keys_output, values_output);

    // input size
    difference_type n = keys_last - keys_first;

    InputIterator2 values_last = values_first + n;

    // compute head flags
    hydra_thrust::detail::temporary_array<FlagType,ExecutionPolicy> head_flags(exec, n);
    hydra_thrust::transform(exec, keys_first, keys_last - 1, keys_first + 1, head_flags.begin() + 1, hydra_thrust::detail::not2(binary_pred));
    head_flags[0] = 1;

    // compute tail flags
    hydra_thrust::detail::temporary_array<FlagType,ExecutionPolicy> tail_flags(exec, n); //COPY INSTEAD OF TRANSFORM
    hydra_thrust::transform(exec, keys_first, keys_last - 1, keys_first + 1, tail_flags.begin(), hydra_thrust::detail::not2(binary_pred));
    tail_flags[n-1] = 1;

    // scan the values by flag
    hydra_thrust::detail::temporary_array<ValueType,ExecutionPolicy> scanned_values(exec, n);
    hydra_thrust::detail::temporary_array<FlagType,ExecutionPolicy>  scanned_tail_flags(exec, n);

    hydra_thrust::inclusive_scan
        (exec,
         hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(values_first,           head_flags.begin())),
         hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(values_last,            head_flags.end())),
         hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(scanned_values.begin(), scanned_tail_flags.begin())),
         detail::reduce_by_key_functor<ValueType, FlagType, BinaryFunction>(binary_op));

    hydra_thrust::exclusive_scan(exec, tail_flags.begin(), tail_flags.end(), scanned_tail_flags.begin(), FlagType(0), hydra_thrust::plus<FlagType>());

    // number of unique keys
    FlagType N = scanned_tail_flags[n - 1] + 1;

    // scatter the keys and accumulated values
    hydra_thrust::scatter_if(exec, keys_first,            keys_last,             scanned_tail_flags.begin(), head_flags.begin(), keys_output);
    hydra_thrust::scatter_if(exec, scanned_values.begin(), scanned_values.end(), scanned_tail_flags.begin(), tail_flags.begin(), values_output);

    return hydra_thrust::make_pair(keys_output + N, values_output + N);
} // end reduce_by_key()


template<typename ExecutionPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2>
__host__ __device__
  hydra_thrust::pair<OutputIterator1,OutputIterator2>
    reduce_by_key(hydra_thrust::execution_policy<ExecutionPolicy> &exec,
                  InputIterator1 keys_first,
                  InputIterator1 keys_last,
                  InputIterator2 values_first,
                  OutputIterator1 keys_output,
                  OutputIterator2 values_output)
{
  typedef typename hydra_thrust::iterator_value<InputIterator1>::type KeyType;

  // use equal_to<KeyType> as default BinaryPredicate
  return hydra_thrust::reduce_by_key(exec, keys_first, keys_last, values_first, keys_output, values_output, hydra_thrust::equal_to<KeyType>());
} // end reduce_by_key()


template<typename ExecutionPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate>
__host__ __device__
  hydra_thrust::pair<OutputIterator1,OutputIterator2>
    reduce_by_key(hydra_thrust::execution_policy<ExecutionPolicy> &exec,
                  InputIterator1 keys_first,
                  InputIterator1 keys_last,
                  InputIterator2 values_first,
                  OutputIterator1 keys_output,
                  OutputIterator2 values_output,
                  BinaryPredicate binary_pred)
{
  typedef typename hydra_thrust::detail::eval_if<
    hydra_thrust::detail::is_output_iterator<OutputIterator2>::value,
    hydra_thrust::iterator_value<InputIterator2>,
    hydra_thrust::iterator_value<OutputIterator2>
  >::type T;

  // use plus<T> as default BinaryFunction
  return hydra_thrust::reduce_by_key(exec,
                               keys_first, keys_last,
                               values_first,
                               keys_output,
                               values_output,
                               binary_pred,
                               hydra_thrust::plus<T>());
} // end reduce_by_key()


} // end namespace generic
} // end namespace detail
} // end namespace system
HYDRA_THRUST_NAMESPACE_END

