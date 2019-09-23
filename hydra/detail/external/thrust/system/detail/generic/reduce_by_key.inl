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


/*! \file reduce_by_key.inl
 *  \brief Inline file for reduce_by_key.h.
 */

#pragma once

#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/thrust/iterator/detail/minimum_system.h>
#include <hydra/detail/external/thrust/detail/type_traits.h>
#include <hydra/detail/external/thrust/detail/type_traits/iterator/is_output_iterator.h>
#include <hydra/detail/external/thrust/detail/type_traits/function_traits.h>
#include <hydra/detail/external/thrust/transform.h>
#include <hydra/detail/external/thrust/scatter.h>
#include <hydra/detail/external/thrust/iterator/zip_iterator.h>
#include <limits>

#include <hydra/detail/external/thrust/detail/internal_functional.h>
#include <hydra/detail/external/thrust/scan.h>
#include <hydra/detail/external/thrust/detail/temporary_array.h>

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


template <typename ValueType, typename TailFlagType, typename AssociativeOperator>
struct reduce_by_key_functor
{
  AssociativeOperator binary_op;
  
  typedef typename HYDRA_EXTERNAL_NS::thrust::tuple<ValueType, TailFlagType> result_type;
  
  __hydra_host__ __hydra_device__
  reduce_by_key_functor(AssociativeOperator _binary_op) : binary_op(_binary_op) {}
  
  __hydra_host__ __hydra_device__
  result_type operator()(result_type a, result_type b)
  {
    return result_type(HYDRA_EXTERNAL_NS::thrust::get<1>(b) ? HYDRA_EXTERNAL_NS::thrust::get<0>(b) : binary_op(HYDRA_EXTERNAL_NS::thrust::get<0>(a), HYDRA_EXTERNAL_NS::thrust::get<0>(b)),
                       HYDRA_EXTERNAL_NS::thrust::get<1>(a) | HYDRA_EXTERNAL_NS::thrust::get<1>(b));
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
__hydra_host__ __hydra_device__
  HYDRA_EXTERNAL_NS::thrust::pair<OutputIterator1,OutputIterator2>
    reduce_by_key(HYDRA_EXTERNAL_NS::thrust::execution_policy<ExecutionPolicy> &exec,
                  InputIterator1 keys_first, 
                  InputIterator1 keys_last,
                  InputIterator2 values_first,
                  OutputIterator1 keys_output,
                  OutputIterator2 values_output,
                  BinaryPredicate binary_pred,
                  BinaryFunction binary_op)
{
    typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<InputIterator1>::difference_type difference_type;

    typedef unsigned int FlagType;  // TODO use difference_type

    // the pseudocode for deducing the type of the temporary used below:
    // 
    // if BinaryFunction is AdaptableBinaryFunction
    //   TemporaryType = AdaptableBinaryFunction::result_type
    // else if OutputIterator2 is a "pure" output iterator
    //   TemporaryType = InputIterator2::value_type
    // else
    //   TemporaryType = OutputIterator2::value_type
    //
    // XXX upon c++0x, TemporaryType needs to be:
    // result_of_adaptable_function<BinaryFunction>::type

    typedef typename HYDRA_EXTERNAL_NS::thrust::detail::eval_if<
      HYDRA_EXTERNAL_NS::thrust::detail::has_result_type<BinaryFunction>::value,
      HYDRA_EXTERNAL_NS::thrust::detail::result_type<BinaryFunction>,
      HYDRA_EXTERNAL_NS::thrust::detail::eval_if<
        HYDRA_EXTERNAL_NS::thrust::detail::is_output_iterator<OutputIterator2>::value,
        HYDRA_EXTERNAL_NS::thrust::iterator_value<InputIterator2>,
        HYDRA_EXTERNAL_NS::thrust::iterator_value<OutputIterator2>
      >
    >::type ValueType;

    if (keys_first == keys_last)
        return HYDRA_EXTERNAL_NS::thrust::make_pair(keys_output, values_output);

    // input size
    difference_type n = keys_last - keys_first;

    InputIterator2 values_last = values_first + n;
    
    // compute head flags
    HYDRA_EXTERNAL_NS::thrust::detail::temporary_array<FlagType,ExecutionPolicy> head_flags(exec, n);
    HYDRA_EXTERNAL_NS::thrust::transform(exec, keys_first, keys_last - 1, keys_first + 1, head_flags.begin() + 1, HYDRA_EXTERNAL_NS::thrust::detail::not2(binary_pred));
    head_flags[0] = 1;

    // compute tail flags
    HYDRA_EXTERNAL_NS::thrust::detail::temporary_array<FlagType,ExecutionPolicy> tail_flags(exec, n); //COPY INSTEAD OF TRANSFORM
    HYDRA_EXTERNAL_NS::thrust::transform(exec, keys_first, keys_last - 1, keys_first + 1, tail_flags.begin(), HYDRA_EXTERNAL_NS::thrust::detail::not2(binary_pred));
    tail_flags[n-1] = 1;

    // scan the values by flag
    HYDRA_EXTERNAL_NS::thrust::detail::temporary_array<ValueType,ExecutionPolicy> scanned_values(exec, n);
    HYDRA_EXTERNAL_NS::thrust::detail::temporary_array<FlagType,ExecutionPolicy>  scanned_tail_flags(exec, n);
    
    HYDRA_EXTERNAL_NS::thrust::inclusive_scan
        (exec,
         HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(values_first,           head_flags.begin())),
         HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(values_last,            head_flags.end())),
         HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(scanned_values.begin(), scanned_tail_flags.begin())),
         detail::reduce_by_key_functor<ValueType, FlagType, BinaryFunction>(binary_op));

    HYDRA_EXTERNAL_NS::thrust::exclusive_scan(exec, tail_flags.begin(), tail_flags.end(), scanned_tail_flags.begin(), FlagType(0), HYDRA_EXTERNAL_NS::thrust::plus<FlagType>());

    // number of unique keys
    FlagType N = scanned_tail_flags[n - 1] + 1;
    
    // scatter the keys and accumulated values    
    HYDRA_EXTERNAL_NS::thrust::scatter_if(exec, keys_first,            keys_last,             scanned_tail_flags.begin(), head_flags.begin(), keys_output);
    HYDRA_EXTERNAL_NS::thrust::scatter_if(exec, scanned_values.begin(), scanned_values.end(), scanned_tail_flags.begin(), tail_flags.begin(), values_output);

    return HYDRA_EXTERNAL_NS::thrust::make_pair(keys_output + N, values_output + N); 
} // end reduce_by_key()


template<typename ExecutionPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2>
__hydra_host__ __hydra_device__
  HYDRA_EXTERNAL_NS::thrust::pair<OutputIterator1,OutputIterator2>
    reduce_by_key(HYDRA_EXTERNAL_NS::thrust::execution_policy<ExecutionPolicy> &exec,
                  InputIterator1 keys_first, 
                  InputIterator1 keys_last,
                  InputIterator2 values_first,
                  OutputIterator1 keys_output,
                  OutputIterator2 values_output)
{
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_value<InputIterator1>::type KeyType;

  // use equal_to<KeyType> as default BinaryPredicate
  return HYDRA_EXTERNAL_NS::thrust::reduce_by_key(exec, keys_first, keys_last, values_first, keys_output, values_output, HYDRA_EXTERNAL_NS::thrust::equal_to<KeyType>());
} // end reduce_by_key()


template<typename ExecutionPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate>
__hydra_host__ __hydra_device__
  HYDRA_EXTERNAL_NS::thrust::pair<OutputIterator1,OutputIterator2>
    reduce_by_key(HYDRA_EXTERNAL_NS::thrust::execution_policy<ExecutionPolicy> &exec,
                  InputIterator1 keys_first, 
                  InputIterator1 keys_last,
                  InputIterator2 values_first,
                  OutputIterator1 keys_output,
                  OutputIterator2 values_output,
                  BinaryPredicate binary_pred)
{
  typedef typename HYDRA_EXTERNAL_NS::thrust::detail::eval_if<
    HYDRA_EXTERNAL_NS::thrust::detail::is_output_iterator<OutputIterator2>::value,
    HYDRA_EXTERNAL_NS::thrust::iterator_value<InputIterator2>,
    HYDRA_EXTERNAL_NS::thrust::iterator_value<OutputIterator2>
  >::type T;

  // use plus<T> as default BinaryFunction
  return HYDRA_EXTERNAL_NS::thrust::reduce_by_key(exec,
                               keys_first, keys_last, 
                               values_first,
                               keys_output,
                               values_output,
                               binary_pred,
                               HYDRA_EXTERNAL_NS::thrust::plus<T>());
} // end reduce_by_key()


} // end namespace generic
} // end namespace detail
} // end namespace system
} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust

HYDRA_EXTERNAL_NAMESPACE_END
