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

#include <hydra/detail/external/hydra_thrust/reduce.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/reduce.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/functional.h>
#include <hydra/detail/external/hydra_thrust/detail/static_assert.h>

HYDRA_THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{


template<typename ExecutionPolicy, typename InputIterator>
__host__ __device__
  typename hydra_thrust::iterator_traits<InputIterator>::value_type
    reduce(hydra_thrust::execution_policy<ExecutionPolicy> &exec, InputIterator first, InputIterator last)
{
  typedef typename hydra_thrust::iterator_value<InputIterator>::type InputType;

  // use InputType(0) as init by default
  return hydra_thrust::reduce(exec, first, last, InputType(0));
} // end reduce()


template<typename ExecutionPolicy, typename InputIterator, typename T>
__host__ __device__
  T reduce(hydra_thrust::execution_policy<ExecutionPolicy> &exec, InputIterator first, InputIterator last, T init)
{
  // use plus<T> by default
  return hydra_thrust::reduce(exec, first, last, init, hydra_thrust::plus<T>());
} // end reduce()


template<typename ExecutionPolicy,
         typename RandomAccessIterator,
         typename OutputType,
         typename BinaryFunction>
__host__ __device__
  OutputType reduce(hydra_thrust::execution_policy<ExecutionPolicy> &,
                    RandomAccessIterator,
                    RandomAccessIterator,
                    OutputType,
                    BinaryFunction)
{
  HYDRA_THRUST_STATIC_ASSERT_MSG(
    (hydra_thrust::detail::depend_on_instantiation<RandomAccessIterator, false>::value)
  , "unimplemented for this system"
  );
  return OutputType();
} // end reduce()


} // end namespace generic
} // end namespace detail
} // end namespace system
HYDRA_THRUST_NAMESPACE_END

