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

#include <hydra/detail/external/thrust/reduce.h>
#include <hydra/detail/external/thrust/system/detail/generic/reduce.h>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/thrust/functional.h>
#include <hydra/detail/external/thrust/detail/static_assert.h>

HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
{
namespace system
{
namespace detail
{
namespace generic
{


template<typename ExecutionPolicy, typename InputIterator>
__hydra_host__ __hydra_device__
  typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<InputIterator>::value_type
    reduce(HYDRA_EXTERNAL_NS::thrust::execution_policy<ExecutionPolicy> &exec, InputIterator first, InputIterator last)
{
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_value<InputIterator>::type InputType;

  // use InputType(0) as init by default
  return HYDRA_EXTERNAL_NS::thrust::reduce(exec, first, last, InputType(0));
} // end reduce()


template<typename ExecutionPolicy, typename InputIterator, typename T>
__hydra_host__ __hydra_device__
  T reduce(HYDRA_EXTERNAL_NS::thrust::execution_policy<ExecutionPolicy> &exec, InputIterator first, InputIterator last, T init)
{
  // use plus<T> by default
  return HYDRA_EXTERNAL_NS::thrust::reduce(exec, first, last, init, HYDRA_EXTERNAL_NS::thrust::plus<T>());
} // end reduce()


template<typename ExecutionPolicy,
         typename RandomAccessIterator,
         typename OutputType,
         typename BinaryFunction>
__hydra_host__ __hydra_device__
  OutputType reduce(HYDRA_EXTERNAL_NS::thrust::execution_policy<ExecutionPolicy> &,
                    RandomAccessIterator,
                    RandomAccessIterator,
                    OutputType,
                    BinaryFunction)
{
  HYDRA_THRUST_STATIC_ASSERT_MSG(
    (HYDRA_EXTERNAL_NS::thrust::detail::depend_on_instantiation<RandomAccessIterator, false>::value)
  , "unimplemented for this system"
  );
  return OutputType();
} // end reduce()


} // end namespace generic
} // end namespace detail
} // end namespace system
} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust

HYDRA_EXTERNAL_NAMESPACE_END
