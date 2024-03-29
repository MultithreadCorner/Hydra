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
#include <hydra/detail/external/hydra_thrust/system/detail/generic/tag.h>
#include <hydra/detail/external/hydra_thrust/detail/internal_functional.h>
#include <hydra/detail/external/hydra_thrust/find.h>
#include <hydra/detail/external/hydra_thrust/logical.h>

HYDRA_THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{


template<typename ExecutionPolicy, typename InputIterator, typename Predicate>
__host__ __device__
bool all_of(hydra_thrust::execution_policy<ExecutionPolicy> &exec, InputIterator first, InputIterator last, Predicate pred)
{
  return hydra_thrust::find_if(exec, first, last, hydra_thrust::detail::not1(pred)) == last;
}


template<typename ExecutionPolicy, typename InputIterator, typename Predicate>
__host__ __device__
bool any_of(hydra_thrust::execution_policy<ExecutionPolicy> &exec, InputIterator first, InputIterator last, Predicate pred)
{
  return hydra_thrust::find_if(exec, first, last, pred) != last;
}


template<typename ExecutionPolicy, typename InputIterator, typename Predicate>
__host__ __device__
bool none_of(hydra_thrust::execution_policy<ExecutionPolicy> &exec, InputIterator first, InputIterator last, Predicate pred)
{
  return !hydra_thrust::any_of(exec, first, last, pred);
}


} // end generic
} // end detail
} // end system
HYDRA_THRUST_NAMESPACE_END

