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
#include <hydra/detail/external/hydra_thrust/system/detail/generic/select_system.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/transform_reduce.h>
#include <hydra/detail/external/hydra_thrust/system/detail/adl/transform_reduce.h>

HYDRA_THRUST_NAMESPACE_BEGIN


__hydra_thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator,
         typename UnaryFunction,
         typename OutputType,
         typename BinaryFunction>
__host__ __device__
  OutputType transform_reduce(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                              InputIterator first,
                              InputIterator last,
                              UnaryFunction unary_op,
                              OutputType init,
                              BinaryFunction binary_op)
{
  using hydra_thrust::system::detail::generic::transform_reduce;
  return transform_reduce(hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)), first, last, unary_op, init, binary_op);
} // end transform_reduce()


template<typename InputIterator,
         typename UnaryFunction,
         typename OutputType,
         typename BinaryFunction>
  OutputType transform_reduce(InputIterator first,
                              InputIterator last,
                              UnaryFunction unary_op,
                              OutputType init,
                              BinaryFunction binary_op)
{
  using hydra_thrust::system::detail::generic::select_system;

  typedef typename hydra_thrust::iterator_system<InputIterator>::type System;

  System system;

  return hydra_thrust::transform_reduce(select_system(system), first, last, unary_op, init, binary_op);
} // end transform_reduce()


HYDRA_THRUST_NAMESPACE_END

