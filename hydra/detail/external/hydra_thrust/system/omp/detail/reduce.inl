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
#include <hydra/detail/external/hydra_thrust/system/omp/detail/reduce.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/default_decomposition.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/reduce_intervals.h>

HYDRA_THRUST_NAMESPACE_BEGIN
namespace system
{
namespace omp
{
namespace detail
{


template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce(execution_policy<DerivedPolicy> &exec,
                    InputIterator first,
                    InputIterator last,
                    OutputType init,
                    BinaryFunction binary_op)
{
  typedef typename hydra_thrust::iterator_difference<InputIterator>::type difference_type;

  const difference_type n = hydra_thrust::distance(first,last);

  // determine first and second level decomposition
  hydra_thrust::system::detail::internal::uniform_decomposition<difference_type> decomp1 = hydra_thrust::system::omp::detail::default_decomposition(n);
  hydra_thrust::system::detail::internal::uniform_decomposition<difference_type> decomp2(decomp1.size() + 1, 1, 1);

  // allocate storage for the initializer and partial sums
  // XXX use select_system for Tag
  hydra_thrust::detail::temporary_array<OutputType,DerivedPolicy> partial_sums(exec, decomp1.size() + 1);

  // set first element of temp array to init
  partial_sums[0] = init;

  // accumulate partial sums (first level reduction)
  hydra_thrust::system::omp::detail::reduce_intervals(exec, first, partial_sums.begin() + 1, binary_op, decomp1);

  // reduce partial sums (second level reduction)
  hydra_thrust::system::omp::detail::reduce_intervals(exec, partial_sums.begin(), partial_sums.begin(), binary_op, decomp2);

  return partial_sums[0];
} // end reduce()


} // end detail
} // end omp
} // end system
HYDRA_THRUST_NAMESPACE_END

