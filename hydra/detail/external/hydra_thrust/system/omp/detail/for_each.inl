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
#include <hydra/detail/external/hydra_thrust/detail/function.h>
#include <hydra/detail/external/hydra_thrust/detail/static_assert.h>
#include <hydra/detail/external/hydra_thrust/distance.h>
#include <hydra/detail/external/hydra_thrust/for_each.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/pragma_omp.h>

HYDRA_THRUST_NAMESPACE_BEGIN
namespace system
{
namespace omp
{
namespace detail
{

template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename Size,
         typename UnaryFunction>
RandomAccessIterator for_each_n(execution_policy<DerivedPolicy> &,
                                RandomAccessIterator first,
                                Size n,
                                UnaryFunction f)
{
  // we're attempting to launch an omp kernel, assert we're compiling with omp support
  // ========================================================================
  // X Note to the user: If you've found this line due to a compiler error, X
  // X you need to enable OpenMP support in your compiler.                  X
  // ========================================================================
  HYDRA_THRUST_STATIC_ASSERT_MSG(
    (hydra_thrust::detail::depend_on_instantiation<
      RandomAccessIterator, (HYDRA_THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == HYDRA_THRUST_TRUE)
    >::value)
  , "OpenMP compiler support is not enabled"
  );

  if (n <= 0) return first;  //empty range

  // create a wrapped function for f
  hydra_thrust::detail::wrapped_function<UnaryFunction,void> wrapped_f(f);

  // use a signed type for the iteration variable or suffer the consequences of warnings
  typedef typename hydra_thrust::iterator_difference<RandomAccessIterator>::type DifferenceType;
  DifferenceType signed_n = n;

  HYDRA_THRUST_PRAGMA_OMP(parallel for)
  for(DifferenceType i = 0;
      i < signed_n;
      ++i)
  {
    RandomAccessIterator temp = first + i;
    wrapped_f(*temp);
  }

  return first + n;
} // end for_each_n()

template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename UnaryFunction>
  RandomAccessIterator for_each(execution_policy<DerivedPolicy> &s,
                                RandomAccessIterator first,
                                RandomAccessIterator last,
                                UnaryFunction f)
{
  return omp::detail::for_each_n(s, first, hydra_thrust::distance(first,last), f);
} // end for_each()

} // end namespace detail
} // end namespace omp
} // end namespace system
HYDRA_THRUST_NAMESPACE_END

