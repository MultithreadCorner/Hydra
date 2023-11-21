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
#include <hydra/detail/external/hydra_thrust/system/detail/generic/mismatch.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/internal_functional.h>
#include <hydra/detail/external/hydra_thrust/find.h>

HYDRA_THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{


template<typename DerivedPolicy, typename InputIterator1, typename InputIterator2>
__host__ __device__
  hydra_thrust::pair<InputIterator1, InputIterator2>
    mismatch(hydra_thrust::execution_policy<DerivedPolicy> &exec,
             InputIterator1 first1,
             InputIterator1 last1,
             InputIterator2 first2)
{
  using namespace hydra_thrust::placeholders;

  return hydra_thrust::mismatch(exec, first1, last1, first2, _1 == _2);
} // end mismatch()


template<typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename BinaryPredicate>
__host__ __device__
  hydra_thrust::pair<InputIterator1, InputIterator2>
    mismatch(hydra_thrust::execution_policy<DerivedPolicy> &exec,
             InputIterator1 first1,
             InputIterator1 last1,
             InputIterator2 first2,
             BinaryPredicate pred)
{
  // Contributed by Erich Elsen
  typedef hydra_thrust::tuple<InputIterator1,InputIterator2> IteratorTuple;
  typedef hydra_thrust::zip_iterator<IteratorTuple>          ZipIterator;

  ZipIterator zipped_first = hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(first1,first2));
  ZipIterator zipped_last  = hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(last1, first2));

  ZipIterator result = hydra_thrust::find_if_not(exec, zipped_first, zipped_last, hydra_thrust::detail::tuple_binary_predicate<BinaryPredicate>(pred));

  return hydra_thrust::make_pair(hydra_thrust::get<0>(result.get_iterator_tuple()),
                           hydra_thrust::get<1>(result.get_iterator_tuple()));
} // end mismatch()


} // end generic
} // end detail
} // end system
HYDRA_THRUST_NAMESPACE_END

