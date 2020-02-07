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

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/reverse.h>
#include <hydra/detail/external/hydra_thrust/advance.h>
#include <hydra/detail/external/hydra_thrust/distance.h>
#include <hydra/detail/external/hydra_thrust/detail/copy.h>
#include <hydra/detail/external/hydra_thrust/swap.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/iterator/reverse_iterator.h>

namespace hydra_thrust
{
namespace system
{
namespace detail
{
namespace generic
{


template<typename ExecutionPolicy, typename BidirectionalIterator>
__host__ __device__
  void reverse(hydra_thrust::execution_policy<ExecutionPolicy> &exec,
               BidirectionalIterator first,
               BidirectionalIterator last)
{
  typedef typename hydra_thrust::iterator_difference<BidirectionalIterator>::type difference_type;

  // find the midpoint of [first,last)
  difference_type N = hydra_thrust::distance(first, last);
  BidirectionalIterator mid(first);
  hydra_thrust::advance(mid, N / 2);

  // swap elements of [first,mid) with [last - 1, mid)
  hydra_thrust::swap_ranges(exec, first, mid, hydra_thrust::make_reverse_iterator(last));
} // end reverse()


template<typename ExecutionPolicy,
         typename BidirectionalIterator,
         typename OutputIterator>
__host__ __device__
  OutputIterator reverse_copy(hydra_thrust::execution_policy<ExecutionPolicy> &exec,
                              BidirectionalIterator first,
                              BidirectionalIterator last,
                              OutputIterator result)
{
  return hydra_thrust::copy(exec,
                      hydra_thrust::make_reverse_iterator(last),
                      hydra_thrust::make_reverse_iterator(first),
                      result);
} // end reverse_copy()


} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace hydra_thrust


