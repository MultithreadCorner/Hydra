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

#include <hydra/detail/external/thrust/detail/config.h>
#include <hydra/detail/external/thrust/system/detail/generic/reverse.h>
#include <hydra/detail/external/thrust/advance.h>
#include <hydra/detail/external/thrust/distance.h>
#include <hydra/detail/external/thrust/detail/copy.h>
#include <hydra/detail/external/thrust/swap.h>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/thrust/iterator/reverse_iterator.h>

HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
{
namespace system
{
namespace detail
{
namespace generic
{


template<typename ExecutionPolicy, typename BidirectionalIterator>
__hydra_host__ __hydra_device__
  void reverse(HYDRA_EXTERNAL_NS::thrust::execution_policy<ExecutionPolicy> &exec,
               BidirectionalIterator first,
               BidirectionalIterator last)
{
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_difference<BidirectionalIterator>::type difference_type;

  // find the midpoint of [first,last)
  difference_type N = HYDRA_EXTERNAL_NS::thrust::distance(first, last);
  BidirectionalIterator mid(first);
  HYDRA_EXTERNAL_NS::thrust::advance(mid, N / 2);

  // swap elements of [first,mid) with [last - 1, mid)
  HYDRA_EXTERNAL_NS::thrust::swap_ranges(exec, first, mid, HYDRA_EXTERNAL_NS::thrust::make_reverse_iterator(last));
} // end reverse()


template<typename ExecutionPolicy,
         typename BidirectionalIterator,
         typename OutputIterator>
__hydra_host__ __hydra_device__
  OutputIterator reverse_copy(HYDRA_EXTERNAL_NS::thrust::execution_policy<ExecutionPolicy> &exec,
                              BidirectionalIterator first,
                              BidirectionalIterator last,
                              OutputIterator result)
{
  return HYDRA_EXTERNAL_NS::thrust::copy(exec,
                      HYDRA_EXTERNAL_NS::thrust::make_reverse_iterator(last),
                      HYDRA_EXTERNAL_NS::thrust::make_reverse_iterator(first),
                      result);
} // end reverse_copy()


} // end namespace generic
} // end namespace detail
} // end namespace system
} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust

HYDRA_EXTERNAL_NAMESPACE_END
