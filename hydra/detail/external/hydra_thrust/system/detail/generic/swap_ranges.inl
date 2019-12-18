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
#include <hydra/detail/external/hydra_thrust/system/detail/generic/swap_ranges.h>
#include <hydra/detail/external/hydra_thrust/tuple.h>
#include <hydra/detail/external/hydra_thrust/iterator/zip_iterator.h>
#include <hydra/detail/external/hydra_thrust/detail/internal_functional.h>
#include <hydra/detail/external/hydra_thrust/for_each.h>

namespace hydra_thrust
{
namespace system
{
namespace detail
{
namespace generic
{
namespace detail
{


// XXX define this here rather than in internal_functional.h
// to avoid circular dependence between swap.h & internal_functional.h
struct swap_pair_elements
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    // use unqualified swap to allow ADL to catch any user-defined swap
    using hydra_thrust::swap;
    swap(hydra_thrust::get<0>(t), hydra_thrust::get<1>(t));
  }
}; // end swap_pair_elements


} // end detail


template<typename DerivedPolicy,
         typename ForwardIterator1,
         typename ForwardIterator2>
__host__ __device__
  ForwardIterator2 swap_ranges(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                               ForwardIterator1 first1,
                               ForwardIterator1 last1,
                               ForwardIterator2 first2)
{
  typedef hydra_thrust::tuple<ForwardIterator1,ForwardIterator2> IteratorTuple;
  typedef hydra_thrust::zip_iterator<IteratorTuple>              ZipIterator;

  ZipIterator result = hydra_thrust::for_each(exec,
                                        hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(first1, first2)),
                                        hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(last1,  first2)),
                                        detail::swap_pair_elements());
  return hydra_thrust::get<1>(result.get_iterator_tuple());
} // end swap_ranges()


} // end generic
} // end detail
} // end system
} // end hydra_thrust

