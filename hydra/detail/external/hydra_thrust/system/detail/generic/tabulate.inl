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
#include <hydra/detail/external/hydra_thrust/system/detail/generic/tabulate.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/transform.h>
#include <hydra/detail/external/hydra_thrust/distance.h>
#include <hydra/detail/external/hydra_thrust/iterator/counting_iterator.h>

namespace hydra_thrust
{
namespace system
{
namespace detail
{
namespace generic
{


template<typename DerivedPolicy,
         typename ForwardIterator,
         typename UnaryOperation>
__host__ __device__
  void tabulate(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                ForwardIterator first,
                ForwardIterator last,
                UnaryOperation unary_op)
{
  typedef typename iterator_difference<ForwardIterator>::type difference_type;

  // by default, counting_iterator uses a 64b difference_type on 32b platforms to avoid overflowing its counter.
  // this causes problems when a zip_iterator is created in transform's implementation -- ForwardIterator is
  // incremented by a 64b difference_type and some compilers warn
  // to avoid this, specify the counting_iterator's difference_type to be the same as ForwardIterator's.
  hydra_thrust::counting_iterator<difference_type, hydra_thrust::use_default, hydra_thrust::use_default, difference_type> iter(0);

  hydra_thrust::transform(exec, iter, iter + hydra_thrust::distance(first, last), first, unary_op);
} // end tabulate()


} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace hydra_thrust


