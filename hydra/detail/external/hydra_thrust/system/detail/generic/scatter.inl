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
#include <hydra/detail/external/hydra_thrust/system/detail/generic/scatter.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/functional.h>
#include <hydra/detail/external/hydra_thrust/transform.h>
#include <hydra/detail/external/hydra_thrust/iterator/permutation_iterator.h>

namespace hydra_thrust
{
namespace system
{
namespace detail
{
namespace generic
{


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator>
__host__ __device__
  void scatter(hydra_thrust::execution_policy<DerivedPolicy> &exec,
               InputIterator1 first,
               InputIterator1 last,
               InputIterator2 map,
               RandomAccessIterator output)
{
  hydra_thrust::transform(exec,
                    first,
                    last,
                    hydra_thrust::make_permutation_iterator(output, map),
                    hydra_thrust::identity<typename hydra_thrust::iterator_value<InputIterator1>::type>());
} // end scatter()


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename RandomAccessIterator>
__host__ __device__
  void scatter_if(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                  InputIterator1 first,
                  InputIterator1 last,
                  InputIterator2 map,
                  InputIterator3 stencil,
                  RandomAccessIterator output)
{
  // default predicate is identity
  typedef typename hydra_thrust::iterator_value<InputIterator3>::type StencilType;
  hydra_thrust::scatter_if(exec, first, last, map, stencil, output, hydra_thrust::identity<StencilType>());
} // end scatter_if()


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename RandomAccessIterator,
         typename Predicate>
__host__ __device__
  void scatter_if(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                  InputIterator1 first,
                  InputIterator1 last,
                  InputIterator2 map,
                  InputIterator3 stencil,
                  RandomAccessIterator output,
                  Predicate pred)
{
  typedef typename hydra_thrust::iterator_value<InputIterator1>::type InputType;
  hydra_thrust::transform_if(exec, first, last, stencil, hydra_thrust::make_permutation_iterator(output, map), hydra_thrust::identity<InputType>(), pred);
} // end scatter_if()


} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace hydra_thrust

