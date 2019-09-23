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
#include <hydra/detail/external/thrust/system/detail/generic/scatter.h>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/thrust/functional.h>
#include <hydra/detail/external/thrust/transform.h>
#include <hydra/detail/external/thrust/iterator/permutation_iterator.h>

HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
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
__hydra_host__ __hydra_device__
  void scatter(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
               InputIterator1 first,
               InputIterator1 last,
               InputIterator2 map,
               RandomAccessIterator output)
{
  HYDRA_EXTERNAL_NS::thrust::transform(exec,
                    first,
                    last,
                    HYDRA_EXTERNAL_NS::thrust::make_permutation_iterator(output, map),
                    HYDRA_EXTERNAL_NS::thrust::identity<typename HYDRA_EXTERNAL_NS::thrust::iterator_value<InputIterator1>::type>());
} // end scatter()


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename RandomAccessIterator>
__hydra_host__ __hydra_device__
  void scatter_if(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                  InputIterator1 first,
                  InputIterator1 last,
                  InputIterator2 map,
                  InputIterator3 stencil,
                  RandomAccessIterator output)
{
  // default predicate is identity
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_value<InputIterator3>::type StencilType;
  HYDRA_EXTERNAL_NS::thrust::scatter_if(exec, first, last, map, stencil, output, HYDRA_EXTERNAL_NS::thrust::identity<StencilType>());
} // end scatter_if()


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename RandomAccessIterator,
         typename Predicate>
__hydra_host__ __hydra_device__
  void scatter_if(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                  InputIterator1 first,
                  InputIterator1 last,
                  InputIterator2 map,
                  InputIterator3 stencil,
                  RandomAccessIterator output,
                  Predicate pred)
{
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_value<InputIterator1>::type InputType;
  HYDRA_EXTERNAL_NS::thrust::transform_if(exec, first, last, stencil, HYDRA_EXTERNAL_NS::thrust::make_permutation_iterator(output, map), HYDRA_EXTERNAL_NS::thrust::identity<InputType>(), pred);
} // end scatter_if()


} // end namespace generic
} // end namespace detail
} // end namespace system
} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
HYDRA_EXTERNAL_NAMESPACE_END
