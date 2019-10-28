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
#include <hydra/detail/external/thrust/system/detail/generic/gather.h>
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
         typename InputIterator,
         typename RandomAccessIterator,
         typename OutputIterator>
__hydra_host__ __hydra_device__
  OutputIterator gather(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                        InputIterator                            map_first,
                        InputIterator                            map_last,
                        RandomAccessIterator                     input_first,
                        OutputIterator                           result)
{
  return HYDRA_EXTERNAL_NS::thrust::transform(exec,
                           HYDRA_EXTERNAL_NS::thrust::make_permutation_iterator(input_first, map_first),
                           HYDRA_EXTERNAL_NS::thrust::make_permutation_iterator(input_first, map_last),
                           result,
                           HYDRA_EXTERNAL_NS::thrust::identity<typename HYDRA_EXTERNAL_NS::thrust::iterator_value<RandomAccessIterator>::type>());
} // end gather()


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator,
         typename OutputIterator>
__hydra_host__ __hydra_device__
  OutputIterator gather_if(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                           InputIterator1                           map_first,
                           InputIterator1                           map_last,
                           InputIterator2                           stencil,
                           RandomAccessIterator                     input_first,
                           OutputIterator                           result)
{
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_value<InputIterator2>::type StencilType;
  return HYDRA_EXTERNAL_NS::thrust::gather_if(exec,
                           map_first,
                           map_last,
                           stencil,
                           input_first,
                           result,
                           HYDRA_EXTERNAL_NS::thrust::identity<StencilType>());
} // end gather_if()


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator,
         typename OutputIterator,
         typename Predicate>
__hydra_host__ __hydra_device__
  OutputIterator gather_if(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                           InputIterator1                           map_first,
                           InputIterator1                           map_last,
                           InputIterator2                           stencil,
                           RandomAccessIterator                     input_first,
                           OutputIterator                           result,
                           Predicate                                pred)
{
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_value<RandomAccessIterator>::type InputType;
  return HYDRA_EXTERNAL_NS::thrust::transform_if(exec,
                              HYDRA_EXTERNAL_NS::thrust::make_permutation_iterator(input_first, map_first),
                              HYDRA_EXTERNAL_NS::thrust::make_permutation_iterator(input_first, map_last),
                              stencil,
                              result,
                              HYDRA_EXTERNAL_NS::thrust::identity<InputType>(),
                              pred);
} // end gather_if()


} // end namespace generic
} // end namespace detail
} // end namespace system
} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust

HYDRA_EXTERNAL_NAMESPACE_END
