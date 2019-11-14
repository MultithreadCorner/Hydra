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


/*! \file gather.inl
 *  \brief Inline file for gather.h.
 */

#include <hydra/detail/external/thrust/gather.h>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/thrust/system/detail/generic/select_system.h>
#include <hydra/detail/external/thrust/system/detail/generic/gather.h>
#include <hydra/detail/external/thrust/system/detail/adl/gather.h>

HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
{


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator,
         typename RandomAccessIterator,
         typename OutputIterator>
__hydra_host__ __hydra_device__
  OutputIterator gather(const HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                        InputIterator                                               map_first,
                        InputIterator                                               map_last,
                        RandomAccessIterator                                        input_first,
                        OutputIterator                                              result)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::gather;
  return gather(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec)), map_first, map_last, input_first, result);
} // end gather()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator,
         typename OutputIterator>
__hydra_host__ __hydra_device__
  OutputIterator gather_if(const HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                           InputIterator1                                              map_first,
                           InputIterator1                                              map_last,
                           InputIterator2                                              stencil,
                           RandomAccessIterator                                        input_first,
                           OutputIterator                                              result)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::gather_if;
  return gather_if(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec)), map_first, map_last, stencil, input_first, result);
} // end gather_if()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator,
         typename OutputIterator,
         typename Predicate>
__hydra_host__ __hydra_device__
  OutputIterator gather_if(const HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                           InputIterator1                                              map_first,
                           InputIterator1                                              map_last,
                           InputIterator2                                              stencil,
                           RandomAccessIterator                                        input_first,
                           OutputIterator                                              result,
                           Predicate                                                   pred)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::gather_if;
  return gather_if(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec)), map_first, map_last, stencil, input_first, result, pred);
} // end gather_if()


template<typename InputIterator,
         typename RandomAccessIterator,
         typename OutputIterator>
  OutputIterator gather(InputIterator        map_first,
                        InputIterator        map_last,
                        RandomAccessIterator input_first,
                        OutputIterator       result)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;

  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<InputIterator>::type        System1; 
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<RandomAccessIterator>::type System2; 
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<OutputIterator>::type       System3; 

  System1 system1;
  System2 system2;
  System3 system3;

  return HYDRA_EXTERNAL_NS::thrust::gather(select_system(system1,system2,system3), map_first, map_last, input_first, result);
} // end gather()


template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator,
         typename OutputIterator>
  OutputIterator gather_if(InputIterator1       map_first,
                           InputIterator1       map_last,
                           InputIterator2       stencil,
                           RandomAccessIterator input_first,
                           OutputIterator       result)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;

  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<InputIterator1>::type       System1; 
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<InputIterator2>::type       System2; 
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<RandomAccessIterator>::type System3; 
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<OutputIterator>::type       System4; 

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;

  return HYDRA_EXTERNAL_NS::thrust::gather_if(select_system(system1,system2,system3,system4), map_first, map_last, stencil, input_first, result);
} // end gather_if()


template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator gather_if(InputIterator1       map_first,
                           InputIterator1       map_last,
                           InputIterator2       stencil,
                           RandomAccessIterator input_first,
                           OutputIterator       result,
                           Predicate            pred)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;

  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<InputIterator1>::type       System1; 
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<InputIterator2>::type       System2; 
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<RandomAccessIterator>::type System3; 
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<OutputIterator>::type       System4; 

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;

  return HYDRA_EXTERNAL_NS::thrust::gather_if(select_system(system1,system2,system3,system4), map_first, map_last, stencil, input_first, result, pred);
} // end gather_if()


} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust


HYDRA_EXTERNAL_NAMESPACE_END
