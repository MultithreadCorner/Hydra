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


/*! \file find.inl
 *  \brief Inline file for find.h
 */

#include <hydra/detail/external/thrust/detail/config.h>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/thrust/system/detail/generic/select_system.h>
#include <hydra/detail/external/thrust/system/detail/generic/find.h>
#include <hydra/detail/external/thrust/system/detail/adl/find.h>

HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
{


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename InputIterator, typename T>
__hydra_host__ __hydra_device__
InputIterator find(const HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                   InputIterator first,
                   InputIterator last,
                   const T& value)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::find;
  return find(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec)), first, last, value);
} // end find()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename InputIterator, typename Predicate>
__hydra_host__ __hydra_device__
InputIterator find_if(const HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                      InputIterator first,
                      InputIterator last,
                      Predicate pred)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::find_if;
  return find_if(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec)), first, last, pred);
} // end find_if()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename InputIterator, typename Predicate>
__hydra_host__ __hydra_device__
InputIterator find_if_not(const HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                          InputIterator first,
                          InputIterator last,
                          Predicate pred)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::find_if_not;
  return find_if_not(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec)), first, last, pred);
} // end find_if_not()


template <typename InputIterator, typename T>
InputIterator find(InputIterator first,
                   InputIterator last,
                   const T& value)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;
  
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<InputIterator>::type System;
  
  System system;
  
  return HYDRA_EXTERNAL_NS::thrust::find(select_system(system), first, last, value);
}

template <typename InputIterator, typename Predicate>
InputIterator find_if(InputIterator first,
                      InputIterator last,
                      Predicate pred)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;
  
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<InputIterator>::type System;
  
  System system;
  
  return HYDRA_EXTERNAL_NS::thrust::find_if(select_system(system), first, last, pred);
}

template <typename InputIterator, typename Predicate>
InputIterator find_if_not(InputIterator first,
                          InputIterator last,
                          Predicate pred)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;
  
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<InputIterator>::type System;
  
  System system;
  
  return HYDRA_EXTERNAL_NS::thrust::find_if_not(select_system(system), first, last, pred);
}


} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust


HYDRA_EXTERNAL_NAMESPACE_END
