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


/*! \file for_each.inl
 *  \brief Inline file for for_each.h.
 */

#include <hydra/detail/external/thrust/detail/config.h>
#include <hydra/detail/external/thrust/for_each.h>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/thrust/system/detail/generic/select_system.h>
#include <hydra/detail/external/thrust/system/detail/generic/for_each.h>
#include <hydra/detail/external/thrust/system/detail/adl/for_each.h>

HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
{

__thrust_exec_check_disable__ 
template<typename DerivedPolicy,
         typename InputIterator,
         typename UnaryFunction>
__hydra_host__ __hydra_device__
  InputIterator for_each(const HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                         InputIterator first,
                         InputIterator last,
                         UnaryFunction f)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::for_each;

  return for_each(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec)), first, last, f);
}


template<typename InputIterator,
         typename UnaryFunction>
InputIterator for_each(InputIterator first,
                       InputIterator last,
                       UnaryFunction f)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<InputIterator>::type System;

  System system;
  return HYDRA_EXTERNAL_NS::thrust::for_each(select_system(system), first, last, f);
} // end for_each()

__thrust_exec_check_disable__ 
template<typename DerivedPolicy, typename InputIterator, typename Size, typename UnaryFunction>
__hydra_host__ __hydra_device__
  InputIterator for_each_n(const HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                           InputIterator first,
                           Size n,
                           UnaryFunction f)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::for_each_n;

  return for_each_n(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec)), first, n, f);
} // end for_each_n()


template<typename InputIterator,
         typename Size,
         typename UnaryFunction>
InputIterator for_each_n(InputIterator first,
                         Size n,
                         UnaryFunction f)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;

  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<InputIterator>::type System;

  System system;
  return HYDRA_EXTERNAL_NS::thrust::for_each_n(select_system(system), first, n, f);
} // end for_each_n()


} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust


HYDRA_EXTERNAL_NAMESPACE_END
