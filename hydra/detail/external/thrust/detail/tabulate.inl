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
#include <hydra/detail/external/thrust/tabulate.h>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/thrust/system/detail/generic/select_system.h>
#include <hydra/detail/external/thrust/system/detail/generic/tabulate.h>
#include <hydra/detail/external/thrust/system/detail/adl/tabulate.h>

HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
{


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename ForwardIterator, typename UnaryOperation>
__hydra_host__ __hydra_device__
  void tabulate(const HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                ForwardIterator first,
                ForwardIterator last,
                UnaryOperation unary_op)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::tabulate;
  return tabulate(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec)), first, last, unary_op);
} // end tabulate()


template<typename ForwardIterator, typename UnaryOperation>
  void tabulate(ForwardIterator first,
                ForwardIterator last,
                UnaryOperation unary_op)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;

  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return HYDRA_EXTERNAL_NS::thrust::tabulate(select_system(system), first, last, unary_op);
} // end tabulate()


} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust


HYDRA_EXTERNAL_NAMESPACE_END
