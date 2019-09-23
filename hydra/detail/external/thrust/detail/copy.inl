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
#include <hydra/detail/external/thrust/detail/copy.h>
#include <hydra/detail/external/thrust/system/detail/generic/select_system.h>
#include <hydra/detail/external/thrust/system/detail/generic/copy.h>
#include <hydra/detail/external/thrust/system/detail/adl/copy.h>

HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
{


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename InputIterator, typename OutputIterator>
__hydra_host__ __hydra_device__
  OutputIterator copy(const HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                      InputIterator first,
                      InputIterator last,
                      OutputIterator result)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::copy;
  return copy(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec)), first, last, result);
} // end copy()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename InputIterator, typename Size, typename OutputIterator>
__hydra_host__ __hydra_device__
  OutputIterator copy_n(const HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                        InputIterator first,
                        Size n,
                        OutputIterator result)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::copy_n;
  return copy_n(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec)), first, n, result);
} // end copy_n()


namespace detail
{


__thrust_exec_check_disable__ // because we might call e.g. std::ostream_iterator's constructor
template<typename System1,
         typename System2,
         typename InputIterator,
         typename OutputIterator>
__hydra_host__ __hydra_device__
  OutputIterator two_system_copy(const HYDRA_EXTERNAL_NS::thrust::execution_policy<System1> &system1,
                                 const HYDRA_EXTERNAL_NS::thrust::execution_policy<System2> &system2,
                                 InputIterator first,
                                 InputIterator last,
                                 OutputIterator result)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;

  return HYDRA_EXTERNAL_NS::thrust::copy(select_system(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(system1)), HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(system2))), first, last, result);
} // end two_system_copy()


__thrust_exec_check_disable__ // because we might call e.g. std::ostream_iterator's constructor
template<typename System1,
         typename System2,
         typename InputIterator,
         typename Size,
         typename OutputIterator>
__hydra_host__ __hydra_device__
  OutputIterator two_system_copy_n(const HYDRA_EXTERNAL_NS::thrust::execution_policy<System1> &system1,
                                   const HYDRA_EXTERNAL_NS::thrust::execution_policy<System2> &system2,
                                   InputIterator first,
                                   Size n,
                                   OutputIterator result)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;

  return HYDRA_EXTERNAL_NS::thrust::copy_n(select_system(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(system1)), HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(system2))), first, n, result);
} // end two_system_copy_n()


} // end detail


template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator first,
                      InputIterator last,
                      OutputIterator result)
{
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<InputIterator>::type  System1;
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<OutputIterator>::type System2;

  System1 system1;
  System2 system2;

  return HYDRA_EXTERNAL_NS::thrust::detail::two_system_copy(system1, system2, first, last, result);
} // end copy()


template<typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_n(InputIterator first,
                        Size n,
                        OutputIterator result)
{
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<InputIterator>::type  System1;
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<OutputIterator>::type System2;

  System1 system1;
  System2 system2;

  return HYDRA_EXTERNAL_NS::thrust::detail::two_system_copy_n(system1, system2, first, n, result);
} // end copy_n()


} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust


HYDRA_EXTERNAL_NAMESPACE_END
