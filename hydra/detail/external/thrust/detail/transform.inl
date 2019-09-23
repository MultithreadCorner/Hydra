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


/*! \file transform.inl
 *  \brief Inline file for transform.h.
 */

#include <hydra/detail/external/thrust/transform.h>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/thrust/system/detail/generic/select_system.h>
#include <hydra/detail/external/thrust/system/detail/generic/transform.h>
#include <hydra/detail/external/thrust/system/detail/adl/transform.h>

HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
{


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction>
__hydra_host__ __hydra_device__
  OutputIterator transform(const HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                           InputIterator first, InputIterator last,
                           OutputIterator result,
                           UnaryFunction op)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::transform;
  return transform(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec)), first, last, result, op);
} // end transform()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryFunction>
__hydra_host__ __hydra_device__
  OutputIterator transform(const HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                           InputIterator1 first1, InputIterator1 last1,
                           InputIterator2 first2,
                           OutputIterator result,
                           BinaryFunction op)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::transform;
  return transform(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec)), first1, last1, first2, result, op);
} // end transform()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator,
         typename ForwardIterator,
         typename UnaryFunction,
         typename Predicate>
__hydra_host__ __hydra_device__
  ForwardIterator transform_if(const HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                               InputIterator first, InputIterator last,
                               ForwardIterator result,
                               UnaryFunction op,
                               Predicate pred)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::transform_if;
  return transform_if(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec)), first, last, result, op, pred);
} // end transform_if()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename ForwardIterator,
         typename UnaryFunction,
         typename Predicate>
__hydra_host__ __hydra_device__
  ForwardIterator transform_if(const HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                               InputIterator1 first, InputIterator1 last,
                               InputIterator2 stencil,
                               ForwardIterator result,
                               UnaryFunction op,
                               Predicate pred)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::transform_if;
  return transform_if(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec)), first, last, stencil, result, op, pred);
} // end transform_if()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename ForwardIterator,
         typename BinaryFunction,
         typename Predicate>
__hydra_host__ __hydra_device__
  ForwardIterator transform_if(const HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                               InputIterator1 first1, InputIterator1 last1,
                               InputIterator2 first2,
                               InputIterator3 stencil,
                               ForwardIterator result,
                               BinaryFunction binary_op,
                               Predicate pred)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::transform_if;
  return transform_if(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec)), first1, last1, first2, stencil, result, binary_op, pred);
} // end transform_if()


template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction>
  OutputIterator transform(InputIterator first,
                           InputIterator last,
                           OutputIterator result,
                           UnaryFunction op)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;

  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<InputIterator>::type  System1;
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<OutputIterator>::type System2;

  System1 system1;
  System2 system2;

  return HYDRA_EXTERNAL_NS::thrust::transform(select_system(system1,system2), first, last, result, op);
} // end transform()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryFunction>
  OutputIterator transform(InputIterator1 first1,
                           InputIterator1 last1,
                           InputIterator2 first2,
                           OutputIterator result,
                           BinaryFunction op)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;

  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<InputIterator1>::type System1;
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<InputIterator2>::type System2;
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<OutputIterator>::type System3;

  System1 system1;
  System2 system2;
  System3 system3;

  return HYDRA_EXTERNAL_NS::thrust::transform(select_system(system1,system2,system3), first1, last1, first2, result, op);
} // end transform()


template<typename InputIterator,
         typename ForwardIterator,
         typename UnaryFunction,
         typename Predicate>
  ForwardIterator transform_if(InputIterator first,
                               InputIterator last,
                               ForwardIterator result,
                               UnaryFunction unary_op,
                               Predicate pred)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;

  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<InputIterator>::type   System1;
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<ForwardIterator>::type System2;

  System1 system1;
  System2 system2;

  return HYDRA_EXTERNAL_NS::thrust::transform_if(select_system(system1,system2), first, last, result, unary_op, pred);
} // end transform_if()


template<typename InputIterator1,
         typename InputIterator2,
         typename ForwardIterator,
         typename UnaryFunction,
         typename Predicate>
  ForwardIterator transform_if(InputIterator1 first,
                               InputIterator1 last,
                               InputIterator2 stencil,
                               ForwardIterator result,
                               UnaryFunction unary_op,
                               Predicate pred)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;

  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<InputIterator1>::type  System1;
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<InputIterator2>::type  System2;
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<ForwardIterator>::type System3;

  System1 system1;
  System2 system2;
  System3 system3;

  return HYDRA_EXTERNAL_NS::thrust::transform_if(select_system(system1,system2,system3), first, last, stencil, result, unary_op, pred);
} // end transform_if()


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename ForwardIterator,
         typename BinaryFunction,
         typename Predicate>
  ForwardIterator transform_if(InputIterator1 first1,
                               InputIterator1 last1,
                               InputIterator2 first2,
                               InputIterator3 stencil,
                               ForwardIterator result,
                               BinaryFunction binary_op,
                               Predicate pred)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;

  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<InputIterator1>::type  System1;
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<InputIterator2>::type  System2;
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<InputIterator3>::type  System3;
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<ForwardIterator>::type System4;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;

  return HYDRA_EXTERNAL_NS::thrust::transform_if(select_system(system1,system2,system3,system4), first1, last1, first2, stencil, result, binary_op, pred);
} // end transform_if()


} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust


HYDRA_EXTERNAL_NAMESPACE_END
