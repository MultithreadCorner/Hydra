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

#pragma once

#include <hydra/detail/external/thrust/detail/config.h>

HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
{
namespace detail
{

struct execution_policy_marker {};

// execution_policy_base serves as a guard against
// inifinite recursion in thrust entry points:
//
// template<typename DerivedPolicy>
// void foo(const HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> &s)
// {
//   using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::foo;
//
//   foo(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(s));
// }
//
// foo is not recursive when
// 1. DerivedPolicy is derived from HYDRA_EXTERNAL_NS::thrust::execution_policy below
// 2. generic::foo takes HYDRA_EXTERNAL_NS::thrust::execution_policy as a parameter
template<typename DerivedPolicy>
struct execution_policy_base : execution_policy_marker {};


template<typename DerivedPolicy>
HYDRA_THRUST_CONSTEXPR __hydra_host__ __hydra_device__
execution_policy_base<DerivedPolicy> &strip_const(const execution_policy_base<DerivedPolicy> &x)
{
  return const_cast<execution_policy_base<DerivedPolicy>&>(x);
}


template<typename DerivedPolicy>
HYDRA_THRUST_CONSTEXPR __hydra_host__ __hydra_device__
DerivedPolicy &derived_cast(execution_policy_base<DerivedPolicy> &x)
{
  return static_cast<DerivedPolicy&>(x);
}


template<typename DerivedPolicy>
HYDRA_THRUST_CONSTEXPR __hydra_host__ __hydra_device__
const DerivedPolicy &derived_cast(const execution_policy_base<DerivedPolicy> &x)
{
  return static_cast<const DerivedPolicy&>(x);
}

} // end detail

template<typename DerivedPolicy>
  struct execution_policy
    : HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy>
{};

} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust


HYDRA_EXTERNAL_NAMESPACE_END
