/*
 *  Copyright 2008-2020 NVIDIA Corporation
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

/*! \file async/scan.h
 *  \brief Functions for asynchronously computing prefix scans.
 */

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/cpp14_required.h>

#if HYDRA_THRUST_CPP_DIALECT >= 2014

#include <hydra/detail/external/hydra_thrust/detail/execution_policy.h>
#include <hydra/detail/external/hydra_thrust/detail/select_system.h>
#include <hydra/detail/external/hydra_thrust/detail/static_assert.h>

#include <hydra/detail/external/hydra_thrust/system/detail/adl/async/scan.h>

#include <hydra/detail/external/hydra_thrust/type_traits/is_execution_policy.h>
#include <hydra/detail/external/hydra_thrust/type_traits/logical_metafunctions.h>
#include <hydra/detail/external/hydra_thrust/type_traits/remove_cvref.h>

#include <hydra/detail/external/hydra_thrust/future.h>

HYDRA_THRUST_NAMESPACE_BEGIN

namespace async
{

// Fallback implementations used when no overloads are found via ADL:
namespace unimplemented
{

template <typename DerivedPolicy,
          typename ForwardIt,
          typename Sentinel,
          typename OutputIt,
          typename BinaryOp>
event<DerivedPolicy>
async_inclusive_scan(hydra_thrust::execution_policy<DerivedPolicy>&,
                     ForwardIt,
                     Sentinel,
                     OutputIt,
                     BinaryOp)
{
  HYDRA_THRUST_STATIC_ASSERT_MSG(
    (hydra_thrust::detail::depend_on_instantiation<ForwardIt, false>::value),
    "this algorithm is not implemented for the specified system"
  );
  return {};
}

template <typename DerivedPolicy,
          typename ForwardIt,
          typename Sentinel,
          typename OutputIt,
          typename InitialValueType,
          typename BinaryOp>
event<DerivedPolicy>
async_exclusive_scan(hydra_thrust::execution_policy<DerivedPolicy>&,
                     ForwardIt,
                     Sentinel,
                     OutputIt,
                     InitialValueType,
                     BinaryOp)
{
  HYDRA_THRUST_STATIC_ASSERT_MSG(
    (hydra_thrust::detail::depend_on_instantiation<ForwardIt, false>::value),
    "this algorithm is not implemented for the specified system"
  );
  return {};
}

} // namespace unimplemented

namespace inclusive_scan_detail
{

// Include fallback implementation for ADL failures
using hydra_thrust::async::unimplemented::async_inclusive_scan;

// Implementation of the hydra_thrust::async::inclusive_scan CPO.
struct inclusive_scan_fn final
{
  template <typename DerivedPolicy,
            typename ForwardIt,
            typename Sentinel,
            typename OutputIt,
            typename BinaryOp>
  auto
  operator()(hydra_thrust::detail::execution_policy_base<DerivedPolicy> const& exec,
             ForwardIt&& first,
             Sentinel&& last,
             OutputIt&& out,
             BinaryOp&& op) const
  // ADL dispatch.
  HYDRA_THRUST_RETURNS(
    async_inclusive_scan(
      hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)),
      HYDRA_THRUST_FWD(first),
      HYDRA_THRUST_FWD(last),
      HYDRA_THRUST_FWD(out),
      HYDRA_THRUST_FWD(op)
    )
  )

  template <typename DerivedPolicy,
            typename ForwardIt,
            typename Sentinel,
            typename OutputIt>
  auto
  operator()(hydra_thrust::detail::execution_policy_base<DerivedPolicy> const& exec,
             ForwardIt&& first,
             Sentinel&& last,
             OutputIt&& out) const
  // ADL dispatch.
  HYDRA_THRUST_RETURNS(
    async_inclusive_scan(
      hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)),
      HYDRA_THRUST_FWD(first),
      HYDRA_THRUST_FWD(last),
      HYDRA_THRUST_FWD(out),
      hydra_thrust::plus<>{}
    )
  )

  template <typename ForwardIt,
            typename Sentinel,
            typename OutputIt,
            typename BinaryOp,
            typename = std::enable_if_t<!is_execution_policy_v<remove_cvref_t<ForwardIt>>>>
  auto operator()(ForwardIt&& first,
                  Sentinel&& last,
                  OutputIt&& out,
                  BinaryOp&& op) const
  // ADL dispatch.
  HYDRA_THRUST_RETURNS(
    async_inclusive_scan(
      hydra_thrust::detail::select_system(
        iterator_system_t<remove_cvref_t<ForwardIt>>{},
        iterator_system_t<remove_cvref_t<OutputIt>>{}
      ),
      HYDRA_THRUST_FWD(first),
      HYDRA_THRUST_FWD(last),
      HYDRA_THRUST_FWD(out),
      HYDRA_THRUST_FWD(op)
    )
  )

  template <typename ForwardIt, typename Sentinel, typename OutputIt>
  auto operator()(ForwardIt&& first, Sentinel&& last, OutputIt&& out) const
  // ADL dispatch.
  HYDRA_THRUST_RETURNS(
    async_inclusive_scan(
      hydra_thrust::detail::select_system(
        iterator_system_t<remove_cvref_t<ForwardIt>>{},
        iterator_system_t<remove_cvref_t<OutputIt>>{}
      ),
      HYDRA_THRUST_FWD(first),
      HYDRA_THRUST_FWD(last),
      HYDRA_THRUST_FWD(out),
      hydra_thrust::plus<>{}
    )
  )
};

} // namespace inclusive_scan_detail

HYDRA_THRUST_INLINE_CONSTANT inclusive_scan_detail::inclusive_scan_fn inclusive_scan{};

namespace exclusive_scan_detail
{

// Include fallback implementation for ADL failures
using hydra_thrust::async::unimplemented::async_exclusive_scan;

// Implementation of the hydra_thrust::async::exclusive_scan CPO.
struct exclusive_scan_fn final
{
  template <typename DerivedPolicy,
            typename ForwardIt,
            typename Sentinel,
            typename OutputIt,
            typename InitialValueType,
            typename BinaryOp>
  auto
  operator()(hydra_thrust::detail::execution_policy_base<DerivedPolicy> const& exec,
             ForwardIt&& first,
             Sentinel&& last,
             OutputIt&& out,
             InitialValueType&& init,
             BinaryOp&& op) const
  // ADL dispatch.
  HYDRA_THRUST_RETURNS(
    async_exclusive_scan(
      hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)),
      HYDRA_THRUST_FWD(first),
      HYDRA_THRUST_FWD(last),
      HYDRA_THRUST_FWD(out),
      HYDRA_THRUST_FWD(init),
      HYDRA_THRUST_FWD(op)
    )
  )

  template <typename DerivedPolicy,
            typename ForwardIt,
            typename Sentinel,
            typename OutputIt,
            typename InitialValueType>
  auto
  operator()(hydra_thrust::detail::execution_policy_base<DerivedPolicy> const& exec,
             ForwardIt&& first,
             Sentinel&& last,
             OutputIt&& out,
             InitialValueType&& init) const
  // ADL dispatch.
  HYDRA_THRUST_RETURNS(
    async_exclusive_scan(
      hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)),
      HYDRA_THRUST_FWD(first),
      HYDRA_THRUST_FWD(last),
      HYDRA_THRUST_FWD(out),
      HYDRA_THRUST_FWD(init),
      hydra_thrust::plus<>{}
    )
  )

  template <typename DerivedPolicy,
            typename ForwardIt,
            typename Sentinel,
            typename OutputIt>
  auto
  operator()(hydra_thrust::detail::execution_policy_base<DerivedPolicy> const& exec,
             ForwardIt&& first,
             Sentinel&& last,
             OutputIt&& out) const
  // ADL dispatch.
  HYDRA_THRUST_RETURNS(
    async_exclusive_scan(
      hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)),
      HYDRA_THRUST_FWD(first),
      HYDRA_THRUST_FWD(last),
      HYDRA_THRUST_FWD(out),
      iterator_value_t<remove_cvref_t<ForwardIt>>{},
      hydra_thrust::plus<>{}
    )
  )

  template <typename ForwardIt,
            typename Sentinel,
            typename OutputIt,
            typename InitialValueType,
            typename BinaryOp,
            typename = std::enable_if_t<!is_execution_policy_v<remove_cvref_t<ForwardIt>>>>
  auto
  operator()(ForwardIt&& first,
             Sentinel&& last,
             OutputIt&& out,
             InitialValueType&& init,
             BinaryOp&& op) const
  // ADL dispatch.
  HYDRA_THRUST_RETURNS(
    async_exclusive_scan(
      hydra_thrust::detail::select_system(
        iterator_system_t<remove_cvref_t<ForwardIt>>{},
        iterator_system_t<remove_cvref_t<OutputIt>>{}
      ),
      HYDRA_THRUST_FWD(first),
      HYDRA_THRUST_FWD(last),
      HYDRA_THRUST_FWD(out),
      HYDRA_THRUST_FWD(init),
      HYDRA_THRUST_FWD(op)
    )
  )

  template <typename ForwardIt,
            typename Sentinel,
            typename OutputIt,
            typename InitialValueType,
            typename = std::enable_if_t<!is_execution_policy_v<remove_cvref_t<ForwardIt>>>>
  auto
  operator()(ForwardIt&& first,
             Sentinel&& last,
             OutputIt&& out,
             InitialValueType&& init) const
  // ADL dispatch.
  HYDRA_THRUST_RETURNS(
    async_exclusive_scan(
      hydra_thrust::detail::select_system(
        iterator_system_t<remove_cvref_t<ForwardIt>>{},
        iterator_system_t<remove_cvref_t<OutputIt>>{}
      ),
      HYDRA_THRUST_FWD(first),
      HYDRA_THRUST_FWD(last),
      HYDRA_THRUST_FWD(out),
      HYDRA_THRUST_FWD(init),
      hydra_thrust::plus<>{}
    )
  )

  template <typename ForwardIt, typename Sentinel, typename OutputIt>
  auto operator()(ForwardIt&& first,
                  Sentinel&& last,
                  OutputIt&& out) const
  // ADL dispatch.
  HYDRA_THRUST_RETURNS(
    async_exclusive_scan(
      hydra_thrust::detail::select_system(
        iterator_system_t<remove_cvref_t<ForwardIt>>{},
        iterator_system_t<remove_cvref_t<OutputIt>>{}
      ),
      HYDRA_THRUST_FWD(first),
      HYDRA_THRUST_FWD(last),
      HYDRA_THRUST_FWD(out),
      iterator_value_t<remove_cvref_t<ForwardIt>>{},
      hydra_thrust::plus<>{}
    )
  )
};

} // namespace exclusive_scan_detail

HYDRA_THRUST_INLINE_CONSTANT exclusive_scan_detail::exclusive_scan_fn exclusive_scan{};

} // namespace async

HYDRA_THRUST_NAMESPACE_END

#endif
