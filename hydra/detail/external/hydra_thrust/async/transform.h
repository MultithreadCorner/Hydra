/*
 *  Copyright 2008-2018 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a transform of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file async/transform.h
 *  \brief Functions for asynchronously transforming a range.
 */

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/cpp11_required.h>
#include <hydra/detail/external/hydra_thrust/detail/modern_gcc_required.h>

#if HYDRA_THRUST_CPP_DIALECT >= 2011 && !defined(HYDRA_THRUST_LEGACY_GCC)

#include <hydra/detail/external/hydra_thrust/detail/static_assert.h>
#include <hydra/detail/external/hydra_thrust/detail/select_system.h>
#include <hydra/detail/external/hydra_thrust/type_traits/remove_cvref.h>
#include <hydra/detail/external/hydra_thrust/system/detail/adl/async/transform.h>

#include <hydra/detail/external/hydra_thrust/event.h>

HYDRA_THRUST_BEGIN_NS

namespace async
{

namespace unimplemented
{

template <
  typename DerivedPolicy
, typename ForwardIt, typename Sentinel, typename OutputIt
, typename UnaryOperation
>
__host__
event<DerivedPolicy>
async_transform(
  hydra_thrust::execution_policy<DerivedPolicy>& exec
, ForwardIt first, Sentinel last, OutputIt output, UnaryOperation op
)
{
  HYDRA_THRUST_STATIC_ASSERT_MSG(
    (hydra_thrust::detail::depend_on_instantiation<ForwardIt, false>::value)
  , "this algorithm is not implemented for the specified system"
  );
  return {};
}

} // namespace unimplemented

namespace transform_detail
{

using hydra_thrust::async::unimplemented::async_transform;

struct transform_fn final
{
  template <
    typename DerivedPolicy
  , typename ForwardIt, typename Sentinel, typename OutputIt
  , typename UnaryOperation
  >
  __host__
  static auto
  call(
    hydra_thrust::detail::execution_policy_base<DerivedPolicy> const& exec
  , ForwardIt&& first, Sentinel&& last
  , OutputIt&& output
  , UnaryOperation&& op
  )
  // ADL dispatch.
  HYDRA_THRUST_DECLTYPE_RETURNS(
    async_transform(
      hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec))
    , HYDRA_THRUST_FWD(first), HYDRA_THRUST_FWD(last)
    , HYDRA_THRUST_FWD(output)
    , HYDRA_THRUST_FWD(op)
    )
  )

  template <
    typename ForwardIt, typename Sentinel, typename OutputIt
  , typename UnaryOperation
  >
  __host__
  static auto call(
    ForwardIt&& first, Sentinel&& last
  , OutputIt&& output
  , UnaryOperation&& op
  )
  HYDRA_THRUST_DECLTYPE_RETURNS(
    transform_fn::call(
      hydra_thrust::detail::select_system(
        typename iterator_system<remove_cvref_t<ForwardIt>>::type{}
      , typename iterator_system<remove_cvref_t<OutputIt>>::type{}
      )
    , HYDRA_THRUST_FWD(first), HYDRA_THRUST_FWD(last)
    , HYDRA_THRUST_FWD(output)
    , HYDRA_THRUST_FWD(op)
    )
  )

  template <typename... Args>
  HYDRA_THRUST_NODISCARD __host__
  auto operator()(Args&&... args) const
  HYDRA_THRUST_DECLTYPE_RETURNS(
    call(HYDRA_THRUST_FWD(args)...)
  )
};

} // namespace tranform_detail

HYDRA_THRUST_INLINE_CONSTANT transform_detail::transform_fn transform{};

} // namespace async

HYDRA_THRUST_END_NS

#endif

