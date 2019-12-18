/*
 *  Copyright 2008-2018 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a for_each of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file async/for_each.h
 *  \brief Functions for asynchronously iterating over the elements of a range.
 */

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/cpp11_required.h>
#include <hydra/detail/external/hydra_thrust/detail/modern_gcc_required.h>

#if HYDRA_THRUST_CPP_DIALECT >= 2011 && !defined(HYDRA_THRUST_LEGACY_GCC)

#include <hydra/detail/external/hydra_thrust/detail/static_assert.h>
#include <hydra/detail/external/hydra_thrust/detail/select_system.h>
#include <hydra/detail/external/hydra_thrust/type_traits/remove_cvref.h>
#include <hydra/detail/external/hydra_thrust/system/detail/adl/async/for_each.h>

#include <hydra/detail/external/hydra_thrust/event.h>

HYDRA_THRUST_BEGIN_NS

namespace async
{

namespace unimplemented
{

template <
  typename DerivedPolicy
, typename ForwardIt, typename Sentinel, typename UnaryFunction
>
__host__
event<DerivedPolicy>
async_for_each(
  hydra_thrust::execution_policy<DerivedPolicy>&, ForwardIt, Sentinel, UnaryFunction
)
{
  HYDRA_THRUST_STATIC_ASSERT_MSG(
    (hydra_thrust::detail::depend_on_instantiation<ForwardIt, false>::value)
  , "this algorithm is not implemented for the specified system"
  );
  return {};
} 

} // namespace unimplemented

namespace for_each_detail
{
    
using hydra_thrust::async::unimplemented::async_for_each;

struct for_each_fn final
{
  template <
    typename DerivedPolicy
  , typename ForwardIt, typename Sentinel, typename UnaryFunction
  >
  __host__
  static auto call(
    hydra_thrust::detail::execution_policy_base<DerivedPolicy> const& exec
  , ForwardIt&& first, Sentinel&& last
  , UnaryFunction&& f 
  )
  // ADL dispatch.
  HYDRA_THRUST_DECLTYPE_RETURNS(
    async_for_each(
      hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec))
    , HYDRA_THRUST_FWD(first), HYDRA_THRUST_FWD(last)
    , HYDRA_THRUST_FWD(f)
    )
  )

  template <typename ForwardIt, typename Sentinel, typename UnaryFunction>
  __host__
  static auto call(ForwardIt&& first, Sentinel&& last, UnaryFunction&& f) 
  HYDRA_THRUST_DECLTYPE_RETURNS(
    for_each_fn::call(
      hydra_thrust::detail::select_system(
        typename iterator_system<remove_cvref_t<ForwardIt>>::type{}
      )
    , HYDRA_THRUST_FWD(first), HYDRA_THRUST_FWD(last)
    , HYDRA_THRUST_FWD(f)
    )
  )

  template <typename... Args>
  HYDRA_THRUST_NODISCARD __host__
  auto operator()(Args&&... args) const
  HYDRA_THRUST_DECLTYPE_RETURNS(
    call(HYDRA_THRUST_FWD(args)...)
  )
};

} // namespace for_each_detail

HYDRA_THRUST_INLINE_CONSTANT for_each_detail::for_each_fn for_each{};

} // namespace async

HYDRA_THRUST_END_NS

#endif

