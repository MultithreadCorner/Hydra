/*
 *  Copyright 2008-2018 NVIDIA Corporation
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

/*! \file async/reduce.h
 *  \brief Functions for asynchronously reducing a range to a single value.
 */

#pragma once

#include <hydra/detail/external/thrust/detail/config.h>
#include <hydra/detail/external/thrust/detail/cpp11_required.h>
#include <hydra/detail/external/thrust/detail/modern_gcc_required.h>

#if HYDRA_THRUST_CPP_DIALECT >= 2011 && !defined(HYDRA_THRUST_LEGACY_GCC)

#include <hydra/detail/external/thrust/detail/static_assert.h>
#include <hydra/detail/external/thrust/detail/select_system.h>
#include <hydra/detail/external/thrust/type_traits/logical_metafunctions.h>
#include <hydra/detail/external/thrust/type_traits/remove_cvref.h>
#include <hydra/detail/external/thrust/type_traits/is_execution_policy.h>
#include <hydra/detail/external/thrust/system/detail/adl/async/reduce.h>

#include <hydra/detail/external/thrust/future.h>
HYDRA_EXTERNAL_NAMESPACE_BEGIN
HYDRA_THRUST_BEGIN_NS

namespace async
{

namespace unimplemented
{

template <
  typename DerivedPolicy
, typename ForwardIt, typename Sentinel, typename T, typename BinaryOp
>
__hydra_host__ 
future<DerivedPolicy, T>
async_reduce(
  HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy>&, ForwardIt, Sentinel, T, BinaryOp
)
{
  HYDRA_THRUST_STATIC_ASSERT_MSG(
    (HYDRA_EXTERNAL_NS::thrust::detail::depend_on_instantiation<ForwardIt, false>::value)
  , "this algorithm is not implemented for the specified system"
  );
  return {};
} 

} // namespace unimplemented

namespace reduce_detail
{

using HYDRA_EXTERNAL_NS::thrust::async::unimplemented::async_reduce;

struct reduce_fn final
{
  template <
    typename DerivedPolicy
  , typename ForwardIt, typename Sentinel, typename T, typename BinaryOp
  >
  __hydra_host__
  static auto call(
    HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> const& exec
  , ForwardIt&& first, Sentinel&& last
  , T&& init
  , BinaryOp&& op
  )
  // ADL dispatch.
  HYDRA_THRUST_DECLTYPE_RETURNS(
    async_reduce(
      HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec))
    , HYDRA_THRUST_FWD(first), HYDRA_THRUST_FWD(last)
    , HYDRA_THRUST_FWD(init)
    , HYDRA_THRUST_FWD(op)
    )
  )

  template <
    typename DerivedPolicy
  , typename ForwardIt, typename Sentinel, typename T
  >
  __hydra_host__
  static auto call(
    HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> const& exec
  , ForwardIt&& first, Sentinel&& last
  , T&& init
  )
  // ADL dispatch.
  HYDRA_THRUST_DECLTYPE_RETURNS(
    async_reduce(
      HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec))
    , HYDRA_THRUST_FWD(first), HYDRA_THRUST_FWD(last)
    , HYDRA_THRUST_FWD(init)
    , HYDRA_EXTERNAL_NS::thrust::plus<remove_cvref_t<T>>{}
    )
  )

  template <
    typename DerivedPolicy
  , typename ForwardIt, typename Sentinel
  >
  __hydra_host__
  static auto
  call(
    HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> const& exec
  , ForwardIt&& first, Sentinel&& last
  )
  // ADL dispatch.
  HYDRA_THRUST_DECLTYPE_RETURNS(
    async_reduce(
      HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec))
    , HYDRA_THRUST_FWD(first), HYDRA_THRUST_FWD(last)
    , typename iterator_traits<remove_cvref_t<ForwardIt>>::value_type{}
    , HYDRA_EXTERNAL_NS::thrust::plus<
        remove_cvref_t<
          typename iterator_traits<remove_cvref_t<ForwardIt>>::value_type
        >
      >{}
    )
  )

  template <typename ForwardIt, typename Sentinel, typename T, typename BinaryOp>
  __hydra_host__
  static auto call(ForwardIt&& first, Sentinel&& last, T&& init, BinaryOp&& op)
  HYDRA_THRUST_DECLTYPE_RETURNS_WITH_SFINAE_CONDITION(
    (negation<is_execution_policy<remove_cvref_t<ForwardIt>>>::value)
  , reduce_fn::call(
      HYDRA_EXTERNAL_NS::thrust::detail::select_system(
        typename iterator_system<remove_cvref_t<ForwardIt>>::type{}
      )
    , HYDRA_THRUST_FWD(first), HYDRA_THRUST_FWD(last)
    , HYDRA_THRUST_FWD(init)
    , HYDRA_THRUST_FWD(op)
    )
  )

  template <typename ForwardIt, typename Sentinel, typename T>
  __hydra_host__
  static auto call(ForwardIt&& first, Sentinel&& last, T&& init)
  HYDRA_THRUST_DECLTYPE_RETURNS_WITH_SFINAE_CONDITION(
    (negation<is_execution_policy<remove_cvref_t<ForwardIt>>>::value)
  , reduce_fn::call(
      HYDRA_EXTERNAL_NS::thrust::detail::select_system(
        typename iterator_system<remove_cvref_t<ForwardIt>>::type{}
      )
    , HYDRA_THRUST_FWD(first), HYDRA_THRUST_FWD(last)
    , HYDRA_THRUST_FWD(init)
    , HYDRA_EXTERNAL_NS::thrust::plus<remove_cvref_t<T>>{}
    )
  )

  template <typename ForwardIt, typename Sentinel>
  __hydra_host__
  static auto call(ForwardIt&& first, Sentinel&& last)
  HYDRA_THRUST_DECLTYPE_RETURNS(
    reduce_fn::call(
      HYDRA_EXTERNAL_NS::thrust::detail::select_system(
        typename iterator_system<remove_cvref_t<ForwardIt>>::type{}
      )
    , HYDRA_THRUST_FWD(first), HYDRA_THRUST_FWD(last)
    , typename iterator_traits<remove_cvref_t<ForwardIt>>::value_type{}
    , HYDRA_EXTERNAL_NS::thrust::plus<
        remove_cvref_t<
          typename iterator_traits<remove_cvref_t<ForwardIt>>::value_type
        >
      >{}
    )
  )

  template <typename... Args>
  HYDRA_THRUST_NODISCARD __hydra_host__ 
  auto operator()(Args&&... args) const
  HYDRA_THRUST_DECLTYPE_RETURNS(
    call(HYDRA_THRUST_FWD(args)...)
  )
};

} // namespace reduce_detail

HYDRA_THRUST_INLINE_CONSTANT reduce_detail::reduce_fn reduce{};

///////////////////////////////////////////////////////////////////////////////

namespace unimplemented
{

template <
  typename DerivedPolicy
, typename ForwardIt, typename Sentinel, typename OutputIt
, typename T, typename BinaryOp
>
__hydra_host__
event<DerivedPolicy>
async_reduce_into(
  HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy>&
, ForwardIt, Sentinel, OutputIt, T, BinaryOp
)
{
  HYDRA_THRUST_STATIC_ASSERT_MSG(
    (HYDRA_EXTERNAL_NS::thrust::detail::depend_on_instantiation<ForwardIt, false>::value)
  , "this algorithm is not implemented for the specified system"
  );
  return {};
} 

} // namespace unimplemented

namespace reduce_into_detail
{

using HYDRA_EXTERNAL_NS::thrust::async::unimplemented::async_reduce_into;

struct reduce_into_fn final
{
  template <
    typename DerivedPolicy
  , typename ForwardIt, typename Sentinel, typename OutputIt
  , typename T, typename BinaryOp
  >
  __hydra_host__
  static auto call(
    HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> const& exec
  , ForwardIt&& first, Sentinel&& last
  , OutputIt&& output
  , T&& init
  , BinaryOp&& op
  )
  // ADL dispatch.
  HYDRA_THRUST_DECLTYPE_RETURNS(
    async_reduce_into(
      HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec))
    , HYDRA_THRUST_FWD(first), HYDRA_THRUST_FWD(last)
    , HYDRA_THRUST_FWD(output)
    , HYDRA_THRUST_FWD(init)
    , HYDRA_THRUST_FWD(op)
    )
  )

  template <
    typename DerivedPolicy
  , typename ForwardIt, typename Sentinel, typename OutputIt
  , typename T
  >
  __hydra_host__
  static auto call(
    HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> const& exec
  , ForwardIt&& first, Sentinel&& last
  , OutputIt&& output
  , T&& init
  )
  // ADL dispatch.
  HYDRA_THRUST_DECLTYPE_RETURNS(
    async_reduce_into(
      HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec))
    , HYDRA_THRUST_FWD(first), HYDRA_THRUST_FWD(last)
    , HYDRA_THRUST_FWD(output)
    , HYDRA_THRUST_FWD(init)
    , HYDRA_EXTERNAL_NS::thrust::plus<remove_cvref_t<T>>{}
    )
  )

  template <
    typename DerivedPolicy
  , typename ForwardIt, typename Sentinel, typename OutputIt
  >
  __hydra_host__
  static auto
  call(
    HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> const& exec
  , ForwardIt&& first, Sentinel&& last
  , OutputIt&& output
  )
  // ADL dispatch.
  HYDRA_THRUST_DECLTYPE_RETURNS(
    async_reduce_into(
      HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec))
    , HYDRA_THRUST_FWD(first), HYDRA_THRUST_FWD(last)
    , HYDRA_THRUST_FWD(output)
    , typename iterator_traits<remove_cvref_t<ForwardIt>>::value_type{}
    , HYDRA_EXTERNAL_NS::thrust::plus<
        remove_cvref_t<
          typename iterator_traits<remove_cvref_t<ForwardIt>>::value_type
        >
      >{}
    )
  )

  template <
    typename ForwardIt, typename Sentinel, typename OutputIt
  , typename T, typename BinaryOp
  >
  __hydra_host__
  static auto call(
    ForwardIt&& first, Sentinel&& last
  , OutputIt&& output
  , T&& init
  , BinaryOp&& op
  )
  HYDRA_THRUST_DECLTYPE_RETURNS_WITH_SFINAE_CONDITION(
    (negation<is_execution_policy<remove_cvref_t<ForwardIt>>>::value)
  , reduce_into_fn::call(
      HYDRA_EXTERNAL_NS::thrust::detail::select_system(
        typename iterator_system<remove_cvref_t<ForwardIt>>::type{}
      , typename iterator_system<remove_cvref_t<OutputIt>>::type{}
      )
    , HYDRA_THRUST_FWD(first), HYDRA_THRUST_FWD(last)
    , HYDRA_THRUST_FWD(output)
    , HYDRA_THRUST_FWD(init)
    , HYDRA_THRUST_FWD(op)
    )
  )

  template <
    typename ForwardIt, typename Sentinel, typename OutputIt
  , typename T
  >
  __hydra_host__
  static auto call(
    ForwardIt&& first, Sentinel&& last
  , OutputIt&& output
  , T&& init
  )
  HYDRA_THRUST_DECLTYPE_RETURNS_WITH_SFINAE_CONDITION(
    (negation<is_execution_policy<remove_cvref_t<ForwardIt>>>::value)
  , reduce_into_fn::call(
      HYDRA_EXTERNAL_NS::thrust::detail::select_system(
        typename iterator_system<remove_cvref_t<ForwardIt>>::type{}
      , typename iterator_system<remove_cvref_t<OutputIt>>::type{}
      )
    , HYDRA_THRUST_FWD(first), HYDRA_THRUST_FWD(last)
    , HYDRA_THRUST_FWD(output)
    , HYDRA_THRUST_FWD(init)
    , HYDRA_EXTERNAL_NS::thrust::plus<remove_cvref_t<T>>{}
    )
  )

  template <
    typename ForwardIt, typename Sentinel, typename OutputIt
  >
  __hydra_host__
  static auto call(
    ForwardIt&& first, Sentinel&& last
  , OutputIt&& output
  )
  HYDRA_THRUST_DECLTYPE_RETURNS(
    reduce_into_fn::call(
      HYDRA_EXTERNAL_NS::thrust::detail::select_system(
        typename iterator_system<remove_cvref_t<ForwardIt>>::type{}
      , typename iterator_system<remove_cvref_t<OutputIt>>::type{}
      )
    , HYDRA_THRUST_FWD(first), HYDRA_THRUST_FWD(last)
    , HYDRA_THRUST_FWD(output)
    , typename iterator_traits<remove_cvref_t<ForwardIt>>::value_type{}
    , HYDRA_EXTERNAL_NS::thrust::plus<
        remove_cvref_t<
          typename iterator_traits<remove_cvref_t<ForwardIt>>::value_type
        >
      >{}
    )
  )

  template <typename... Args>
  HYDRA_THRUST_NODISCARD __hydra_host__ 
  auto operator()(Args&&... args) const
  HYDRA_THRUST_DECLTYPE_RETURNS(
    call(HYDRA_THRUST_FWD(args)...)
  )
};

} // namespace reduce_into_detail

HYDRA_THRUST_INLINE_CONSTANT reduce_into_detail::reduce_into_fn reduce_into{};

} // namespace async

HYDRA_THRUST_END_NS
HYDRA_EXTERNAL_NAMESPACE_END
#endif

