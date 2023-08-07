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

// Portions of this code are derived from
//
// Manjunath Kudlur's Carbon library
//
// and
//
// Based on Boost.Phoenix v1.2
// Copyright (c) 2001-2002 Joel de Guzman

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/functional/composite.h>
#include <hydra/detail/external/hydra_thrust/detail/functional/operators/assignment_operator.h>
#include <hydra/detail/external/hydra_thrust/functional.h>
#include <hydra/detail/external/hydra_thrust/type_traits/logical_metafunctions.h>

#include <type_traits>

HYDRA_THRUST_NAMESPACE_BEGIN

namespace detail
{
namespace functional
{

template<typename Eval>
  __host__ __device__
  constexpr actor<Eval>
    ::actor()
      : eval_type()
{}

template<typename Eval>
  __host__ __device__
  actor<Eval>
    ::actor(const Eval &base)
      : eval_type(base)
{}

template<typename Eval>
  __host__ __device__
  typename apply_actor<
    typename actor<Eval>::eval_type,
    typename hydra_thrust::null_type
  >::type
    actor<Eval>
      ::operator()(void) const
{
  return eval_type::eval(hydra_thrust::null_type());
} // end basic_environment::operator()

// actor::operator() needs to construct a tuple of references to its
// arguments. To make this work with hydra_thrust::reference<T>, we need to
// detect hydra_thrust proxy references and store them as T rather than T&.
// This check ensures that the forwarding references passed into
// actor::operator() are either:
// - T&& if and only if T is a hydra_thrust::reference<U>, or
// - T& for any other types.
// This struct provides a nicer diagnostic for when these conditions aren't
// met.
template <typename T>
using actor_check_ref_type =
  hydra_thrust::detail::integral_constant<bool,
    ( std::is_lvalue_reference<T>::value ||
      hydra_thrust::detail::is_wrapped_reference<T>::value )>;

template <typename... Ts>
using actor_check_ref_types =
  hydra_thrust::conjunction<actor_check_ref_type<Ts>...>;

template<typename Eval>
template<typename... Ts>
__host__ __device__
typename apply_actor<typename actor<Eval>::eval_type,
                     hydra_thrust::tuple<eval_ref<Ts>...>>::type
actor<Eval>::operator()(Ts&&... ts) const
{
  static_assert(actor_check_ref_types<Ts...>::value,
                "Actor evaluations only support rvalue references to "
                "hydra_thrust::reference subclasses.");
  using tuple_type = hydra_thrust::tuple<eval_ref<Ts>...>;
  return eval_type::eval(tuple_type(HYDRA_THRUST_FWD(ts)...));
} // end actor<Eval>::operator()

template<typename Eval>
  template<typename T>
    __host__ __device__
    typename assign_result<Eval,T>::type
      actor<Eval>
        ::operator=(const T& _1) const
{
  return do_assign(*this,_1);
} // end actor::operator=()

} // end functional
} // end detail
HYDRA_THRUST_NAMESPACE_END
