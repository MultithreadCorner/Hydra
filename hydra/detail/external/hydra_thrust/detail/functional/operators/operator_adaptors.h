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

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/functional/argument.h>
#include <hydra/detail/external/hydra_thrust/detail/type_deduction.h>
#include <hydra/detail/external/hydra_thrust/tuple.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/type_traits/void_t.h>

#include <type_traits>

HYDRA_THRUST_NAMESPACE_BEGIN
namespace detail
{
namespace functional
{

// Adapts a transparent unary functor from functional.h (e.g. hydra_thrust::negate<>)
// into the Eval interface.
template <typename UnaryFunctor>
struct transparent_unary_operator
{
  template <typename>
  using operator_type = UnaryFunctor;

  template <typename Env>
  using argument =
  typename hydra_thrust::detail::eval_if<
    hydra_thrust::tuple_size<Env>::value != 1,
    hydra_thrust::detail::identity_<hydra_thrust::null_type>,
    hydra_thrust::detail::functional::argument_helper<0, Env>
  >::type;

  template <typename Env>
  struct result_type_impl
  {
    using type = decltype(
      std::declval<UnaryFunctor>()(std::declval<argument<Env>>()));
  };

  template <typename Env>
  using result_type =
  typename hydra_thrust::detail::eval_if<
    std::is_same<hydra_thrust::null_type, argument<Env>>::value,
    hydra_thrust::detail::identity_<hydra_thrust::null_type>,
    result_type_impl<Env>
  >::type;

  template <typename Env>
  struct result
  {
    using op_type = UnaryFunctor;
    using type = result_type<Env>;
  };

  template <typename Env>
  __host__ __device__
  result_type<Env> eval(Env&& e) const
  HYDRA_THRUST_RETURNS(UnaryFunctor{}(hydra_thrust::get<0>(HYDRA_THRUST_FWD(e))))
};


// Adapts a transparent binary functor from functional.h (e.g. hydra_thrust::less<>)
// into the Eval interface.
template <typename BinaryFunctor>
struct transparent_binary_operator
{
  template <typename>
  using operator_type = BinaryFunctor;

  template <typename Env>
  using first_argument =
    typename hydra_thrust::detail::eval_if<
      hydra_thrust::tuple_size<Env>::value != 2,
      hydra_thrust::detail::identity_<hydra_thrust::null_type>,
      hydra_thrust::detail::functional::argument_helper<0, Env>
    >::type;

  template <typename Env>
  using second_argument =
    typename hydra_thrust::detail::eval_if<
      hydra_thrust::tuple_size<Env>::value != 2,
      hydra_thrust::detail::identity_<hydra_thrust::null_type>,
      hydra_thrust::detail::functional::argument_helper<1, Env>
    >::type;

  template <typename Env>
  struct result_type_impl
  {
    using type = decltype(
      std::declval<BinaryFunctor>()(std::declval<first_argument<Env>>(),
                                    std::declval<second_argument<Env>>()));
  };

  template <typename Env>
  using result_type =
    typename hydra_thrust::detail::eval_if<
      (std::is_same<hydra_thrust::null_type, first_argument<Env>>::value ||
       std::is_same<hydra_thrust::null_type, second_argument<Env>>::value),
      hydra_thrust::detail::identity_<hydra_thrust::null_type>,
      result_type_impl<Env>
    >::type;

  template <typename Env>
  struct result
  {
    using op_type = BinaryFunctor;
    using type = result_type<Env>;
  };

  template <typename Env>
  __host__ __device__
  result_type<Env> eval(Env&& e) const
  HYDRA_THRUST_RETURNS(BinaryFunctor{}(hydra_thrust::get<0>(e), hydra_thrust::get<1>(e)))
};

} // end functional
} // end detail
HYDRA_THRUST_NAMESPACE_END

