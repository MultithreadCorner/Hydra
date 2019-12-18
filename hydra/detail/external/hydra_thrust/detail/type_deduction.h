// Copyright (c)      2018 NVIDIA Corporation
//                         (Bryce Adelstein Lelbach <brycelelbach@gmail.com>)
// Copyright (c) 2013-2018 Eric Niebler (`HYDRA_THRUST_RETURNS`, etc)
// Copyright (c) 2016-2018 Casey Carter (`HYDRA_THRUST_RETURNS`, etc)
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/cpp11_required.h>

#if HYDRA_THRUST_CPP_DIALECT >= 2011

#include <hydra/detail/external/hydra_thrust/detail/preprocessor.h>

#include <utility>
#include <type_traits>

///////////////////////////////////////////////////////////////////////////////

/// \def HYDRA_THRUST_FWD(x)
/// \brief Performs universal forwarding of a universal reference.
///
#define HYDRA_THRUST_FWD(x) ::std::forward<decltype(x)>(x)

/// \def HYDRA_THRUST_MVCAP(x)
/// \brief Capture `x` into a lambda by moving.
///
#define HYDRA_THRUST_MVCAP(x) x = ::std::move(x)

/// \def HYDRA_THRUST_RETOF(invocable, ...)
/// \brief Expands to the type returned by invoking an instance of the invocable
///        type \a invocable with parameters of type \c __VA_ARGS__. Must
///        be called with 1 or fewer parameters to the invocable.
///
#define HYDRA_THRUST_RETOF(...)   HYDRA_THRUST_PP_DISPATCH(HYDRA_THRUST_RETOF, __VA_ARGS__)
#define HYDRA_THRUST_RETOF1(C)    decltype(::std::declval<C>()())
#define HYDRA_THRUST_RETOF2(C, V) decltype(::std::declval<C>()(::std::declval<V>()))

/// \def HYDRA_THRUST_RETURNS(...)
/// \brief Expands to a function definition that returns the expression
///        \c __VA_ARGS__.
///
#define HYDRA_THRUST_RETURNS(...)                                                   \
  noexcept(noexcept(__VA_ARGS__))                                             \
  { return (__VA_ARGS__); }                                                   \
  /**/

/// \def HYDRA_THRUST_DECLTYPE_RETURNS(...)
/// \brief Expands to a function definition, including a trailing returning
///        type, that returns the expression \c __VA_ARGS__.
///
#define HYDRA_THRUST_DECLTYPE_RETURNS(...)                                          \
  noexcept(noexcept(__VA_ARGS__))                                             \
  -> decltype(__VA_ARGS__)                                                    \
  { return (__VA_ARGS__); }                                                   \
  /**/

/// \def HYDRA_THRUST_DECLTYPE_RETURNS_WITH_SFINAE_CONDITION(condition, ...)
/// \brief Expands to a function definition, including a trailing returning
///        type, that returns the expression \c __VA_ARGS__. It shall only 
///        participate in overload resolution if \c condition is \c true.
///
#define HYDRA_THRUST_DECLTYPE_RETURNS_WITH_SFINAE_CONDITION(condition, ...)         \
  noexcept(noexcept(__VA_ARGS__))                                             \
  -> typename std::enable_if<condition, decltype(__VA_ARGS__)>::type          \
  { return (__VA_ARGS__); }                                                   \
  /**/

///////////////////////////////////////////////////////////////////////////////

#endif // HYDRA_THRUST_CPP_DIALECT >= 2011

