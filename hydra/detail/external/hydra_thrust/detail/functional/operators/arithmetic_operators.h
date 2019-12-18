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
#include <hydra/detail/external/hydra_thrust/detail/functional/actor.h>
#include <hydra/detail/external/hydra_thrust/detail/functional/composite.h>
#include <hydra/detail/external/hydra_thrust/detail/functional/operators/operator_adaptors.h>
#include <hydra/detail/external/hydra_thrust/functional.h>

namespace hydra_thrust
{
namespace detail
{
namespace functional
{

template<typename Eval>
__host__ __device__
actor<
  composite<
    unary_operator<hydra_thrust::negate>,
    actor<Eval>
  >
>
__host__ __device__
operator-(const actor<Eval> &_1)
{
  return compose(unary_operator<hydra_thrust::negate>(), _1);
} // end operator-()

// there's no standard unary_plus functional, so roll an ad hoc one here
template<typename T>
  struct unary_plus
    : public hydra_thrust::unary_function<T,T>
{
  __host__ __device__ T operator()(const T &x) const {return +x;}
}; // end unary_plus

template<typename Eval>
__host__ __device__
actor<
  composite<
    unary_operator<unary_plus>,
    actor<Eval>
  >
>
operator+(const actor<Eval> &_1)
{
  return compose(unary_operator<unary_plus>(), _1);
} // end operator+()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    binary_operator<hydra_thrust::plus>,
    actor<T1>,
    typename as_actor<T2>::type
  >
>
operator+(const actor<T1> &_1, const T2 &_2)
{
  return compose(binary_operator<hydra_thrust::plus>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator+()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    binary_operator<hydra_thrust::plus>,
    typename as_actor<T1>::type,
    actor<T2>
  >
>
operator+(const T1 &_1, const actor<T2> &_2)
{
  return compose(binary_operator<hydra_thrust::plus>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator+()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    binary_operator<hydra_thrust::plus>,
    actor<T1>,
    actor<T2>
  >
>
operator+(const actor<T1> &_1, const actor<T2> &_2)
{
  return compose(binary_operator<hydra_thrust::plus>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator+()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    binary_operator<hydra_thrust::minus>,
    typename as_actor<T1>::type,
    actor<T2>
  >
>
operator-(const T1 &_1, const actor<T2> &_2)
{
  return compose(binary_operator<hydra_thrust::minus>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator-()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    binary_operator<hydra_thrust::minus>,
    actor<T1>,
    typename as_actor<T2>::type
  >
>
operator-(const actor<T1> &_1, const T2 &_2)
{
  return compose(binary_operator<hydra_thrust::minus>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator-()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    binary_operator<hydra_thrust::minus>,
    actor<T1>,
    actor<T2>
  >
>
operator-(const actor<T1> &_1, const actor<T2> &_2)
{
  return compose(binary_operator<hydra_thrust::minus>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator-()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    binary_operator<hydra_thrust::multiplies>,
    typename as_actor<T1>::type,
    actor<T2>
  >
>
operator*(const T1 &_1, const actor<T2> &_2)
{
  return compose(binary_operator<hydra_thrust::multiplies>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator*()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    binary_operator<hydra_thrust::multiplies>,
    actor<T1>,
    typename as_actor<T2>::type
  >
>
operator*(const actor<T1> &_1, const T2 &_2)
{
  return compose(binary_operator<hydra_thrust::multiplies>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator*()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    binary_operator<hydra_thrust::multiplies>,
    actor<T1>,
    actor<T2>
  >
>
operator*(const actor<T1> &_1, const actor<T2> &_2)
{
  return compose(binary_operator<hydra_thrust::multiplies>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator*()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    binary_operator<hydra_thrust::divides>,
    actor<T1>,
    typename as_actor<T2>::type
  >
>
operator/(const actor<T1> &_1, const T2 &_2)
{
  return compose(binary_operator<hydra_thrust::divides>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator/()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    binary_operator<hydra_thrust::divides>,
    typename as_actor<T1>::type,
    actor<T2>
  >
>
operator/(const T1 &_1, const actor<T2> &_2)
{
  return compose(binary_operator<hydra_thrust::divides>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator/()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    binary_operator<hydra_thrust::divides>,
    actor<T1>,
    actor<T2>
  >
>
operator/(const actor<T1> &_1, const actor<T2> &_2)
{
  return compose(binary_operator<hydra_thrust::divides>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator/()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    binary_operator<hydra_thrust::modulus>,
    actor<T1>,
    typename as_actor<T2>::type
  >
>
operator%(const actor<T1> &_1, const T2 &_2)
{
  return compose(binary_operator<hydra_thrust::modulus>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator%()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    binary_operator<hydra_thrust::modulus>,
    typename as_actor<T1>::type,
    actor<T2>
  >
>
operator%(const T1 &_1, const actor<T2> &_2)
{
  return compose(binary_operator<hydra_thrust::modulus>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator%()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    binary_operator<hydra_thrust::modulus>,
    actor<T1>,
    actor<T2>
  >
>
operator%(const actor<T1> &_1, const actor<T2> &_2)
{
  return compose(binary_operator<hydra_thrust::modulus>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator%()

// there's no standard prefix_increment functional, so roll an ad hoc one here
template<typename T>
  struct prefix_increment
    : public hydra_thrust::unary_function<T&,T&>
{
  __host__ __device__ T& operator()(T &x) const { return ++x; }
}; // end prefix_increment

template<typename Eval>
__host__ __device__
actor<
  composite<
    unary_operator<prefix_increment>,
    actor<Eval>
  >
>
operator++(const actor<Eval> &_1)
{
  return compose(unary_operator<prefix_increment>(), _1);
} // end operator++()

// there's no standard suffix_increment functional, so roll an ad hoc one here
template<typename T>
  struct suffix_increment
    : public hydra_thrust::unary_function<T&,T>
{
  __host__ __device__ T operator()(T &x) const { return x++; }
}; // end suffix_increment

template<typename Eval>
__host__ __device__
actor<
  composite<
    unary_operator<suffix_increment>,
    actor<Eval>
  >
>
operator++(const actor<Eval> &_1, int)
{
  return compose(unary_operator<suffix_increment>(), _1);
} // end operator++()

// there's no standard prefix_decrement functional, so roll an ad hoc one here
template<typename T>
  struct prefix_decrement
    : public hydra_thrust::unary_function<T&,T&>
{
  __host__ __device__ T& operator()(T &x) const { return --x; }
}; // end prefix_decrement

template<typename Eval>
__host__ __device__
actor<
  composite<
    unary_operator<prefix_decrement>,
    actor<Eval>
  >
>
operator--(const actor<Eval> &_1)
{
  return compose(unary_operator<prefix_decrement>(), _1);
} // end operator--()

// there's no standard suffix_decrement functional, so roll an ad hoc one here
template<typename T>
  struct suffix_decrement
    : public hydra_thrust::unary_function<T&,T>
{
  __host__ __device__ T operator()(T &x) const { return x--; }
}; // end suffix_decrement

template<typename Eval>
__host__ __device__
actor<
  composite<
    unary_operator<suffix_decrement>,
    actor<Eval>
  >
>
operator--(const actor<Eval> &_1, int)
{
  return compose(unary_operator<suffix_decrement>(), _1);
} // end operator--()

} // end functional
} // end detail
} // end hydra_thrust

