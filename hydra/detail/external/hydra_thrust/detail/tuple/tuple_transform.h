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

#include <hydra/detail/external/hydra_thrust/tuple.h>
#include <hydra/detail/external/hydra_thrust/detail/tuple/tuple_meta_transform.h>

namespace hydra_thrust
{

namespace detail
{

template<typename Tuple,
         template<typename> class UnaryMetaFunction,
         typename UnaryFunction,
         typename IndexSequence = hydra_thrust::__make_index_sequence<hydra_thrust::tuple_size<Tuple>::value>>
  struct tuple_transform_functor;


template<typename Tuple,
         template<typename> class UnaryMetaFunction,
         typename UnaryFunction,
         size_t... I>
  struct tuple_transform_functor<Tuple,UnaryMetaFunction,UnaryFunction,hydra_thrust::__index_sequence<I...>>
{
  static __host__
  typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
  do_it_on_the_host(const Tuple &t, UnaryFunction f)
  {
    typedef typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type XfrmTuple;

    return XfrmTuple(f(hydra_thrust::get<I>(t))...);
  }

  static __host__ __device__
  typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
  do_it_on_the_host_or_device(const Tuple &t, UnaryFunction f)
  {
    typedef typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type XfrmTuple;

    return XfrmTuple(f(hydra_thrust::get<I>(t))...);
  }
};


template<template<typename> class UnaryMetaFunction,
         typename Tuple,
         typename UnaryFunction>
typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
tuple_host_transform(const Tuple &t, UnaryFunction f)
{
  return tuple_transform_functor<Tuple,UnaryMetaFunction,UnaryFunction>::do_it_on_the_host(t,f);
}

template<template<typename> class UnaryMetaFunction,
         typename Tuple,
         typename UnaryFunction>
typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
__host__ __device__
tuple_host_device_transform(const Tuple &t, UnaryFunction f)
{
  return tuple_transform_functor<Tuple,UnaryMetaFunction,UnaryFunction>::do_it_on_the_host_or_device(t,f);
}

} // end detail

} // end hydra_thrust

