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

namespace hydra_thrust
{

namespace detail
{

template<typename Tuple,
         template<typename> class UnaryMetaFunction,
         typename IndexSequence = hydra_thrust::__make_index_sequence<hydra_thrust::tuple_size<Tuple>::value>>
  struct tuple_meta_transform;

template<typename Tuple,
         template<typename> class UnaryMetaFunction,
         size_t... I>
  struct tuple_meta_transform<Tuple, UnaryMetaFunction, hydra_thrust::__index_sequence<I...>>
{
  typedef hydra_thrust::tuple<
    typename UnaryMetaFunction<typename hydra_thrust::tuple_element<I,Tuple>::type>::type...
  > type;
};

} // end detail

} // end hydra_thrust

