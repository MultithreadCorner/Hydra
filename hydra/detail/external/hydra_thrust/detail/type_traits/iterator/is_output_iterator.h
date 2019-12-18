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
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits/is_metafunction_defined.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/any_assign.h>

namespace hydra_thrust
{

namespace detail
{


template<typename T>
  struct is_void_like
    : hydra_thrust::detail::or_<
        hydra_thrust::detail::is_void<T>,
        hydra_thrust::detail::is_same<T,hydra_thrust::detail::any_assign>
      >
{}; // end is_void_like


template<typename T>
  struct lazy_is_void_like
    : is_void_like<typename T::type>
{}; // end lazy_is_void_like


// XXX this meta function should first check that T is actually an iterator
//
//     if hydra_thrust::iterator_value<T> is defined and hydra_thrust::iterator_value<T>::type == void
//       return false
//     else
//       return true
template<typename T>
  struct is_output_iterator
    : eval_if<
        is_metafunction_defined<hydra_thrust::iterator_value<T> >::value,
        lazy_is_void_like<hydra_thrust::iterator_value<T> >,
        hydra_thrust::detail::true_type
      >::type
{
}; // end is_output_iterator

} // end detail

} // end hydra_thrust

