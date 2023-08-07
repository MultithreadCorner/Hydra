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

#include <hydra/detail/external/hydra_thrust/iterator/iterator_adaptor.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/minimum_system.h>

HYDRA_THRUST_NAMESPACE_BEGIN

template<typename,typename> class permutation_iterator;


namespace detail
{

template<typename ElementIterator,
         typename IndexIterator>
  struct permutation_iterator_base
{
  typedef typename hydra_thrust::iterator_system<ElementIterator>::type System1;
  typedef typename hydra_thrust::iterator_system<IndexIterator>::type System2;

  typedef hydra_thrust::iterator_adaptor<
    permutation_iterator<ElementIterator,IndexIterator>,
    IndexIterator,
    typename hydra_thrust::iterator_value<ElementIterator>::type,
    typename detail::minimum_system<System1,System2>::type,
    hydra_thrust::use_default,
    typename hydra_thrust::iterator_reference<ElementIterator>::type
  > type;
}; // end permutation_iterator_base

} // end detail

HYDRA_THRUST_NAMESPACE_END

