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
#include <hydra/detail/external/hydra_thrust/device_malloc.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/select_system.h>
#include <hydra/detail/external/hydra_thrust/detail/malloc_and_free.h>

HYDRA_THRUST_NAMESPACE_BEGIN

hydra_thrust::device_ptr<void> device_malloc(const std::size_t n)
{
  using hydra_thrust::system::detail::generic::select_system;

  typedef hydra_thrust::iterator_system< hydra_thrust::device_ptr<void> >::type system;

  // XXX lower to select_system(system) here
  system s;

  return hydra_thrust::device_ptr<void>(hydra_thrust::malloc(s, n).get());
} // end device_malloc()


template<typename T>
  hydra_thrust::device_ptr<T> device_malloc(const std::size_t n)
{
  using hydra_thrust::system::detail::generic::select_system;

  typedef hydra_thrust::iterator_system< hydra_thrust::device_ptr<void> >::type system;

  // XXX lower to select_system(system) here
  system s;

  return hydra_thrust::device_ptr<T>(hydra_thrust::malloc<T>(s,n).get());
} // end device_malloc()

HYDRA_THRUST_NAMESPACE_END
