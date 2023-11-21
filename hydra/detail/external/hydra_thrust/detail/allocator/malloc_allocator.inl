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
#include <hydra/detail/external/hydra_thrust/detail/allocator/malloc_allocator.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/select_system.h>
#include <hydra/detail/external/hydra_thrust/system/detail/bad_alloc.h>
#include <hydra/detail/external/hydra_thrust/detail/raw_pointer_cast.h>
#include <hydra/detail/external/hydra_thrust/detail/malloc_and_free.h>

HYDRA_THRUST_NAMESPACE_BEGIN
namespace detail
{


template<typename T, typename System, typename Pointer>
  typename malloc_allocator<T,System,Pointer>::pointer
    malloc_allocator<T,System,Pointer>
      ::allocate(typename malloc_allocator<T,System,Pointer>::size_type cnt)
{
  using hydra_thrust::system::detail::generic::select_system;

  // XXX should use a hypothetical hydra_thrust::static_pointer_cast here
  System system;

  pointer result = hydra_thrust::malloc<T>(select_system(system), cnt);

  if(result.get() == 0)
  {
    throw hydra_thrust::system::detail::bad_alloc("malloc_allocator::allocate: malloc failed");
  } // end if

  return result;
} // end malloc_allocator::allocate()


template<typename T, typename System, typename Pointer>
  void malloc_allocator<T,System,Pointer>
    ::deallocate(typename malloc_allocator<T,System,Pointer>::pointer p, typename malloc_allocator<T,System,Pointer>::size_type)
{
  using hydra_thrust::system::detail::generic::select_system;

  System system;
  hydra_thrust::free(select_system(system), p);
} // end malloc_allocator


} // end detail
HYDRA_THRUST_NAMESPACE_END

