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
#include <hydra/detail/external/hydra_thrust/detail/type_traits/pointer_traits.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/memory.h>
#include <hydra/detail/external/hydra_thrust/system/detail/adl/malloc_and_free.h>
#include <hydra/detail/external/hydra_thrust/detail/static_assert.h>
#include <hydra/detail/external/hydra_thrust/detail/malloc_and_free.h>

HYDRA_THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{


template<typename DerivedPolicy, typename Size>
__host__ __device__
  void malloc(hydra_thrust::execution_policy<DerivedPolicy> &, Size)
{
  HYDRA_THRUST_STATIC_ASSERT_MSG(
    (hydra_thrust::detail::depend_on_instantiation<Size, false>::value)
  , "unimplemented for this system"
  );
}


template<typename T, typename DerivedPolicy>
__host__ __device__
  hydra_thrust::pointer<T,DerivedPolicy>
    malloc(hydra_thrust::execution_policy<DerivedPolicy> &exec, std::size_t n)
{
  hydra_thrust::pointer<void,DerivedPolicy> void_ptr = hydra_thrust::malloc(exec, sizeof(T) * n);

  return pointer<T,DerivedPolicy>(static_cast<T*>(void_ptr.get()));
} // end malloc()


template<typename DerivedPolicy, typename Pointer>
__host__ __device__
  void free(hydra_thrust::execution_policy<DerivedPolicy> &, Pointer)
{
  HYDRA_THRUST_STATIC_ASSERT_MSG(
    (hydra_thrust::detail::depend_on_instantiation<Pointer, false>::value)
  , "unimplemented for this system"
  );
}


template<typename DerivedPolicy, typename Pointer1, typename Pointer2>
__host__ __device__
void assign_value(hydra_thrust::execution_policy<DerivedPolicy> &, Pointer1, Pointer2)
{
  HYDRA_THRUST_STATIC_ASSERT_MSG(
    (hydra_thrust::detail::depend_on_instantiation<Pointer1, false>::value)
  , "unimplemented for this system"
  );
}


template<typename DerivedPolicy, typename Pointer>
__host__ __device__
void get_value(hydra_thrust::execution_policy<DerivedPolicy> &, Pointer)
{
  HYDRA_THRUST_STATIC_ASSERT_MSG(
    (hydra_thrust::detail::depend_on_instantiation<Pointer, false>::value)
  , "unimplemented for this system"
  );
}


template<typename DerivedPolicy, typename Pointer1, typename Pointer2>
__host__ __device__
void iter_swap(hydra_thrust::execution_policy<DerivedPolicy> &, Pointer1, Pointer2)
{
  HYDRA_THRUST_STATIC_ASSERT_MSG(
    (hydra_thrust::detail::depend_on_instantiation<Pointer1, false>::value)
  , "unimplemented for this system"
  );
}


} // end generic
} // end detail
} // end system
HYDRA_THRUST_NAMESPACE_END

