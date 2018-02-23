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

#include <hydra/detail/external/thrust/detail/config.h>
#include <hydra/detail/external/thrust/detail/type_traits/pointer_traits.h>
#include <hydra/detail/external/thrust/system/detail/generic/memory.h>
#include <hydra/detail/external/thrust/system/detail/generic/select_system.h>
#include <hydra/detail/external/thrust/system/detail/adl/malloc_and_free.h>
#include <hydra/detail/external/thrust/detail/static_assert.h>
#include <hydra/detail/external/thrust/detail/malloc_and_free.h>

HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
{
namespace system
{
namespace detail
{
namespace generic
{


template<typename DerivedPolicy, typename Size>
__hydra_host__ __hydra_device__
  void malloc(thrust::execution_policy<DerivedPolicy> &, Size)
{
  // unimplemented
  HYDRA_THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<Size, false>::value) );
}


template<typename T, typename DerivedPolicy>
__hydra_host__ __hydra_device__
  thrust::pointer<T,DerivedPolicy>
    malloc(thrust::execution_policy<DerivedPolicy> &exec, std::size_t n)
{
  thrust::pointer<void,DerivedPolicy> void_ptr = thrust::malloc(exec, sizeof(T) * n);

  return pointer<T,DerivedPolicy>(static_cast<T*>(void_ptr.get()));
} // end malloc()


template<typename DerivedPolicy, typename Pointer>
__hydra_host__ __hydra_device__
  void free(thrust::execution_policy<DerivedPolicy> &, Pointer)
{
  // unimplemented
  HYDRA_THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<Pointer, false>::value) );
}


template<typename DerivedPolicy, typename Pointer1, typename Pointer2>
__hydra_host__ __hydra_device__
void assign_value(thrust::execution_policy<DerivedPolicy> &, Pointer1, Pointer2)
{
  // unimplemented
  HYDRA_THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<Pointer1, false>::value) );
}


template<typename DerivedPolicy, typename Pointer>
__hydra_host__ __hydra_device__
void get_value(thrust::execution_policy<DerivedPolicy> &, Pointer)
{
  // unimplemented
  HYDRA_THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<Pointer, false>::value) );
}


template<typename Pointer1, typename Pointer2>
__hydra_host__ __hydra_device__
void iter_swap(tag, Pointer1, Pointer2)
{
  // unimplemented
  HYDRA_THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<Pointer1, false>::value) );
}


} // end generic
} // end detail
} // end system
} // end thrust

HYDRA_EXTERNAL_NAMESPACE_END
