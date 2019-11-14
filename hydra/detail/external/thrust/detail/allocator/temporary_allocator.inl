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
#include <hydra/detail/external/thrust/detail/allocator/temporary_allocator.h>
#include <hydra/detail/external/thrust/detail/temporary_buffer.h>
#include <hydra/detail/external/thrust/system/detail/bad_alloc.h>
#include <cassert>

#ifdef __CUDACC__
#include <hydra/detail/external/thrust/system/cuda/detail/terminate.h>
#endif

HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
{
namespace detail
{


template<typename T, typename System>
__hydra_host__ __hydra_device__
  typename temporary_allocator<T,System>::pointer
    temporary_allocator<T,System>
      ::allocate(typename temporary_allocator<T,System>::size_type cnt)
{
  pointer_and_size result = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<T>(system(), cnt);

  // handle failure
  if(result.second < cnt)
  {
    // deallocate and throw
    // note that we pass cnt to deallocate, not a value derived from result.second
    deallocate(result.first, cnt);

#if !defined(__CUDA_ARCH__)
    throw HYDRA_EXTERNAL_NS::thrust::system::detail::bad_alloc("temporary_buffer::allocate: get_temporary_buffer failed");
#else
    HYDRA_EXTERNAL_NS::thrust::system::cuda::detail::terminate_with_message("temporary_buffer::allocate: get_temporary_buffer failed");
#endif
  } // end if

  return result.first;
} // end temporary_allocator::allocate()


template<typename T, typename System>
__hydra_host__ __hydra_device__
  void temporary_allocator<T,System>
    ::deallocate(typename temporary_allocator<T,System>::pointer p, typename temporary_allocator<T,System>::size_type)
{
  return HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(system(), p);
} // end temporary_allocator


} // end detail
} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust

HYDRA_EXTERNAL_NAMESPACE_END
