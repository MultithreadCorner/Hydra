/******************************************************************************
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

// TODO: Move into system::cuda

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/cpp11_required.h>
#include <hydra/detail/external/hydra_thrust/detail/modern_gcc_required.h>

#if HYDRA_THRUST_CPP_DIALECT >= 2011 && !defined(HYDRA_THRUST_LEGACY_GCC)

#if HYDRA_THRUST_DEVICE_COMPILER == HYDRA_THRUST_DEVICE_COMPILER_NVCC

#include <hydra/detail/external/hydra_thrust/system/cuda/config.h>

#include <hydra/detail/external/hydra_thrust/detail/type_deduction.h>
#include <hydra/detail/external/hydra_thrust/detail/cstdint.h>
#include <hydra/detail/external/hydra_thrust/detail/execute_with_allocator.h>
#include <hydra/detail/external/hydra_thrust/system/cuda/memory_resource.h>
#include <hydra/detail/external/hydra_thrust/memory/detail/host_system_resource.h>
#include <hydra/detail/external/hydra_thrust/mr/allocator.h>
#include <hydra/detail/external/hydra_thrust/mr/disjoint_sync_pool.h>
#include <hydra/detail/external/hydra_thrust/mr/sync_pool.h>
#include <hydra/detail/external/hydra_thrust/per_device_resource.h>

HYDRA_THRUST_BEGIN_NS

namespace system { namespace cuda { namespace detail
{

using default_async_host_resource =
  hydra_thrust::mr::synchronized_pool_resource<
    hydra_thrust::host_memory_resource
  >;

template <typename DerivedPolicy>
auto get_async_host_allocator(
  hydra_thrust::detail::execution_policy_base<DerivedPolicy>&
)
HYDRA_THRUST_DECLTYPE_RETURNS(
  hydra_thrust::mr::stateless_resource_allocator<
    hydra_thrust::detail::uint8_t, default_async_host_resource
  >{}
)

///////////////////////////////////////////////////////////////////////////////

using default_async_device_resource =
  hydra_thrust::mr::disjoint_synchronized_pool_resource<
    hydra_thrust::system::cuda::memory_resource
  , hydra_thrust::mr::new_delete_resource
  >;

template <typename DerivedPolicy>
auto get_async_device_allocator(
  hydra_thrust::detail::execution_policy_base<DerivedPolicy>&
)
HYDRA_THRUST_DECLTYPE_RETURNS(
  hydra_thrust::per_device_allocator<
    hydra_thrust::detail::uint8_t, default_async_device_resource, par_t
  >{}
)

template <typename Allocator, template <typename> class BaseSystem>
auto get_async_device_allocator(
  hydra_thrust::detail::execute_with_allocator<Allocator, BaseSystem>& exec
)
HYDRA_THRUST_DECLTYPE_RETURNS(exec.get_allocator())

template <typename Allocator, template <typename> class BaseSystem>
auto get_async_device_allocator(
  hydra_thrust::detail::execute_with_allocator_and_dependencies<
    Allocator, BaseSystem
  >& exec
)
HYDRA_THRUST_DECLTYPE_RETURNS(exec.get_allocator())

///////////////////////////////////////////////////////////////////////////////

using default_async_universal_host_pinned_resource =
  hydra_thrust::mr::synchronized_pool_resource<
    hydra_thrust::system::cuda::universal_host_pinned_memory_resource
  >;

template <typename DerivedPolicy>
auto get_async_universal_host_pinned_allocator(
  hydra_thrust::detail::execution_policy_base<DerivedPolicy>&
)
HYDRA_THRUST_DECLTYPE_RETURNS(
  hydra_thrust::mr::stateless_resource_allocator<
    hydra_thrust::detail::uint8_t, default_async_universal_host_pinned_resource
  >{}
)

}}} // namespace system::cuda::detail

HYDRA_THRUST_END_NS

#endif // HYDRA_THRUST_DEVICE_COMPILER == HYDRA_THRUST_DEVICE_COMPILER_NVCC

#endif

