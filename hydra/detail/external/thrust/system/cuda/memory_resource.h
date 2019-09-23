/*
 *  Copyright 2018 NVIDIA Corporation
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

/*! \file cuda/memory_resource.h
 *  \brief Memory resources for the CUDA system.
 */

#pragma once

#include <hydra/detail/external/thrust/mr/memory_resource.h>
#include <hydra/detail/external/thrust/system/cuda/detail/guarded_cuda_runtime_api.h>
#include <hydra/detail/external/thrust/system/cuda/pointer.h>
#include <hydra/detail/external/thrust/system/detail/bad_alloc.h>
#include <hydra/detail/external/thrust/system/cuda/error.h>
#include <hydra/detail/external/thrust/system/cuda/detail/util.h>

#include <hydra/detail/external/thrust/memory/detail/host_system_resource.h>
HYDRA_EXTERNAL_NAMESPACE_BEGIN
HYDRA_THRUST_BEGIN_NS

namespace system
{
namespace cuda
{

//! \cond
namespace detail
{

    typedef cudaError_t (*allocation_fn)(void **, std::size_t);
    typedef cudaError_t (*deallocation_fn)(void *);

    template<allocation_fn Alloc, deallocation_fn Dealloc, typename Pointer>
    class cuda_memory_resource HYDRA_THRUST_FINAL : public mr::memory_resource<Pointer>
    {
    public:
        Pointer do_allocate(std::size_t bytes, std::size_t alignment = HYDRA_THRUST_MR_DEFAULT_ALIGNMENT) HYDRA_THRUST_OVERRIDE
        {
            (void)alignment;

            void * ret;
            cudaError_t status = Alloc(&ret, bytes);

            if (status != cudaSuccess)
            {
                throw HYDRA_EXTERNAL_NS::thrust::system::detail::bad_alloc(HYDRA_EXTERNAL_NS::thrust::cuda_category().message(status).c_str());
            }

            return Pointer(ret);
        }

        void do_deallocate(Pointer p, std::size_t bytes, std::size_t alignment) HYDRA_THRUST_OVERRIDE
        {
            (void)bytes;
            (void)alignment;

            cudaError_t status = Dealloc(HYDRA_EXTERNAL_NS::thrust::detail::pointer_traits<Pointer>::get(p));

            if (status != cudaSuccess)
            {
                HYDRA_EXTERNAL_NS::thrust::cuda_cub::throw_on_error(status, "CUDA free failed");
            }
        }
    };

    inline cudaError_t cudaMallocManaged(void ** ptr, std::size_t bytes)
    {
        return ::cudaMallocManaged(ptr, bytes, cudaMemAttachGlobal);
    }

    typedef detail::cuda_memory_resource<cudaMalloc, cudaFree,
        HYDRA_EXTERNAL_NS::thrust::cuda::pointer<void> >
        device_memory_resource;
    typedef detail::cuda_memory_resource<detail::cudaMallocManaged, cudaFree,
        HYDRA_EXTERNAL_NS::thrust::cuda::pointer<void> >
        managed_memory_resource;
    typedef detail::cuda_memory_resource<cudaMallocHost, cudaFreeHost,
        HYDRA_EXTERNAL_NS::thrust::host_memory_resource::pointer>
        pinned_memory_resource;

} // end detail
//! \endcond

/*! The memory resource for the CUDA system. Uses <tt>cudaMalloc</tt> and wraps the result with \p cuda::pointer. */
typedef detail::device_memory_resource memory_resource;
/*! The universal memory resource for the CUDA system. Uses <tt>cudaMallocManaged</tt> and wraps the result with \p cuda::pointer. */
typedef detail::managed_memory_resource universal_memory_resource;
/*! The host pinned memory resource for the CUDA system. Uses <tt>cudaMallocHost</tt> and wraps the result with \p cuda::pointer. */
typedef detail::pinned_memory_resource universal_host_pinned_memory_resource;

} // end cuda
} // end system

HYDRA_THRUST_END_NS
HYDRA_EXTERNAL_NAMESPACE_END
