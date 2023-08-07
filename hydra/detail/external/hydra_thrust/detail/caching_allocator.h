/*
 *  Copyright 2020 NVIDIA Corporation
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
#include <hydra/detail/external/hydra_thrust/mr/allocator.h>
#include <hydra/detail/external/hydra_thrust/mr/disjoint_tls_pool.h>
#include <hydra/detail/external/hydra_thrust/mr/new.h>
#include <hydra/detail/external/hydra_thrust/mr/device_memory_resource.h>

HYDRA_THRUST_NAMESPACE_BEGIN

namespace detail
{
inline
hydra_thrust::mr::allocator<
    char,
    hydra_thrust::mr::disjoint_unsynchronized_pool_resource<
        hydra_thrust::device_memory_resource,
        hydra_thrust::mr::new_delete_resource
    >
> single_device_tls_caching_allocator()
{
    return {
        &hydra_thrust::mr::tls_disjoint_pool(
            hydra_thrust::mr::get_global_resource<hydra_thrust::device_memory_resource>(),
            hydra_thrust::mr::get_global_resource<hydra_thrust::mr::new_delete_resource>()
        )
    };
}
}

HYDRA_THRUST_NAMESPACE_END
