/*
 *  Copyright 2018-2020 NVIDIA Corporation
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

/*! \file tbb/memory_resource.h
 *  \brief Memory resources for the TBB system.
 */

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/mr/new.h>
#include <hydra/detail/external/hydra_thrust/mr/fancy_pointer_resource.h>

#include <hydra/detail/external/hydra_thrust/system/tbb/pointer.h>

HYDRA_THRUST_NAMESPACE_BEGIN
namespace system { namespace tbb
{

//! \cond
namespace detail
{
    typedef hydra_thrust::mr::fancy_pointer_resource<
        hydra_thrust::mr::new_delete_resource,
        hydra_thrust::tbb::pointer<void>
    > native_resource;

    typedef hydra_thrust::mr::fancy_pointer_resource<
        hydra_thrust::mr::new_delete_resource,
        hydra_thrust::tbb::universal_pointer<void>
    > universal_native_resource;
} // namespace detail
//! \endcond

/*! \addtogroup memory_resources Memory Resources
 *  \ingroup memory_management
 *  \{
 */

/*! The memory resource for the TBB system. Uses \p mr::new_delete_resource and
 *  tags it with \p tbb::pointer.
 */
typedef detail::native_resource memory_resource;
/*! The unified memory resource for the TBB system. Uses
 *  \p mr::new_delete_resource and tags it with \p tbb::universal_pointer.
 */
typedef detail::universal_native_resource universal_memory_resource;
/*! An alias for \p tbb::universal_memory_resource. */
typedef detail::native_resource universal_host_pinned_memory_resource;

/*! \} // memory_resources
 */

}} // namespace system::tbb

HYDRA_THRUST_NAMESPACE_END
