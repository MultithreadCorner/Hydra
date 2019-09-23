/*
 *  Copyright 2008-2018 NVIDIA Corporation
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

#include <hydra/detail/external/thrust/detail/config.h>
#include <hydra/detail/external/thrust/detail/allocator_aware_execution_policy.h>
#include <hydra/detail/external/thrust/system/tbb/detail/execution_policy.h>

HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
{
namespace system
{
namespace tbb
{
namespace detail
{


struct par_t : HYDRA_EXTERNAL_NS::thrust::system::tbb::detail::execution_policy<par_t>,
  HYDRA_EXTERNAL_NS::thrust::detail::allocator_aware_execution_policy<
    HYDRA_EXTERNAL_NS::thrust::system::tbb::detail::execution_policy>
{
  __hydra_host__ __hydra_device__
  par_t() : HYDRA_EXTERNAL_NS::thrust::system::tbb::detail::execution_policy<par_t>() {}
};


} // end detail


static const detail::par_t par;


} // end tbb
} // end system


// alias par here
namespace tbb
{


using HYDRA_EXTERNAL_NS::thrust::system::tbb::par;


} // end tbb
} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust

HYDRA_EXTERNAL_NAMESPACE_END
