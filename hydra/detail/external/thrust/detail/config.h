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
/*! \file config.h
 *  \brief Defines platform configuration.
 */

#pragma once


#ifndef HYDRA_EXTERNAL_NAMESPACE_BEGIN
#define HYDRA_EXTERNAL_NAMESPACE_BEGIN namespace hydra { namespace detail { namespace external {
#endif //HYDRA_EXTERNAL_NAMESPACE_BEGIN

#ifndef HYDRA_EXTERNAL_NAMESPACE_END
#define HYDRA_EXTERNAL_NAMESPACE_END                   }                  }                    }
#endif //HYDRA_EXTERNAL_NAMESPACE_END

#ifndef HYDRA_EXTERNAL_NS
#define HYDRA_EXTERNAL_NS hydra::detail::external
#endif //HYDRA_EXTERNAL_NS



#ifndef HYDRA_THRUST_BEGIN_NS
#define HYDRA_THRUST_BEGIN_NS namespace thrust {
#endif

#ifndef HYDRA_THRUST_END_NS
#define HYDRA_THRUST_END_NS }
#endif

#include <hydra/detail/external/thrust/detail/config/config.h>


#define __hydra_host__ __host__

#define __hydra_device__ __device__

#define __hydra_dual__ __host__ __device__
