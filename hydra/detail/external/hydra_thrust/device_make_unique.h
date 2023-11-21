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


/*! \file device_make_unique.h
 *  \brief A factory function for creating `unique_ptr`s to device objects.
 */

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/cpp11_required.h>

#if HYDRA_THRUST_CPP_DIALECT >= 2011

#include <hydra/detail/external/hydra_thrust/allocate_unique.h>
#include <hydra/detail/external/hydra_thrust/device_new.h>
#include <hydra/detail/external/hydra_thrust/device_ptr.h>
#include <hydra/detail/external/hydra_thrust/device_allocator.h>
#include <hydra/detail/external/hydra_thrust/detail/type_deduction.h>

HYDRA_THRUST_NAMESPACE_BEGIN

///////////////////////////////////////////////////////////////////////////////

template <typename T, typename... Args>
__host__
auto device_make_unique(Args&&... args)
  HYDRA_THRUST_TRAILING_RETURN(decltype(
    uninitialized_allocate_unique<T>(device_allocator<T>{})
  ))
{
#if !defined(HYDRA_THRUST_DOXYGEN) // This causes Doxygen to choke for some reason.
  // FIXME: This is crude - we construct an unnecessary T on the host for
  // `device_new`. We need a proper dispatched `construct` algorithm to
  // do this properly.
  auto p = uninitialized_allocate_unique<T>(device_allocator<T>{});
  device_new<T>(p.get(), T(HYDRA_THRUST_FWD(args)...));
  return p;
#endif
}

///////////////////////////////////////////////////////////////////////////////

HYDRA_THRUST_NAMESPACE_END

#endif // HYDRA_THRUST_CPP_DIALECT >= 2011
