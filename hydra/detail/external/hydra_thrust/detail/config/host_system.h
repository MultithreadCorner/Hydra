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

// reserve 0 for undefined
#define HYDRA_THRUST_HOST_SYSTEM_CPP    1
#define HYDRA_THRUST_HOST_SYSTEM_OMP    2
#define HYDRA_THRUST_HOST_SYSTEM_TBB    3

#ifndef HYDRA_THRUST_HOST_SYSTEM
#define HYDRA_THRUST_HOST_SYSTEM HYDRA_THRUST_HOST_SYSTEM_CPP
#endif // HYDRA_THRUST_HOST_SYSTEM

#ifdef HYDRA_THRUST_HOST_BACKEND
#  error HYDRA_THRUST_HOST_BACKEND is no longer supported; use HYDRA_THRUST_HOST_SYSTEM instead.
#endif // HYDRA_THRUST_HOST_BACKEND

#if HYDRA_THRUST_HOST_SYSTEM == HYDRA_THRUST_HOST_SYSTEM_CPP
#define __HYDRA_THRUST_HOST_SYSTEM_NAMESPACE cpp
#elif HYDRA_THRUST_HOST_SYSTEM == HYDRA_THRUST_HOST_SYSTEM_OMP
#define __HYDRA_THRUST_HOST_SYSTEM_NAMESPACE omp
#elif HYDRA_THRUST_HOST_SYSTEM == HYDRA_THRUST_HOST_SYSTEM_TBB
#define __HYDRA_THRUST_HOST_SYSTEM_NAMESPACE tbb
#endif

#define __HYDRA_THRUST_HOST_SYSTEM_ROOT hydra/detail/external/hydra_thrust/system/__HYDRA_THRUST_HOST_SYSTEM_NAMESPACE

