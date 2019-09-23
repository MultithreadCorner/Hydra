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
#define HYDRA_THRUST_DEVICE_SYSTEM_CUDA    1
#define HYDRA_THRUST_DEVICE_SYSTEM_OMP     2
#define HYDRA_THRUST_DEVICE_SYSTEM_TBB     3
#define HYDRA_THRUST_DEVICE_SYSTEM_CPP     4

#ifndef HYDRA_THRUST_DEVICE_SYSTEM
#define HYDRA_THRUST_DEVICE_SYSTEM HYDRA_THRUST_DEVICE_SYSTEM_CUDA
#endif // HYDRA_THRUST_DEVICE_SYSTEM

// XXX make the use of HYDRA_THRUST_DEVICE_BACKEND an error in Thrust 1.7
// XXX eliminate the following in Thrust 1.7

#define HYDRA_THRUST_DEVICE_BACKEND_CUDA HYDRA_THRUST_DEVICE_SYSTEM_CUDA
#define HYDRA_THRUST_DEVICE_BACKEND_OMP  HYDRA_THRUST_DEVICE_SYSTEM_OMP
#define HYDRA_THRUST_DEVICE_BACKEND_TBB  HYDRA_THRUST_DEVICE_SYSTEM_TBB

#ifdef HYDRA_THRUST_DEVICE_BACKEND
#  if HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_MSVC
#    pragma message("----------------------------------------------------------------------------------")
#    pragma message("| WARNING: HYDRA_THRUST_DEVICE_BACKEND is deprecated; use HYDRA_THRUST_DEVICE_SYSTEM instead |")
#    pragma message("----------------------------------------------------------------------------------")
#  else
#    warning ----------------------------------------------------------------------------------
#    warning | WARNING: HYDRA_THRUST_DEVICE_BACKEND is deprecated; use HYDRA_THRUST_DEVICE_SYSTEM instead |
#    warning ----------------------------------------------------------------------------------
#  endif // HYDRA_THRUST_HOST_COMPILER
#  undef HYDRA_THRUST_DEVICE_SYSTEM
#  define HYDRA_THRUST_DEVICE_SYSTEM HYDRA_THRUST_DEVICE_BACKEND
#endif // HYDRA_THRUST_DEVICE_BACKEND

#if HYDRA_THRUST_DEVICE_SYSTEM == HYDRA_THRUST_DEVICE_SYSTEM_CUDA
#define __HYDRA_THRUST_DEVICE_SYSTEM_NAMESPACE cuda
#elif HYDRA_THRUST_DEVICE_SYSTEM == HYDRA_THRUST_DEVICE_SYSTEM_OMP
#define __HYDRA_THRUST_DEVICE_SYSTEM_NAMESPACE omp
#elif HYDRA_THRUST_DEVICE_SYSTEM == HYDRA_THRUST_DEVICE_SYSTEM_TBB
#define __HYDRA_THRUST_DEVICE_SYSTEM_NAMESPACE tbb
#elif HYDRA_THRUST_DEVICE_SYSTEM == HYDRA_THRUST_DEVICE_SYSTEM_CPP
#define __HYDRA_THRUST_DEVICE_SYSTEM_NAMESPACE cpp
#endif

#define __HYDRA_THRUST_DEVICE_SYSTEM_ROOT hydra/detail/external/thrust/system/__HYDRA_THRUST_DEVICE_SYSTEM_NAMESPACE

