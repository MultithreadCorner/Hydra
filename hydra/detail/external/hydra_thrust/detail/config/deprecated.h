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

/*! \file deprecated.h
 *  \brief Defines the HYDRA_THRUST_DEPRECATED macro
 */

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config/compiler.h>
#include <hydra/detail/external/hydra_thrust/detail/config/cpp_dialect.h>

#if defined(CUB_IGNORE_DEPRECATED_API) && !defined(HYDRA_THRUST_IGNORE_DEPRECATED_API)
#  define HYDRA_THRUST_IGNORE_DEPRECATED_API
#endif

#ifdef HYDRA_THRUST_IGNORE_DEPRECATED_API
#  define HYDRA_THRUST_DEPRECATED
#elif HYDRA_THRUST_CPP_DIALECT >= 2014
#  define HYDRA_THRUST_DEPRECATED [[deprecated]]
#elif HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_MSVC
#  define HYDRA_THRUST_DEPRECATED __declspec(deprecated)
#elif HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_CLANG
#  define HYDRA_THRUST_DEPRECATED __attribute__((deprecated))
#elif HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_GCC
#  define HYDRA_THRUST_DEPRECATED __attribute__((deprecated))
#else
#  define HYDRA_THRUST_DEPRECATED
#endif
