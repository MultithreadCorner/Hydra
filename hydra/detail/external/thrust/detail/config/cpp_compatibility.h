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

#include <hydra/detail/external/thrust/detail/config/cpp_dialect.h>

#include <cstddef>

#if HYDRA_THRUST_CPP_DIALECT >= 2011
#  ifndef __has_cpp_attribute
#    define __has_cpp_attribute(X) 0
#  endif

#  if __has_cpp_attribute(nodiscard)
#    define HYDRA_THRUST_NODISCARD [[nodiscard]]
#  endif

#  define HYDRA_THRUST_CONSTEXPR constexpr
#  define HYDRA_THRUST_OVERRIDE override
#  define HYDRA_THRUST_DEFAULT = default;
#  define HYDRA_THRUST_NOEXCEPT noexcept
#  define HYDRA_THRUST_FINAL final
#else
#  define HYDRA_THRUST_CONSTEXPR
#  define HYDRA_THRUST_OVERRIDE
#  define HYDRA_THRUST_DEFAULT {}
#  define HYDRA_THRUST_NOEXCEPT throw()
#  define HYDRA_THRUST_FINAL
#endif

#ifndef HYDRA_THRUST_NODISCARD
#  define HYDRA_THRUST_NODISCARD
#endif

// FIXME: Combine HYDRA_THRUST_INLINE_CONSTANT and
// HYDRA_THRUST_INLINE_INTEGRAL_MEMBER_CONSTANT into one macro when NVCC properly
// supports `constexpr` globals in host and device code.
#ifdef __CUDA_ARCH__
// FIXME: Add this when NVCC supports inline variables.
//#  if   HYDRA_THRUST_CPP_DIALECT >= 2017
//#    define HYDRA_THRUST_INLINE_CONSTANT                 inline constexpr
//#    define HYDRA_THRUST_INLINE_INTEGRAL_MEMBER_CONSTANT inline constexpr
#  if HYDRA_THRUST_CPP_DIALECT >= 2011
#    define HYDRA_THRUST_INLINE_CONSTANT                 static constexpr
#    define HYDRA_THRUST_INLINE_INTEGRAL_MEMBER_CONSTANT static constexpr
#  else
#    define HYDRA_THRUST_INLINE_CONSTANT                 static const __device__
#    define HYDRA_THRUST_INLINE_INTEGRAL_MEMBER_CONSTANT static const
#  endif
#else
// FIXME: Add this when NVCC supports inline variables.
//#  if   HYDRA_THRUST_CPP_DIALECT >= 2017
//#    define HYDRA_THRUST_INLINE_CONSTANT                 inline constexpr
//#    define HYDRA_THRUST_INLINE_INTEGRAL_MEMBER_CONSTANT inline constexpr
#  if HYDRA_THRUST_CPP_DIALECT >= 2011
#    define HYDRA_THRUST_INLINE_CONSTANT                 static constexpr
#    define HYDRA_THRUST_INLINE_INTEGRAL_MEMBER_CONSTANT static constexpr
#  else
#    define HYDRA_THRUST_INLINE_CONSTANT                 static const
#    define HYDRA_THRUST_INLINE_INTEGRAL_MEMBER_CONSTANT static const
#  endif
#endif

