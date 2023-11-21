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

/*! \file cpp_dialect.h
 *  \brief Detect the version of the C++ standard used by the compiler.
 */

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config/compiler.h>

// Deprecation warnings may be silenced by defining the following macros. These
// may be combined.
// - HYDRA_THRUST_IGNORE_DEPRECATED_CPP_DIALECT:
//   Ignore all deprecated C++ dialects and outdated compilers.
// - HYDRA_THRUST_IGNORE_DEPRECATED_CPP_11:
//   Ignore deprecation warnings when compiling with C++11. C++03 and outdated
//   compilers will still issue warnings.
// - HYDRA_THRUST_IGNORE_DEPRECATED_COMPILER
//   Ignore deprecation warnings when using deprecated compilers. Compiling
//   with C++03 and C++11 will still issue warnings.

// Check for the CUB opt-outs as well:
#if !defined(HYDRA_THRUST_IGNORE_DEPRECATED_CPP_DIALECT) && \
     defined(CUB_IGNORE_DEPRECATED_CPP_DIALECT)
#  define    HYDRA_THRUST_IGNORE_DEPRECATED_CPP_DIALECT
#endif
#if !defined(HYDRA_THRUST_IGNORE_DEPRECATED_CPP_11) && \
     defined(CUB_IGNORE_DEPRECATED_CPP_11)
#  define    HYDRA_THRUST_IGNORE_DEPRECATED_CPP_11
#endif
#if !defined(HYDRA_THRUST_IGNORE_DEPRECATED_COMPILER) && \
     defined(CUB_IGNORE_DEPRECATED_COMPILER)
#  define    HYDRA_THRUST_IGNORE_DEPRECATED_COMPILER
#endif

#ifdef HYDRA_THRUST_IGNORE_DEPRECATED_CPP_DIALECT
#  define HYDRA_THRUST_IGNORE_DEPRECATED_CPP_11
#  define HYDRA_THRUST_IGNORE_DEPRECATED_COMPILER
#endif

// Define this to override the built-in detection.
#ifndef HYDRA_THRUST_CPP_DIALECT

// MSVC does not define __cplusplus correctly. _MSVC_LANG is used instead.
// This macro is only defined in MSVC 2015U3+.
#  ifdef _MSVC_LANG // Do not replace with HYDRA_THRUST_HOST_COMPILER test (see above)
// MSVC2015 reports C++14 but lacks extended constexpr support. Treat as C++11.
#    if HYDRA_THRUST_MSVC_VERSION < 1910 && _MSVC_LANG > 201103L /* MSVC < 2017 && CPP > 2011 */
#      define HYDRA_THRUST_CPLUSPLUS 201103L /* Fix to 2011 */
#    else
#      define HYDRA_THRUST_CPLUSPLUS _MSVC_LANG /* We'll trust this for now. */
#    endif // MSVC 2015 C++14 fix
#  else
#    define HYDRA_THRUST_CPLUSPLUS __cplusplus
#  endif

// Detect current dialect:
#  if HYDRA_THRUST_CPLUSPLUS < 201103L
#    define HYDRA_THRUST_CPP_DIALECT 2003
#  elif HYDRA_THRUST_CPLUSPLUS < 201402L
#    define HYDRA_THRUST_CPP_DIALECT 2011
#  elif HYDRA_THRUST_CPLUSPLUS < 201703L
#    define HYDRA_THRUST_CPP_DIALECT 2014
#  elif HYDRA_THRUST_CPLUSPLUS == 201703L
#    define HYDRA_THRUST_CPP_DIALECT 2017
#  elif HYDRA_THRUST_CPLUSPLUS > 201703L // unknown, but is higher than 2017.
#    define HYDRA_THRUST_CPP_DIALECT 2020
#  endif

#  undef HYDRA_THRUST_CPLUSPLUS // cleanup

#endif // !HYDRA_THRUST_CPP_DIALECT

// Define HYDRA_THRUST_COMPILER_DEPRECATION macro:
#if HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_MSVC
#  define HYDRA_THRUST_COMP_DEPR_IMPL(msg) \
    __pragma(message(__FILE__ ":" HYDRA_THRUST_COMP_DEPR_IMPL0(__LINE__) ": warning: " #msg))
#  define HYDRA_THRUST_COMP_DEPR_IMPL0(x) HYDRA_THRUST_COMP_DEPR_IMPL1(x)
#  define HYDRA_THRUST_COMP_DEPR_IMPL1(x) #x
#else // clang / gcc:
#  define HYDRA_THRUST_COMP_DEPR_IMPL(msg) HYDRA_THRUST_COMP_DEPR_IMPL0(GCC warning #msg)
#  define HYDRA_THRUST_COMP_DEPR_IMPL0(expr) _Pragma(#expr)
#  define HYDRA_THRUST_COMP_DEPR_IMPL1 /* intentionally blank */
#endif

#define HYDRA_THRUST_COMPILER_DEPRECATION(REQ) \
  HYDRA_THRUST_COMP_DEPR_IMPL(Thrust requires at least REQ. Define HYDRA_THRUST_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.)

#define HYDRA_THRUST_COMPILER_DEPRECATION_SOFT(REQ, CUR) \
  HYDRA_THRUST_COMP_DEPR_IMPL(Thrust requires at least REQ. CUR is deprecated but still supported. CUR support will be removed in a future release. Define HYDRA_THRUST_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.)

#ifndef HYDRA_THRUST_IGNORE_DEPRECATED_COMPILER

// Compiler checks:
#  if HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_GCC && HYDRA_THRUST_GCC_VERSION < 50000
     HYDRA_THRUST_COMPILER_DEPRECATION(GCC 5.0);
#  elif HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_CLANG && HYDRA_THRUST_CLANG_VERSION < 70000
     HYDRA_THRUST_COMPILER_DEPRECATION(Clang 7.0);
#  elif HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_MSVC && HYDRA_THRUST_MSVC_VERSION < 1910
     // <2017. Hard upgrade message:
     HYDRA_THRUST_COMPILER_DEPRECATION(MSVC 2019 (19.20/16.0/14.20));
#  elif HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_MSVC && HYDRA_THRUST_MSVC_VERSION < 1920
     // >=2017, <2019. Soft deprecation message:
     HYDRA_THRUST_COMPILER_DEPRECATION_SOFT(MSVC 2019 (19.20/16.0/14.20), MSVC 2017);
#  endif

#endif // HYDRA_THRUST_IGNORE_DEPRECATED_COMPILER

#ifndef HYDRA_THRUST_IGNORE_DEPRECATED_DIALECT

// Dialect checks:
#  if HYDRA_THRUST_CPP_DIALECT < 2011
     // <C++11. Hard upgrade message:
     HYDRA_THRUST_COMPILER_DEPRECATION(C++14);
#  elif HYDRA_THRUST_CPP_DIALECT == 2011 && !defined(HYDRA_THRUST_IGNORE_DEPRECATED_CPP_11)
     // =C++11. Soft upgrade message:
     HYDRA_THRUST_COMPILER_DEPRECATION_SOFT(C++14, C++11);
#  endif

#endif // HYDRA_THRUST_IGNORE_DEPRECATED_DIALECT

#undef HYDRA_THRUST_COMPILER_DEPRECATION_SOFT
#undef HYDRA_THRUST_COMPILER_DEPRECATION
#undef HYDRA_THRUST_COMP_DEPR_IMPL
#undef HYDRA_THRUST_COMP_DEPR_IMPL0
#undef HYDRA_THRUST_COMP_DEPR_IMPL1
