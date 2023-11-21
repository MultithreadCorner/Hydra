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

/*! \file compiler.h
 *  \brief Compiler-specific configuration
 */

#pragma once

// enumerate host compilers we know about
#define HYDRA_THRUST_HOST_COMPILER_UNKNOWN 0
#define HYDRA_THRUST_HOST_COMPILER_MSVC    1
#define HYDRA_THRUST_HOST_COMPILER_GCC     2
#define HYDRA_THRUST_HOST_COMPILER_CLANG   3
#define HYDRA_THRUST_HOST_COMPILER_INTEL   4

// enumerate device compilers we know about
#define HYDRA_THRUST_DEVICE_COMPILER_UNKNOWN 0
#define HYDRA_THRUST_DEVICE_COMPILER_MSVC    1
#define HYDRA_THRUST_DEVICE_COMPILER_GCC     2
#define HYDRA_THRUST_DEVICE_COMPILER_CLANG   3
#define HYDRA_THRUST_DEVICE_COMPILER_NVCC    4

// figure out which host compiler we're using
// XXX we should move the definition of HYDRA_THRUST_DEPRECATED out of this logic
#if   defined(_MSC_VER)
#define HYDRA_THRUST_HOST_COMPILER HYDRA_THRUST_HOST_COMPILER_MSVC
#define HYDRA_THRUST_MSVC_VERSION _MSC_VER
#define HYDRA_THRUST_MSVC_VERSION_FULL _MSC_FULL_VER
#elif defined(__ICC)
#define HYDRA_THRUST_HOST_COMPILER HYDRA_THRUST_HOST_COMPILER_INTEL
#elif defined(__clang__)
#define HYDRA_THRUST_HOST_COMPILER HYDRA_THRUST_HOST_COMPILER_CLANG
#define HYDRA_THRUST_CLANG_VERSION (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#elif defined(__GNUC__)
#define HYDRA_THRUST_HOST_COMPILER HYDRA_THRUST_HOST_COMPILER_GCC
#define HYDRA_THRUST_GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#if (HYDRA_THRUST_GCC_VERSION >= 50000)
#define HYDRA_THRUST_MODERN_GCC
#else
#define HYDRA_THRUST_LEGACY_GCC
#endif
#else
#define HYDRA_THRUST_HOST_COMPILER HYDRA_THRUST_HOST_COMPILER_UNKNOWN
#endif // HYDRA_THRUST_HOST_COMPILER

// figure out which device compiler we're using
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
#define HYDRA_THRUST_DEVICE_COMPILER HYDRA_THRUST_DEVICE_COMPILER_NVCC
#elif HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_MSVC
#define HYDRA_THRUST_DEVICE_COMPILER HYDRA_THRUST_DEVICE_COMPILER_MSVC
#elif HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_GCC
#define HYDRA_THRUST_DEVICE_COMPILER HYDRA_THRUST_DEVICE_COMPILER_GCC
#elif HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_CLANG
// CUDA-capable clang should behave similar to NVCC.
#if defined(__CUDA__)
#define HYDRA_THRUST_DEVICE_COMPILER HYDRA_THRUST_DEVICE_COMPILER_NVCC
#else
#define HYDRA_THRUST_DEVICE_COMPILER HYDRA_THRUST_DEVICE_COMPILER_CLANG
#endif
#else
#define HYDRA_THRUST_DEVICE_COMPILER HYDRA_THRUST_DEVICE_COMPILER_UNKNOWN
#endif

// is the device compiler capable of compiling omp?
#if defined(_OPENMP) || defined(_NVHPC_STDPAR_OPENMP)
#define HYDRA_THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE HYDRA_THRUST_TRUE
#else
#define HYDRA_THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE HYDRA_THRUST_FALSE
#endif // _OPENMP


#if (HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_MSVC) && !defined(__CUDA_ARCH__)
  #define HYDRA_THRUST_DISABLE_MSVC_WARNING_BEGIN(x)                                \
    __pragma(warning(push))                                                   \
    __pragma(warning(disable : x))                                            \
    /**/
  #define HYDRA_THRUST_DISABLE_MSVC_WARNING_END(x)                                  \
    __pragma(warning(pop))                                                    \
    /**/
#else
  #define HYDRA_THRUST_DISABLE_MSVC_WARNING_BEGIN(x)
  #define HYDRA_THRUST_DISABLE_MSVC_WARNING_END(x)
#endif

#if (HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_CLANG) && !defined(__CUDA_ARCH__)
  #define HYDRA_THRUST_IGNORE_CLANG_WARNING_IMPL(x)                                 \
    HYDRA_THRUST_PP_STRINGIZE(clang diagnostic ignored x)                           \
    /**/
  #define HYDRA_THRUST_IGNORE_CLANG_WARNING(x)                                      \
    HYDRA_THRUST_IGNORE_CLANG_WARNING_IMPL(HYDRA_THRUST_PP_STRINGIZE(x))                  \
    /**/

  #define HYDRA_THRUST_DISABLE_CLANG_WARNING_BEGIN(x)                               \
    _Pragma("clang diagnostic push")                                          \
    _Pragma(HYDRA_THRUST_IGNORE_CLANG_WARNING(x))                                   \
    /**/
  #define HYDRA_THRUST_DISABLE_CLANG_WARNING_END(x)                                 \
    _Pragma("clang diagnostic pop")                                           \
    /**/
#else
  #define HYDRA_THRUST_DISABLE_CLANG_WARNING_BEGIN(x)
  #define HYDRA_THRUST_DISABLE_CLANG_WARNING_END(x)
#endif

#if (HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_GCC) && !defined(__CUDA_ARCH__)
  #define HYDRA_THRUST_IGNORE_GCC_WARNING_IMPL(x)                                   \
    HYDRA_THRUST_PP_STRINGIZE(GCC diagnostic ignored x)                             \
    /**/
  #define HYDRA_THRUST_IGNORE_GCC_WARNING(x)                                        \
    HYDRA_THRUST_IGNORE_GCC_WARNING_IMPL(HYDRA_THRUST_PP_STRINGIZE(x))                    \
    /**/

  #define HYDRA_THRUST_DISABLE_GCC_WARNING_BEGIN(x)                                 \
    _Pragma("GCC diagnostic push")                                            \
    _Pragma(HYDRA_THRUST_IGNORE_GCC_WARNING(x))                                     \
    /**/
  #define HYDRA_THRUST_DISABLE_GCC_WARNING_END(x)                                   \
    _Pragma("GCC diagnostic pop")                                             \
    /**/
#else
  #define HYDRA_THRUST_DISABLE_GCC_WARNING_BEGIN(x)
  #define HYDRA_THRUST_DISABLE_GCC_WARNING_END(x)
#endif

#define HYDRA_THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_BEGIN               \
  HYDRA_THRUST_DISABLE_MSVC_WARNING_BEGIN(4244 4267)                                \
  /**/
#define HYDRA_THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END                 \
  HYDRA_THRUST_DISABLE_MSVC_WARNING_END(4244 4267)                                  \
  /**/
#define HYDRA_THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING(x)                  \
  HYDRA_THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_BEGIN                     \
  x;                                                                          \
  HYDRA_THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END                       \
  /**/

#define HYDRA_THRUST_DISABLE_MSVC_FORCING_VALUE_TO_BOOL_WARNING_BEGIN               \
  HYDRA_THRUST_DISABLE_MSVC_WARNING_BEGIN(4800)                                     \
  /**/
#define HYDRA_THRUST_DISABLE_MSVC_FORCING_VALUE_TO_BOOL_WARNING_END                 \
  HYDRA_THRUST_DISABLE_MSVC_WARNING_END(4800)                                       \
  /**/
#define HYDRA_THRUST_DISABLE_MSVC_FORCING_VALUE_TO_BOOL_WARNING(x)                  \
  HYDRA_THRUST_DISABLE_MSVC_FORCING_VALUE_TO_BOOL_WARNING_BEGIN                     \
  x;                                                                          \
  HYDRA_THRUST_DISABLE_MSVC_FORCING_VALUE_TO_BOOL_WARNING_END                       \
  /**/

#define HYDRA_THRUST_DISABLE_CLANG_SELF_ASSIGNMENT_WARNING_BEGIN                    \
  HYDRA_THRUST_DISABLE_CLANG_WARNING_BEGIN(-Wself-assign)                           \
  /**/
#define HYDRA_THRUST_DISABLE_CLANG_SELF_ASSIGNMENT_WARNING_END                      \
  HYDRA_THRUST_DISABLE_CLANG_WARNING_END(-Wself-assign)                             \
  /**/
#define HYDRA_THRUST_DISABLE_CLANG_SELF_ASSIGNMENT_WARNING(x)                       \
  HYDRA_THRUST_DISABLE_CLANG_SELF_ASSIGNMENT_WARNING_BEGIN                          \
  x;                                                                          \
  HYDRA_THRUST_DISABLE_CLANG_SELF_ASSIGNMENT_WARNING_END                            \
  /**/

#define HYDRA_THRUST_DISABLE_CLANG_AND_GCC_INITIALIZER_REORDERING_WARNING_BEGIN     \
  HYDRA_THRUST_DISABLE_CLANG_WARNING_BEGIN(-Wreorder)                               \
  HYDRA_THRUST_DISABLE_GCC_WARNING_BEGIN(-Wreorder)                                 \
  /**/
#define HYDRA_THRUST_DISABLE_CLANG_AND_GCC_INITIALIZER_REORDERING_WARNING_END       \
  HYDRA_THRUST_DISABLE_CLANG_WARNING_END(-Wreorder)                                 \
  HYDRA_THRUST_DISABLE_GCC_WARNING_END(-Wreorder)                                   \
  /**/
#define HYDRA_THRUST_DISABLE_CLANG_AND_GCC_INITIALIZER_REORDERING_WARNING(x)        \
  HYDRA_THRUST_DISABLE_CLANG_AND_GCC_INITIALIZER_REORDERING_WARNING_BEGIN           \
  x;                                                                          \
  HYDRA_THRUST_DISABLE_CLANG_AND_GCC_INITIALIZER_REORDERING_WARNING_END             \
  /**/


