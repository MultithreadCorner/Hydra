///////////////////////////////////////////////////////////////////////////////
//  Copyright (c)      2018 NVIDIA Corporation
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

/*! \file is_trivially_relocatable.h
 *  \brief <a href="https://wg21.link/P1144R0">P1144R0</a>'s
 *         \c is_trivially_relocatable, an extensible type trait indicating
 *         whether a type can be bitwise copied (e.g. via \c memcpy).
 */

#pragma once

#include <hydra/detail/external/thrust/detail/config.h>
#include <hydra/detail/external/thrust/detail/static_assert.h>
#include <hydra/detail/external/thrust/detail/type_traits.h>
#include <hydra/detail/external/thrust/type_traits/is_contiguous_iterator.h>

#if HYDRA_THRUST_CPP_DIALECT >= 2011
  #include <type_traits>
#endif
HYDRA_EXTERNAL_NAMESPACE_BEGIN
HYDRA_THRUST_BEGIN_NS

namespace detail
{

template <typename T>
struct is_trivially_relocatable_impl;

} // namespace detail

/// Unary metafunction returns \c true_type if \c T is \a TriviallyRelocatable, 
/// e.g. can be bitwise copied (with a facility like \c memcpy), and
/// \c false_type otherwise.
template <typename T>
#if HYDRA_THRUST_CPP_DIALECT >= 2011
using is_trivially_relocatable =
#else
struct is_trivially_relocatable :
#endif
  detail::is_trivially_relocatable_impl<T>
#if HYDRA_THRUST_CPP_DIALECT < 2011
{}
#endif
;

#if HYDRA_THRUST_CPP_DIALECT >= 2014
/// <code>constexpr bool</code> that is \c true if \c T is
/// \a TriviallyRelocatable e.g. can be copied bitwise (with a facility like
/// \c memcpy), and \c false otherwise.
template <typename T>
constexpr bool is_trivially_relocatable_v = is_trivially_relocatable<T>::value;
#endif

/// Unary metafunction returns \c true_type if \c From is \a TriviallyRelocatable
/// to \c To, e.g. can be bitwise copied (with a facility like \c memcpy), and
/// \c false_type otherwise.
template <typename From, typename To>
#if HYDRA_THRUST_CPP_DIALECT >= 2011
using is_trivially_relocatable_to =
#else
struct is_trivially_relocatable_to :
#endif
  integral_constant<
    bool
  , detail::is_same<From, To>::value && is_trivially_relocatable<To>::value
  >
#if HYDRA_THRUST_CPP_DIALECT < 2011
{}
#endif
;

#if HYDRA_THRUST_CPP_DIALECT >= 2014
/// <code>constexpr bool</code> that is \c true if \c From is 
/// \a TriviallyRelocatable to \c To, e.g. can be copied bitwise (with a
/// facility like \c memcpy), and \c false otherwise.
template <typename From, typename To>
constexpr bool is_trivially_relocatable_to_v
  = is_trivially_relocatable_to<From, To>::value;
#endif

/// Unary metafunction that returns \c true_type if the element type of
/// \c FromIterator is \a TriviallyRelocatable to the element type of
/// \c ToIterator, and \c false_type otherwise.
template <typename FromIterator, typename ToIterator>
#if HYDRA_THRUST_CPP_DIALECT >= 2011
using is_indirectly_trivially_relocatable_to =
#else
struct is_indirectly_trivially_relocatable_to :
#endif
  integral_constant<
    bool
  ,    is_contiguous_iterator<FromIterator>::value
    && is_contiguous_iterator<ToIterator>::value
    && is_trivially_relocatable_to<
         typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<FromIterator>::value_type,
         typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<ToIterator>::value_type
       >::value
  >
#if HYDRA_THRUST_CPP_DIALECT < 2011
{}
#endif
;

#if HYDRA_THRUST_CPP_DIALECT >= 2014
/// <code>constexpr bool</code> that is \c true if the element type of
/// \c FromIterator is \a TriviallyRelocatable to the element type of
/// \c ToIterator, and \c false otherwise.
template <typename FromIterator, typename ToIterator>
constexpr bool is_trivial_relocatable_sequence_copy_v
  = is_indirectly_trivially_relocatable_to<FromIterator, ToIterator>::value;
#endif

/// Customization point that can be customized to indicate that a type \c T is
/// \a TriviallyRelocatable, e.g. can be copied bitwise (with a facility like
/// \c memcpy).
template <typename T>
struct proclaim_trivially_relocatable : false_type {};

/// Declares that the type \c T is \a TriviallyRelocatable by specializing
/// `HYDRA_EXTERNAL_NS::thrust::proclaim_trivially_relocatable`.
#define HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(T)                              \
  HYDRA_EXTERNAL_NAMESPACE_BEGIN                                              \
  HYDRA_THRUST_BEGIN_NS                                                             \
  template <>                                                                 \
  struct proclaim_trivially_relocatable<T> : ::HYDRA_EXTERNAL_NS::thrust::true_type {};          \
  HYDRA_THRUST_END_NS                                                               \
  HYDRA_EXTERNAL_NAMESPACE_END                                                \
  /**/

///////////////////////////////////////////////////////////////////////////////

namespace detail
{

// There is no way to actually detect the libstdc++ version; __GLIBCXX__
// is always set to the date of libstdc++ being packaged, not the release
// day or version. This means that we can't detect the libstdc++ version,
// except when compiling with GCC.
//
// Therefore, for the best approximation of is_trivially_copyable, we need to
// handle three distinct cases:
// 1) GCC above 5, or another C++11 compiler not using libstdc++: use the
//      standard trait directly.
// 2) A C++11 compiler using libstdc++ that provides the intrinsic: use the
//      intrinsic.
// 3) Any other case (essentially: compiling without C++11): has_trivial_assign.

#ifndef __has_feature
    #define __has_feature(x) 0
#endif

template <typename T>
struct is_trivially_copyable_impl
    : integral_constant<
        bool,
        #if HYDRA_THRUST_CPP_DIALECT >= 2011
            #if defined(__GLIBCXX__) && __has_feature(is_trivially_copyable)
                __is_trivially_copyable(T)
            #elif HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_GCC && HYDRA_THRUST_GCC_VERSION >= 50000
                std::is_trivially_copyable<T>::value
            #else
                has_trivial_assign<T>::value
            #endif
        #else
            has_trivial_assign<T>::value
        #endif
    >
{
};

// https://wg21.link/P1144R0#wording-inheritance
template <typename T>
struct is_trivially_relocatable_impl
    : integral_constant<
        bool,
        is_trivially_copyable_impl<T>::value
            || proclaim_trivially_relocatable<T>::value
    >
{};

template <typename T, std::size_t N>
struct is_trivially_relocatable_impl<T[N]> : is_trivially_relocatable_impl<T> {};

} // namespace detail

HYDRA_THRUST_END_NS
HYDRA_EXTERNAL_NAMESPACE_END

#if HYDRA_THRUST_DEVICE_SYSTEM == HYDRA_THRUST_DEVICE_SYSTEM_CUDA

#include <hydra/detail/external/thrust/system/cuda/detail/guarded_cuda_runtime_api.h>

HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(char1)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(char2)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(char3)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(char4)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(uchar1)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(uchar2)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(uchar3)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(uchar4)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(short1)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(short2)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(short3)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(short4)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(ushort1)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(ushort2)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(ushort3)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(ushort4)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(int1)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(int2)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(int3)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(int4)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(uint1)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(uint2)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(uint3)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(uint4)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(long1)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(long2)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(long3)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(long4)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(ulong1)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(ulong2)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(ulong3)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(ulong4)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(longlong1)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(longlong2)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(longlong3)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(longlong4)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(ulonglong1)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(ulonglong2)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(ulonglong3)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(ulonglong4)

struct __half;
struct __half2;

HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(__half)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(__half2)

HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(float1)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(float2)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(float3)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(float4)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(double1)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(double2)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(double3)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(double4)
#endif

