// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2015 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HYDRA_EIGEN_MACROS_H
#define HYDRA_EIGEN_MACROS_H




#define HYDRA_EIGEN_WORLD_VERSION 3
#define HYDRA_EIGEN_MAJOR_VERSION 3
#define HYDRA_EIGEN_MINOR_VERSION 4

#define HYDRA_EIGEN_VERSION_AT_LEAST(x,y,z) (HYDRA_EIGEN_WORLD_VERSION>x || (HYDRA_EIGEN_WORLD_VERSION>=x && \
                                      (HYDRA_EIGEN_MAJOR_VERSION>y || (HYDRA_EIGEN_MAJOR_VERSION>=y && \
                                                                 HYDRA_EIGEN_MINOR_VERSION>=z))))

// Compiler identification, HYDRA_EIGEN_COMP_*

/// \internal HYDRA_EIGEN_COMP_GNUC set to 1 for all compilers compatible with GCC
#ifdef __GNUC__
  #define HYDRA_EIGEN_COMP_GNUC 1
#else
  #define HYDRA_EIGEN_COMP_GNUC 0
#endif

/// \internal HYDRA_EIGEN_COMP_CLANG set to major+minor version (e.g., 307 for clang 3.7) if the compiler is clang
#if defined(__clang__)
  #define HYDRA_EIGEN_COMP_CLANG (__clang_major__*100+__clang_minor__)
#else
  #define HYDRA_EIGEN_COMP_CLANG 0
#endif


/// \internal HYDRA_EIGEN_COMP_LLVM set to 1 if the compiler backend is llvm
#if defined(__llvm__)
  #define HYDRA_EIGEN_COMP_LLVM 1
#else
  #define HYDRA_EIGEN_COMP_LLVM 0
#endif

/// \internal HYDRA_EIGEN_COMP_ICC set to __INTEL_COMPILER if the compiler is Intel compiler, 0 otherwise
#if defined(__INTEL_COMPILER)
  #define HYDRA_EIGEN_COMP_ICC __INTEL_COMPILER
#else
  #define HYDRA_EIGEN_COMP_ICC 0
#endif

/// \internal HYDRA_EIGEN_COMP_MINGW set to 1 if the compiler is mingw
#if defined(__MINGW32__)
  #define HYDRA_EIGEN_COMP_MINGW 1
#else
  #define HYDRA_EIGEN_COMP_MINGW 0
#endif

/// \internal HYDRA_EIGEN_COMP_SUNCC set to 1 if the compiler is Solaris Studio
#if defined(__SUNPRO_CC)
  #define HYDRA_EIGEN_COMP_SUNCC 1
#else
  #define HYDRA_EIGEN_COMP_SUNCC 0
#endif

/// \internal HYDRA_EIGEN_COMP_MSVC set to _MSC_VER if the compiler is Microsoft Visual C++, 0 otherwise.
#if defined(_MSC_VER)
  #define HYDRA_EIGEN_COMP_MSVC _MSC_VER
#else
  #define HYDRA_EIGEN_COMP_MSVC 0
#endif

// For the record, here is a table summarizing the possible values for HYDRA_EIGEN_COMP_MSVC:
//  name  ver   MSC_VER
//  2008    9      1500
//  2010   10      1600
//  2012   11      1700
//  2013   12      1800
//  2015   14      1900
//  "15"   15      1900

/// \internal HYDRA_EIGEN_COMP_MSVC_STRICT set to 1 if the compiler is really Microsoft Visual C++ and not ,e.g., ICC or clang-cl
#if HYDRA_EIGEN_COMP_MSVC && !(HYDRA_EIGEN_COMP_ICC || HYDRA_EIGEN_COMP_LLVM || HYDRA_EIGEN_COMP_CLANG)
  #define HYDRA_EIGEN_COMP_MSVC_STRICT _MSC_VER
#else
  #define HYDRA_EIGEN_COMP_MSVC_STRICT 0
#endif

/// \internal HYDRA_EIGEN_COMP_IBM set to 1 if the compiler is IBM XL C++
#if defined(__IBMCPP__) || defined(__xlc__)
  #define HYDRA_EIGEN_COMP_IBM 1
#else
  #define HYDRA_EIGEN_COMP_IBM 0
#endif

/// \internal HYDRA_EIGEN_COMP_PGI set to 1 if the compiler is Portland Group Compiler
#if defined(__PGI)
  #define HYDRA_EIGEN_COMP_PGI 1
#else
  #define HYDRA_EIGEN_COMP_PGI 0
#endif

/// \internal HYDRA_EIGEN_COMP_ARM set to 1 if the compiler is ARM Compiler
#if defined(__CC_ARM) || defined(__ARMCC_VERSION)
  #define HYDRA_EIGEN_COMP_ARM 1
#else
  #define HYDRA_EIGEN_COMP_ARM 0
#endif

/// \internal HYDRA_EIGEN_COMP_ARM set to 1 if the compiler is ARM Compiler
#if defined(__EMSCRIPTEN__)
  #define HYDRA_EIGEN_COMP_EMSCRIPTEN 1
#else
  #define HYDRA_EIGEN_COMP_EMSCRIPTEN 0
#endif


/// \internal HYDRA_EIGEN_GNUC_STRICT set to 1 if the compiler is really GCC and not a compatible compiler (e.g., ICC, clang, mingw, etc.)
#if HYDRA_EIGEN_COMP_GNUC && !(HYDRA_EIGEN_COMP_CLANG || HYDRA_EIGEN_COMP_ICC || HYDRA_EIGEN_COMP_MINGW || HYDRA_EIGEN_COMP_PGI || HYDRA_EIGEN_COMP_IBM || HYDRA_EIGEN_COMP_ARM || HYDRA_EIGEN_COMP_EMSCRIPTEN)
  #define HYDRA_EIGEN_COMP_GNUC_STRICT 1
#else
  #define HYDRA_EIGEN_COMP_GNUC_STRICT 0
#endif


#if HYDRA_EIGEN_COMP_GNUC
  #define HYDRA_EIGEN_GNUC_AT_LEAST(x,y) ((__GNUC__==x && __GNUC_MINOR__>=y) || __GNUC__>x)
  #define HYDRA_EIGEN_GNUC_AT_MOST(x,y)  ((__GNUC__==x && __GNUC_MINOR__<=y) || __GNUC__<x)
  #define HYDRA_EIGEN_GNUC_AT(x,y)       ( __GNUC__==x && __GNUC_MINOR__==y )
#else
  #define HYDRA_EIGEN_GNUC_AT_LEAST(x,y) 0
  #define HYDRA_EIGEN_GNUC_AT_MOST(x,y)  0
  #define HYDRA_EIGEN_GNUC_AT(x,y)       0
#endif

// FIXME: could probably be removed as we do not support gcc 3.x anymore
#if HYDRA_EIGEN_COMP_GNUC && (__GNUC__ <= 3)
#define HYDRA_EIGEN_GCC3_OR_OLDER 1
#else
#define HYDRA_EIGEN_GCC3_OR_OLDER 0
#endif


// Architecture identification, HYDRA_EIGEN_ARCH_*

#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64)
  #define HYDRA_EIGEN_ARCH_x86_64 1
#else
  #define HYDRA_EIGEN_ARCH_x86_64 0
#endif

#if defined(__i386__) || defined(_M_IX86) || defined(_X86_) || defined(__i386)
  #define HYDRA_EIGEN_ARCH_i386 1
#else
  #define HYDRA_EIGEN_ARCH_i386 0
#endif

#if HYDRA_EIGEN_ARCH_x86_64 || HYDRA_EIGEN_ARCH_i386
  #define HYDRA_EIGEN_ARCH_i386_OR_x86_64 1
#else
  #define HYDRA_EIGEN_ARCH_i386_OR_x86_64 0
#endif

/// \internal HYDRA_EIGEN_ARCH_ARM set to 1 if the architecture is ARM
#if defined(__arm__)
  #define HYDRA_EIGEN_ARCH_ARM 1
#else
  #define HYDRA_EIGEN_ARCH_ARM 0
#endif

/// \internal HYDRA_EIGEN_ARCH_ARM64 set to 1 if the architecture is ARM64
#if defined(__aarch64__)
  #define HYDRA_EIGEN_ARCH_ARM64 1
#else
  #define HYDRA_EIGEN_ARCH_ARM64 0
#endif

#if HYDRA_EIGEN_ARCH_ARM || HYDRA_EIGEN_ARCH_ARM64
  #define HYDRA_EIGEN_ARCH_ARM_OR_ARM64 1
#else
  #define HYDRA_EIGEN_ARCH_ARM_OR_ARM64 0
#endif

/// \internal HYDRA_EIGEN_ARCH_MIPS set to 1 if the architecture is MIPS
#if defined(__mips__) || defined(__mips)
  #define HYDRA_EIGEN_ARCH_MIPS 1
#else
  #define HYDRA_EIGEN_ARCH_MIPS 0
#endif

/// \internal HYDRA_EIGEN_ARCH_SPARC set to 1 if the architecture is SPARC
#if defined(__sparc__) || defined(__sparc)
  #define HYDRA_EIGEN_ARCH_SPARC 1
#else
  #define HYDRA_EIGEN_ARCH_SPARC 0
#endif

/// \internal HYDRA_EIGEN_ARCH_IA64 set to 1 if the architecture is Intel Itanium
#if defined(__ia64__)
  #define HYDRA_EIGEN_ARCH_IA64 1
#else
  #define HYDRA_EIGEN_ARCH_IA64 0
#endif

/// \internal HYDRA_EIGEN_ARCH_PPC set to 1 if the architecture is PowerPC
#if defined(__powerpc__) || defined(__ppc__) || defined(_M_PPC)
  #define HYDRA_EIGEN_ARCH_PPC 1
#else
  #define HYDRA_EIGEN_ARCH_PPC 0
#endif



// Operating system identification, HYDRA_EIGEN_OS_*

/// \internal HYDRA_EIGEN_OS_UNIX set to 1 if the OS is a unix variant
#if defined(__unix__) || defined(__unix)
  #define HYDRA_EIGEN_OS_UNIX 1
#else
  #define HYDRA_EIGEN_OS_UNIX 0
#endif

/// \internal HYDRA_EIGEN_OS_LINUX set to 1 if the OS is based on Linux kernel
#if defined(__linux__)
  #define HYDRA_EIGEN_OS_LINUX 1
#else
  #define HYDRA_EIGEN_OS_LINUX 0
#endif

/// \internal HYDRA_EIGEN_OS_ANDROID set to 1 if the OS is Android
// note: ANDROID is defined when using ndk_build, __ANDROID__ is defined when using a standalone toolchain.
#if defined(__ANDROID__) || defined(ANDROID)
  #define HYDRA_EIGEN_OS_ANDROID 1
#else
  #define HYDRA_EIGEN_OS_ANDROID 0
#endif

/// \internal HYDRA_EIGEN_OS_GNULINUX set to 1 if the OS is GNU Linux and not Linux-based OS (e.g., not android)
#if defined(__gnu_linux__) && !(HYDRA_EIGEN_OS_ANDROID)
  #define HYDRA_EIGEN_OS_GNULINUX 1
#else
  #define HYDRA_EIGEN_OS_GNULINUX 0
#endif

/// \internal HYDRA_EIGEN_OS_BSD set to 1 if the OS is a BSD variant
#if defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__bsdi__) || defined(__DragonFly__)
  #define HYDRA_EIGEN_OS_BSD 1
#else
  #define HYDRA_EIGEN_OS_BSD 0
#endif

/// \internal HYDRA_EIGEN_OS_MAC set to 1 if the OS is MacOS
#if defined(__APPLE__)
  #define HYDRA_EIGEN_OS_MAC 1
#else
  #define HYDRA_EIGEN_OS_MAC 0
#endif

/// \internal HYDRA_EIGEN_OS_QNX set to 1 if the OS is QNX
#if defined(__QNX__)
  #define HYDRA_EIGEN_OS_QNX 1
#else
  #define HYDRA_EIGEN_OS_QNX 0
#endif

/// \internal HYDRA_EIGEN_OS_WIN set to 1 if the OS is Windows based
#if defined(_WIN32)
  #define HYDRA_EIGEN_OS_WIN 1
#else
  #define HYDRA_EIGEN_OS_WIN 0
#endif

/// \internal HYDRA_EIGEN_OS_WIN64 set to 1 if the OS is Windows 64bits
#if defined(_WIN64)
  #define HYDRA_EIGEN_OS_WIN64 1
#else
  #define HYDRA_EIGEN_OS_WIN64 0
#endif

/// \internal HYDRA_EIGEN_OS_WINCE set to 1 if the OS is Windows CE
#if defined(_WIN32_WCE)
  #define HYDRA_EIGEN_OS_WINCE 1
#else
  #define HYDRA_EIGEN_OS_WINCE 0
#endif

/// \internal HYDRA_EIGEN_OS_CYGWIN set to 1 if the OS is Windows/Cygwin
#if defined(__CYGWIN__)
  #define HYDRA_EIGEN_OS_CYGWIN 1
#else
  #define HYDRA_EIGEN_OS_CYGWIN 0
#endif

/// \internal HYDRA_EIGEN_OS_WIN_STRICT set to 1 if the OS is really Windows and not some variants
#if HYDRA_EIGEN_OS_WIN && !( HYDRA_EIGEN_OS_WINCE || HYDRA_EIGEN_OS_CYGWIN )
  #define HYDRA_EIGEN_OS_WIN_STRICT 1
#else
  #define HYDRA_EIGEN_OS_WIN_STRICT 0
#endif

/// \internal HYDRA_EIGEN_OS_SUN set to 1 if the OS is SUN
#if (defined(sun) || defined(__sun)) && !(defined(__SVR4) || defined(__svr4__))
  #define HYDRA_EIGEN_OS_SUN 1
#else
  #define HYDRA_EIGEN_OS_SUN 0
#endif

/// \internal HYDRA_EIGEN_OS_SOLARIS set to 1 if the OS is Solaris
#if (defined(sun) || defined(__sun)) && (defined(__SVR4) || defined(__svr4__))
  #define HYDRA_EIGEN_OS_SOLARIS 1
#else
  #define HYDRA_EIGEN_OS_SOLARIS 0
#endif



#if HYDRA_EIGEN_GNUC_AT_MOST(4,3) && !HYDRA_EIGEN_COMP_CLANG
  // see bug 89
  #define HYDRA_EIGEN_SAFE_TO_USE_STANDARD_ASSERT_MACRO 0
#else
  #define HYDRA_EIGEN_SAFE_TO_USE_STANDARD_ASSERT_MACRO 1
#endif

// This macro can be used to prevent from macro expansion, e.g.:
//   std::max HYDRA_EIGEN_NOT_A_MACRO(a,b)
#define HYDRA_EIGEN_NOT_A_MACRO

#ifdef HYDRA_EIGEN_DEFAULT_TO_ROW_MAJOR
#define HYDRA_EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION HYDRA_EXTERNAL_NS::Eigen::RowMajor
#else
#define HYDRA_EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION HYDRA_EXTERNAL_NS::Eigen::ColMajor
#endif

#ifndef HYDRA_EIGEN_DEFAULT_DENSE_INDEX_TYPE
#define HYDRA_EIGEN_DEFAULT_DENSE_INDEX_TYPE std::ptrdiff_t
#endif

// Cross compiler wrapper around LLVM's __has_builtin
#ifdef __has_builtin
#  define HYDRA_EIGEN_HAS_BUILTIN(x) __has_builtin(x)
#else
#  define HYDRA_EIGEN_HAS_BUILTIN(x) 0
#endif

// A Clang feature extension to determine compiler features.
// We use it to determine 'cxx_rvalue_references'
#ifndef __has_feature
# define __has_feature(x) 0
#endif

// Upperbound on the C++ version to use.
// Expected values are 03, 11, 14, 17, etc.
// By default, let's use an arbitrarily large C++ version.
#ifndef HYDRA_EIGEN_MAX_CPP_VER
#define HYDRA_EIGEN_MAX_CPP_VER 99
#endif

#if HYDRA_EIGEN_MAX_CPP_VER>=11 && (defined(__cplusplus) && (__cplusplus >= 201103L) || HYDRA_EIGEN_COMP_MSVC >= 1900)
#define HYDRA_EIGEN_HAS_CXX11 1
#else
#define HYDRA_EIGEN_HAS_CXX11 0
#endif


// Do we support r-value references?
#ifndef HYDRA_EIGEN_HAS_RVALUE_REFERENCES
#if HYDRA_EIGEN_MAX_CPP_VER>=11 && \
    (__has_feature(cxx_rvalue_references) || \
    (defined(__cplusplus) && __cplusplus >= 201103L) || \
    (HYDRA_EIGEN_COMP_MSVC >= 1600))
  #define HYDRA_EIGEN_HAS_RVALUE_REFERENCES 1
#else
  #define HYDRA_EIGEN_HAS_RVALUE_REFERENCES 0
#endif
#endif

// Does the compiler support C99?
#ifndef HYDRA_EIGEN_HAS_C99_MATH
#if HYDRA_EIGEN_MAX_CPP_VER>=11 && \
    ((defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901))       \
  || (defined(__GNUC__) && defined(_GLIBCXX_USE_C99)) \
  || (defined(_LIBCPP_VERSION) && !defined(_MSC_VER)))
  #define HYDRA_EIGEN_HAS_C99_MATH 1
#else
  #define HYDRA_EIGEN_HAS_C99_MATH 0
#endif
#endif

// Does the compiler support result_of?
#ifndef HYDRA_EIGEN_HAS_STD_RESULT_OF
#if HYDRA_EIGEN_MAX_CPP_VER>=11 && ((__has_feature(cxx_lambdas) || (defined(__cplusplus) && __cplusplus >= 201103L)))
#define HYDRA_EIGEN_HAS_STD_RESULT_OF 1
#else
#define HYDRA_EIGEN_HAS_STD_RESULT_OF 0
#endif
#endif

// Does the compiler support variadic templates?
#ifndef HYDRA_EIGEN_HAS_VARIADIC_TEMPLATES
#if HYDRA_EIGEN_MAX_CPP_VER>=11 && (__cplusplus > 199711L || HYDRA_EIGEN_COMP_MSVC >= 1900) \
  && ( !defined(__NVCC__) || !HYDRA_EIGEN_ARCH_ARM_OR_ARM64  )
    // ^^ Disable the use of variadic templates when compiling with versions of nvcc older than 8.0 on ARM devices:
    //    this prevents nvcc from crashing when compiling Eigen on Tegra X1
#define HYDRA_EIGEN_HAS_VARIADIC_TEMPLATES 1
#else
#define HYDRA_EIGEN_HAS_VARIADIC_TEMPLATES 0
#endif
#endif

// Does the compiler fully support const expressions? (as in c++14)
#ifndef HYDRA_EIGEN_HAS_CONSTEXPR

#ifdef __CUDACC__
// Const expressions are supported provided that c++11 is enabled and we're using either clang or nvcc 7.5 or above
#if HYDRA_EIGEN_MAX_CPP_VER>=14 && (__cplusplus > 199711L && defined(__CUDACC_VER__) && (HYDRA_EIGEN_COMP_CLANG || __CUDACC_VER__ >= 70500))
  #define HYDRA_EIGEN_HAS_CONSTEXPR 1
#endif
#elif HYDRA_EIGEN_MAX_CPP_VER>=14 && (__has_feature(cxx_relaxed_constexpr) || (defined(__cplusplus) && __cplusplus >= 201402L) || \
  (HYDRA_EIGEN_GNUC_AT_LEAST(4,8) && (__cplusplus > 199711L)))
#define HYDRA_EIGEN_HAS_CONSTEXPR 1
#endif

#ifndef HYDRA_EIGEN_HAS_CONSTEXPR
#define HYDRA_EIGEN_HAS_CONSTEXPR 0
#endif

#endif

// Does the compiler support C++11 math?
// Let's be conservative and enable the default C++11 implementation only if we are sure it exists
#ifndef HYDRA_EIGEN_HAS_CXX11_MATH
  #if HYDRA_EIGEN_MAX_CPP_VER>=11 && ((__cplusplus > 201103L) || (__cplusplus >= 201103L) && (HYDRA_EIGEN_COMP_GNUC_STRICT || HYDRA_EIGEN_COMP_CLANG || HYDRA_EIGEN_COMP_MSVC || HYDRA_EIGEN_COMP_ICC)  \
      && (HYDRA_EIGEN_ARCH_i386_OR_x86_64) && (HYDRA_EIGEN_OS_GNULINUX || HYDRA_EIGEN_OS_WIN_STRICT || HYDRA_EIGEN_OS_MAC))
    #define HYDRA_EIGEN_HAS_CXX11_MATH 1
  #else
    #define HYDRA_EIGEN_HAS_CXX11_MATH 0
  #endif
#endif

// Does the compiler support proper C++11 containers?
#ifndef HYDRA_EIGEN_HAS_CXX11_CONTAINERS
  #if    HYDRA_EIGEN_MAX_CPP_VER>=11 && \
         ((__cplusplus > 201103L) \
      || ((__cplusplus >= 201103L) && (HYDRA_EIGEN_COMP_GNUC_STRICT || HYDRA_EIGEN_COMP_CLANG || HYDRA_EIGEN_COMP_ICC>=1400)) \
      || HYDRA_EIGEN_COMP_MSVC >= 1900)
    #define HYDRA_EIGEN_HAS_CXX11_CONTAINERS 1
  #else
    #define HYDRA_EIGEN_HAS_CXX11_CONTAINERS 0
  #endif
#endif

// Does the compiler support C++11 noexcept?
#ifndef HYDRA_EIGEN_HAS_CXX11_NOEXCEPT
  #if    HYDRA_EIGEN_MAX_CPP_VER>=11 && \
         (__has_feature(cxx_noexcept) \
      || (__cplusplus > 201103L) \
      || ((__cplusplus >= 201103L) && (HYDRA_EIGEN_COMP_GNUC_STRICT || HYDRA_EIGEN_COMP_CLANG || HYDRA_EIGEN_COMP_ICC>=1400)) \
      || HYDRA_EIGEN_COMP_MSVC >= 1900)
    #define HYDRA_EIGEN_HAS_CXX11_NOEXCEPT 1
  #else
    #define HYDRA_EIGEN_HAS_CXX11_NOEXCEPT 0
  #endif
#endif

/** Allows to disable some optimizations which might affect the accuracy of the result.
  * Such optimization are enabled by default, and set HYDRA_EIGEN_FAST_MATH to 0 to disable them.
  * They currently include:
  *   - single precision ArrayBase::sin() and ArrayBase::cos() for SSE and AVX vectorization.
  */
#ifndef HYDRA_EIGEN_FAST_MATH
#define HYDRA_EIGEN_FAST_MATH 1
#endif

#define HYDRA_EIGEN_DEBUG_VAR(x) std::cerr << #x << " = " << x << std::endl;

// concatenate two tokens
#define HYDRA_EIGEN_CAT2(a,b) a ## b
#define HYDRA_EIGEN_CAT(a,b) HYDRA_EIGEN_CAT2(a,b)

#define HYDRA_EIGEN_COMMA ,

// convert a token to a string
#define HYDRA_EIGEN_MAKESTRING2(a) #a
#define HYDRA_EIGEN_MAKESTRING(a) HYDRA_EIGEN_MAKESTRING2(a)

// HYDRA_EIGEN_STRONG_INLINE is a stronger version of the inline, using __forceinline on MSVC,
// but it still doesn't use GCC's always_inline. This is useful in (common) situations where MSVC needs forceinline
// but GCC is still doing fine with just inline.
#if HYDRA_EIGEN_COMP_MSVC || HYDRA_EIGEN_COMP_ICC
#define HYDRA_EIGEN_STRONG_INLINE __forceinline
#else
#define HYDRA_EIGEN_STRONG_INLINE inline
#endif

// HYDRA_EIGEN_ALWAYS_INLINE is the stronget, it has the effect of making the function inline and adding every possible
// attribute to maximize inlining. This should only be used when really necessary: in particular,
// it uses __attribute__((always_inline)) on GCC, which most of the time is useless and can severely harm compile times.
// FIXME with the always_inline attribute,
// gcc 3.4.x and 4.1 reports the following compilation error:
//   Eval.h:91: sorry, unimplemented: inlining failed in call to 'const HYDRA_EXTERNAL_NS::Eigen::Eval<Derived> HYDRA_EXTERNAL_NS::Eigen::MatrixBase<Scalar, Derived>::eval() const'
//    : function body not available
//   See also bug 1367
#if HYDRA_EIGEN_GNUC_AT_LEAST(4,2)
#define HYDRA_EIGEN_ALWAYS_INLINE __attribute__((always_inline)) inline
#else
#define HYDRA_EIGEN_ALWAYS_INLINE HYDRA_EIGEN_STRONG_INLINE
#endif

#if HYDRA_EIGEN_COMP_GNUC
#define HYDRA_EIGEN_DONT_INLINE __attribute__((noinline))
#elif HYDRA_EIGEN_COMP_MSVC
#define HYDRA_EIGEN_DONT_INLINE __declspec(noinline)
#else
#define HYDRA_EIGEN_DONT_INLINE
#endif

#if HYDRA_EIGEN_COMP_GNUC
#define HYDRA_EIGEN_PERMISSIVE_EXPR __extension__
#else
#define HYDRA_EIGEN_PERMISSIVE_EXPR
#endif

// this macro allows to get rid of linking errors about multiply defined functions.
//  - static is not very good because it prevents definitions from different object files to be merged.
//           So static causes the resulting linked executable to be bloated with multiple copies of the same function.
//  - inline is not perfect either as it unwantedly hints the compiler toward inlining the function.
#define HYDRA_EIGEN_DECLARE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS
#define HYDRA_EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS inline

#ifdef NDEBUG
# ifndef HYDRA_EIGEN_NO_DEBUG
#  define HYDRA_EIGEN_NO_DEBUG
# endif
#endif

// eigen_plain_assert is where we implement the workaround for the assert() bug in GCC <= 4.3, see bug 89
#ifdef HYDRA_EIGEN_NO_DEBUG
  #define eigen_plain_assert(x)
#else
  #if HYDRA_EIGEN_SAFE_TO_USE_STANDARD_ASSERT_MACRO
    HYDRA_EXTERNAL_NAMESPACE_BEGIN namespace Eigen {
    namespace internal {
    inline bool copy_bool(bool b) { return b; }
    }
    } HYDRA_EXTERNAL_NAMESPACE_END
    #define eigen_plain_assert(x) assert(x)
  #else
    // work around bug 89
    #include <cstdlib>   // for abort
    #include <iostream>  // for std::cerr

    HYDRA_EXTERNAL_NAMESPACE_BEGIN namespace Eigen {
    namespace internal {
    // trivial function copying a bool. Must be HYDRA_EIGEN_DONT_INLINE, so we implement it after including Eigen headers.
    // see bug 89.
    namespace {
    HYDRA_EIGEN_DONT_INLINE bool copy_bool(bool b) { return b; }
    }
    inline void assert_fail(const char *condition, const char *function, const char *file, int line)
    {
      std::cerr << "assertion failed: " << condition << " in function " << function << " at " << file << ":" << line << std::endl;
      abort();
    }
    }
    } HYDRA_EXTERNAL_NAMESPACE_END
    #define eigen_plain_assert(x) \
      do { \
        if(!HYDRA_EXTERNAL_NS::Eigen::internal::copy_bool(x)) \
          HYDRA_EXTERNAL_NS::Eigen::internal::assert_fail(HYDRA_EIGEN_MAKESTRING(x), __PRETTY_FUNCTION__, __FILE__, __LINE__); \
      } while(false)
  #endif
#endif

// eigen_assert can be overridden
#ifndef eigen_assert
#define eigen_assert(x) eigen_plain_assert(x)
#endif

#ifdef HYDRA_EIGEN_INTERNAL_DEBUGGING
#define eigen_internal_assert(x) eigen_assert(x)
#else
#define eigen_internal_assert(x)
#endif

#ifdef HYDRA_EIGEN_NO_DEBUG
#define HYDRA_EIGEN_ONLY_USED_FOR_DEBUG(x) HYDRA_EIGEN_UNUSED_VARIABLE(x)
#else
#define HYDRA_EIGEN_ONLY_USED_FOR_DEBUG(x)
#endif

#ifndef HYDRA_EIGEN_NO_DEPRECATED_WARNING
  #if HYDRA_EIGEN_COMP_GNUC
    #define HYDRA_EIGEN_DEPRECATED __attribute__((deprecated))
  #elif HYDRA_EIGEN_COMP_MSVC
    #define HYDRA_EIGEN_DEPRECATED __declspec(deprecated)
  #else
    #define HYDRA_EIGEN_DEPRECATED
  #endif
#else
  #define HYDRA_EIGEN_DEPRECATED
#endif

#if HYDRA_EIGEN_COMP_GNUC
#define HYDRA_EIGEN_UNUSED __attribute__((unused))
#else
#define HYDRA_EIGEN_UNUSED
#endif

// Suppresses 'unused variable' warnings.
HYDRA_EXTERNAL_NAMESPACE_BEGIN namespace Eigen {
  namespace internal {
    template<typename T> HYDRA_EIGEN_DEVICE_FUNC void ignore_unused_variable(const T&) {}
  }
} HYDRA_EXTERNAL_NAMESPACE_END
#define HYDRA_EIGEN_UNUSED_VARIABLE(var) HYDRA_EXTERNAL_NS::Eigen::internal::ignore_unused_variable(var);

#if !defined(HYDRA_EIGEN_ASM_COMMENT)
  #if HYDRA_EIGEN_COMP_GNUC && (HYDRA_EIGEN_ARCH_i386_OR_x86_64 || HYDRA_EIGEN_ARCH_ARM_OR_ARM64)
    #define HYDRA_EIGEN_ASM_COMMENT(X)  __asm__("#" X)
  #else
    #define HYDRA_EIGEN_ASM_COMMENT(X)
  #endif
#endif


//------------------------------------------------------------------------------------------
// Static and dynamic alignment control
//
// The main purpose of this section is to define HYDRA_EIGEN_MAX_ALIGN_BYTES and HYDRA_EIGEN_MAX_STATIC_ALIGN_BYTES
// as the maximal boundary in bytes on which dynamically and statically allocated data may be alignment respectively.
// The values of HYDRA_EIGEN_MAX_ALIGN_BYTES and HYDRA_EIGEN_MAX_STATIC_ALIGN_BYTES can be specified by the user. If not,
// a default value is automatically computed based on architecture, compiler, and OS.
//
// This section also defines macros HYDRA_EIGEN_ALIGN_TO_BOUNDARY(N) and the shortcuts HYDRA_EIGEN_ALIGN{8,16,32,_MAX}
// to be used to declare statically aligned buffers.
//------------------------------------------------------------------------------------------


/* HYDRA_EIGEN_ALIGN_TO_BOUNDARY(n) forces data to be n-byte aligned. This is used to satisfy SIMD requirements.
 * However, we do that EVEN if vectorization (HYDRA_EIGEN_VECTORIZE) is disabled,
 * so that vectorization doesn't affect binary compatibility.
 *
 * If we made alignment depend on whether or not HYDRA_EIGEN_VECTORIZE is defined, it would be impossible to link
 * vectorized and non-vectorized code.
 */
#if (defined __CUDACC__)
  #define HYDRA_EIGEN_ALIGN_TO_BOUNDARY(n) __align__(n)
#elif HYDRA_EIGEN_COMP_GNUC || HYDRA_EIGEN_COMP_PGI || HYDRA_EIGEN_COMP_IBM || HYDRA_EIGEN_COMP_ARM
  #define HYDRA_EIGEN_ALIGN_TO_BOUNDARY(n) __attribute__((aligned(n)))
#elif HYDRA_EIGEN_COMP_MSVC
  #define HYDRA_EIGEN_ALIGN_TO_BOUNDARY(n) __declspec(align(n))
#elif HYDRA_EIGEN_COMP_SUNCC
  // FIXME not sure about this one:
  #define HYDRA_EIGEN_ALIGN_TO_BOUNDARY(n) __attribute__((aligned(n)))
#else
  #error Please tell me what is the equivalent of __attribute__((aligned(n))) for your compiler
#endif

// If the user explicitly disable vectorization, then we also disable alignment
#if defined(HYDRA_EIGEN_DONT_VECTORIZE)
  #define HYDRA_EIGEN_IDEAL_MAX_ALIGN_BYTES 0
#elif defined(HYDRA_EIGEN_VECTORIZE_AVX512)
  // 64 bytes static alignmeent is preferred only if really required
  #define HYDRA_EIGEN_IDEAL_MAX_ALIGN_BYTES 64
#elif defined(__AVX__)
  // 32 bytes static alignmeent is preferred only if really required
  #define HYDRA_EIGEN_IDEAL_MAX_ALIGN_BYTES 32
#else
  #define HYDRA_EIGEN_IDEAL_MAX_ALIGN_BYTES 16
#endif


// HYDRA_EIGEN_MIN_ALIGN_BYTES defines the minimal value for which the notion of explicit alignment makes sense
#define HYDRA_EIGEN_MIN_ALIGN_BYTES 16

// Defined the boundary (in bytes) on which the data needs to be aligned. Note
// that unless HYDRA_EIGEN_ALIGN is defined and not equal to 0, the data may not be
// aligned at all regardless of the value of this #define.

#if (defined(HYDRA_EIGEN_DONT_ALIGN_STATICALLY) || defined(HYDRA_EIGEN_DONT_ALIGN))  && defined(HYDRA_EIGEN_MAX_STATIC_ALIGN_BYTES) && HYDRA_EIGEN_MAX_STATIC_ALIGN_BYTES>0
#error HYDRA_EIGEN_MAX_STATIC_ALIGN_BYTES and HYDRA_EIGEN_DONT_ALIGN[_STATICALLY] are both defined with HYDRA_EIGEN_MAX_STATIC_ALIGN_BYTES!=0. Use HYDRA_EIGEN_MAX_STATIC_ALIGN_BYTES=0 as a synonym of HYDRA_EIGEN_DONT_ALIGN_STATICALLY.
#endif

// HYDRA_EIGEN_DONT_ALIGN_STATICALLY and HYDRA_EIGEN_DONT_ALIGN are deprectated
// They imply HYDRA_EIGEN_MAX_STATIC_ALIGN_BYTES=0
#if defined(HYDRA_EIGEN_DONT_ALIGN_STATICALLY) || defined(HYDRA_EIGEN_DONT_ALIGN)
  #ifdef HYDRA_EIGEN_MAX_STATIC_ALIGN_BYTES
    #undef HYDRA_EIGEN_MAX_STATIC_ALIGN_BYTES
  #endif
  #define HYDRA_EIGEN_MAX_STATIC_ALIGN_BYTES 0
#endif

#ifndef HYDRA_EIGEN_MAX_STATIC_ALIGN_BYTES

  // Try to automatically guess what is the best default value for HYDRA_EIGEN_MAX_STATIC_ALIGN_BYTES

  // 16 byte alignment is only useful for vectorization. Since it affects the ABI, we need to enable
  // 16 byte alignment on all platforms where vectorization might be enabled. In theory we could always
  // enable alignment, but it can be a cause of problems on some platforms, so we just disable it in
  // certain common platform (compiler+architecture combinations) to avoid these problems.
  // Only static alignment is really problematic (relies on nonstandard compiler extensions),
  // try to keep heap alignment even when we have to disable static alignment.
  #if HYDRA_EIGEN_COMP_GNUC && !(HYDRA_EIGEN_ARCH_i386_OR_x86_64 || HYDRA_EIGEN_ARCH_ARM_OR_ARM64 || HYDRA_EIGEN_ARCH_PPC || HYDRA_EIGEN_ARCH_IA64)
  #define HYDRA_EIGEN_GCC_AND_ARCH_DOESNT_WANT_STACK_ALIGNMENT 1
  #elif HYDRA_EIGEN_ARCH_ARM_OR_ARM64 && HYDRA_EIGEN_COMP_GNUC_STRICT && HYDRA_EIGEN_GNUC_AT_MOST(4, 6)
  // Old versions of GCC on ARM, at least 4.4, were once seen to have buggy static alignment support.
  // Not sure which version fixed it, hopefully it doesn't affect 4.7, which is still somewhat in use.
  // 4.8 and newer seem definitely unaffected.
  #define HYDRA_EIGEN_GCC_AND_ARCH_DOESNT_WANT_STACK_ALIGNMENT 1
  #else
  #define HYDRA_EIGEN_GCC_AND_ARCH_DOESNT_WANT_STACK_ALIGNMENT 0
  #endif

  // static alignment is completely disabled with GCC 3, Sun Studio, and QCC/QNX
  #if !HYDRA_EIGEN_GCC_AND_ARCH_DOESNT_WANT_STACK_ALIGNMENT \
  && !HYDRA_EIGEN_GCC3_OR_OLDER \
  && !HYDRA_EIGEN_COMP_SUNCC \
  && !HYDRA_EIGEN_OS_QNX
    #define HYDRA_EIGEN_ARCH_WANTS_STACK_ALIGNMENT 1
  #else
    #define HYDRA_EIGEN_ARCH_WANTS_STACK_ALIGNMENT 0
  #endif

  #if HYDRA_EIGEN_ARCH_WANTS_STACK_ALIGNMENT
    #define HYDRA_EIGEN_MAX_STATIC_ALIGN_BYTES HYDRA_EIGEN_IDEAL_MAX_ALIGN_BYTES
  #else
    #define HYDRA_EIGEN_MAX_STATIC_ALIGN_BYTES 0
  #endif

#endif

// If HYDRA_EIGEN_MAX_ALIGN_BYTES is defined, then it is considered as an upper bound for HYDRA_EIGEN_MAX_ALIGN_BYTES
#if defined(HYDRA_EIGEN_MAX_ALIGN_BYTES) && HYDRA_EIGEN_MAX_ALIGN_BYTES<HYDRA_EIGEN_MAX_STATIC_ALIGN_BYTES
#undef HYDRA_EIGEN_MAX_STATIC_ALIGN_BYTES
#define HYDRA_EIGEN_MAX_STATIC_ALIGN_BYTES HYDRA_EIGEN_MAX_ALIGN_BYTES
#endif

#if HYDRA_EIGEN_MAX_STATIC_ALIGN_BYTES==0 && !defined(HYDRA_EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT)
  #define HYDRA_EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT
#endif

// At this stage, HYDRA_EIGEN_MAX_STATIC_ALIGN_BYTES>0 is the true test whether we want to align arrays on the stack or not.
// It takes into account both the user choice to explicitly enable/disable alignment (by settting HYDRA_EIGEN_MAX_STATIC_ALIGN_BYTES)
// and the architecture config (HYDRA_EIGEN_ARCH_WANTS_STACK_ALIGNMENT).
// Henceforth, only HYDRA_EIGEN_MAX_STATIC_ALIGN_BYTES should be used.


// Shortcuts to HYDRA_EIGEN_ALIGN_TO_BOUNDARY
#define HYDRA_EIGEN_ALIGN8  HYDRA_EIGEN_ALIGN_TO_BOUNDARY(8)
#define HYDRA_EIGEN_ALIGN16 HYDRA_EIGEN_ALIGN_TO_BOUNDARY(16)
#define HYDRA_EIGEN_ALIGN32 HYDRA_EIGEN_ALIGN_TO_BOUNDARY(32)
#define HYDRA_EIGEN_ALIGN64 HYDRA_EIGEN_ALIGN_TO_BOUNDARY(64)
#if HYDRA_EIGEN_MAX_STATIC_ALIGN_BYTES>0
#define HYDRA_EIGEN_ALIGN_MAX HYDRA_EIGEN_ALIGN_TO_BOUNDARY(HYDRA_EIGEN_MAX_STATIC_ALIGN_BYTES)
#else
#define HYDRA_EIGEN_ALIGN_MAX
#endif


// Dynamic alignment control

#if defined(HYDRA_EIGEN_DONT_ALIGN) && defined(HYDRA_EIGEN_MAX_ALIGN_BYTES) && HYDRA_EIGEN_MAX_ALIGN_BYTES>0
#error HYDRA_EIGEN_MAX_ALIGN_BYTES and HYDRA_EIGEN_DONT_ALIGN are both defined with HYDRA_EIGEN_MAX_ALIGN_BYTES!=0. Use HYDRA_EIGEN_MAX_ALIGN_BYTES=0 as a synonym of HYDRA_EIGEN_DONT_ALIGN.
#endif

#ifdef HYDRA_EIGEN_DONT_ALIGN
  #ifdef HYDRA_EIGEN_MAX_ALIGN_BYTES
    #undef HYDRA_EIGEN_MAX_ALIGN_BYTES
  #endif
  #define HYDRA_EIGEN_MAX_ALIGN_BYTES 0
#elif !defined(HYDRA_EIGEN_MAX_ALIGN_BYTES)
  #define HYDRA_EIGEN_MAX_ALIGN_BYTES HYDRA_EIGEN_IDEAL_MAX_ALIGN_BYTES
#endif

#if HYDRA_EIGEN_IDEAL_MAX_ALIGN_BYTES > HYDRA_EIGEN_MAX_ALIGN_BYTES
#define HYDRA_EIGEN_DEFAULT_ALIGN_BYTES HYDRA_EIGEN_IDEAL_MAX_ALIGN_BYTES
#else
#define HYDRA_EIGEN_DEFAULT_ALIGN_BYTES HYDRA_EIGEN_MAX_ALIGN_BYTES
#endif


#ifndef HYDRA_EIGEN_UNALIGNED_VECTORIZE
#define HYDRA_EIGEN_UNALIGNED_VECTORIZE 1
#endif

//----------------------------------------------------------------------


#ifdef HYDRA_EIGEN_DONT_USE_RESTRICT_KEYWORD
  #define HYDRA_EIGEN_RESTRICT
#endif
#ifndef HYDRA_EIGEN_RESTRICT
  #define HYDRA_EIGEN_RESTRICT __restrict
#endif

#ifndef HYDRA_EIGEN_STACK_ALLOCATION_LIMIT
// 131072 == 128 KB
#define HYDRA_EIGEN_STACK_ALLOCATION_LIMIT 131072
#endif

#ifndef HYDRA_EIGEN_DEFAULT_IO_FORMAT
#ifdef HYDRA_EIGEN_MAKING_DOCS
// format used in Eigen's documentation
// needed to define it here as escaping characters in CMake add_definition's argument seems very problematic.
#define HYDRA_EIGEN_DEFAULT_IO_FORMAT HYDRA_EXTERNAL_NS::Eigen::IOFormat(3, 0, " ", "\n", "", "")
#else
#define HYDRA_EIGEN_DEFAULT_IO_FORMAT HYDRA_EXTERNAL_NS::Eigen::IOFormat()
#endif
#endif

// just an empty macro !
#define HYDRA_EIGEN_EMPTY

#if HYDRA_EIGEN_COMP_MSVC_STRICT && (HYDRA_EIGEN_COMP_MSVC < 1900 ||  defined(__CUDACC_VER__)) // for older MSVC versions, as well as 1900 && CUDA 8, using the base operator is sufficient (cf Bugs 1000, 1324)
  #define HYDRA_EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Derived) \
    using Base::operator =;
#elif HYDRA_EIGEN_COMP_CLANG // workaround clang bug (see http://forum.kde.org/viewtopic.php?f=74&t=102653)
  #define HYDRA_EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Derived) \
    using Base::operator =; \
    HYDRA_EIGEN_DEVICE_FUNC HYDRA_EIGEN_STRONG_INLINE Derived& operator=(const Derived& other) { Base::operator=(other); return *this; } \
    template <typename OtherDerived> \
    HYDRA_EIGEN_DEVICE_FUNC HYDRA_EIGEN_STRONG_INLINE Derived& operator=(const DenseBase<OtherDerived>& other) { Base::operator=(other.derived()); return *this; }
#else
  #define HYDRA_EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Derived) \
    using Base::operator =; \
    HYDRA_EIGEN_DEVICE_FUNC HYDRA_EIGEN_STRONG_INLINE Derived& operator=(const Derived& other) \
    { \
      Base::operator=(other); \
      return *this; \
    }
#endif


/** \internal
 * \brief Macro to manually inherit assignment operators.
 * This is necessary, because the implicitly defined assignment operator gets deleted when a custom operator= is defined.
 */
#define HYDRA_EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Derived) HYDRA_EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Derived)

/**
* Just a side note. Commenting within defines works only by documenting
* behind the object (via '!<'). Comments cannot be multi-line and thus
* we have these extra long lines. What is confusing doxygen over here is
* that we use '\' and basically have a bunch of typedefs with their
* documentation in a single line.
**/

#define HYDRA_EIGEN_GENERIC_PUBLIC_INTERFACE(Derived) \
  typedef typename HYDRA_EXTERNAL_NS::Eigen::internal::traits<Derived>::Scalar Scalar; /*!< \brief Numeric type, e.g. float, double, int or std::complex<float>. */ \
  typedef typename HYDRA_EXTERNAL_NS::Eigen::NumTraits<Scalar>::Real RealScalar; /*!< \brief The underlying numeric type for composed scalar types. \details In cases where Scalar is e.g. std::complex<T>, T were corresponding to RealScalar. */ \
  typedef typename Base::CoeffReturnType CoeffReturnType; /*!< \brief The return type for coefficient access. \details Depending on whether the object allows direct coefficient access (e.g. for a MatrixXd), this type is either 'const Scalar&' or simply 'Scalar' for objects that do not allow direct coefficient access. */ \
  typedef typename HYDRA_EXTERNAL_NS::Eigen::internal::ref_selector<Derived>::type Nested; \
  typedef typename HYDRA_EXTERNAL_NS::Eigen::internal::traits<Derived>::StorageKind StorageKind; \
  typedef typename HYDRA_EXTERNAL_NS::Eigen::internal::traits<Derived>::StorageIndex StorageIndex; \
  enum { RowsAtCompileTime = HYDRA_EXTERNAL_NS::Eigen::internal::traits<Derived>::RowsAtCompileTime, \
        ColsAtCompileTime = HYDRA_EXTERNAL_NS::Eigen::internal::traits<Derived>::ColsAtCompileTime, \
        Flags = HYDRA_EXTERNAL_NS::Eigen::internal::traits<Derived>::Flags, \
        SizeAtCompileTime = Base::SizeAtCompileTime, \
        MaxSizeAtCompileTime = Base::MaxSizeAtCompileTime, \
        IsVectorAtCompileTime = Base::IsVectorAtCompileTime }; \
  using Base::derived; \
  using Base::const_cast_derived;


// FIXME Maybe the HYDRA_EIGEN_DENSE_PUBLIC_INTERFACE could be removed as importing PacketScalar is rarely needed
#define HYDRA_EIGEN_DENSE_PUBLIC_INTERFACE(Derived) \
  HYDRA_EIGEN_GENERIC_PUBLIC_INTERFACE(Derived) \
  typedef typename Base::PacketScalar PacketScalar;


#define HYDRA_EIGEN_PLAIN_ENUM_MIN(a,b) (((int)a <= (int)b) ? (int)a : (int)b)
#define HYDRA_EIGEN_PLAIN_ENUM_MAX(a,b) (((int)a >= (int)b) ? (int)a : (int)b)

// HYDRA_EIGEN_SIZE_MIN_PREFER_DYNAMIC gives the min between compile-time sizes. 0 has absolute priority, followed by 1,
// followed by Dynamic, followed by other finite values. The reason for giving Dynamic the priority over
// finite values is that min(3, Dynamic) should be Dynamic, since that could be anything between 0 and 3.
#define HYDRA_EIGEN_SIZE_MIN_PREFER_DYNAMIC(a,b) (((int)a == 0 || (int)b == 0) ? 0 \
                           : ((int)a == 1 || (int)b == 1) ? 1 \
                           : ((int)a == Dynamic || (int)b == Dynamic) ? Dynamic \
                           : ((int)a <= (int)b) ? (int)a : (int)b)

// HYDRA_EIGEN_SIZE_MIN_PREFER_FIXED is a variant of HYDRA_EIGEN_SIZE_MIN_PREFER_DYNAMIC comparing MaxSizes. The difference is that finite values
// now have priority over Dynamic, so that min(3, Dynamic) gives 3. Indeed, whatever the actual value is
// (between 0 and 3), it is not more than 3.
#define HYDRA_EIGEN_SIZE_MIN_PREFER_FIXED(a,b)  (((int)a == 0 || (int)b == 0) ? 0 \
                           : ((int)a == 1 || (int)b == 1) ? 1 \
                           : ((int)a == Dynamic && (int)b == Dynamic) ? Dynamic \
                           : ((int)a == Dynamic) ? (int)b \
                           : ((int)b == Dynamic) ? (int)a \
                           : ((int)a <= (int)b) ? (int)a : (int)b)

// see HYDRA_EIGEN_SIZE_MIN_PREFER_DYNAMIC. No need for a separate variant for MaxSizes here.
#define HYDRA_EIGEN_SIZE_MAX(a,b) (((int)a == Dynamic || (int)b == Dynamic) ? Dynamic \
                           : ((int)a >= (int)b) ? (int)a : (int)b)

#define HYDRA_EIGEN_LOGICAL_XOR(a,b) (((a) || (b)) && !((a) && (b)))

#define HYDRA_EIGEN_IMPLIES(a,b) (!(a) || (b))

// the expression type of a standard coefficient wise binary operation
#define HYDRA_EIGEN_CWISE_BINARY_RETURN_TYPE(LHS,RHS,OPNAME) \
    CwiseBinaryOp< \
      HYDRA_EIGEN_CAT(HYDRA_EIGEN_CAT(internal::scalar_,OPNAME),_op)< \
          typename internal::traits<LHS>::Scalar, \
          typename internal::traits<RHS>::Scalar \
      >, \
      const LHS, \
      const RHS \
    >

#define HYDRA_EIGEN_MAKE_CWISE_BINARY_OP(METHOD,OPNAME) \
  template<typename OtherDerived> \
  HYDRA_EIGEN_DEVICE_FUNC HYDRA_EIGEN_STRONG_INLINE const HYDRA_EIGEN_CWISE_BINARY_RETURN_TYPE(Derived,OtherDerived,OPNAME) \
  (METHOD)(const HYDRA_EIGEN_CURRENT_STORAGE_BASE_CLASS<OtherDerived> &other) const \
  { \
    return HYDRA_EIGEN_CWISE_BINARY_RETURN_TYPE(Derived,OtherDerived,OPNAME)(derived(), other.derived()); \
  }

#define HYDRA_EIGEN_SCALAR_BINARY_SUPPORTED(OPNAME,TYPEA,TYPEB) \
  (HYDRA_EXTERNAL_NS::Eigen::internal::has_ReturnType<HYDRA_EXTERNAL_NS::Eigen::ScalarBinaryOpTraits<TYPEA,TYPEB,HYDRA_EIGEN_CAT(HYDRA_EIGEN_CAT(HYDRA_EXTERNAL_NS::Eigen::internal::scalar_,OPNAME),_op)<TYPEA,TYPEB>  > >::value)

#define HYDRA_EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE(EXPR,SCALAR,OPNAME) \
  CwiseBinaryOp<HYDRA_EIGEN_CAT(HYDRA_EIGEN_CAT(internal::scalar_,OPNAME),_op)<typename internal::traits<EXPR>::Scalar,SCALAR>, const EXPR, \
                const typename internal::plain_constant_type<EXPR,SCALAR>::type>

#define HYDRA_EIGEN_SCALAR_BINARYOP_EXPR_RETURN_TYPE(SCALAR,EXPR,OPNAME) \
  CwiseBinaryOp<HYDRA_EIGEN_CAT(HYDRA_EIGEN_CAT(internal::scalar_,OPNAME),_op)<SCALAR,typename internal::traits<EXPR>::Scalar>, \
                const typename internal::plain_constant_type<EXPR,SCALAR>::type, const EXPR>

// Workaround for MSVC 2010 (see ML thread "patch with compile for for MSVC 2010")
#if HYDRA_EIGEN_COMP_MSVC_STRICT<=1600
#define HYDRA_EIGEN_MSVC10_WORKAROUND_BINARYOP_RETURN_TYPE(X) typename internal::enable_if<true,X>::type
#else
#define HYDRA_EIGEN_MSVC10_WORKAROUND_BINARYOP_RETURN_TYPE(X) X
#endif

#define HYDRA_EIGEN_MAKE_SCALAR_BINARY_OP_ONTHERIGHT(METHOD,OPNAME) \
  template <typename T> HYDRA_EIGEN_DEVICE_FUNC inline \
  HYDRA_EIGEN_MSVC10_WORKAROUND_BINARYOP_RETURN_TYPE(const HYDRA_EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE(Derived,typename internal::promote_scalar_arg<Scalar HYDRA_EIGEN_COMMA T HYDRA_EIGEN_COMMA HYDRA_EIGEN_SCALAR_BINARY_SUPPORTED(OPNAME,Scalar,T)>::type,OPNAME))\
  (METHOD)(const T& scalar) const { \
    typedef typename internal::promote_scalar_arg<Scalar,T,HYDRA_EIGEN_SCALAR_BINARY_SUPPORTED(OPNAME,Scalar,T)>::type PromotedT; \
    return HYDRA_EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE(Derived,PromotedT,OPNAME)(derived(), \
           typename internal::plain_constant_type<Derived,PromotedT>::type(derived().rows(), derived().cols(), internal::scalar_constant_op<PromotedT>(scalar))); \
  }

#define HYDRA_EIGEN_MAKE_SCALAR_BINARY_OP_ONTHELEFT(METHOD,OPNAME) \
  template <typename T> HYDRA_EIGEN_DEVICE_FUNC inline friend \
  HYDRA_EIGEN_MSVC10_WORKAROUND_BINARYOP_RETURN_TYPE(const HYDRA_EIGEN_SCALAR_BINARYOP_EXPR_RETURN_TYPE(typename internal::promote_scalar_arg<Scalar HYDRA_EIGEN_COMMA T HYDRA_EIGEN_COMMA HYDRA_EIGEN_SCALAR_BINARY_SUPPORTED(OPNAME,T,Scalar)>::type,Derived,OPNAME)) \
  (METHOD)(const T& scalar, const StorageBaseType& matrix) { \
    typedef typename internal::promote_scalar_arg<Scalar,T,HYDRA_EIGEN_SCALAR_BINARY_SUPPORTED(OPNAME,T,Scalar)>::type PromotedT; \
    return HYDRA_EIGEN_SCALAR_BINARYOP_EXPR_RETURN_TYPE(PromotedT,Derived,OPNAME)( \
           typename internal::plain_constant_type<Derived,PromotedT>::type(matrix.derived().rows(), matrix.derived().cols(), internal::scalar_constant_op<PromotedT>(scalar)), matrix.derived()); \
  }

#define HYDRA_EIGEN_MAKE_SCALAR_BINARY_OP(METHOD,OPNAME) \
  HYDRA_EIGEN_MAKE_SCALAR_BINARY_OP_ONTHELEFT(METHOD,OPNAME) \
  HYDRA_EIGEN_MAKE_SCALAR_BINARY_OP_ONTHERIGHT(METHOD,OPNAME)


#ifdef HYDRA_EIGEN_EXCEPTIONS
#  define HYDRA_EIGEN_THROW_X(X) throw X
#  define HYDRA_EIGEN_THROW throw
#  define HYDRA_EIGEN_TRY try
#  define HYDRA_EIGEN_CATCH(X) catch (X)
#else
#  ifdef __CUDA_ARCH__
#    define HYDRA_EIGEN_THROW_X(X) asm("trap;")
#    define HYDRA_EIGEN_THROW asm("trap;")
#  else
#    define HYDRA_EIGEN_THROW_X(X) std::abort()
#    define HYDRA_EIGEN_THROW std::abort()
#  endif
#  define HYDRA_EIGEN_TRY if (true)
#  define HYDRA_EIGEN_CATCH(X) else
#endif


#if HYDRA_EIGEN_HAS_CXX11_NOEXCEPT
#   define HYDRA_EIGEN_INCLUDE_TYPE_TRAITS
#   define HYDRA_EIGEN_NOEXCEPT noexcept
#   define HYDRA_EIGEN_NOEXCEPT_IF(x) noexcept(x)
#   define HYDRA_EIGEN_NO_THROW noexcept(true)
#   define HYDRA_EIGEN_EXCEPTION_SPEC(X) noexcept(false)
#else
#   define HYDRA_EIGEN_NOEXCEPT
#   define HYDRA_EIGEN_NOEXCEPT_IF(x)
#   define HYDRA_EIGEN_NO_THROW throw()
#   define HYDRA_EIGEN_EXCEPTION_SPEC(X) throw(X)
#endif

#endif // HYDRA_EIGEN_MACROS_H
