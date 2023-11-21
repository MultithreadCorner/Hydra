//  Copyright (c) 2006-7 John Maddock
//  Copyright (c) 2021 Matt Borland
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_MATH_TOOLS_CONFIG_HPP
#define HYDRA_BOOST_MATH_TOOLS_CONFIG_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <hydra/detail/external/hydra_boost/math/tools/is_standalone.hpp>

// Minimum language standard transition
#ifdef _MSVC_LANG
#  if _MSVC_LANG < 201402L
#    pragma warning("The minimum language standard to use Boost.Math will be C++14 starting in July 2023 (Boost 1.82 release)");
#  endif
#else
#  if __cplusplus < 201402L
#    warning "The minimum language standard to use Boost.Math will be C++14 starting in July 2023 (Boost 1.82 release)"
#  endif
#endif

#ifndef HYDRA_BOOST_MATH_STANDALONE
#include <hydra/detail/external/hydra_boost/config.hpp>

#else // Things from boost/config that are required, and easy to replicate

#define HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION
#define HYDRA_BOOST_MATH_NO_REAL_CONCEPT_TESTS
#define HYDRA_BOOST_MATH_NO_DISTRIBUTION_CONCEPT_TESTS
#define HYDRA_BOOST_MATH_NO_LEXICAL_CAST

// Since Boost.Multiprecision is in active development some tests do not fully cooperate yet.
#define HYDRA_BOOST_MATH_NO_MP_TESTS

#if (__cplusplus > 201400L || _MSVC_LANG > 201400L)
#define HYDRA_BOOST_CXX14_CONSTEXPR constexpr
#else
#define HYDRA_BOOST_CXX14_CONSTEXPR
#define HYDRA_BOOST_NO_CXX14_CONSTEXPR
#endif // HYDRA_BOOST_CXX14_CONSTEXPR

#if (__cplusplus > 201700L || _MSVC_LANG > 201700L)
#define HYDRA_BOOST_IF_CONSTEXPR if constexpr

// Clang on mac provides the execution header with none of the functionality. TODO: Check back on this
// https://en.cppreference.com/w/cpp/compiler_support "Standardization of Parallelism TS"
#if !__has_include(<execution>) || (defined(__APPLE__) && defined(__clang__))
#define HYDRA_BOOST_NO_CXX17_HDR_EXECUTION
#endif
#else
#define HYDRA_BOOST_IF_CONSTEXPR if
#define HYDRA_BOOST_NO_CXX17_IF_CONSTEXPR
#define HYDRA_BOOST_NO_CXX17_HDR_EXECUTION
#endif

#if __cpp_lib_gcd_lcm >= 201606L
#define HYDRA_BOOST_MATH_HAS_CXX17_NUMERIC
#endif

#define HYDRA_BOOST_JOIN(X, Y) HYDRA_BOOST_DO_JOIN(X, Y)
#define HYDRA_BOOST_DO_JOIN(X, Y) HYDRA_BOOST_DO_JOIN2(X,Y)
#define HYDRA_BOOST_DO_JOIN2(X, Y) X##Y

#define HYDRA_BOOST_STRINGIZE(X) HYDRA_BOOST_DO_STRINGIZE(X)
#define HYDRA_BOOST_DO_STRINGIZE(X) #X

#ifdef HYDRA_BOOST_DISABLE_THREADS // No threads, do nothing
// Detect thread support via STL implementation
#elif defined(__has_include)
#  if !__has_include(<thread>) || !__has_include(<mutex>) || !__has_include(<future>) || !__has_include(<atomic>)
#     define HYDRA_BOOST_DISABLE_THREADS
#  else
#     define HYDRA_BOOST_HAS_THREADS
#  endif 
#else
#  define HYDRA_BOOST_HAS_THREADS // The default assumption is that the machine has threads
#endif // Thread Support

#ifdef HYDRA_BOOST_DISABLE_THREADS
#  define HYDRA_BOOST_NO_CXX11_HDR_ATOMIC
#  define HYDRA_BOOST_NO_CXX11_HDR_FUTURE
#  define HYDRA_BOOST_NO_CXX11_HDR_THREAD
#  define HYDRA_BOOST_NO_CXX11_THREAD_LOCAL
#endif // HYDRA_BOOST_DISABLE_THREADS

#ifdef __GNUC__
#  if !defined(__EXCEPTIONS) && !defined(HYDRA_BOOST_NO_EXCEPTIONS)
#     define HYDRA_BOOST_NO_EXCEPTIONS
#  endif
   //
   // Make sure we have some std lib headers included so we can detect __GXX_RTTI:
   //
#  include <algorithm>  // for min and max
#  include <limits>
#  ifndef __GXX_RTTI
#     ifndef HYDRA_BOOST_NO_TYPEID
#        define HYDRA_BOOST_NO_TYPEID
#     endif
#     ifndef HYDRA_BOOST_NO_RTTI
#        define HYDRA_BOOST_NO_RTTI
#     endif
#  endif
#endif

#if !defined(HYDRA_BOOST_NOINLINE)
#  if defined(_MSC_VER)
#    define HYDRA_BOOST_NOINLINE __declspec(noinline)
#  elif defined(__GNUC__) && __GNUC__ > 3
     // Clang also defines __GNUC__ (as 4)
#    if defined(__CUDACC__)
       // nvcc doesn't always parse __noinline__,
       // see: https://svn.boost.org/trac/boost/ticket/9392
#      define HYDRA_BOOST_NOINLINE __attribute__ ((noinline))
#    elif defined(__HIP__)
       // See https://github.com/boostorg/config/issues/392
#      define HYDRA_BOOST_NOINLINE __attribute__ ((noinline))
#    else
#      define HYDRA_BOOST_NOINLINE __attribute__ ((__noinline__))
#    endif
#  else
#    define HYDRA_BOOST_NOINLINE
#  endif
#endif

#if !defined(HYDRA_BOOST_FORCEINLINE)
#  if defined(_MSC_VER)
#    define HYDRA_BOOST_FORCEINLINE __forceinline
#  elif defined(__GNUC__) && __GNUC__ > 3
     // Clang also defines __GNUC__ (as 4)
#    define HYDRA_BOOST_FORCEINLINE inline __attribute__ ((__always_inline__))
#  else
#    define HYDRA_BOOST_FORCEINLINE inline
#  endif
#endif

#endif // HYDRA_BOOST_MATH_STANDALONE

// Support compilers with P0024R2 implemented without linking TBB
// https://en.cppreference.com/w/cpp/compiler_support
#if !defined(HYDRA_BOOST_NO_CXX17_HDR_EXECUTION) && defined(HYDRA_BOOST_HAS_THREADS)
#  define HYDRA_BOOST_MATH_EXEC_COMPATIBLE
#endif

// Attributes from C++14 and newer
#ifdef __has_cpp_attribute

// C++17
#if (__cplusplus >= 201703L || _MSVC_LANG >= 201703L)
#  if __has_cpp_attribute(maybe_unused)
#    define HYDRA_BOOST_MATH_MAYBE_UNUSED [[maybe_unused]]
#  endif
#endif

#endif // Standalone config

// If attributes are not defined make sure we don't have compiler errors
#ifndef HYDRA_BOOST_MATH_MAYBE_UNUSED
#  define HYDRA_BOOST_MATH_MAYBE_UNUSED 
#endif

// C++23
#if __cplusplus > 202002L || _MSVC_LANG > 202002L
#  if __GNUC__ >= 13
     // libstdc++3 only defines to/from_chars for std::float128_t when one of these defines are set
     // otherwise we're right out of luck...
#    if defined(_GLIBCXX_LDOUBLE_IS_IEEE_BINARY128) || defined(_GLIBCXX_HAVE_FLOAT128_MATH)
#    include <cstring> // std::strlen is used with from_chars
#    include <charconv>
#    include <stdfloat>
#    define HYDRA_BOOST_MATH_USE_CHARCONV_FOR_CONVERSION
#endif
#  endif
#endif

#include <algorithm>  // for min and max
#include <limits>
#include <cmath>
#include <climits>
#include <cfloat>

#include <hydra/detail/external/hydra_boost/math/tools/user.hpp>

#if (defined(__NetBSD__) || defined(__EMSCRIPTEN__)\
   || (defined(__hppa) && !defined(__OpenBSD__)) || (defined(__NO_LONG_DOUBLE_MATH) && (DBL_MANT_DIG != LDBL_MANT_DIG))) \
   && !defined(HYDRA_BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS)
//#  define HYDRA_BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
#endif

#ifdef __IBMCPP__
//
// For reasons I don't understand, the tests with IMB's compiler all
// pass at long double precision, but fail with real_concept, those tests
// are disabled for now.  (JM 2012).
#ifndef HYDRA_BOOST_MATH_NO_REAL_CONCEPT_TESTS
#  define HYDRA_BOOST_MATH_NO_REAL_CONCEPT_TESTS
#endif // HYDRA_BOOST_MATH_NO_REAL_CONCEPT_TESTS
#endif
#ifdef sun
// Any use of __float128 in program startup code causes a segfault  (tested JM 2015, Solaris 11).
#  define HYDRA_BOOST_MATH_DISABLE_FLOAT128
#endif
#ifdef __HAIKU__
//
// Not sure what's up with the math detection on Haiku, but linking fails with
// float128 code enabled, and we don't have an implementation of __expl, so
// disabling long double functions for now as well.
#  define HYDRA_BOOST_MATH_DISABLE_FLOAT128
#  define HYDRA_BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
#endif
#if (defined(macintosh) || defined(__APPLE__) || defined(__APPLE_CC__)) && ((LDBL_MANT_DIG == 106) || (__LDBL_MANT_DIG__ == 106)) && !defined(HYDRA_BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS)
//
// Darwin's rather strange "double double" is rather hard to
// support, it should be possible given enough effort though...
//
#  define HYDRA_BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
#endif
#if !defined(HYDRA_BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS) && (LDBL_MANT_DIG == 106) && (LDBL_MIN_EXP > DBL_MIN_EXP)
//
// Generic catch all case for gcc's "double-double" long double type.
// We do not support this as it's not even remotely IEEE conforming:
//
#  define HYDRA_BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
#endif
#if defined(unix) && defined(__INTEL_COMPILER) && (__INTEL_COMPILER <= 1000) && !defined(HYDRA_BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS)
//
// Intel compiler prior to version 10 has sporadic problems
// calling the long double overloads of the std lib math functions:
// calling ::powl is OK, but std::pow(long double, long double) 
// may segfault depending upon the value of the arguments passed 
// and the specific Linux distribution.
//
// We'll be conservative and disable long double support for this compiler.
//
// Comment out this #define and try building the tests to determine whether
// your Intel compiler version has this issue or not.
//
#  define HYDRA_BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
#endif
#if defined(unix) && defined(__INTEL_COMPILER)
//
// Intel compiler has sporadic issues compiling std::fpclassify depending on
// the exact OS version used.  Use our own code for this as we know it works
// well on Intel processors:
//
#define HYDRA_BOOST_MATH_DISABLE_STD_FPCLASSIFY
#endif

#if defined(_MSC_VER) && !defined(_WIN32_WCE)
   // Better safe than sorry, our tests don't support hardware exceptions:
#  define HYDRA_BOOST_MATH_CONTROL_FP _control87(MCW_EM,MCW_EM)
#endif

#ifdef __IBMCPP__
#  define HYDRA_BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS
#endif

#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901))
#  define HYDRA_BOOST_MATH_USE_C99
#endif

#if (defined(__hpux) && !defined(__hppa))
#  define HYDRA_BOOST_MATH_USE_C99
#endif

#if defined(__GNUC__) && defined(_GLIBCXX_USE_C99)
#  define HYDRA_BOOST_MATH_USE_C99
#endif

#if defined(_LIBCPP_VERSION) && !defined(_MSC_VER)
#  define HYDRA_BOOST_MATH_USE_C99
#endif

#if defined(__CYGWIN__) || defined(__HP_aCC) || defined(__INTEL_COMPILER) \
  || defined(HYDRA_BOOST_NO_NATIVE_LONG_DOUBLE_FP_CLASSIFY) \
  || (defined(__GNUC__) && !defined(HYDRA_BOOST_MATH_USE_C99))\
  || defined(HYDRA_BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS)
#  define HYDRA_BOOST_MATH_NO_NATIVE_LONG_DOUBLE_FP_CLASSIFY
#endif

#if defined(__SUNPRO_CC) && (__SUNPRO_CC <= 0x590)

namespace hydra_boost { namespace math { namespace tools { namespace detail {
template <typename T>
struct type {};

template <typename T, T n>
struct non_type {};
}}}} // Namespace boost, math tools, detail

#  define HYDRA_BOOST_MATH_EXPLICIT_TEMPLATE_TYPE(t)              hydra_boost::math::tools::detail::type<t>* = 0
#  define HYDRA_BOOST_MATH_EXPLICIT_TEMPLATE_TYPE_SPEC(t)         hydra_boost::math::tools::detail::type<t>*
#  define HYDRA_BOOST_MATH_EXPLICIT_TEMPLATE_NON_TYPE(t, v)       hydra_boost::math::tools::detail::non_type<t, v>* = 0
#  define HYDRA_BOOST_MATH_EXPLICIT_TEMPLATE_NON_TYPE_SPEC(t, v)  hydra_boost::math::tools::detail::non_type<t, v>*

#  define HYDRA_BOOST_MATH_APPEND_EXPLICIT_TEMPLATE_TYPE(t)         \
             , HYDRA_BOOST_MATH_EXPLICIT_TEMPLATE_TYPE(t)
#  define HYDRA_BOOST_MATH_APPEND_EXPLICIT_TEMPLATE_TYPE_SPEC(t)    \
             , HYDRA_BOOST_MATH_EXPLICIT_TEMPLATE_TYPE_SPEC(t)
#  define HYDRA_BOOST_MATH_APPEND_EXPLICIT_TEMPLATE_NON_TYPE(t, v)  \
             , HYDRA_BOOST_MATH_EXPLICIT_TEMPLATE_NON_TYPE(t, v)
#  define HYDRA_BOOST_MATH_APPEND_EXPLICIT_TEMPLATE_NON_TYPE_SPEC(t, v)  \
             , HYDRA_BOOST_MATH_EXPLICIT_TEMPLATE_NON_TYPE_SPEC(t, v)

#else

// no workaround needed: expand to nothing

#  define HYDRA_BOOST_MATH_EXPLICIT_TEMPLATE_TYPE(t)
#  define HYDRA_BOOST_MATH_EXPLICIT_TEMPLATE_TYPE_SPEC(t)
#  define HYDRA_BOOST_MATH_EXPLICIT_TEMPLATE_NON_TYPE(t, v)
#  define HYDRA_BOOST_MATH_EXPLICIT_TEMPLATE_NON_TYPE_SPEC(t, v)

#  define HYDRA_BOOST_MATH_APPEND_EXPLICIT_TEMPLATE_TYPE(t)
#  define HYDRA_BOOST_MATH_APPEND_EXPLICIT_TEMPLATE_TYPE_SPEC(t)
#  define HYDRA_BOOST_MATH_APPEND_EXPLICIT_TEMPLATE_NON_TYPE(t, v)
#  define HYDRA_BOOST_MATH_APPEND_EXPLICIT_TEMPLATE_NON_TYPE_SPEC(t, v)


#endif // __SUNPRO_CC

#if (defined(__SUNPRO_CC) || defined(__hppa) || defined(__GNUC__)) && !defined(HYDRA_BOOST_MATH_SMALL_CONSTANT)
// Sun's compiler emits a hard error if a constant underflows,
// as does aCC on PA-RISC, while gcc issues a large number of warnings:
#  define HYDRA_BOOST_MATH_SMALL_CONSTANT(x) 0.0
#else
#  define HYDRA_BOOST_MATH_SMALL_CONSTANT(x) x
#endif

//
// Tune performance options for specific compilers:
//
#ifdef _MSC_VER
#  define HYDRA_BOOST_MATH_POLY_METHOD 2
#if _MSC_VER <= 1900
#  define HYDRA_BOOST_MATH_RATIONAL_METHOD 1
#else
#  define HYDRA_BOOST_MATH_RATIONAL_METHOD 2
#endif
#if _MSC_VER > 1900
#  define HYDRA_BOOST_MATH_INT_TABLE_TYPE(RT, IT) RT
#  define HYDRA_BOOST_MATH_INT_VALUE_SUFFIX(RV, SUF) RV##.0L
#endif

#elif defined(__INTEL_COMPILER)
#  define HYDRA_BOOST_MATH_POLY_METHOD 2
#  define HYDRA_BOOST_MATH_RATIONAL_METHOD 1

#elif defined(__GNUC__)
#if __GNUC__ < 4
#  define HYDRA_BOOST_MATH_POLY_METHOD 3
#  define HYDRA_BOOST_MATH_RATIONAL_METHOD 3
#  define HYDRA_BOOST_MATH_INT_TABLE_TYPE(RT, IT) RT
#  define HYDRA_BOOST_MATH_INT_VALUE_SUFFIX(RV, SUF) RV##.0L
#else
#  define HYDRA_BOOST_MATH_POLY_METHOD 3
#  define HYDRA_BOOST_MATH_RATIONAL_METHOD 3
#endif

#elif defined(__clang__)

#if __clang__ > 6
#  define HYDRA_BOOST_MATH_POLY_METHOD 3
#  define HYDRA_BOOST_MATH_RATIONAL_METHOD 3
#  define HYDRA_BOOST_MATH_INT_TABLE_TYPE(RT, IT) RT
#  define HYDRA_BOOST_MATH_INT_VALUE_SUFFIX(RV, SUF) RV##.0L
#endif

#endif

//
// noexcept support:
//
#include <type_traits>
#define HYDRA_BOOST_MATH_NOEXCEPT(T) noexcept(std::is_floating_point<T>::value)
#define HYDRA_BOOST_MATH_IS_FLOAT(T) (std::is_floating_point<T>::value)

//
// The maximum order of polynomial that will be evaluated 
// via an unrolled specialisation:
//
#ifndef HYDRA_BOOST_MATH_MAX_POLY_ORDER
#  define HYDRA_BOOST_MATH_MAX_POLY_ORDER 20
#endif 
//
// Set the method used to evaluate polynomials and rationals:
//
#ifndef HYDRA_BOOST_MATH_POLY_METHOD
#  define HYDRA_BOOST_MATH_POLY_METHOD 2
#endif 
#ifndef HYDRA_BOOST_MATH_RATIONAL_METHOD
#  define HYDRA_BOOST_MATH_RATIONAL_METHOD 1
#endif 
//
// decide whether to store constants as integers or reals:
//
#ifndef HYDRA_BOOST_MATH_INT_TABLE_TYPE
#  define HYDRA_BOOST_MATH_INT_TABLE_TYPE(RT, IT) IT
#endif
#ifndef HYDRA_BOOST_MATH_INT_VALUE_SUFFIX
#  define HYDRA_BOOST_MATH_INT_VALUE_SUFFIX(RV, SUF) RV##SUF
#endif
//
// And then the actual configuration:
//
#if defined(HYDRA_BOOST_MATH_STANDALONE) && defined(_GLIBCXX_USE_FLOAT128) && defined(__GNUC__) && defined(__GNUC_MINOR__) && defined(__GNUC_PATCHLEVEL__) && !defined(__STRICT_ANSI__) \
   && !defined(HYDRA_BOOST_MATH_DISABLE_FLOAT128) && !defined(HYDRA_BOOST_MATH_USE_FLOAT128)
#  define HYDRA_BOOST_MATH_USE_FLOAT128
#elif defined(HYDRA_BOOST_HAS_FLOAT128) && !defined(HYDRA_BOOST_MATH_USE_FLOAT128)
#  define HYDRA_BOOST_MATH_USE_FLOAT128
#endif
#ifdef HYDRA_BOOST_MATH_USE_FLOAT128
//
// Only enable this when the compiler really is GCC as clang and probably 
// intel too don't support __float128 yet :-(
//
#  if defined(__INTEL_COMPILER) && defined(__GNUC__)
#    if (__GNUC__ > 4) || ((__GNUC__ == 4) && (__GNUC_MINOR__ >= 6))
#      define HYDRA_BOOST_MATH_FLOAT128_TYPE __float128
#    endif
#  elif defined(__GNUC__)
#      define HYDRA_BOOST_MATH_FLOAT128_TYPE __float128
#  endif

#  ifndef HYDRA_BOOST_MATH_FLOAT128_TYPE
#      define HYDRA_BOOST_MATH_FLOAT128_TYPE _Quad
#  endif
#endif
//
// Check for WinCE with no iostream support:
//
#if defined(_WIN32_WCE) && !defined(__SGI_STL_PORT)
#  define HYDRA_BOOST_MATH_NO_LEXICAL_CAST
#endif

//
// Helper macro for controlling the FP behaviour:
//
#ifndef HYDRA_BOOST_MATH_CONTROL_FP
#  define HYDRA_BOOST_MATH_CONTROL_FP
#endif
//
// Helper macro for using statements:
//
#define HYDRA_BOOST_MATH_STD_USING_CORE \
   using std::abs;\
   using std::acos;\
   using std::cos;\
   using std::fmod;\
   using std::modf;\
   using std::tan;\
   using std::asin;\
   using std::cosh;\
   using std::frexp;\
   using std::pow;\
   using std::tanh;\
   using std::atan;\
   using std::exp;\
   using std::ldexp;\
   using std::sin;\
   using std::atan2;\
   using std::fabs;\
   using std::log;\
   using std::sinh;\
   using std::ceil;\
   using std::floor;\
   using std::log10;\
   using std::sqrt;

#define HYDRA_BOOST_MATH_STD_USING HYDRA_BOOST_MATH_STD_USING_CORE

namespace hydra_boost{ namespace math{
namespace tools
{

template <class T>
inline T max HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T a, T b, T c) HYDRA_BOOST_MATH_NOEXCEPT(T)
{
   return (std::max)((std::max)(a, b), c);
}

template <class T>
inline T max HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T a, T b, T c, T d) HYDRA_BOOST_MATH_NOEXCEPT(T)
{
   return (std::max)((std::max)(a, b), (std::max)(c, d));
}

} // namespace tools

template <class T>
void suppress_unused_variable_warning(const T&) HYDRA_BOOST_MATH_NOEXCEPT(T)
{
}

namespace detail{

template <class T>
struct is_integer_for_rounding
{
   static constexpr bool value = std::is_integral<T>::value || (std::numeric_limits<T>::is_specialized && std::numeric_limits<T>::is_integer);
};

}

}} // namespace hydra_boost namespace math

#ifdef __GLIBC_PREREQ
#  if __GLIBC_PREREQ(2,14)
#     define HYDRA_BOOST_MATH_HAVE_FIXED_GLIBC
#  endif
#endif

#if ((defined(__linux__) && !defined(__UCLIBC__) && !defined(HYDRA_BOOST_MATH_HAVE_FIXED_GLIBC)) || defined(__QNX__) || defined(__IBMCPP__))
//
// This code was introduced in response to this glibc bug: http://sourceware.org/bugzilla/show_bug.cgi?id=2445
// Basically powl and expl can return garbage when the result is small and certain exception flags are set
// on entrance to these functions.  This appears to have been fixed in Glibc 2.14 (May 2011).
// Much more information in this message thread: https://groups.google.com/forum/#!topic/boost-list/ZT99wtIFlb4
//

#include <cfenv>

#  ifdef FE_ALL_EXCEPT

namespace hydra_boost{ namespace math{
   namespace detail
   {
   struct fpu_guard
   {
      fpu_guard()
      {
         fegetexceptflag(&m_flags, FE_ALL_EXCEPT);
         feclearexcept(FE_ALL_EXCEPT);
      }
      ~fpu_guard()
      {
         fesetexceptflag(&m_flags, FE_ALL_EXCEPT);
      }
   private:
      fexcept_t m_flags;
   };

   } // namespace detail
   }} // namespaces

#    define HYDRA_BOOST_FPU_EXCEPTION_GUARD hydra_boost::math::detail::fpu_guard local_guard_object;
#    define HYDRA_BOOST_MATH_INSTRUMENT_FPU do{ fexcept_t cpu_flags; fegetexceptflag(&cpu_flags, FE_ALL_EXCEPT); HYDRA_BOOST_MATH_INSTRUMENT_VARIABLE(cpu_flags); } while(0); 

#  else

#    define HYDRA_BOOST_FPU_EXCEPTION_GUARD
#    define HYDRA_BOOST_MATH_INSTRUMENT_FPU

#  endif

#else // All other platforms.
#  define HYDRA_BOOST_FPU_EXCEPTION_GUARD
#  define HYDRA_BOOST_MATH_INSTRUMENT_FPU
#endif

#ifdef HYDRA_BOOST_MATH_INSTRUMENT

#  include <iostream>
#  include <iomanip>
#  include <typeinfo>

#  define HYDRA_BOOST_MATH_INSTRUMENT_CODE(x) \
      std::cout << std::setprecision(35) << __FILE__ << ":" << __LINE__ << " " << x << std::endl;
#  define HYDRA_BOOST_MATH_INSTRUMENT_VARIABLE(name) HYDRA_BOOST_MATH_INSTRUMENT_CODE(#name << " = " << name)

#else

#  define HYDRA_BOOST_MATH_INSTRUMENT_CODE(x)
#  define HYDRA_BOOST_MATH_INSTRUMENT_VARIABLE(name)

#endif

//
// Thread local storage:
//
#ifndef HYDRA_BOOST_DISABLE_THREADS
#  define HYDRA_BOOST_MATH_THREAD_LOCAL thread_local
#else
#  define HYDRA_BOOST_MATH_THREAD_LOCAL 
#endif

//
// Some mingw flavours have issues with thread_local and types with non-trivial destructors
// See https://sourceforge.net/p/mingw-w64/bugs/527/
//
#if (defined(__MINGW32__) && (__GNUC__ < 9) && !defined(__clang__))
#  define HYDRA_BOOST_MATH_NO_THREAD_LOCAL_WITH_NON_TRIVIAL_TYPES
#endif


//
// Can we have constexpr tables?
//
#if (!defined(HYDRA_BOOST_NO_CXX14_CONSTEXPR)) || (defined(_MSC_VER) && _MSC_VER >= 1910)
#define HYDRA_BOOST_MATH_HAVE_CONSTEXPR_TABLES
#define HYDRA_BOOST_MATH_CONSTEXPR_TABLE_FUNCTION constexpr
#else
#define HYDRA_BOOST_MATH_CONSTEXPR_TABLE_FUNCTION
#endif


#endif // HYDRA_BOOST_MATH_TOOLS_CONFIG_HPP




