// Copyright John Maddock 2008.
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_MATH_TR1_HPP
#define HYDRA_BOOST_MATH_TR1_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <math.h> // So we can check which std C lib we're using

#ifdef __cplusplus

#include <hydra/detail/external/hydra_boost/math/tools/is_standalone.hpp>
#include <hydra/detail/external/hydra_boost/math/tools/assert.hpp>

namespace hydra_boost{ namespace math{ namespace tr1{ extern "C"{

#endif // __cplusplus

#ifndef HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION
#define HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION /**/
#endif

// we need to import/export our code only if the user has specifically
// asked for it by defining either HYDRA_BOOST_ALL_DYN_LINK if they want all boost
// libraries to be dynamically linked, or HYDRA_BOOST_MATH_TR1_DYN_LINK
// if they want just this one to be dynamically liked:
#if defined(HYDRA_BOOST_ALL_DYN_LINK) || defined(HYDRA_BOOST_MATH_TR1_DYN_LINK)
// export if this is our own source, otherwise import:
#ifdef HYDRA_BOOST_MATH_TR1_SOURCE
# define HYDRA_BOOST_MATH_TR1_DECL HYDRA_BOOST_SYMBOL_EXPORT
#else
# define HYDRA_BOOST_MATH_TR1_DECL HYDRA_BOOST_SYMBOL_IMPORT
#endif  // HYDRA_BOOST_MATH_TR1_SOURCE
#else
#  define HYDRA_BOOST_MATH_TR1_DECL
#endif  // DYN_LINK
//
// Set any throw specifications on the C99 extern "C" functions - these have to be
// the same as used in the std lib if any.
//
#if defined(__GLIBC__) && defined(__THROW)
#  define HYDRA_BOOST_MATH_C99_THROW_SPEC __THROW
#else
#  define HYDRA_BOOST_MATH_C99_THROW_SPEC
#endif

//
// Now set up the libraries to link against:
// Not compatible with standalone mode
//
#ifndef HYDRA_BOOST_MATH_STANDALONE
#include <hydra/detail/external/hydra_boost/config.hpp>
#if !defined(HYDRA_BOOST_MATH_TR1_NO_LIB) && !defined(HYDRA_BOOST_MATH_TR1_SOURCE) \
   && !defined(HYDRA_BOOST_ALL_NO_LIB) && defined(__cplusplus)
#  define HYDRA_BOOST_LIB_NAME hydra_boost_math_c99
#  if defined(HYDRA_BOOST_MATH_TR1_DYN_LINK) || defined(HYDRA_BOOST_ALL_DYN_LINK)
#     define HYDRA_BOOST_DYN_LINK
#  endif
#  include <hydra/detail/external/hydra_boost/config/auto_link.hpp>
#endif
#if !defined(HYDRA_BOOST_MATH_TR1_NO_LIB) && !defined(HYDRA_BOOST_MATH_TR1_SOURCE) \
   && !defined(HYDRA_BOOST_ALL_NO_LIB) && defined(__cplusplus)
#  define HYDRA_BOOST_LIB_NAME hydra_boost_math_c99f
#  if defined(HYDRA_BOOST_MATH_TR1_DYN_LINK) || defined(HYDRA_BOOST_ALL_DYN_LINK)
#     define HYDRA_BOOST_DYN_LINK
#  endif
#  include <hydra/detail/external/hydra_boost/config/auto_link.hpp>
#endif
#if !defined(HYDRA_BOOST_MATH_TR1_NO_LIB) && !defined(HYDRA_BOOST_MATH_TR1_SOURCE) \
   && !defined(HYDRA_BOOST_ALL_NO_LIB) && defined(__cplusplus) \
   && !defined(HYDRA_BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS)
#  define HYDRA_BOOST_LIB_NAME hydra_boost_math_c99l
#  if defined(HYDRA_BOOST_MATH_TR1_DYN_LINK) || defined(HYDRA_BOOST_ALL_DYN_LINK)
#     define HYDRA_BOOST_DYN_LINK
#  endif
#  include <hydra/detail/external/hydra_boost/config/auto_link.hpp>
#endif
#if !defined(HYDRA_BOOST_MATH_TR1_NO_LIB) && !defined(HYDRA_BOOST_MATH_TR1_SOURCE) \
   && !defined(HYDRA_BOOST_ALL_NO_LIB) && defined(__cplusplus)
#  define HYDRA_BOOST_LIB_NAME hydra_boost_math_tr1
#  if defined(HYDRA_BOOST_MATH_TR1_DYN_LINK) || defined(HYDRA_BOOST_ALL_DYN_LINK)
#     define HYDRA_BOOST_DYN_LINK
#  endif
#  include <hydra/detail/external/hydra_boost/config/auto_link.hpp>
#endif
#if !defined(HYDRA_BOOST_MATH_TR1_NO_LIB) && !defined(HYDRA_BOOST_MATH_TR1_SOURCE) \
   && !defined(HYDRA_BOOST_ALL_NO_LIB) && defined(__cplusplus)
#  define HYDRA_BOOST_LIB_NAME hydra_boost_math_tr1f
#  if defined(HYDRA_BOOST_MATH_TR1_DYN_LINK) || defined(HYDRA_BOOST_ALL_DYN_LINK)
#     define HYDRA_BOOST_DYN_LINK
#  endif
#  include <hydra/detail/external/hydra_boost/config/auto_link.hpp>
#endif
#if !defined(HYDRA_BOOST_MATH_TR1_NO_LIB) && !defined(HYDRA_BOOST_MATH_TR1_SOURCE) \
   && !defined(HYDRA_BOOST_ALL_NO_LIB) && defined(__cplusplus) \
   && !defined(HYDRA_BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS)
#  define HYDRA_BOOST_LIB_NAME hydra_boost_math_tr1l
#  if defined(HYDRA_BOOST_MATH_TR1_DYN_LINK) || defined(HYDRA_BOOST_ALL_DYN_LINK)
#     define HYDRA_BOOST_DYN_LINK
#  endif
#  include <hydra/detail/external/hydra_boost/config/auto_link.hpp>
#endif
#else // Standalone mode
#  if defined(_MSC_VER) && !defined(HYDRA_BOOST_ALL_NO_LIB)
#    pragma message("Auto linking of TR1 is not supported in standalone mode")
#  endif
#endif // HYDRA_BOOST_MATH_STANDALONE

#if !(defined(__INTEL_COMPILER) && defined(__APPLE__)) && !(defined(__FLT_EVAL_METHOD__) && !defined(__cplusplus))
#if !defined(FLT_EVAL_METHOD)
typedef float float_t;
typedef double double_t;
#elif FLT_EVAL_METHOD == -1
typedef float float_t;
typedef double double_t;
#elif FLT_EVAL_METHOD == 0
typedef float float_t;
typedef double double_t;
#elif FLT_EVAL_METHOD == 1
typedef double float_t;
typedef double double_t;
#else
typedef long double float_t;
typedef long double double_t;
#endif
#endif

// C99 Functions:
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_acosh HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_acoshf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_acoshl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;

double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_asinh HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_asinhf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_asinhl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;

double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_atanh HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_atanhf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_atanhl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;

double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_cbrt HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_cbrtf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_cbrtl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;

double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_copysign HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x, double y) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_copysignf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x, float y) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_copysignl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x, long double y) HYDRA_BOOST_MATH_C99_THROW_SPEC;

double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_erf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_erff HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_erfl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;

double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_erfc HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_erfcf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_erfcl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
#if 0
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_exp2 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_exp2f HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_exp2l HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
#endif
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_expm1 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_expm1f HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_expm1l HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
#if 0
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_fdim HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x, double y) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_fdimf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x, float y) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_fdiml HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x, long double y) HYDRA_BOOST_MATH_C99_THROW_SPEC;
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_fma HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x, double y, double z) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_fmaf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x, float y, float z) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_fmal HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x, long double y, long double z) HYDRA_BOOST_MATH_C99_THROW_SPEC;
#endif
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_fmax HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x, double y) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_fmaxf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x, float y) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_fmaxl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x, long double y) HYDRA_BOOST_MATH_C99_THROW_SPEC;

double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_fmin HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x, double y) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_fminf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x, float y) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_fminl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x, long double y) HYDRA_BOOST_MATH_C99_THROW_SPEC;

double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_hypot HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x, double y) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_hypotf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x, float y) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_hypotl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x, long double y) HYDRA_BOOST_MATH_C99_THROW_SPEC;
#if 0
int HYDRA_BOOST_MATH_TR1_DECL hydra_boost_ilogb HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
int HYDRA_BOOST_MATH_TR1_DECL hydra_boost_ilogbf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
int HYDRA_BOOST_MATH_TR1_DECL hydra_boost_ilogbl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
#endif
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_lgamma HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_lgammaf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_lgammal HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;

long long HYDRA_BOOST_MATH_TR1_DECL hydra_boost_llround HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long long HYDRA_BOOST_MATH_TR1_DECL hydra_boost_llroundf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long long HYDRA_BOOST_MATH_TR1_DECL hydra_boost_llroundl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;

double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_log1p HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_log1pf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_log1pl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
#if 0
double HYDRA_BOOST_MATH_TR1_DECL log2 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL log2f HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL log2l HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;

double HYDRA_BOOST_MATH_TR1_DECL logb HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL logbf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL logbl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long HYDRA_BOOST_MATH_TR1_DECL lrint HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long HYDRA_BOOST_MATH_TR1_DECL lrintf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long HYDRA_BOOST_MATH_TR1_DECL lrintl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
#endif
long HYDRA_BOOST_MATH_TR1_DECL hydra_boost_lround HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long HYDRA_BOOST_MATH_TR1_DECL hydra_boost_lroundf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long HYDRA_BOOST_MATH_TR1_DECL hydra_boost_lroundl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
#if 0
double HYDRA_BOOST_MATH_TR1_DECL nan HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(const char *str) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL nanf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(const char *str) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL nanl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(const char *str) HYDRA_BOOST_MATH_C99_THROW_SPEC;
double HYDRA_BOOST_MATH_TR1_DECL nearbyint HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL nearbyintf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL nearbyintl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
#endif
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_nextafter HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x, double y) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_nextafterf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x, float y) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_nextafterl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x, long double y) HYDRA_BOOST_MATH_C99_THROW_SPEC;

double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_nexttoward HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x, long double y) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_nexttowardf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x, long double y) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_nexttowardl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x, long double y) HYDRA_BOOST_MATH_C99_THROW_SPEC;
#if 0
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_remainder HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x, double y) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_remainderf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x, float y) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_remainderl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x, long double y) HYDRA_BOOST_MATH_C99_THROW_SPEC;
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_remquo HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x, double y, int *pquo) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_remquof HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x, float y, int *pquo) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_remquol HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x, long double y, int *pquo) HYDRA_BOOST_MATH_C99_THROW_SPEC;
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_rint HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_rintf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_rintl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
#endif
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_round HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_roundf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_roundl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
#if 0
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_scalbln HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x, long ex) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_scalblnf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x, long ex) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_scalblnl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x, long ex) HYDRA_BOOST_MATH_C99_THROW_SPEC;
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_scalbn HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x, int ex) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_scalbnf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x, int ex) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_scalbnl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x, int ex) HYDRA_BOOST_MATH_C99_THROW_SPEC;
#endif
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_tgamma HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_tgammaf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_tgammal HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;

double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_trunc HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_truncf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_truncl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;

// [5.2.1.1] associated Laguerre polynomials:
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_assoc_laguerre HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, unsigned m, double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_assoc_laguerref HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, unsigned m, float x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_assoc_laguerrel HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, unsigned m, long double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;

// [5.2.1.2] associated Legendre functions:
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_assoc_legendre HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned l, unsigned m, double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_assoc_legendref HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned l, unsigned m, float x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_assoc_legendrel HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned l, unsigned m, long double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;

// [5.2.1.3] beta function:
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_beta HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x, double y) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_betaf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x, float y) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_betal HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x, long double y) HYDRA_BOOST_MATH_C99_THROW_SPEC;

// [5.2.1.4] (complete) elliptic integral of the first kind:
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_comp_ellint_1 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double k) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_comp_ellint_1f HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float k) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_comp_ellint_1l HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double k) HYDRA_BOOST_MATH_C99_THROW_SPEC;

// [5.2.1.5] (complete) elliptic integral of the second kind:
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_comp_ellint_2 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double k) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_comp_ellint_2f HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float k) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_comp_ellint_2l HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double k) HYDRA_BOOST_MATH_C99_THROW_SPEC;

// [5.2.1.6] (complete) elliptic integral of the third kind:
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_comp_ellint_3 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double k, double nu) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_comp_ellint_3f HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float k, float nu) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_comp_ellint_3l HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double k, long double nu) HYDRA_BOOST_MATH_C99_THROW_SPEC;
#if 0
// [5.2.1.7] confluent hypergeometric functions:
double HYDRA_BOOST_MATH_TR1_DECL conf_hyperg HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double a, double c, double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL conf_hypergf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float a, float c, float x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL conf_hypergl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double a, long double c, long double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
#endif
// [5.2.1.8] regular modified cylindrical Bessel functions:
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_cyl_bessel_i HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double nu, double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_cyl_bessel_if HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float nu, float x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_cyl_bessel_il HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double nu, long double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;

// [5.2.1.9] cylindrical Bessel functions (of the first kind):
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_cyl_bessel_j HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double nu, double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_cyl_bessel_jf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float nu, float x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_cyl_bessel_jl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double nu, long double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;

// [5.2.1.10] irregular modified cylindrical Bessel functions:
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_cyl_bessel_k HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double nu, double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_cyl_bessel_kf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float nu, float x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_cyl_bessel_kl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double nu, long double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;

// [5.2.1.11] cylindrical Neumann functions HYDRA_BOOST_MATH_C99_THROW_SPEC;
// cylindrical Bessel functions (of the second kind):
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_cyl_neumann HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double nu, double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_cyl_neumannf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float nu, float x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_cyl_neumannl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double nu, long double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;

// [5.2.1.12] (incomplete) elliptic integral of the first kind:
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_ellint_1 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double k, double phi) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_ellint_1f HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float k, float phi) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_ellint_1l HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double k, long double phi) HYDRA_BOOST_MATH_C99_THROW_SPEC;

// [5.2.1.13] (incomplete) elliptic integral of the second kind:
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_ellint_2 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double k, double phi) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_ellint_2f HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float k, float phi) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_ellint_2l HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double k, long double phi) HYDRA_BOOST_MATH_C99_THROW_SPEC;

// [5.2.1.14] (incomplete) elliptic integral of the third kind:
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_ellint_3 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double k, double nu, double phi) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_ellint_3f HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float k, float nu, float phi) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_ellint_3l HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double k, long double nu, long double phi) HYDRA_BOOST_MATH_C99_THROW_SPEC;

// [5.2.1.15] exponential integral:
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_expint HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_expintf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_expintl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;

// [5.2.1.16] Hermite polynomials:
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_hermite HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_hermitef HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, float x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_hermitel HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, long double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;

#if 0
// [5.2.1.17] hypergeometric functions:
double HYDRA_BOOST_MATH_TR1_DECL hyperg HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double a, double b, double c, double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hypergf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float a, float b, float c, float x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hypergl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double a, long double b, long double c,
long double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
#endif

// [5.2.1.18] Laguerre polynomials:
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_laguerre HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_laguerref HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, float x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_laguerrel HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, long double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;

// [5.2.1.19] Legendre polynomials:
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_legendre HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned l, double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_legendref HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned l, float x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_legendrel HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned l, long double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;

// [5.2.1.20] Riemann zeta function:
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_riemann_zeta HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_riemann_zetaf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_riemann_zetal HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double) HYDRA_BOOST_MATH_C99_THROW_SPEC;

// [5.2.1.21] spherical Bessel functions (of the first kind):
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_sph_bessel HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_sph_besself HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, float x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_sph_bessell HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, long double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;

// [5.2.1.22] spherical associated Legendre functions:
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_sph_legendre HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned l, unsigned m, double theta) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_sph_legendref HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned l, unsigned m, float theta) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_sph_legendrel HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned l, unsigned m, long double theta) HYDRA_BOOST_MATH_C99_THROW_SPEC;

// [5.2.1.23] spherical Neumann functions HYDRA_BOOST_MATH_C99_THROW_SPEC;
// spherical Bessel functions (of the second kind):
double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_sph_neumann HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
float HYDRA_BOOST_MATH_TR1_DECL hydra_boost_sph_neumannf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, float x) HYDRA_BOOST_MATH_C99_THROW_SPEC;
long double HYDRA_BOOST_MATH_TR1_DECL hydra_boost_sph_neumannl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, long double x) HYDRA_BOOST_MATH_C99_THROW_SPEC;

#ifdef __cplusplus

}}}}  // namespaces

#include <hydra/detail/external/hydra_boost/math/tools/promotion.hpp>

namespace hydra_boost{ namespace math{ namespace tr1{
//
// Declare overload of the functions which forward to the
// C interfaces:
//
// C99 Functions:
inline double acosh HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x)
{ return hydra_boost::math::tr1::hydra_boost_acosh HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline float acoshf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{ return hydra_boost::math::tr1::hydra_boost_acoshf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline long double acoshl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{ return hydra_boost::math::tr1::hydra_boost_acoshl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline float acosh HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{ return hydra_boost::math::tr1::acoshf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline long double acosh HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{ return hydra_boost::math::tr1::acoshl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
template <class T>
inline typename tools::promote_args<T>::type acosh HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T x)
{ return hydra_boost::math::tr1::acosh HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(static_cast<typename tools::promote_args<T>::type>(x)); }

inline double asinh HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x)
{ return hydra_boost::math::tr1::hydra_boost_asinh HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline float asinhf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{ return hydra_boost::math::tr1::hydra_boost_asinhf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline long double asinhl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{ return hydra_boost::math::tr1::hydra_boost_asinhl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline float asinh HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{ return hydra_boost::math::tr1::asinhf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline long double asinh HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{ return hydra_boost::math::tr1::asinhl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
template <class T>
inline typename tools::promote_args<T>::type asinh HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T x)
{ return hydra_boost::math::tr1::asinh HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(static_cast<typename tools::promote_args<T>::type>(x)); }

inline double atanh HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x)
{ return hydra_boost::math::tr1::hydra_boost_atanh HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline float atanhf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{ return hydra_boost::math::tr1::hydra_boost_atanhf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline long double atanhl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{ return hydra_boost::math::tr1::hydra_boost_atanhl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline float atanh HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{ return hydra_boost::math::tr1::atanhf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline long double atanh HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{ return hydra_boost::math::tr1::atanhl(x); }
template <class T>
inline typename tools::promote_args<T>::type atanh HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T x)
{ return hydra_boost::math::tr1::atanh HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(static_cast<typename tools::promote_args<T>::type>(x)); }

inline double cbrt HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x)
{ return hydra_boost::math::tr1::hydra_boost_cbrt HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline float cbrtf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{ return hydra_boost::math::tr1::hydra_boost_cbrtf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline long double cbrtl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{ return hydra_boost::math::tr1::hydra_boost_cbrtl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline float cbrt HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{ return hydra_boost::math::tr1::cbrtf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline long double cbrt HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{ return hydra_boost::math::tr1::cbrtl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
template <class T>
inline typename tools::promote_args<T>::type cbrt HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T x)
{ return hydra_boost::math::tr1::cbrt HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(static_cast<typename tools::promote_args<T>::type>(x)); }

inline double copysign HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x, double y)
{ return hydra_boost::math::tr1::hydra_boost_copysign HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, y); }
inline float copysignf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x, float y)
{ return hydra_boost::math::tr1::hydra_boost_copysignf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, y); }
inline long double copysignl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x, long double y)
{ return hydra_boost::math::tr1::hydra_boost_copysignl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, y); }
inline float copysign HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x, float y)
{ return hydra_boost::math::tr1::copysignf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, y); }
inline long double copysign HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x, long double y)
{ return hydra_boost::math::tr1::copysignl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, y); }
template <class T1, class T2>
inline typename tools::promote_args<T1, T2>::type copysign HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T1 x, T2 y)
{ return hydra_boost::math::tr1::copysign HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(static_cast<typename tools::promote_args<T1, T2>::type>(x), static_cast<typename tools::promote_args<T1, T2>::type>(y)); }

inline double erf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x)
{ return hydra_boost::math::tr1::hydra_boost_erf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline float erff HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{ return hydra_boost::math::tr1::hydra_boost_erff HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline long double erfl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{ return hydra_boost::math::tr1::hydra_boost_erfl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline float erf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{ return hydra_boost::math::tr1::erff HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline long double erf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{ return hydra_boost::math::tr1::erfl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
template <class T>
inline typename tools::promote_args<T>::type erf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T x)
{ return hydra_boost::math::tr1::erf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(static_cast<typename tools::promote_args<T>::type>(x)); }

inline double erfc HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x)
{ return hydra_boost::math::tr1::hydra_boost_erfc HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline float erfcf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{ return hydra_boost::math::tr1::hydra_boost_erfcf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline long double erfcl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{ return hydra_boost::math::tr1::hydra_boost_erfcl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline float erfc HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{ return hydra_boost::math::tr1::erfcf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline long double erfc HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{ return hydra_boost::math::tr1::erfcl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
template <class T>
inline typename tools::promote_args<T>::type erfc HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T x)
{ return hydra_boost::math::tr1::erfc HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(static_cast<typename tools::promote_args<T>::type>(x)); }

#if 0
double exp2 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x);
float exp2f HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x);
long double exp2l HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x);
#endif

inline float expm1f HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{ return hydra_boost::math::tr1::hydra_boost_expm1f HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline double expm1 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x)
{ return hydra_boost::math::tr1::hydra_boost_expm1 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline long double expm1l HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{ return hydra_boost::math::tr1::hydra_boost_expm1l HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline float expm1 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{ return hydra_boost::math::tr1::expm1f HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline long double expm1 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{ return hydra_boost::math::tr1::expm1l HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
template <class T>
inline typename tools::promote_args<T>::type expm1 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T x)
{ return hydra_boost::math::tr1::expm1 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(static_cast<typename tools::promote_args<T>::type>(x)); }

#if 0
double fdim HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x, double y);
float fdimf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x, float y);
long double fdiml HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x, long double y);
double fma HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x, double y, double z);
float fmaf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x, float y, float z);
long double fmal HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x, long double y, long double z);
#endif
inline double fmax HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x, double y)
{ return hydra_boost::math::tr1::hydra_boost_fmax HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, y); }
inline float fmaxf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x, float y)
{ return hydra_boost::math::tr1::hydra_boost_fmaxf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, y); }
inline long double fmaxl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x, long double y)
{ return hydra_boost::math::tr1::hydra_boost_fmaxl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, y); }
inline float fmax HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x, float y)
{ return hydra_boost::math::tr1::fmaxf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, y); }
inline long double fmax HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x, long double y)
{ return hydra_boost::math::tr1::fmaxl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, y); }
template <class T1, class T2>
inline typename tools::promote_args<T1, T2>::type fmax HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T1 x, T2 y)
{ return hydra_boost::math::tr1::fmax HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(static_cast<typename tools::promote_args<T1, T2>::type>(x), static_cast<typename tools::promote_args<T1, T2>::type>(y)); }

inline double fmin HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x, double y)
{ return hydra_boost::math::tr1::hydra_boost_fmin HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, y); }
inline float fminf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x, float y)
{ return hydra_boost::math::tr1::hydra_boost_fminf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, y); }
inline long double fminl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x, long double y)
{ return hydra_boost::math::tr1::hydra_boost_fminl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, y); }
inline float fmin HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x, float y)
{ return hydra_boost::math::tr1::fminf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, y); }
inline long double fmin HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x, long double y)
{ return hydra_boost::math::tr1::fminl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, y); }
template <class T1, class T2>
inline typename tools::promote_args<T1, T2>::type fmin HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T1 x, T2 y)
{ return hydra_boost::math::tr1::fmin HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(static_cast<typename tools::promote_args<T1, T2>::type>(x), static_cast<typename tools::promote_args<T1, T2>::type>(y)); }

inline float hypotf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x, float y)
{ return hydra_boost::math::tr1::hydra_boost_hypotf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, y); }
inline double hypot HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x, double y)
{ return hydra_boost::math::tr1::hydra_boost_hypot HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, y); }
inline long double hypotl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x, long double y)
{ return hydra_boost::math::tr1::hydra_boost_hypotl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, y); }
inline float hypot HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x, float y)
{ return hydra_boost::math::tr1::hypotf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, y); }
inline long double hypot HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x, long double y)
{ return hydra_boost::math::tr1::hypotl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, y); }
template <class T1, class T2>
inline typename tools::promote_args<T1, T2>::type hypot HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T1 x, T2 y)
{ return hydra_boost::math::tr1::hypot HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(static_cast<typename tools::promote_args<T1, T2>::type>(x), static_cast<typename tools::promote_args<T1, T2>::type>(y)); }

#if 0
int ilogb HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x);
int ilogbf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x);
int ilogbl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x);
#endif

inline float lgammaf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{ return hydra_boost::math::tr1::hydra_boost_lgammaf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline double lgamma HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x)
{ return hydra_boost::math::tr1::hydra_boost_lgamma HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline long double lgammal HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{ return hydra_boost::math::tr1::hydra_boost_lgammal HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline float lgamma HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{ return hydra_boost::math::tr1::lgammaf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline long double lgamma HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{ return hydra_boost::math::tr1::lgammal HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
template <class T>
inline typename tools::promote_args<T>::type lgamma HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T x)
{ return hydra_boost::math::tr1::lgamma HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(static_cast<typename tools::promote_args<T>::type>(x)); }

#ifdef HYDRA_BOOST_HAS_LONG_LONG
#if 0
long long llrint HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x);
long long llrintf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x);
long long llrintl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x);
#endif

inline long long llroundf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{ return hydra_boost::math::tr1::hydra_boost_llroundf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline long long llround HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x)
{ return hydra_boost::math::tr1::hydra_boost_llround HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline long long llroundl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{ return hydra_boost::math::tr1::hydra_boost_llroundl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline long long llround HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{ return hydra_boost::math::tr1::llroundf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline long long llround HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{ return hydra_boost::math::tr1::llroundl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
template <class T>
inline long long llround HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T x)
{ return llround HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(static_cast<double>(x)); }
#endif

inline float log1pf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{ return hydra_boost::math::tr1::hydra_boost_log1pf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline double log1p HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x)
{ return hydra_boost::math::tr1::hydra_boost_log1p HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline long double log1pl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{ return hydra_boost::math::tr1::hydra_boost_log1pl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline float log1p HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{ return hydra_boost::math::tr1::log1pf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline long double log1p HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{ return hydra_boost::math::tr1::log1pl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
template <class T>
inline typename tools::promote_args<T>::type log1p HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T x)
{ return hydra_boost::math::tr1::log1p HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(static_cast<typename tools::promote_args<T>::type>(x)); }
#if 0
double log2 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x);
float log2f HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x);
long double log2l HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x);

double logb HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x);
float logbf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x);
long double logbl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x);
long lrint HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x);
long lrintf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x);
long lrintl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x);
#endif
inline long lroundf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{ return hydra_boost::math::tr1::hydra_boost_lroundf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline long lround HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x)
{ return hydra_boost::math::tr1::hydra_boost_lround HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline long lroundl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{ return hydra_boost::math::tr1::hydra_boost_lroundl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline long lround HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{ return hydra_boost::math::tr1::lroundf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline long lround HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{ return hydra_boost::math::tr1::lroundl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
template <class T>
long lround HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T x)
{ return hydra_boost::math::tr1::lround HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(static_cast<double>(x)); }
#if 0
double nan HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(const char *str);
float nanf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(const char *str);
long double nanl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(const char *str);
double nearbyint HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x);
float nearbyintf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x);
long double nearbyintl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x);
#endif
inline float nextafterf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x, float y)
{ return hydra_boost::math::tr1::hydra_boost_nextafterf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, y); }
inline double nextafter HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x, double y)
{ return hydra_boost::math::tr1::hydra_boost_nextafter HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, y); }
inline long double nextafterl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x, long double y)
{ return hydra_boost::math::tr1::hydra_boost_nextafterl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, y); }
inline float nextafter HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x, float y)
{ return hydra_boost::math::tr1::nextafterf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, y); }
inline long double nextafter HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x, long double y)
{ return hydra_boost::math::tr1::nextafterl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, y); }
template <class T1, class T2>
inline typename tools::promote_args<T1, T2>::type nextafter HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T1 x, T2 y)
{ return hydra_boost::math::tr1::nextafter HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(static_cast<typename tools::promote_args<T1, T2>::type>(x), static_cast<typename tools::promote_args<T1, T2>::type>(y)); }

inline float nexttowardf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x, float y)
{ return hydra_boost::math::tr1::hydra_boost_nexttowardf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, y); }
inline double nexttoward HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x, double y)
{ return hydra_boost::math::tr1::hydra_boost_nexttoward HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, y); }
inline long double nexttowardl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x, long double y)
{ return hydra_boost::math::tr1::hydra_boost_nexttowardl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, y); }
inline float nexttoward HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x, float y)
{ return hydra_boost::math::tr1::nexttowardf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, y); }
inline long double nexttoward HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x, long double y)
{ return hydra_boost::math::tr1::nexttowardl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, y); }
template <class T1, class T2>
inline typename tools::promote_args<T1, T2>::type nexttoward HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T1 x, T2 y)
{ return static_cast<typename tools::promote_args<T1, T2>::type>(hydra_boost::math::tr1::nexttoward HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(static_cast<typename tools::promote_args<T1, T2>::type>(x), static_cast<long double>(y))); }
#if 0
double remainder HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x, double y);
float remainderf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x, float y);
long double remainderl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x, long double y);
double remquo HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x, double y, int *pquo);
float remquof HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x, float y, int *pquo);
long double remquol HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x, long double y, int *pquo);
double rint HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x);
float rintf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x);
long double rintl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x);
#endif
inline float roundf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{ return hydra_boost::math::tr1::hydra_boost_roundf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline double round HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x)
{ return hydra_boost::math::tr1::hydra_boost_round HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline long double roundl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{ return hydra_boost::math::tr1::hydra_boost_roundl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline float round HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{ return hydra_boost::math::tr1::roundf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline long double round HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{ return hydra_boost::math::tr1::roundl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
template <class T>
inline typename tools::promote_args<T>::type round HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T x)
{ return hydra_boost::math::tr1::round HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(static_cast<typename tools::promote_args<T>::type>(x)); }
#if 0
double scalbln HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x, long ex);
float scalblnf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x, long ex);
long double scalblnl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x, long ex);
double scalbn HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x, int ex);
float scalbnf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x, int ex);
long double scalbnl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x, int ex);
#endif
inline float tgammaf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{ return hydra_boost::math::tr1::hydra_boost_tgammaf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline double tgamma HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x)
{ return hydra_boost::math::tr1::hydra_boost_tgamma HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline long double tgammal HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{ return hydra_boost::math::tr1::hydra_boost_tgammal HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline float tgamma HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{ return hydra_boost::math::tr1::tgammaf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline long double tgamma HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{ return hydra_boost::math::tr1::tgammal HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
template <class T>
inline typename tools::promote_args<T>::type tgamma HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T x)
{ return hydra_boost::math::tr1::tgamma HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(static_cast<typename tools::promote_args<T>::type>(x)); }

inline float truncf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{ return hydra_boost::math::tr1::hydra_boost_truncf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline double trunc HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x)
{ return hydra_boost::math::tr1::hydra_boost_trunc HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline long double truncl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{ return hydra_boost::math::tr1::hydra_boost_truncl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline float trunc HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{ return hydra_boost::math::tr1::truncf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline long double trunc HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{ return hydra_boost::math::tr1::truncl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
template <class T>
inline typename tools::promote_args<T>::type trunc HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T x)
{ return hydra_boost::math::tr1::trunc HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(static_cast<typename tools::promote_args<T>::type>(x)); }

# define NO_MACRO_EXPAND /**/
// C99 macros defined as C++ templates
template<class T> bool signbit NO_MACRO_EXPAND(T)
{ static_assert(sizeof(T) == 0, "Undefined behavior; this template should never be instantiated"); return false; } // must not be instantiated
template<> bool HYDRA_BOOST_MATH_TR1_DECL signbit<float> NO_MACRO_EXPAND(float x);
template<> bool HYDRA_BOOST_MATH_TR1_DECL signbit<double> NO_MACRO_EXPAND(double x);
template<> bool HYDRA_BOOST_MATH_TR1_DECL signbit<long double> NO_MACRO_EXPAND(long double x);

template<class T> int fpclassify NO_MACRO_EXPAND(T)
{ static_assert(sizeof(T) == 0, "Undefined behavior; this template should never be instantiated"); return false; } // must not be instantiated
template<> int HYDRA_BOOST_MATH_TR1_DECL fpclassify<float> NO_MACRO_EXPAND(float x);
template<> int HYDRA_BOOST_MATH_TR1_DECL fpclassify<double> NO_MACRO_EXPAND(double x);
template<> int HYDRA_BOOST_MATH_TR1_DECL fpclassify<long double> NO_MACRO_EXPAND(long double x);

template<class T> bool isfinite NO_MACRO_EXPAND(T)
{ static_assert(sizeof(T) == 0, "Undefined behavior; this template should never be instantiated"); return false; } // must not be instantiated
template<> bool HYDRA_BOOST_MATH_TR1_DECL isfinite<float> NO_MACRO_EXPAND(float x);
template<> bool HYDRA_BOOST_MATH_TR1_DECL isfinite<double> NO_MACRO_EXPAND(double x);
template<> bool HYDRA_BOOST_MATH_TR1_DECL isfinite<long double> NO_MACRO_EXPAND(long double x);

template<class T> bool isinf NO_MACRO_EXPAND(T)
{ static_assert(sizeof(T) == 0, "Undefined behavior; this template should never be instantiated"); return false; } // must not be instantiated
template<> bool HYDRA_BOOST_MATH_TR1_DECL isinf<float> NO_MACRO_EXPAND(float x);
template<> bool HYDRA_BOOST_MATH_TR1_DECL isinf<double> NO_MACRO_EXPAND(double x);
template<> bool HYDRA_BOOST_MATH_TR1_DECL isinf<long double> NO_MACRO_EXPAND(long double x);

template<class T> bool isnan NO_MACRO_EXPAND(T)
{ static_assert(sizeof(T) == 0, "Undefined behavior; this template should never be instantiated"); return false; } // must not be instantiated
template<> bool HYDRA_BOOST_MATH_TR1_DECL isnan<float> NO_MACRO_EXPAND(float x);
template<> bool HYDRA_BOOST_MATH_TR1_DECL isnan<double> NO_MACRO_EXPAND(double x);
template<> bool HYDRA_BOOST_MATH_TR1_DECL isnan<long double> NO_MACRO_EXPAND(long double x);

template<class T> bool isnormal NO_MACRO_EXPAND(T)
{ static_assert(sizeof(T) == 0, "Undefined behavior; this template should never be instantiated"); return false; } // must not be instantiated
template<> bool HYDRA_BOOST_MATH_TR1_DECL isnormal<float> NO_MACRO_EXPAND(float x);
template<> bool HYDRA_BOOST_MATH_TR1_DECL isnormal<double> NO_MACRO_EXPAND(double x);
template<> bool HYDRA_BOOST_MATH_TR1_DECL isnormal<long double> NO_MACRO_EXPAND(long double x);

#undef NO_MACRO_EXPAND

// [5.2.1.1] associated Laguerre polynomials:
inline float assoc_laguerref HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, unsigned m, float x)
{ return hydra_boost::math::tr1::hydra_boost_assoc_laguerref HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(n, m, x); }
inline double assoc_laguerre HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, unsigned m, double x)
{ return hydra_boost::math::tr1::hydra_boost_assoc_laguerre HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(n, m, x); }
inline long double assoc_laguerrel HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, unsigned m, long double x)
{ return hydra_boost::math::tr1::hydra_boost_assoc_laguerrel HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(n, m, x); }
inline float assoc_laguerre HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, unsigned m, float x)
{ return hydra_boost::math::tr1::assoc_laguerref HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(n, m, x); }
inline long double assoc_laguerre HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, unsigned m, long double x)
{ return hydra_boost::math::tr1::assoc_laguerrel HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(n, m, x); }
template <class T>
inline typename tools::promote_args<T>::type assoc_laguerre HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, unsigned m, T x)
{ return hydra_boost::math::tr1::assoc_laguerre HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(n, m, static_cast<typename tools::promote_args<T>::type>(x)); }

// [5.2.1.2] associated Legendre functions:
inline float assoc_legendref HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned l, unsigned m, float x)
{ return hydra_boost::math::tr1::hydra_boost_assoc_legendref HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(l, m, x); }
inline double assoc_legendre HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned l, unsigned m, double x)
{ return hydra_boost::math::tr1::hydra_boost_assoc_legendre HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(l, m, x); }
inline long double assoc_legendrel HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned l, unsigned m, long double x)
{ return hydra_boost::math::tr1::hydra_boost_assoc_legendrel HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(l, m, x); }
inline float assoc_legendre HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned l, unsigned m, float x)
{ return hydra_boost::math::tr1::assoc_legendref HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(l, m, x); }
inline long double assoc_legendre HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned l, unsigned m, long double x)
{ return hydra_boost::math::tr1::assoc_legendrel HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(l, m, x); }
template <class T>
inline typename tools::promote_args<T>::type assoc_legendre HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned l, unsigned m, T x)
{ return hydra_boost::math::tr1::assoc_legendre HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(l, m, static_cast<typename tools::promote_args<T>::type>(x)); }

// [5.2.1.3] beta function:
inline float betaf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x, float y)
{ return hydra_boost::math::tr1::hydra_boost_betaf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, y); }
inline double beta HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x, double y)
{ return hydra_boost::math::tr1::hydra_boost_beta HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, y); }
inline long double betal HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x, long double y)
{ return hydra_boost::math::tr1::hydra_boost_betal HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, y); }
inline float beta HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x, float y)
{ return hydra_boost::math::tr1::betaf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, y); }
inline long double beta HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x, long double y)
{ return hydra_boost::math::tr1::betal HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, y); }
template <class T1, class T2>
inline typename tools::promote_args<T1, T2>::type beta HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T2 x, T1 y)
{ return hydra_boost::math::tr1::beta HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(static_cast<typename tools::promote_args<T1, T2>::type>(x), static_cast<typename tools::promote_args<T1, T2>::type>(y)); }

// [5.2.1.4] (complete) elliptic integral of the first kind:
inline float comp_ellint_1f HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float k)
{ return hydra_boost::math::tr1::hydra_boost_comp_ellint_1f HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(k); }
inline double comp_ellint_1 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double k)
{ return hydra_boost::math::tr1::hydra_boost_comp_ellint_1 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(k); }
inline long double comp_ellint_1l HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double k)
{ return hydra_boost::math::tr1::hydra_boost_comp_ellint_1l HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(k); }
inline float comp_ellint_1 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float k)
{ return hydra_boost::math::tr1::comp_ellint_1f HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(k); }
inline long double comp_ellint_1 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double k)
{ return hydra_boost::math::tr1::comp_ellint_1l HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(k); }
template <class T>
inline typename tools::promote_args<T>::type comp_ellint_1 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T k)
{ return hydra_boost::math::tr1::comp_ellint_1 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(static_cast<typename tools::promote_args<T>::type>(k)); }

// [5.2.1.5]  (complete) elliptic integral of the second kind:
inline float comp_ellint_2f(float k)
{ return hydra_boost::math::tr1::hydra_boost_comp_ellint_2f(k); }
inline double comp_ellint_2(double k)
{ return hydra_boost::math::tr1::hydra_boost_comp_ellint_2(k); }
inline long double comp_ellint_2l(long double k)
{ return hydra_boost::math::tr1::hydra_boost_comp_ellint_2l(k); }
inline float comp_ellint_2(float k)
{ return hydra_boost::math::tr1::comp_ellint_2f(k); }
inline long double comp_ellint_2(long double k)
{ return hydra_boost::math::tr1::comp_ellint_2l(k); }
template <class T>
inline typename tools::promote_args<T>::type comp_ellint_2(T k)
{ return hydra_boost::math::tr1::comp_ellint_2(static_cast<typename tools::promote_args<T>::type> HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(k)); }

// [5.2.1.6]  (complete) elliptic integral of the third kind:
inline float comp_ellint_3f(float k, float nu)
{ return hydra_boost::math::tr1::hydra_boost_comp_ellint_3f(k, nu); }
inline double comp_ellint_3(double k, double nu)
{ return hydra_boost::math::tr1::hydra_boost_comp_ellint_3(k, nu); }
inline long double comp_ellint_3l(long double k, long double nu)
{ return hydra_boost::math::tr1::hydra_boost_comp_ellint_3l(k, nu); }
inline float comp_ellint_3(float k, float nu)
{ return hydra_boost::math::tr1::comp_ellint_3f(k, nu); }
inline long double comp_ellint_3(long double k, long double nu)
{ return hydra_boost::math::tr1::comp_ellint_3l(k, nu); }
template <class T1, class T2>
inline typename tools::promote_args<T1, T2>::type comp_ellint_3(T1 k, T2 nu)
{ return hydra_boost::math::tr1::comp_ellint_3(static_cast<typename tools::promote_args<T1, T2>::type> HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(k), static_cast<typename tools::promote_args<T1, T2>::type> HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(nu)); }

#if 0
// [5.2.1.7] confluent hypergeometric functions:
double conf_hyperg HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double a, double c, double x);
float conf_hypergf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float a, float c, float x);
long double conf_hypergl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double a, long double c, long double x);
#endif

// [5.2.1.8] regular modified cylindrical Bessel functions:
inline float cyl_bessel_if HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float nu, float x)
{ return hydra_boost::math::tr1::hydra_boost_cyl_bessel_if HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(nu, x); }
inline double cyl_bessel_i HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double nu, double x)
{ return hydra_boost::math::tr1::hydra_boost_cyl_bessel_i HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(nu, x); }
inline long double cyl_bessel_il HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double nu, long double x)
{ return hydra_boost::math::tr1::hydra_boost_cyl_bessel_il HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(nu, x); }
inline float cyl_bessel_i HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float nu, float x)
{ return hydra_boost::math::tr1::cyl_bessel_if HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(nu, x); }
inline long double cyl_bessel_i HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double nu, long double x)
{ return hydra_boost::math::tr1::cyl_bessel_il HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(nu, x); }
template <class T1, class T2>
inline typename tools::promote_args<T1, T2>::type cyl_bessel_i HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T1 nu, T2 x)
{ return hydra_boost::math::tr1::cyl_bessel_i HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(static_cast<typename tools::promote_args<T1, T2>::type>(nu), static_cast<typename tools::promote_args<T1, T2>::type>(x)); }

// [5.2.1.9] cylindrical Bessel functions (of the first kind):
inline float cyl_bessel_jf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float nu, float x)
{ return hydra_boost::math::tr1::hydra_boost_cyl_bessel_jf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(nu, x); }
inline double cyl_bessel_j HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double nu, double x)
{ return hydra_boost::math::tr1::hydra_boost_cyl_bessel_j HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(nu, x); }
inline long double cyl_bessel_jl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double nu, long double x)
{ return hydra_boost::math::tr1::hydra_boost_cyl_bessel_jl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(nu, x); }
inline float cyl_bessel_j HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float nu, float x)
{ return hydra_boost::math::tr1::cyl_bessel_jf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(nu, x); }
inline long double cyl_bessel_j HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double nu, long double x)
{ return hydra_boost::math::tr1::cyl_bessel_jl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(nu, x); }
template <class T1, class T2>
inline typename tools::promote_args<T1, T2>::type cyl_bessel_j HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T1 nu, T2 x)
{ return hydra_boost::math::tr1::cyl_bessel_j HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(static_cast<typename tools::promote_args<T1, T2>::type>(nu), static_cast<typename tools::promote_args<T1, T2>::type>(x)); }

// [5.2.1.10] irregular modified cylindrical Bessel functions:
inline float cyl_bessel_kf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float nu, float x)
{ return hydra_boost::math::tr1::hydra_boost_cyl_bessel_kf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(nu, x); }
inline double cyl_bessel_k HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double nu, double x)
{ return hydra_boost::math::tr1::hydra_boost_cyl_bessel_k HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(nu, x); }
inline long double cyl_bessel_kl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double nu, long double x)
{ return hydra_boost::math::tr1::hydra_boost_cyl_bessel_kl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(nu, x); }
inline float cyl_bessel_k HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float nu, float x)
{ return hydra_boost::math::tr1::cyl_bessel_kf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(nu, x); }
inline long double cyl_bessel_k HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double nu, long double x)
{ return hydra_boost::math::tr1::cyl_bessel_kl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(nu, x); }
template <class T1, class T2>
inline typename tools::promote_args<T1, T2>::type cyl_bessel_k HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T1 nu, T2 x)
{ return hydra_boost::math::tr1::cyl_bessel_k HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(static_cast<typename tools::promote_args<T1, T2>::type> HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(nu), static_cast<typename tools::promote_args<T1, T2>::type>(x)); }

// [5.2.1.11] cylindrical Neumann functions;
// cylindrical Bessel functions (of the second kind):
inline float cyl_neumannf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float nu, float x)
{ return hydra_boost::math::tr1::hydra_boost_cyl_neumannf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(nu, x); }
inline double cyl_neumann HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double nu, double x)
{ return hydra_boost::math::tr1::hydra_boost_cyl_neumann HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(nu, x); }
inline long double cyl_neumannl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double nu, long double x)
{ return hydra_boost::math::tr1::hydra_boost_cyl_neumannl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(nu, x); }
inline float cyl_neumann HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float nu, float x)
{ return hydra_boost::math::tr1::cyl_neumannf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(nu, x); }
inline long double cyl_neumann HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double nu, long double x)
{ return hydra_boost::math::tr1::cyl_neumannl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(nu, x); }
template <class T1, class T2>
inline typename tools::promote_args<T1, T2>::type cyl_neumann HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T1 nu, T2 x)
{ return hydra_boost::math::tr1::cyl_neumann HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(static_cast<typename tools::promote_args<T1, T2>::type>(nu), static_cast<typename tools::promote_args<T1, T2>::type>(x)); }

// [5.2.1.12] (incomplete) elliptic integral of the first kind:
inline float ellint_1f HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float k, float phi)
{ return hydra_boost::math::tr1::hydra_boost_ellint_1f HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(k, phi); }
inline double ellint_1 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double k, double phi)
{ return hydra_boost::math::tr1::hydra_boost_ellint_1 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(k, phi); }
inline long double ellint_1l HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double k, long double phi)
{ return hydra_boost::math::tr1::hydra_boost_ellint_1l HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(k, phi); }
inline float ellint_1 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float k, float phi)
{ return hydra_boost::math::tr1::ellint_1f HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(k, phi); }
inline long double ellint_1 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double k, long double phi)
{ return hydra_boost::math::tr1::ellint_1l HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(k, phi); }
template <class T1, class T2>
inline typename tools::promote_args<T1, T2>::type ellint_1 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T1 k, T2 phi)
{ return hydra_boost::math::tr1::ellint_1 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(static_cast<typename tools::promote_args<T1, T2>::type>(k), static_cast<typename tools::promote_args<T1, T2>::type>(phi)); }

// [5.2.1.13] (incomplete) elliptic integral of the second kind:
inline float ellint_2f HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float k, float phi)
{ return hydra_boost::math::tr1::hydra_boost_ellint_2f HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(k, phi); }
inline double ellint_2 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double k, double phi)
{ return hydra_boost::math::tr1::hydra_boost_ellint_2 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(k, phi); }
inline long double ellint_2l HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double k, long double phi)
{ return hydra_boost::math::tr1::hydra_boost_ellint_2l HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(k, phi); }
inline float ellint_2 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float k, float phi)
{ return hydra_boost::math::tr1::ellint_2f HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(k, phi); }
inline long double ellint_2 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double k, long double phi)
{ return hydra_boost::math::tr1::ellint_2l HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(k, phi); }
template <class T1, class T2>
inline typename tools::promote_args<T1, T2>::type ellint_2 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T1 k, T2 phi)
{ return hydra_boost::math::tr1::ellint_2 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(static_cast<typename tools::promote_args<T1, T2>::type>(k), static_cast<typename tools::promote_args<T1, T2>::type>(phi)); }

// [5.2.1.14] (incomplete) elliptic integral of the third kind:
inline float ellint_3f HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float k, float nu, float phi)
{ return hydra_boost::math::tr1::hydra_boost_ellint_3f HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(k, nu, phi); }
inline double ellint_3 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double k, double nu, double phi)
{ return hydra_boost::math::tr1::hydra_boost_ellint_3 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(k, nu, phi); }
inline long double ellint_3l HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double k, long double nu, long double phi)
{ return hydra_boost::math::tr1::hydra_boost_ellint_3l HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(k, nu, phi); }
inline float ellint_3 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float k, float nu, float phi)
{ return hydra_boost::math::tr1::ellint_3f HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(k, nu, phi); }
inline long double ellint_3 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double k, long double nu, long double phi)
{ return hydra_boost::math::tr1::ellint_3l HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(k, nu, phi); }
template <class T1, class T2, class T3>
inline typename tools::promote_args<T1, T2, T3>::type ellint_3 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T1 k, T2 nu, T3 phi)
{ return hydra_boost::math::tr1::ellint_3 HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(static_cast<typename tools::promote_args<T1, T2, T3>::type>(k), static_cast<typename tools::promote_args<T1, T2, T3>::type>(nu), static_cast<typename tools::promote_args<T1, T2, T3>::type>(phi)); }

// [5.2.1.15] exponential integral:
inline float expintf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{ return hydra_boost::math::tr1::hydra_boost_expintf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline double expint HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double x)
{ return hydra_boost::math::tr1::hydra_boost_expint HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline long double expintl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{ return hydra_boost::math::tr1::hydra_boost_expintl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline float expint HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{ return hydra_boost::math::tr1::expintf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
inline long double expint HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{ return hydra_boost::math::tr1::expintl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x); }
template <class T>
inline typename tools::promote_args<T>::type expint HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T x)
{ return hydra_boost::math::tr1::expint HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(static_cast<typename tools::promote_args<T>::type>(x)); }

// [5.2.1.16] Hermite polynomials:
inline float hermitef HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, float x)
{ return hydra_boost::math::tr1::hydra_boost_hermitef HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(n, x); }
inline double hermite HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, double x)
{ return hydra_boost::math::tr1::hydra_boost_hermite HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(n, x); }
inline long double hermitel HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, long double x)
{ return hydra_boost::math::tr1::hydra_boost_hermitel HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(n, x); }
inline float hermite HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, float x)
{ return hydra_boost::math::tr1::hermitef HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(n, x); }
inline long double hermite HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, long double x)
{ return hydra_boost::math::tr1::hermitel HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(n, x); }
template <class T>
inline typename tools::promote_args<T>::type hermite HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, T x)
{ return hydra_boost::math::tr1::hermite HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(n, static_cast<typename tools::promote_args<T>::type>(x)); }

#if 0
// [5.2.1.17] hypergeometric functions:
double hyperg HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double a, double b, double c, double x);
float hypergf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float a, float b, float c, float x);
long double hypergl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double a, long double b, long double c,
long double x);
#endif

// [5.2.1.18] Laguerre polynomials:
inline float laguerref HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, float x)
{ return hydra_boost::math::tr1::hydra_boost_laguerref HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(n, x); }
inline double laguerre HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, double x)
{ return hydra_boost::math::tr1::hydra_boost_laguerre HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(n, x); }
inline long double laguerrel HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, long double x)
{ return hydra_boost::math::tr1::hydra_boost_laguerrel HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(n, x); }
inline float laguerre HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, float x)
{ return hydra_boost::math::tr1::laguerref HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(n, x); }
inline long double laguerre HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, long double x)
{ return hydra_boost::math::tr1::laguerrel HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(n, x); }
template <class T>
inline typename tools::promote_args<T>::type laguerre HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, T x)
{ return hydra_boost::math::tr1::laguerre HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(n, static_cast<typename tools::promote_args<T>::type>(x)); }

// [5.2.1.19] Legendre polynomials:
inline float legendref HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned l, float x)
{ return hydra_boost::math::tr1::hydra_boost_legendref HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(l, x); }
inline double legendre HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned l, double x)
{ return hydra_boost::math::tr1::hydra_boost_legendre HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(l, x); }
inline long double legendrel HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned l, long double x)
{ return hydra_boost::math::tr1::hydra_boost_legendrel HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(l, x); }
inline float legendre HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned l, float x)
{ return hydra_boost::math::tr1::legendref HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(l, x); }
inline long double legendre HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned l, long double x)
{ return hydra_boost::math::tr1::legendrel HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(l, x); }
template <class T>
inline typename tools::promote_args<T>::type legendre HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned l, T x)
{ return hydra_boost::math::tr1::legendre HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(l, static_cast<typename tools::promote_args<T>::type>(x)); }

// [5.2.1.20] Riemann zeta function:
inline float riemann_zetaf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float z)
{ return hydra_boost::math::tr1::hydra_boost_riemann_zetaf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(z); }
inline double riemann_zeta HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(double z)
{ return hydra_boost::math::tr1::hydra_boost_riemann_zeta HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(z); }
inline long double riemann_zetal HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double z)
{ return hydra_boost::math::tr1::hydra_boost_riemann_zetal HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(z); }
inline float riemann_zeta HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(float z)
{ return hydra_boost::math::tr1::riemann_zetaf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(z); }
inline long double riemann_zeta HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(long double z)
{ return hydra_boost::math::tr1::riemann_zetal HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(z); }
template <class T>
inline typename tools::promote_args<T>::type riemann_zeta HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T z)
{ return hydra_boost::math::tr1::riemann_zeta HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(static_cast<typename tools::promote_args<T>::type>(z)); }

// [5.2.1.21] spherical Bessel functions (of the first kind):
inline float sph_besself HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, float x)
{ return hydra_boost::math::tr1::hydra_boost_sph_besself HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(n, x); }
inline double sph_bessel HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, double x)
{ return hydra_boost::math::tr1::hydra_boost_sph_bessel HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(n, x); }
inline long double sph_bessell HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, long double x)
{ return hydra_boost::math::tr1::hydra_boost_sph_bessell HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(n, x); }
inline float sph_bessel HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, float x)
{ return hydra_boost::math::tr1::sph_besself HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(n, x); }
inline long double sph_bessel HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, long double x)
{ return hydra_boost::math::tr1::sph_bessell HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(n, x); }
template <class T>
inline typename tools::promote_args<T>::type sph_bessel HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, T x)
{ return hydra_boost::math::tr1::sph_bessel HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(n, static_cast<typename tools::promote_args<T>::type>(x)); }

// [5.2.1.22] spherical associated Legendre functions:
inline float sph_legendref HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned l, unsigned m, float theta)
{ return hydra_boost::math::tr1::hydra_boost_sph_legendref HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(l, m, theta); }
inline double sph_legendre HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned l, unsigned m, double theta)
{ return hydra_boost::math::tr1::hydra_boost_sph_legendre HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(l, m, theta); }
inline long double sph_legendrel HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned l, unsigned m, long double theta)
{ return hydra_boost::math::tr1::hydra_boost_sph_legendrel HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(l, m, theta); }
inline float sph_legendre HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned l, unsigned m, float theta)
{ return hydra_boost::math::tr1::sph_legendref HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(l, m, theta); }
inline long double sph_legendre HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned l, unsigned m, long double theta)
{ return hydra_boost::math::tr1::sph_legendrel HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(l, m, theta); }
template <class T>
inline typename tools::promote_args<T>::type sph_legendre HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned l, unsigned m, T theta)
{ return hydra_boost::math::tr1::sph_legendre HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(l, m, static_cast<typename tools::promote_args<T>::type>(theta)); }

// [5.2.1.23] spherical Neumann functions;
// spherical Bessel functions (of the second kind):
inline float sph_neumannf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, float x)
{ return hydra_boost::math::tr1::hydra_boost_sph_neumannf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(n, x); }
inline double sph_neumann HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, double x)
{ return hydra_boost::math::tr1::hydra_boost_sph_neumann HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(n, x); }
inline long double sph_neumannl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, long double x)
{ return hydra_boost::math::tr1::hydra_boost_sph_neumannl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(n, x); }
inline float sph_neumann HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, float x)
{ return hydra_boost::math::tr1::sph_neumannf HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(n, x); }
inline long double sph_neumann HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, long double x)
{ return hydra_boost::math::tr1::sph_neumannl HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(n, x); }
template <class T>
inline typename tools::promote_args<T>::type sph_neumann HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(unsigned n, T x)
{ return hydra_boost::math::tr1::sph_neumann HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(n, static_cast<typename tools::promote_args<T>::type>(x)); }

}}} // namespaces

#else // __cplusplus

#include <hydra/detail/external/hydra_boost/math/tr1_c_macros.ipp>

#endif // __cplusplus

#endif // HYDRA_BOOST_MATH_TR1_HPP

