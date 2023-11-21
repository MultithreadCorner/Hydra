///////////////////////////////////////////////////////////////////////////////
// Copyright Christopher Kormanyos 2014.
// Copyright John Maddock 2014.
// Copyright Paul Bristow 2014.
// Distributed under the Boost Software License,
// Version 1.0. (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Implement quadruple-precision <cmath> support.

#ifndef HYDRA_BOOST_MATH_CSTDFLOAT_CMATH_2014_02_15_HPP_
#define HYDRA_BOOST_MATH_CSTDFLOAT_CMATH_2014_02_15_HPP_

#include <hydra/detail/external/hydra_boost/math/cstdfloat/cstdfloat_types.hpp>
#include <hydra/detail/external/hydra_boost/math/cstdfloat/cstdfloat_limits.hpp>

#if defined(HYDRA_BOOST_CSTDFLOAT_HAS_INTERNAL_FLOAT128_T) && defined(HYDRA_BOOST_MATH_USE_FLOAT128) && !defined(HYDRA_BOOST_CSTDFLOAT_NO_LIBQUADMATH_SUPPORT)

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <type_traits>
#include <memory>
#include <hydra/detail/external/hydra_boost/math/tools/assert.hpp>
#include <hydra/detail/external/hydra_boost/math/tools/nothrow.hpp>
#include <hydra/detail/external/hydra_boost/math/tools/throw_exception.hpp>

#if defined(_WIN32) && defined(__GNUC__)
  // Several versions of Mingw and probably cygwin too have broken
  // libquadmath implementations that segfault as soon as you call
  // expq or any function that depends on it.
#define HYDRA_BOOST_CSTDFLOAT_BROKEN_FLOAT128_MATH_FUNCTIONS
#endif

// Here is a helper function used for raising the value of a given
// floating-point type to the power of n, where n has integral type.
namespace hydra_boost {
   namespace math {
      namespace cstdfloat {
         namespace detail {

            template<class float_type, class integer_type>
            inline float_type pown(const float_type& x, const integer_type p)
            {
               const bool isneg = (x < 0);
               const bool isnan = (x != x);
               const bool isinf = ((!isneg) ? bool(+x > (std::numeric_limits<float_type>::max)())
                  : bool(-x > (std::numeric_limits<float_type>::max)()));

               if (isnan) { return x; }

               if (isinf) { return std::numeric_limits<float_type>::quiet_NaN(); }

               const bool       x_is_neg = (x < 0);
               const float_type abs_x = (x_is_neg ? -x : x);

               if (p < static_cast<integer_type>(0))
               {
                  if (abs_x < (std::numeric_limits<float_type>::min)())
                  {
                     return (x_is_neg ? -std::numeric_limits<float_type>::infinity()
                        : +std::numeric_limits<float_type>::infinity());
                  }
                  else
                  {
                     return float_type(1) / pown(x, static_cast<integer_type>(-p));
                  }
               }

               if (p == static_cast<integer_type>(0))
               {
                  return float_type(1);
               }
               else
               {
                  if (p == static_cast<integer_type>(1)) { return x; }

                  if (abs_x > (std::numeric_limits<float_type>::max)())
                  {
                     return (x_is_neg ? -std::numeric_limits<float_type>::infinity()
                        : +std::numeric_limits<float_type>::infinity());
                  }

                  if      (p == static_cast<integer_type>(2)) { return  (x * x); }
                  else if (p == static_cast<integer_type>(3)) { return ((x * x) * x); }
                  else if (p == static_cast<integer_type>(4)) { const float_type x2 = (x * x); return (x2 * x2); }
                  else
                  {
                     // The variable xn stores the binary powers of x.
                     float_type result(((p % integer_type(2)) != integer_type(0)) ? x : float_type(1));
                     float_type xn(x);

                     integer_type p2 = p;

                     while (integer_type(p2 /= 2) != integer_type(0))
                     {
                        // Square xn for each binary power.
                        xn *= xn;

                        const bool has_binary_power = (integer_type(p2 % integer_type(2)) != integer_type(0));

                        if (has_binary_power)
                        {
                           // Multiply the result with each binary power contained in the exponent.
                           result *= xn;
                        }
                     }

                     return result;
                  }
               }
            }

         }
      }
   }
} // hydra_boost::math::cstdfloat::detail

// We will now define preprocessor symbols representing quadruple-precision <cmath> functions.
#if defined(__INTEL_COMPILER)
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_LDEXP  __ldexpq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_FREXP  __frexpq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_FABS   __fabsq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_FLOOR  __floorq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_CEIL   __ceilq
#if !defined(HYDRA_BOOST_CSTDFLOAT_FLOAT128_SQRT)
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_SQRT   __sqrtq
#endif
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_TRUNC  __truncq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_EXP    __expq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_EXPM1  __expm1q
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_POW    __powq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_LOG    __logq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_LOG10  __log10q
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_SIN    __sinq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_COS    __cosq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_TAN    __tanq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ASIN   __asinq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ACOS   __acosq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ATAN   __atanq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_SINH   __sinhq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_COSH   __coshq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_TANH   __tanhq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ASINH  __asinhq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ACOSH  __acoshq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ATANH  __atanhq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_FMOD   __fmodq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ATAN2  __atan2q
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_LGAMMA __lgammaq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_TGAMMA __tgammaq
//   begin more functions
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_REMAINDER   __remainderq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_REMQUO      __remquoq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_FMA         __fmaq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_FMAX        __fmaxq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_FMIN        __fminq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_FDIM        __fdimq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_NAN         __nanq
//#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_EXP2      __exp2q
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_LOG2        __log2q
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_LOG1P       __log1pq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_CBRT        __cbrtq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_HYPOT       __hypotq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ERF         __erfq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ERFC        __erfcq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_LLROUND     __llroundq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_LROUND      __lroundq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ROUND       __roundq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_NEARBYINT   __nearbyintq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_LLRINT      __llrintq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_LRINT       __lrintq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_RINT        __rintq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_MODF        __modfq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_SCALBLN     __scalblnq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_SCALBN      __scalbnq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ILOGB       __ilogbq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_LOGB        __logbq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_NEXTAFTER   __nextafterq
//#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_NEXTTOWARD  __nexttowardq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_COPYSIGN     __copysignq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_SIGNBIT      __signbitq
//#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_FPCLASSIFY __fpclassifyq
//#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISFINITE   __isfiniteq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISINF        __isinfq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISNAN        __isnanq
//#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISNORMAL   __isnormalq
//#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISGREATER  __isgreaterq
//#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISGREATEREQUAL __isgreaterequalq
//#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISLESS         __islessq
//#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISLESSEQUAL    __islessequalq
//#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISLESSGREATER  __islessgreaterq
//#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISUNORDERED    __isunorderedq
//   end more functions
#elif defined(__GNUC__)
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_LDEXP  ldexpq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_FREXP  frexpq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_FABS   fabsq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_FLOOR  floorq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_CEIL   ceilq
#if !defined(HYDRA_BOOST_CSTDFLOAT_FLOAT128_SQRT)
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_SQRT   sqrtq
#endif
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_TRUNC  truncq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_POW    powq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_LOG    logq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_LOG10  log10q
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_SIN    sinq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_COS    cosq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_TAN    tanq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ASIN   asinq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ACOS   acosq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ATAN   atanq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_FMOD   fmodq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ATAN2  atan2q
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_LGAMMA lgammaq
#if !defined(HYDRA_BOOST_CSTDFLOAT_BROKEN_FLOAT128_MATH_FUNCTIONS)
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_EXP    expq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_EXPM1  expm1q
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_SINH   sinhq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_COSH   coshq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_TANH   tanhq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ASINH  asinhq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ACOSH  acoshq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ATANH  atanhq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_TGAMMA tgammaq
#else // HYDRA_BOOST_CSTDFLOAT_BROKEN_FLOAT128_MATH_FUNCTIONS
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_EXP    expq_patch
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_SINH   sinhq_patch
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_COSH   coshq_patch
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_TANH   tanhq_patch
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ASINH  asinhq_patch
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ACOSH  acoshq_patch
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ATANH  atanhq_patch
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_TGAMMA tgammaq_patch
#endif // HYDRA_BOOST_CSTDFLOAT_BROKEN_FLOAT128_MATH_FUNCTIONS
//   begin more functions
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_REMAINDER   remainderq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_REMQUO      remquoq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_FMA         fmaq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_FMAX        fmaxq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_FMIN        fminq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_FDIM        fdimq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_NAN         nanq
//#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_EXP2      exp2q
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_LOG2        log2q
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_LOG1P       log1pq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_CBRT        cbrtq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_HYPOT       hypotq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ERF         erfq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ERFC        erfcq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_LLROUND     llroundq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_LROUND      lroundq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ROUND       roundq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_NEARBYINT   nearbyintq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_LLRINT      llrintq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_LRINT       lrintq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_RINT        rintq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_MODF        modfq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_SCALBLN     scalblnq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_SCALBN      scalbnq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ILOGB       ilogbq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_LOGB        logbq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_NEXTAFTER   nextafterq
//#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_NEXTTOWARD nexttowardq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_COPYSIGN    copysignq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_SIGNBIT     signbitq
//#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_FPCLASSIFY fpclassifyq
//#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISFINITE   isfiniteq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISINF        isinfq
#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISNAN        isnanq
//#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISNORMAL   isnormalq
//#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISGREATER  isgreaterq
//#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISGREATEREQUAL isgreaterequalq
//#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISLESS         islessq
//#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISLESSEQUAL    islessequalq
//#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISLESSGREATER  islessgreaterq
//#define HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISUNORDERED    isunorderedq
//   end more functions
#endif

// Implement quadruple-precision <cmath> functions in the namespace
// hydra_boost::math::cstdfloat::detail. Subsequently inject these into the
// std namespace via *using* directive.

// Begin with some forward function declarations. Also implement patches
// for compilers that have broken float128 exponential functions.

extern "C" int quadmath_snprintf(char*, std::size_t, const char*, ...) HYDRA_BOOST_MATH_NOTHROW;

extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_LDEXP(hydra_boost::math::cstdfloat::detail::float_internal128_t, int) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_FREXP(hydra_boost::math::cstdfloat::detail::float_internal128_t, int*) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_FABS(hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_FLOOR(hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_CEIL(hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_SQRT(hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_TRUNC(hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_POW(hydra_boost::math::cstdfloat::detail::float_internal128_t, hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_LOG(hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_LOG10(hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_SIN(hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_COS(hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_TAN(hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_ASIN(hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_ACOS(hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_ATAN(hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_FMOD(hydra_boost::math::cstdfloat::detail::float_internal128_t, hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_ATAN2(hydra_boost::math::cstdfloat::detail::float_internal128_t, hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_LGAMMA(hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;

//   begin more functions
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t  HYDRA_BOOST_CSTDFLOAT_FLOAT128_REMAINDER(hydra_boost::math::cstdfloat::detail::float_internal128_t, hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t  HYDRA_BOOST_CSTDFLOAT_FLOAT128_REMQUO(hydra_boost::math::cstdfloat::detail::float_internal128_t, hydra_boost::math::cstdfloat::detail::float_internal128_t, int*) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t  HYDRA_BOOST_CSTDFLOAT_FLOAT128_FMA(hydra_boost::math::cstdfloat::detail::float_internal128_t, hydra_boost::math::cstdfloat::detail::float_internal128_t, hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t  HYDRA_BOOST_CSTDFLOAT_FLOAT128_FMAX(hydra_boost::math::cstdfloat::detail::float_internal128_t, hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t  HYDRA_BOOST_CSTDFLOAT_FLOAT128_FMIN(hydra_boost::math::cstdfloat::detail::float_internal128_t, hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t  HYDRA_BOOST_CSTDFLOAT_FLOAT128_FDIM(hydra_boost::math::cstdfloat::detail::float_internal128_t, hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t  HYDRA_BOOST_CSTDFLOAT_FLOAT128_NAN(const char*) HYDRA_BOOST_MATH_NOTHROW;
//extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t  HYDRA_BOOST_CSTDFLOAT_FLOAT128_EXP2         (hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t  HYDRA_BOOST_CSTDFLOAT_FLOAT128_LOG2(hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t  HYDRA_BOOST_CSTDFLOAT_FLOAT128_LOG1P(hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t  HYDRA_BOOST_CSTDFLOAT_FLOAT128_CBRT(hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t  HYDRA_BOOST_CSTDFLOAT_FLOAT128_HYPOT(hydra_boost::math::cstdfloat::detail::float_internal128_t, hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t  HYDRA_BOOST_CSTDFLOAT_FLOAT128_ERF(hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t  HYDRA_BOOST_CSTDFLOAT_FLOAT128_ERFC(hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" long long int                                HYDRA_BOOST_CSTDFLOAT_FLOAT128_LLROUND(hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" long int                                   HYDRA_BOOST_CSTDFLOAT_FLOAT128_LROUND(hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t  HYDRA_BOOST_CSTDFLOAT_FLOAT128_ROUND(hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t  HYDRA_BOOST_CSTDFLOAT_FLOAT128_NEARBYINT(hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" long long int                                HYDRA_BOOST_CSTDFLOAT_FLOAT128_LLRINT(hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" long int                                   HYDRA_BOOST_CSTDFLOAT_FLOAT128_LRINT(hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t  HYDRA_BOOST_CSTDFLOAT_FLOAT128_RINT(hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t  HYDRA_BOOST_CSTDFLOAT_FLOAT128_MODF(hydra_boost::math::cstdfloat::detail::float_internal128_t, hydra_boost::math::cstdfloat::detail::float_internal128_t*) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t  HYDRA_BOOST_CSTDFLOAT_FLOAT128_SCALBLN(hydra_boost::math::cstdfloat::detail::float_internal128_t, long int) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t  HYDRA_BOOST_CSTDFLOAT_FLOAT128_SCALBN(hydra_boost::math::cstdfloat::detail::float_internal128_t, int) HYDRA_BOOST_MATH_NOTHROW;
extern "C" int                                      HYDRA_BOOST_CSTDFLOAT_FLOAT128_ILOGB(hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t  HYDRA_BOOST_CSTDFLOAT_FLOAT128_LOGB(hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t  HYDRA_BOOST_CSTDFLOAT_FLOAT128_NEXTAFTER(hydra_boost::math::cstdfloat::detail::float_internal128_t, hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
//extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t  HYDRA_BOOST_CSTDFLOAT_FLOAT128_NEXTTOWARD   (hydra_boost::math::cstdfloat::detail::float_internal128_t, hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t  HYDRA_BOOST_CSTDFLOAT_FLOAT128_COPYSIGN(hydra_boost::math::cstdfloat::detail::float_internal128_t, hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" int                                                  HYDRA_BOOST_CSTDFLOAT_FLOAT128_SIGNBIT(hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
//extern "C" int                                                HYDRA_BOOST_CSTDFLOAT_FLOAT128_FPCLASSIFY   (hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
//extern "C" int                                                HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISFINITE      (hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" int                                                  HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISINF(hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
extern "C" int                                                  HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISNAN(hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
//extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t  HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISNORMAL   (hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
//extern "C" int                                                HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISGREATER   (hydra_boost::math::cstdfloat::detail::float_internal128_t, hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
//extern "C" int                                                HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISGREATEREQUAL(hydra_boost::math::cstdfloat::detail::float_internal128_t, hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
//extern "C" int                                                HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISLESS      (hydra_boost::math::cstdfloat::detail::float_internal128_t, hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
//extern "C" int                                                HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISLESSEQUAL   (hydra_boost::math::cstdfloat::detail::float_internal128_t, hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
//extern "C" int                                                HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISLESSGREATER(hydra_boost::math::cstdfloat::detail::float_internal128_t, hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
//extern "C" int                                                HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISUNORDERED   (hydra_boost::math::cstdfloat::detail::float_internal128_t, hydra_boost::math::cstdfloat::detail::float_internal128_t) HYDRA_BOOST_MATH_NOTHROW;
 //   end more functions

#if !defined(HYDRA_BOOST_CSTDFLOAT_BROKEN_FLOAT128_MATH_FUNCTIONS)

extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_EXP(hydra_boost::math::cstdfloat::detail::float_internal128_t x) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_EXPM1(hydra_boost::math::cstdfloat::detail::float_internal128_t x) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_SINH(hydra_boost::math::cstdfloat::detail::float_internal128_t x) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_COSH(hydra_boost::math::cstdfloat::detail::float_internal128_t x) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_TANH(hydra_boost::math::cstdfloat::detail::float_internal128_t x) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_ASINH(hydra_boost::math::cstdfloat::detail::float_internal128_t x) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_ACOSH(hydra_boost::math::cstdfloat::detail::float_internal128_t x) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_ATANH(hydra_boost::math::cstdfloat::detail::float_internal128_t x) HYDRA_BOOST_MATH_NOTHROW;
extern "C" hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_TGAMMA(hydra_boost::math::cstdfloat::detail::float_internal128_t x) HYDRA_BOOST_MATH_NOTHROW;
 
#else // HYDRA_BOOST_CSTDFLOAT_BROKEN_FLOAT128_MATH_FUNCTIONS

// Forward declaration of the patched exponent function, exp(x).
inline hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_EXP(hydra_boost::math::cstdfloat::detail::float_internal128_t x);

inline hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_EXPM1(hydra_boost::math::cstdfloat::detail::float_internal128_t x)
{
   // Compute exp(x) - 1 for x small.

   // Use an order-12 Pade approximation of the exponential function.
   // PadeApproximant[Exp[x] - 1, {x, 0, 12, 12}].

   typedef hydra_boost::math::cstdfloat::detail::float_internal128_t float_type;

   float_type sum;

   if (x > HYDRA_BOOST_FLOAT128_C(0.693147180559945309417232121458176568075500134360255))
   {
      sum = ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_EXP(x) - float_type(1);
   }
   else
   {
      const float_type x2 = (x * x);

      const float_type top = (((((  float_type(HYDRA_BOOST_FLOAT128_C(2.4087176110456818621091195109360728010934088788572E-13))  * x2
                                  + float_type(HYDRA_BOOST_FLOAT128_C(9.2735628025258751691201101171038802842096241836000E-10))) * x2
                                  + float_type(HYDRA_BOOST_FLOAT128_C(9.0806726962333369656024118266681195742980640005812E-07))) * x2
                                  + float_type(HYDRA_BOOST_FLOAT128_C(3.1055900621118012422360248447204968944099378881988E-04))) * x2
                                  + float_type(HYDRA_BOOST_FLOAT128_C(3.6231884057971014492753623188405797101449275362319E-02))) * x2
                                  + float_type(HYDRA_BOOST_FLOAT128_C(1.00000000000000000000000000000000000000000000000000000)))
                                  ;

      const float_type bot = ((((((((((((  float_type(HYDRA_BOOST_FLOAT128_C(+7.7202487533515444298369215094104897470942592271063E-16))  * x
                                         + float_type(HYDRA_BOOST_FLOAT128_C(-1.2043588055228409310545597554680364005467044394286E-13))) * x
                                         + float_type(HYDRA_BOOST_FLOAT128_C(+9.2735628025258751691201101171038802842096241836000E-12))) * x
                                         + float_type(HYDRA_BOOST_FLOAT128_C(-4.6367814012629375845600550585519401421048120918000E-10))) * x
                                         + float_type(HYDRA_BOOST_FLOAT128_C(+1.6692413044546575304416198210786984511577323530480E-08))) * x
                                         + float_type(HYDRA_BOOST_FLOAT128_C(-4.5403363481166684828012059133340597871490320002906E-07))) * x
                                         + float_type(HYDRA_BOOST_FLOAT128_C(+9.5347063310450038138825324180015255530129672006102E-06))) * x
                                         + float_type(HYDRA_BOOST_FLOAT128_C(-1.5527950310559006211180124223602484472049689440994E-04))) * x
                                         + float_type(HYDRA_BOOST_FLOAT128_C(+1.9409937888198757763975155279503105590062111801242E-03))) * x
                                         + float_type(HYDRA_BOOST_FLOAT128_C(-1.8115942028985507246376811594202898550724637681159E-02))) * x
                                         + float_type(HYDRA_BOOST_FLOAT128_C(+1.1956521739130434782608695652173913043478260869565E-01))) * x
                                         + float_type(HYDRA_BOOST_FLOAT128_C(-0.50000000000000000000000000000000000000000000000000000))) * x
                                         + float_type(HYDRA_BOOST_FLOAT128_C(+1.00000000000000000000000000000000000000000000000000000)))
                                         ;

      sum = (x * top) / bot;
   }

   return sum;
}
inline hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_EXP(hydra_boost::math::cstdfloat::detail::float_internal128_t x)
{
   // Patch the expq() function for a subset of broken GCC compilers
   // like GCC 4.7, 4.8 on MinGW.

   typedef hydra_boost::math::cstdfloat::detail::float_internal128_t float_type;

   // Scale the argument x to the range (-ln2 < x < ln2).
   constexpr float_type one_over_ln2 = float_type(HYDRA_BOOST_FLOAT128_C(1.44269504088896340735992468100189213742664595415299));
   const float_type x_over_ln2 = x * one_over_ln2;

   int n;

   if (x != x)
   {
      // The argument is NaN.
      return std::numeric_limits<float_type>::quiet_NaN();
   }
   else if (::HYDRA_BOOST_CSTDFLOAT_FLOAT128_FABS(x) > HYDRA_BOOST_FLOAT128_C(+0.693147180559945309417232121458176568075500134360255))
   {
      // The absolute value of the argument exceeds ln2.
      n = static_cast<int>(::HYDRA_BOOST_CSTDFLOAT_FLOAT128_FLOOR(x_over_ln2));
   }
   else if (::HYDRA_BOOST_CSTDFLOAT_FLOAT128_FABS(x) < HYDRA_BOOST_FLOAT128_C(+0.693147180559945309417232121458176568075500134360255))
   {
      // The absolute value of the argument is less than ln2.
      n = 0;
   }
   else
   {
      // The absolute value of the argument is exactly equal to ln2 (in the sense of floating-point equality).
      return float_type(2);
   }

   // Check if the argument is very near an integer.
   const float_type floor_of_x = ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_FLOOR(x);

   if (::HYDRA_BOOST_CSTDFLOAT_FLOAT128_FABS(x - floor_of_x) < float_type(HYDRA_BOOST_CSTDFLOAT_FLOAT128_EPS))
   {
      // Return e^n for arguments very near an integer.
      return hydra_boost::math::cstdfloat::detail::pown(HYDRA_BOOST_FLOAT128_C(2.71828182845904523536028747135266249775724709369996), static_cast<std::int_fast32_t>(floor_of_x));
   }

   // Compute the scaled argument alpha.
   const float_type alpha = x - (n * HYDRA_BOOST_FLOAT128_C(0.693147180559945309417232121458176568075500134360255));

   // Compute the polynomial approximation of expm1(alpha) and add to it
   // in order to obtain the scaled result.
   const float_type scaled_result = ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_EXPM1(alpha) + float_type(1);

   // Rescale the result and return it.
   return scaled_result * hydra_boost::math::cstdfloat::detail::pown(float_type(2), n);
}
inline hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_SINH(hydra_boost::math::cstdfloat::detail::float_internal128_t x)
{
   // Patch the sinhq() function for a subset of broken GCC compilers
   // like GCC 4.7, 4.8 on MinGW.
   typedef hydra_boost::math::cstdfloat::detail::float_internal128_t float_type;

   // Here, we use the following:
   // Set: ex  = exp(x)
   // Set: em1 = expm1(x)
   // Then
   // sinh(x) = (ex - 1/ex) / 2         ; for |x| >= 1
   // sinh(x) = (2em1 + em1^2) / (2ex)  ; for |x| < 1

   const float_type ex = ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_EXP(x);

   if (::HYDRA_BOOST_CSTDFLOAT_FLOAT128_FABS(x) < float_type(+1))
   {
      const float_type em1 = ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_EXPM1(x);

      return ((em1 * 2) + (em1 * em1)) / (ex * 2);
   }
   else
   {
      return (ex - (float_type(1) / ex)) / 2;
   }
}
inline hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_COSH(hydra_boost::math::cstdfloat::detail::float_internal128_t x)
{
   // Patch the coshq() function for a subset of broken GCC compilers
   // like GCC 4.7, 4.8 on MinGW.
   typedef hydra_boost::math::cstdfloat::detail::float_internal128_t float_type;
   const float_type ex = ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_EXP(x);
   return (ex + (float_type(1) / ex)) / 2;
}
inline hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_TANH(hydra_boost::math::cstdfloat::detail::float_internal128_t x)
{
   // Patch the tanhq() function for a subset of broken GCC compilers
   // like GCC 4.7, 4.8 on MinGW.
   typedef hydra_boost::math::cstdfloat::detail::float_internal128_t float_type;
   const float_type ex_plus = ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_EXP(x);
   const float_type ex_minus = (float_type(1) / ex_plus);
   return (ex_plus - ex_minus) / (ex_plus + ex_minus);
}
inline hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_ASINH(hydra_boost::math::cstdfloat::detail::float_internal128_t x) HYDRA_BOOST_MATH_NOTHROW
{
   // Patch the asinh() function since quadmath does not have it.
   typedef hydra_boost::math::cstdfloat::detail::float_internal128_t float_type;
   return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_LOG(x + ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_SQRT((x * x) + float_type(1)));
}
inline hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_ACOSH(hydra_boost::math::cstdfloat::detail::float_internal128_t x) HYDRA_BOOST_MATH_NOTHROW
{
   // Patch the acosh() function since quadmath does not have it.
   typedef hydra_boost::math::cstdfloat::detail::float_internal128_t float_type;
   const float_type zp(x + float_type(1));
   const float_type zm(x - float_type(1));

   return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_LOG(x + (zp * ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_SQRT(zm / zp)));
}
inline hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_ATANH(hydra_boost::math::cstdfloat::detail::float_internal128_t x) HYDRA_BOOST_MATH_NOTHROW
{
   // Patch the atanh() function since quadmath does not have it.
   typedef hydra_boost::math::cstdfloat::detail::float_internal128_t float_type;
   return (::HYDRA_BOOST_CSTDFLOAT_FLOAT128_LOG(float_type(1) + x)
      - ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_LOG(float_type(1) - x)) / 2;
}
inline hydra_boost::math::cstdfloat::detail::float_internal128_t HYDRA_BOOST_CSTDFLOAT_FLOAT128_TGAMMA(hydra_boost::math::cstdfloat::detail::float_internal128_t x) HYDRA_BOOST_MATH_NOTHROW
{
   // Patch the tgammaq() function for a subset of broken GCC compilers
   // like GCC 4.7, 4.8 on MinGW.
   typedef hydra_boost::math::cstdfloat::detail::float_internal128_t float_type;

   if (x > float_type(0))
   {
      return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_EXP(::HYDRA_BOOST_CSTDFLOAT_FLOAT128_LGAMMA(x));
   }
   else if (x < float_type(0))
   {
      // For x < 0, compute tgamma(-x) and use the reflection formula.
      const float_type positive_x = -x;
      float_type gamma_value = ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_TGAMMA(positive_x);
      const float_type floor_of_positive_x = ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_FLOOR(positive_x);

      // Take the reflection checks (slightly adapted) from <hydra/detail/external/hydra_boost/math/gamma.hpp>.
      const bool floor_of_z_is_equal_to_z = (positive_x == ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_FLOOR(positive_x));

      constexpr float_type my_pi = HYDRA_BOOST_FLOAT128_C(3.14159265358979323846264338327950288419716939937511);

      if (floor_of_z_is_equal_to_z)
      {
         const bool is_odd = ((std::int32_t(floor_of_positive_x) % std::int32_t(2)) != std::int32_t(0));

         return (is_odd ? -std::numeric_limits<float_type>::infinity()
            : +std::numeric_limits<float_type>::infinity());
      }

      const float_type sinpx_value = x * ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_SIN(my_pi * x);

      gamma_value *= sinpx_value;

      const bool result_is_too_large_to_represent = ((::HYDRA_BOOST_CSTDFLOAT_FLOAT128_FABS(gamma_value) < float_type(1))
         && (((std::numeric_limits<float_type>::max)() * ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_FABS(gamma_value)) < my_pi));

      if (result_is_too_large_to_represent)
      {
         const bool is_odd = ((std::int32_t(floor_of_positive_x) % std::int32_t(2)) != std::int32_t(0));

         return (is_odd ? -std::numeric_limits<float_type>::infinity()
            : +std::numeric_limits<float_type>::infinity());
      }

      gamma_value = -my_pi / gamma_value;

      if ((gamma_value > float_type(0)) || (gamma_value < float_type(0)))
      {
         return gamma_value;
      }
      else
      {
         // The value of gamma is too small to represent. Return 0.0 here.
         return float_type(0);
      }
   }
   else
   {
      // Gamma of zero is complex infinity. Return NaN here.
      return std::numeric_limits<float_type>::quiet_NaN();
   }
}
#endif // HYDRA_BOOST_CSTDFLOAT_BROKEN_FLOAT128_MATH_FUNCTIONS

// Define the quadruple-precision <cmath> functions in the namespace hydra_boost::math::cstdfloat::detail.

namespace hydra_boost {
   namespace math {
      namespace cstdfloat {
         namespace detail {
            inline   hydra_boost::math::cstdfloat::detail::float_internal128_t ldexp(hydra_boost::math::cstdfloat::detail::float_internal128_t x, int n) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_LDEXP(x, n); }
            inline   hydra_boost::math::cstdfloat::detail::float_internal128_t frexp(hydra_boost::math::cstdfloat::detail::float_internal128_t x, int* pn) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_FREXP(x, pn); }
            inline   hydra_boost::math::cstdfloat::detail::float_internal128_t fabs(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_FABS(x); }
            inline   hydra_boost::math::cstdfloat::detail::float_internal128_t abs(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_FABS(x); }
            inline   hydra_boost::math::cstdfloat::detail::float_internal128_t floor(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_FLOOR(x); }
            inline   hydra_boost::math::cstdfloat::detail::float_internal128_t ceil(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_CEIL(x); }
            inline   hydra_boost::math::cstdfloat::detail::float_internal128_t sqrt(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_SQRT(x); }
            inline   hydra_boost::math::cstdfloat::detail::float_internal128_t trunc(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_TRUNC(x); }
            inline   hydra_boost::math::cstdfloat::detail::float_internal128_t exp(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_EXP(x); }
            inline   hydra_boost::math::cstdfloat::detail::float_internal128_t expm1(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_EXPM1(x); }
            inline   hydra_boost::math::cstdfloat::detail::float_internal128_t pow(hydra_boost::math::cstdfloat::detail::float_internal128_t x, hydra_boost::math::cstdfloat::detail::float_internal128_t a) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_POW(x, a); }
            inline   hydra_boost::math::cstdfloat::detail::float_internal128_t pow(hydra_boost::math::cstdfloat::detail::float_internal128_t x, int a) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_POW(x, hydra_boost::math::cstdfloat::detail::float_internal128_t(a)); }
            inline   hydra_boost::math::cstdfloat::detail::float_internal128_t log(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_LOG(x); }
            inline   hydra_boost::math::cstdfloat::detail::float_internal128_t log10(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_LOG10(x); }
            inline   hydra_boost::math::cstdfloat::detail::float_internal128_t sin(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_SIN(x); }
            inline   hydra_boost::math::cstdfloat::detail::float_internal128_t cos(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_COS(x); }
            inline   hydra_boost::math::cstdfloat::detail::float_internal128_t tan(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_TAN(x); }
            inline   hydra_boost::math::cstdfloat::detail::float_internal128_t asin(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_ASIN(x); }
            inline   hydra_boost::math::cstdfloat::detail::float_internal128_t acos(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_ACOS(x); }
            inline   hydra_boost::math::cstdfloat::detail::float_internal128_t atan(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_ATAN(x); }
            inline   hydra_boost::math::cstdfloat::detail::float_internal128_t sinh(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_SINH(x); }
            inline   hydra_boost::math::cstdfloat::detail::float_internal128_t cosh(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_COSH(x); }
            inline   hydra_boost::math::cstdfloat::detail::float_internal128_t tanh(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_TANH(x); }
            inline   hydra_boost::math::cstdfloat::detail::float_internal128_t asinh(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_ASINH(x); }
            inline   hydra_boost::math::cstdfloat::detail::float_internal128_t acosh(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_ACOSH(x); }
            inline   hydra_boost::math::cstdfloat::detail::float_internal128_t atanh(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_ATANH(x); }
            inline   hydra_boost::math::cstdfloat::detail::float_internal128_t fmod(hydra_boost::math::cstdfloat::detail::float_internal128_t a, hydra_boost::math::cstdfloat::detail::float_internal128_t b) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_FMOD(a, b); }
            inline   hydra_boost::math::cstdfloat::detail::float_internal128_t atan2(hydra_boost::math::cstdfloat::detail::float_internal128_t y, hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_ATAN2(y, x); }
            inline   hydra_boost::math::cstdfloat::detail::float_internal128_t lgamma(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_LGAMMA(x); }
            inline   hydra_boost::math::cstdfloat::detail::float_internal128_t tgamma(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_TGAMMA(x); }
            //   begin more functions
            inline hydra_boost::math::cstdfloat::detail::float_internal128_t  remainder(hydra_boost::math::cstdfloat::detail::float_internal128_t x, hydra_boost::math::cstdfloat::detail::float_internal128_t y) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_REMAINDER(x, y); }
            inline hydra_boost::math::cstdfloat::detail::float_internal128_t  remquo(hydra_boost::math::cstdfloat::detail::float_internal128_t x, hydra_boost::math::cstdfloat::detail::float_internal128_t y, int* z) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_REMQUO(x, y, z); }
            inline hydra_boost::math::cstdfloat::detail::float_internal128_t  fma(hydra_boost::math::cstdfloat::detail::float_internal128_t x, hydra_boost::math::cstdfloat::detail::float_internal128_t y, hydra_boost::math::cstdfloat::detail::float_internal128_t z) { return HYDRA_BOOST_CSTDFLOAT_FLOAT128_FMA(x, y, z); }

            inline hydra_boost::math::cstdfloat::detail::float_internal128_t  fmax(hydra_boost::math::cstdfloat::detail::float_internal128_t x, hydra_boost::math::cstdfloat::detail::float_internal128_t y) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_FMAX(x, y); }
            template <class T>
            inline typename std::enable_if<
               std::is_convertible<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value
               && !std::is_same<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value, hydra_boost::math::cstdfloat::detail::float_internal128_t>::type
               fmax(hydra_boost::math::cstdfloat::detail::float_internal128_t x, T y) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_FMAX(x, y); }
            template <class T>
            inline typename std::enable_if<
               std::is_convertible<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value
               && !std::is_same<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value, hydra_boost::math::cstdfloat::detail::float_internal128_t>::type
               fmax(T x, hydra_boost::math::cstdfloat::detail::float_internal128_t y) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_FMAX(x, y); }
            inline hydra_boost::math::cstdfloat::detail::float_internal128_t  fmin(hydra_boost::math::cstdfloat::detail::float_internal128_t x, hydra_boost::math::cstdfloat::detail::float_internal128_t y) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_FMIN(x, y); }
            template <class T>
            inline typename std::enable_if<
               std::is_convertible<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value
               && !std::is_same<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value, hydra_boost::math::cstdfloat::detail::float_internal128_t>::type
               fmin(hydra_boost::math::cstdfloat::detail::float_internal128_t x, T y) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_FMIN(x, y); }
            template <class T>
            inline typename std::enable_if<
               std::is_convertible<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value
               && !std::is_same<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value, hydra_boost::math::cstdfloat::detail::float_internal128_t>::type
               fmin(T x, hydra_boost::math::cstdfloat::detail::float_internal128_t y) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_FMIN(x, y); }

            inline hydra_boost::math::cstdfloat::detail::float_internal128_t  fdim(hydra_boost::math::cstdfloat::detail::float_internal128_t x, hydra_boost::math::cstdfloat::detail::float_internal128_t y) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_FDIM(x, y); }
            inline hydra_boost::math::cstdfloat::detail::float_internal128_t  nanq(const char* x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_NAN(x); }
            inline hydra_boost::math::cstdfloat::detail::float_internal128_t  exp2(hydra_boost::math::cstdfloat::detail::float_internal128_t x)
            {
               return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_POW(hydra_boost::math::cstdfloat::detail::float_internal128_t(2), x);
            }
            inline hydra_boost::math::cstdfloat::detail::float_internal128_t  log2(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_LOG2(x); }
            inline hydra_boost::math::cstdfloat::detail::float_internal128_t  log1p(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_LOG1P(x); }
            inline hydra_boost::math::cstdfloat::detail::float_internal128_t  cbrt(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_CBRT(x); }
            inline hydra_boost::math::cstdfloat::detail::float_internal128_t  hypot(hydra_boost::math::cstdfloat::detail::float_internal128_t x, hydra_boost::math::cstdfloat::detail::float_internal128_t y, hydra_boost::math::cstdfloat::detail::float_internal128_t z) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_SQRT(x*x + y * y + z * z); }
            inline hydra_boost::math::cstdfloat::detail::float_internal128_t  hypot(hydra_boost::math::cstdfloat::detail::float_internal128_t x, hydra_boost::math::cstdfloat::detail::float_internal128_t y) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_HYPOT(x, y); }
            template <class T>
            inline typename std::enable_if<
               std::is_convertible<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value
               && !std::is_same<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value, hydra_boost::math::cstdfloat::detail::float_internal128_t>::type
               hypot(hydra_boost::math::cstdfloat::detail::float_internal128_t x, T y) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_HYPOT(x, y); }
            template <class T>
            inline typename std::enable_if<
               std::is_convertible<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value
               && !std::is_same<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value, hydra_boost::math::cstdfloat::detail::float_internal128_t>::type
               hypot(T x, hydra_boost::math::cstdfloat::detail::float_internal128_t y) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_HYPOT(x, y); }


            inline hydra_boost::math::cstdfloat::detail::float_internal128_t  erf(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_ERF(x); }
            inline hydra_boost::math::cstdfloat::detail::float_internal128_t  erfc(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_ERFC(x); }
            inline long long int                                        llround(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_LLROUND(x); }
            inline long int                                             lround(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_LROUND(x); }
            inline hydra_boost::math::cstdfloat::detail::float_internal128_t  round(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_ROUND(x); }
            inline hydra_boost::math::cstdfloat::detail::float_internal128_t  nearbyint(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_NEARBYINT(x); }
            inline long long int                                        llrint(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_LLRINT(x); }
            inline long int                                             lrint(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_LRINT(x); }
            inline hydra_boost::math::cstdfloat::detail::float_internal128_t  rint(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_RINT(x); }
            inline hydra_boost::math::cstdfloat::detail::float_internal128_t  modf(hydra_boost::math::cstdfloat::detail::float_internal128_t x, hydra_boost::math::cstdfloat::detail::float_internal128_t* y) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_MODF(x, y); }
            inline hydra_boost::math::cstdfloat::detail::float_internal128_t  scalbln(hydra_boost::math::cstdfloat::detail::float_internal128_t x, long int y) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_SCALBLN(x, y); }
            inline hydra_boost::math::cstdfloat::detail::float_internal128_t  scalbn(hydra_boost::math::cstdfloat::detail::float_internal128_t x, int y) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_SCALBN(x, y); }
            inline int                                                  ilogb(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_ILOGB(x); }
            inline hydra_boost::math::cstdfloat::detail::float_internal128_t  logb(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_LOGB(x); }
            inline hydra_boost::math::cstdfloat::detail::float_internal128_t  nextafter(hydra_boost::math::cstdfloat::detail::float_internal128_t x, hydra_boost::math::cstdfloat::detail::float_internal128_t y) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_NEXTAFTER(x, y); }
            inline hydra_boost::math::cstdfloat::detail::float_internal128_t  nexttoward(hydra_boost::math::cstdfloat::detail::float_internal128_t x, hydra_boost::math::cstdfloat::detail::float_internal128_t y) { return -(::HYDRA_BOOST_CSTDFLOAT_FLOAT128_NEXTAFTER(-x, -y)); }
            inline hydra_boost::math::cstdfloat::detail::float_internal128_t  copysign   HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(hydra_boost::math::cstdfloat::detail::float_internal128_t x, hydra_boost::math::cstdfloat::detail::float_internal128_t y) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_COPYSIGN(x, y); }
            inline bool                                                 signbit   HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_SIGNBIT(x); }
            inline int                                                  fpclassify HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(hydra_boost::math::cstdfloat::detail::float_internal128_t x)
            {
               if (::HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISNAN(x))
                  return FP_NAN;
               else if (::HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISINF(x))
                  return FP_INFINITE;
               else if (x == HYDRA_BOOST_FLOAT128_C(0.0))
                  return FP_ZERO;

               if (::HYDRA_BOOST_CSTDFLOAT_FLOAT128_FABS(x) < HYDRA_BOOST_CSTDFLOAT_FLOAT128_MIN)
                  return FP_SUBNORMAL;
               else
                  return FP_NORMAL;
            }
            inline bool                                      isfinite   HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(hydra_boost::math::cstdfloat::detail::float_internal128_t x)
            {
               return !::HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISNAN(x) && !::HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISINF(x);
            }
            inline bool                                      isinf      HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISINF(x); }
            inline bool                                      isnan      HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISNAN(x); }
            inline bool                                      isnormal   HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(hydra_boost::math::cstdfloat::detail::float_internal128_t x) { return hydra_boost::math::cstdfloat::detail::fpclassify HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x) == FP_NORMAL; }
            inline bool                                      isgreater      HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(hydra_boost::math::cstdfloat::detail::float_internal128_t x, hydra_boost::math::cstdfloat::detail::float_internal128_t y)
            {
               if (isnan HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x) || isnan HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(y))
                  return false;
               return x > y;
            }
            template <class T>
            inline typename std::enable_if<
               std::is_convertible<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value
               && !std::is_same<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value, hydra_boost::math::cstdfloat::detail::float_internal128_t>::type
               isgreater HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(hydra_boost::math::cstdfloat::detail::float_internal128_t x, T y) { return isgreater HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, (hydra_boost::math::cstdfloat::detail::float_internal128_t)y); }
            template <class T>
            inline typename std::enable_if<
               std::is_convertible<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value
               && !std::is_same<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value, hydra_boost::math::cstdfloat::detail::float_internal128_t>::type
               isgreater HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T x, hydra_boost::math::cstdfloat::detail::float_internal128_t y) { return isgreater HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION((hydra_boost::math::cstdfloat::detail::float_internal128_t)x, y); }

            inline bool                                      isgreaterequal HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(hydra_boost::math::cstdfloat::detail::float_internal128_t x, hydra_boost::math::cstdfloat::detail::float_internal128_t y)
            {
               if (isnan HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x) || isnan HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(y))
                  return false;
               return x >= y;
            }
            template <class T>
            inline typename std::enable_if<
               std::is_convertible<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value
               && !std::is_same<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value, hydra_boost::math::cstdfloat::detail::float_internal128_t>::type
               isgreaterequal HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(hydra_boost::math::cstdfloat::detail::float_internal128_t x, T y) { return isgreaterequal HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, (hydra_boost::math::cstdfloat::detail::float_internal128_t)y); }
            template <class T>
            inline typename std::enable_if<
               std::is_convertible<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value
               && !std::is_same<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value, hydra_boost::math::cstdfloat::detail::float_internal128_t>::type
               isgreaterequal HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T x, hydra_boost::math::cstdfloat::detail::float_internal128_t y) { return isgreaterequal HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION((hydra_boost::math::cstdfloat::detail::float_internal128_t)x, y); }

            inline bool                                      isless      HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(hydra_boost::math::cstdfloat::detail::float_internal128_t x, hydra_boost::math::cstdfloat::detail::float_internal128_t y)
            {
               if (isnan HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x) || isnan HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(y))
                  return false;
               return x < y;
            }
            template <class T>
            inline typename std::enable_if<
               std::is_convertible<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value
               && !std::is_same<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value, hydra_boost::math::cstdfloat::detail::float_internal128_t>::type
               isless HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(hydra_boost::math::cstdfloat::detail::float_internal128_t x, T y) { return isless HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, (hydra_boost::math::cstdfloat::detail::float_internal128_t)y); }
            template <class T>
            inline typename std::enable_if<
               std::is_convertible<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value
               && !std::is_same<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value, hydra_boost::math::cstdfloat::detail::float_internal128_t>::type
               isless HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T x, hydra_boost::math::cstdfloat::detail::float_internal128_t y) { return isless HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION((hydra_boost::math::cstdfloat::detail::float_internal128_t)x, y); }


            inline bool                                      islessequal   HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(hydra_boost::math::cstdfloat::detail::float_internal128_t x, hydra_boost::math::cstdfloat::detail::float_internal128_t y)
            {
               if (isnan HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x) || isnan HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(y))
                  return false;
               return x <= y;
            }
            template <class T>
            inline typename std::enable_if<
               std::is_convertible<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value
               && !std::is_same<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value, hydra_boost::math::cstdfloat::detail::float_internal128_t>::type
               islessequal HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(hydra_boost::math::cstdfloat::detail::float_internal128_t x, T y) { return islessequal HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, (hydra_boost::math::cstdfloat::detail::float_internal128_t)y); }
            template <class T>
            inline typename std::enable_if<
               std::is_convertible<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value
               && !std::is_same<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value, hydra_boost::math::cstdfloat::detail::float_internal128_t>::type
               islessequal HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T x, hydra_boost::math::cstdfloat::detail::float_internal128_t y) { return islessequal HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION((hydra_boost::math::cstdfloat::detail::float_internal128_t)x, y); }


            inline bool                                      islessgreater   HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(hydra_boost::math::cstdfloat::detail::float_internal128_t x, hydra_boost::math::cstdfloat::detail::float_internal128_t y)
            {
               if (isnan HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x) || isnan HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(y))
                  return false;
               return (x < y) || (x > y);
            }
            template <class T>
            inline typename std::enable_if<
               std::is_convertible<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value
               && !std::is_same<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value, hydra_boost::math::cstdfloat::detail::float_internal128_t>::type
               islessgreater HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(hydra_boost::math::cstdfloat::detail::float_internal128_t x, T y) { return islessgreater HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, (hydra_boost::math::cstdfloat::detail::float_internal128_t)y); }
            template <class T>
            inline typename std::enable_if<
               std::is_convertible<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value
               && !std::is_same<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value, hydra_boost::math::cstdfloat::detail::float_internal128_t>::type
               islessgreater HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T x, hydra_boost::math::cstdfloat::detail::float_internal128_t y) { return islessgreater HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION((hydra_boost::math::cstdfloat::detail::float_internal128_t)x, y); }


            inline bool                                      isunordered   HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(hydra_boost::math::cstdfloat::detail::float_internal128_t x, hydra_boost::math::cstdfloat::detail::float_internal128_t y) { return ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISNAN(x) || ::HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISNAN(y); }
            template <class T>
            inline typename std::enable_if<
               std::is_convertible<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value
               && !std::is_same<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value, hydra_boost::math::cstdfloat::detail::float_internal128_t>::type
               isunordered HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(hydra_boost::math::cstdfloat::detail::float_internal128_t x, T y) { return isunordered HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(x, (hydra_boost::math::cstdfloat::detail::float_internal128_t)y); }
            template <class T>
            inline typename std::enable_if<
               std::is_convertible<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value
               && !std::is_same<T, hydra_boost::math::cstdfloat::detail::float_internal128_t>::value, hydra_boost::math::cstdfloat::detail::float_internal128_t>::type
               isunordered HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T x, hydra_boost::math::cstdfloat::detail::float_internal128_t y) { return isunordered HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION((hydra_boost::math::cstdfloat::detail::float_internal128_t)x, y); }


            //   end more functions
         }
      }
   }
} // hydra_boost::math::cstdfloat::detail

// We will now inject the quadruple-precision <cmath> functions
// into the std namespace. This is done via *using* directive.
namespace std
{
   using hydra_boost::math::cstdfloat::detail::ldexp;
   using hydra_boost::math::cstdfloat::detail::frexp;
   using hydra_boost::math::cstdfloat::detail::fabs;

#if !(defined(_GLIBCXX_USE_FLOAT128) && defined(__GNUC__) && (__GNUC__ >= 7))
#if (defined(__clang__) && !(!defined(__STRICT_ANSI__) && defined(_GLIBCXX_USE_FLOAT128))) || (__GNUC__ <= 6 && !defined(__clang__)) 
   // workaround for clang using libstdc++ and old GCC
   using hydra_boost::math::cstdfloat::detail::abs;
#endif
#endif

   using hydra_boost::math::cstdfloat::detail::floor;
   using hydra_boost::math::cstdfloat::detail::ceil;
   using hydra_boost::math::cstdfloat::detail::sqrt;
   using hydra_boost::math::cstdfloat::detail::trunc;
   using hydra_boost::math::cstdfloat::detail::exp;
   using hydra_boost::math::cstdfloat::detail::expm1;
   using hydra_boost::math::cstdfloat::detail::pow;
   using hydra_boost::math::cstdfloat::detail::log;
   using hydra_boost::math::cstdfloat::detail::log10;
   using hydra_boost::math::cstdfloat::detail::sin;
   using hydra_boost::math::cstdfloat::detail::cos;
   using hydra_boost::math::cstdfloat::detail::tan;
   using hydra_boost::math::cstdfloat::detail::asin;
   using hydra_boost::math::cstdfloat::detail::acos;
   using hydra_boost::math::cstdfloat::detail::atan;
   using hydra_boost::math::cstdfloat::detail::sinh;
   using hydra_boost::math::cstdfloat::detail::cosh;
   using hydra_boost::math::cstdfloat::detail::tanh;
   using hydra_boost::math::cstdfloat::detail::asinh;
   using hydra_boost::math::cstdfloat::detail::acosh;
   using hydra_boost::math::cstdfloat::detail::atanh;
   using hydra_boost::math::cstdfloat::detail::fmod;
   using hydra_boost::math::cstdfloat::detail::atan2;
   using hydra_boost::math::cstdfloat::detail::lgamma;
   using hydra_boost::math::cstdfloat::detail::tgamma;

   //   begin more functions
   using hydra_boost::math::cstdfloat::detail::remainder;
   using hydra_boost::math::cstdfloat::detail::remquo;
   using hydra_boost::math::cstdfloat::detail::fma;
   using hydra_boost::math::cstdfloat::detail::fmax;
   using hydra_boost::math::cstdfloat::detail::fmin;
   using hydra_boost::math::cstdfloat::detail::fdim;
   using hydra_boost::math::cstdfloat::detail::nanq;
   using hydra_boost::math::cstdfloat::detail::exp2;
   using hydra_boost::math::cstdfloat::detail::log2;
   using hydra_boost::math::cstdfloat::detail::log1p;
   using hydra_boost::math::cstdfloat::detail::cbrt;
   using hydra_boost::math::cstdfloat::detail::hypot;
   using hydra_boost::math::cstdfloat::detail::erf;
   using hydra_boost::math::cstdfloat::detail::erfc;
   using hydra_boost::math::cstdfloat::detail::llround;
   using hydra_boost::math::cstdfloat::detail::lround;
   using hydra_boost::math::cstdfloat::detail::round;
   using hydra_boost::math::cstdfloat::detail::nearbyint;
   using hydra_boost::math::cstdfloat::detail::llrint;
   using hydra_boost::math::cstdfloat::detail::lrint;
   using hydra_boost::math::cstdfloat::detail::rint;
   using hydra_boost::math::cstdfloat::detail::modf;
   using hydra_boost::math::cstdfloat::detail::scalbln;
   using hydra_boost::math::cstdfloat::detail::scalbn;
   using hydra_boost::math::cstdfloat::detail::ilogb;
   using hydra_boost::math::cstdfloat::detail::logb;
   using hydra_boost::math::cstdfloat::detail::nextafter;
   using hydra_boost::math::cstdfloat::detail::nexttoward;
   using hydra_boost::math::cstdfloat::detail::copysign;
   using hydra_boost::math::cstdfloat::detail::signbit;
   using hydra_boost::math::cstdfloat::detail::fpclassify;
   using hydra_boost::math::cstdfloat::detail::isfinite;
   using hydra_boost::math::cstdfloat::detail::isinf;
   using hydra_boost::math::cstdfloat::detail::isnan;
   using hydra_boost::math::cstdfloat::detail::isnormal;
   using hydra_boost::math::cstdfloat::detail::isgreater;
   using hydra_boost::math::cstdfloat::detail::isgreaterequal;
   using hydra_boost::math::cstdfloat::detail::isless;
   using hydra_boost::math::cstdfloat::detail::islessequal;
   using hydra_boost::math::cstdfloat::detail::islessgreater;
   using hydra_boost::math::cstdfloat::detail::isunordered;
   //   end more functions

   //
   // Very basic iostream operator:
   //
   inline std::ostream& operator << (std::ostream& os, __float128 m_value)
   {
      std::streamsize digits = os.precision();
      std::ios_base::fmtflags f = os.flags();
      std::string s;

      char buf[100];
      std::unique_ptr<char[]> buf2;
      std::string format = "%";
      if (f & std::ios_base::showpos)
         format += "+";
      if (f & std::ios_base::showpoint)
         format += "#";
      format += ".*";
      if (digits == 0)
         digits = 36;
      format += "Q";
      if (f & std::ios_base::scientific)
         format += "e";
      else if (f & std::ios_base::fixed)
         format += "f";
      else
         format += "g";

      int v = quadmath_snprintf(buf, 100, format.c_str(), digits, m_value);

      if ((v < 0) || (v >= 99))
      {
         int v_max = v;
         buf2.reset(new char[v + 3]);
         v = quadmath_snprintf(&buf2[0], v_max + 3, format.c_str(), digits, m_value);
         if (v >= v_max + 3)
         {
            HYDRA_BOOST_MATH_THROW_EXCEPTION(std::runtime_error("Formatting of float128_type failed."));
         }
         s = &buf2[0];
      }
      else
         s = buf;
      std::streamsize ss = os.width();
      if (ss > static_cast<std::streamsize>(s.size()))
      {
         char fill = os.fill();
         if ((os.flags() & std::ios_base::left) == std::ios_base::left)
            s.append(static_cast<std::string::size_type>(ss - s.size()), fill);
         else
            s.insert(static_cast<std::string::size_type>(0), static_cast<std::string::size_type>(ss - s.size()), fill);
      }

      return os << s;
   }


} // namespace std

// We will now remove the preprocessor symbols representing quadruple-precision <cmath>
// functions from the preprocessor.

#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_LDEXP
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_FREXP
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_FABS
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_FLOOR
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_CEIL
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_SQRT
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_TRUNC
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_EXP
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_EXPM1
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_POW
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_LOG
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_LOG10
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_SIN
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_COS
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_TAN
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_ASIN
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_ACOS
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_ATAN
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_SINH
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_COSH
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_TANH
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_ASINH
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_ACOSH
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_ATANH
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_FMOD
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_ATAN2
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_LGAMMA
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_TGAMMA

//   begin more functions
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_REMAINDER
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_REMQUO
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_FMA
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_FMAX
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_FMIN
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_FDIM
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_NAN
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_EXP2
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_LOG2
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_LOG1P
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_CBRT
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_HYPOT
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_ERF
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_ERFC
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_LLROUND
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_LROUND
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_ROUND
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_NEARBYINT
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_LLRINT
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_LRINT
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_RINT
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_MODF
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_SCALBLN
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_SCALBN
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_ILOGB
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_LOGB
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_NEXTAFTER
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_NEXTTOWARD
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_COPYSIGN
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_SIGNBIT
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_FPCLASSIFY
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISFINITE
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISINF
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISNAN
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISNORMAL
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISGREATER
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISGREATEREQUAL
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISLESS
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISLESSEQUAL
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISLESSGREATER
#undef HYDRA_BOOST_CSTDFLOAT_FLOAT128_ISUNORDERED
//   end more functions

#endif // Not HYDRA_BOOST_CSTDFLOAT_NO_LIBQUADMATH_SUPPORT (i.e., the user would like to have libquadmath support)

#endif // HYDRA_BOOST_MATH_CSTDFLOAT_CMATH_2014_02_15_HPP_

