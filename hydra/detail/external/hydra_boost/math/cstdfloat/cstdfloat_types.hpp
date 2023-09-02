///////////////////////////////////////////////////////////////////////////////
// Copyright Christopher Kormanyos 2014.
// Copyright John Maddock 2014.
// Copyright Paul Bristow 2014.
// Distributed under the Boost Software License,
// Version 1.0. (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Implement the types for floating-point typedefs having specified widths.

#ifndef HYDRA_BOOST_MATH_CSTDFLOAT_TYPES_2014_01_09_HPP_
  #define HYDRA_BOOST_MATH_CSTDFLOAT_TYPES_2014_01_09_HPP_

  #include <cfloat>
  #include <limits>
  #include <hydra/detail/external/hydra_boost/math/tools/config.hpp>

  // This is the beginning of the preamble.

  // In this preamble, the preprocessor is used to query certain
  // preprocessor definitions from <cfloat>. Based on the results
  // of these queries, an attempt is made to automatically detect
  // the presence of built-in floating-point types having specified
  // widths. These are *thought* to be conformant with IEEE-754,
  // whereby an unequivocal test based on std::numeric_limits<>
  // follows below.

  // In addition, various macros that are used for initializing
  // floating-point literal values having specified widths and
  // some basic min/max values are defined.

  // First, we will pre-load certain preprocessor definitions
  // with a dummy value.

  #define HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH  0

  #define HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT16_NATIVE_TYPE  0
  #define HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT32_NATIVE_TYPE  0
  #define HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT64_NATIVE_TYPE  0
  #define HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT80_NATIVE_TYPE  0
  #define HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT128_NATIVE_TYPE 0

  // Ensure that the compiler has a radix-2 floating-point representation.
  #if (!defined(FLT_RADIX) || ((defined(FLT_RADIX) && (FLT_RADIX != 2))))
    #error The compiler does not support any radix-2 floating-point types required for <hydra/detail/external/hydra_boost/cstdfloat.hpp>.
  #endif

  // Check if built-in float is equivalent to float16_t, float32_t, float64_t, float80_t, or float128_t.
  #if(defined(FLT_MANT_DIG) && defined(FLT_MAX_EXP))
    #if  ((FLT_MANT_DIG == 11) && (FLT_MAX_EXP == 16) && (HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT16_NATIVE_TYPE == 0))
      #define HYDRA_BOOST_CSTDFLOAT_FLOAT16_NATIVE_TYPE float
      #undef  HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH
      #define HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH 16
      #undef  HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT16_NATIVE_TYPE
      #define HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT16_NATIVE_TYPE  1
      #define HYDRA_BOOST_FLOAT16_C(x)  (x ## F)
      #define HYDRA_BOOST_CSTDFLOAT_FLOAT_16_MIN  FLT_MIN
      #define HYDRA_BOOST_CSTDFLOAT_FLOAT_16_MAX  FLT_MAX
    #elif((FLT_MANT_DIG == 24) && (FLT_MAX_EXP == 128) && (HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT32_NATIVE_TYPE == 0))
      #define HYDRA_BOOST_CSTDFLOAT_FLOAT32_NATIVE_TYPE float
      #undef  HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH
      #define HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH 32
      #undef  HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT32_NATIVE_TYPE
      #define HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT32_NATIVE_TYPE  1
      #define HYDRA_BOOST_FLOAT32_C(x)  (x ## F)
      #define HYDRA_BOOST_CSTDFLOAT_FLOAT_32_MIN  FLT_MIN
      #define HYDRA_BOOST_CSTDFLOAT_FLOAT_32_MAX  FLT_MAX
    #elif((FLT_MANT_DIG == 53) && (FLT_MAX_EXP == 1024) && (HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT64_NATIVE_TYPE == 0))
      #define HYDRA_BOOST_CSTDFLOAT_FLOAT64_NATIVE_TYPE float
      #undef  HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH
      #define HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH 64
      #undef  HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT64_NATIVE_TYPE
      #define HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT64_NATIVE_TYPE  1
      #define HYDRA_BOOST_FLOAT64_C(x)  (x ## F)
      #define HYDRA_BOOST_CSTDFLOAT_FLOAT_64_MIN  FLT_MIN
      #define HYDRA_BOOST_CSTDFLOAT_FLOAT_64_MAX  FLT_MAX
    #elif((FLT_MANT_DIG == 64) && (FLT_MAX_EXP == 16384) && (HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT80_NATIVE_TYPE == 0))
      #define HYDRA_BOOST_CSTDFLOAT_FLOAT80_NATIVE_TYPE float
      #undef  HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH
      #define HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH 80
      #undef  HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT80_NATIVE_TYPE
      #define HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT80_NATIVE_TYPE  1
      #define HYDRA_BOOST_FLOAT80_C(x)  (x ## F)
      #define HYDRA_BOOST_CSTDFLOAT_FLOAT_80_MIN  FLT_MIN
      #define HYDRA_BOOST_CSTDFLOAT_FLOAT_80_MAX  FLT_MAX
    #elif((FLT_MANT_DIG == 113) && (FLT_MAX_EXP == 16384) && (HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT128_NATIVE_TYPE == 0))
      #define HYDRA_BOOST_CSTDFLOAT_FLOAT128_NATIVE_TYPE float
      #undef  HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH
      #define HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH 128
      #undef  HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT128_NATIVE_TYPE
      #define HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT128_NATIVE_TYPE  1
      #define HYDRA_BOOST_FLOAT128_C(x)  (x ## F)
      #define HYDRA_BOOST_CSTDFLOAT_FLOAT_128_MIN  FLT_MIN
      #define HYDRA_BOOST_CSTDFLOAT_FLOAT_128_MAX  FLT_MAX
    #endif
  #endif

  // Check if built-in double is equivalent to float16_t, float32_t, float64_t, float80_t, or float128_t.
  #if(defined(DBL_MANT_DIG) && defined(DBL_MAX_EXP))
    #if  ((DBL_MANT_DIG == 11) && (DBL_MAX_EXP == 16) && (HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT16_NATIVE_TYPE == 0))
      #define HYDRA_BOOST_CSTDFLOAT_FLOAT16_NATIVE_TYPE double
      #undef  HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH
      #define HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH 16
      #undef  HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT16_NATIVE_TYPE
      #define HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT16_NATIVE_TYPE  1
      #define HYDRA_BOOST_FLOAT16_C(x)  (x)
      #define HYDRA_BOOST_CSTDFLOAT_FLOAT_16_MIN  DBL_MIN
      #define HYDRA_BOOST_CSTDFLOAT_FLOAT_16_MAX  DBL_MAX
    #elif((DBL_MANT_DIG == 24) && (DBL_MAX_EXP == 128) && (HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT32_NATIVE_TYPE == 0))
      #define HYDRA_BOOST_CSTDFLOAT_FLOAT32_NATIVE_TYPE double
      #undef  HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH
      #define HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH 32
      #undef  HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT32_NATIVE_TYPE
      #define HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT32_NATIVE_TYPE  1
      #define HYDRA_BOOST_FLOAT32_C(x)  (x)
      #define HYDRA_BOOST_CSTDFLOAT_FLOAT_32_MIN  DBL_MIN
      #define HYDRA_BOOST_CSTDFLOAT_FLOAT_32_MAX  DBL_MAX
    #elif((DBL_MANT_DIG == 53) && (DBL_MAX_EXP == 1024) && (HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT64_NATIVE_TYPE == 0))
      #define HYDRA_BOOST_CSTDFLOAT_FLOAT64_NATIVE_TYPE double
      #undef  HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH
      #define HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH 64
      #undef  HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT64_NATIVE_TYPE
      #define HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT64_NATIVE_TYPE  1
      #define HYDRA_BOOST_FLOAT64_C(x)  (x)
      #define HYDRA_BOOST_CSTDFLOAT_FLOAT_64_MIN  DBL_MIN
      #define HYDRA_BOOST_CSTDFLOAT_FLOAT_64_MAX  DBL_MAX
    #elif((DBL_MANT_DIG == 64) && (DBL_MAX_EXP == 16384) && (HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT80_NATIVE_TYPE == 0))
      #define HYDRA_BOOST_CSTDFLOAT_FLOAT80_NATIVE_TYPE double
      #undef  HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH
      #define HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH 80
      #undef  HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT80_NATIVE_TYPE
      #define HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT80_NATIVE_TYPE  1
      #define HYDRA_BOOST_FLOAT80_C(x)  (x)
      #define HYDRA_BOOST_CSTDFLOAT_FLOAT_80_MIN  DBL_MIN
      #define HYDRA_BOOST_CSTDFLOAT_FLOAT_80_MAX  DBL_MAX
    #elif((DBL_MANT_DIG == 113) && (DBL_MAX_EXP == 16384) && (HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT128_NATIVE_TYPE == 0))
      #define HYDRA_BOOST_CSTDFLOAT_FLOAT128_NATIVE_TYPE double
      #undef  HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH
      #define HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH 128
      #undef  HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT128_NATIVE_TYPE
      #define HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT128_NATIVE_TYPE  1
      #define HYDRA_BOOST_FLOAT128_C(x)  (x)
      #define HYDRA_BOOST_CSTDFLOAT_FLOAT_128_MIN  DBL_MIN
      #define HYDRA_BOOST_CSTDFLOAT_FLOAT_128_MAX  DBL_MAX
    #endif
  #endif

  // Disable check long double capability even if supported by compiler since some math runtime
  // implementations are broken for long double.
  #ifndef HYDRA_BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    // Check if built-in long double is equivalent to float16_t, float32_t, float64_t, float80_t, or float128_t.
    #if(defined(LDBL_MANT_DIG) && defined(LDBL_MAX_EXP))
      #if  ((LDBL_MANT_DIG == 11) && (LDBL_MAX_EXP == 16) && (HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT16_NATIVE_TYPE == 0))
        #define HYDRA_BOOST_CSTDFLOAT_FLOAT16_NATIVE_TYPE long double
        #undef  HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH
        #define HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH 16
        #undef  HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT16_NATIVE_TYPE
        #define HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT16_NATIVE_TYPE  1
        #define HYDRA_BOOST_FLOAT16_C(x)  (x ## L)
        #define HYDRA_BOOST_CSTDFLOAT_FLOAT_16_MIN  LDBL_MIN
        #define HYDRA_BOOST_CSTDFLOAT_FLOAT_16_MAX  LDBL_MAX
      #elif((LDBL_MANT_DIG == 24) && (LDBL_MAX_EXP == 128) && (HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT32_NATIVE_TYPE == 0))
        #define HYDRA_BOOST_CSTDFLOAT_FLOAT32_NATIVE_TYPE long double
        #undef  HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH
        #define HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH 32
        #undef  HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT32_NATIVE_TYPE
        #define HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT32_NATIVE_TYPE  1
        #define HYDRA_BOOST_FLOAT32_C(x)  (x ## L)
        #define HYDRA_BOOST_CSTDFLOAT_FLOAT_32_MIN  LDBL_MIN
        #define HYDRA_BOOST_CSTDFLOAT_FLOAT_32_MAX  LDBL_MAX
      #elif((LDBL_MANT_DIG == 53) && (LDBL_MAX_EXP == 1024) && (HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT64_NATIVE_TYPE == 0))
        #define HYDRA_BOOST_CSTDFLOAT_FLOAT64_NATIVE_TYPE long double
        #undef  HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH
        #define HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH 64
        #undef  HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT64_NATIVE_TYPE
        #define HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT64_NATIVE_TYPE  1
        #define HYDRA_BOOST_FLOAT64_C(x)  (x ## L)
        #define HYDRA_BOOST_CSTDFLOAT_FLOAT_64_MIN  LDBL_MIN
        #define HYDRA_BOOST_CSTDFLOAT_FLOAT_64_MAX  LDBL_MAX
      #elif((LDBL_MANT_DIG == 64) && (LDBL_MAX_EXP == 16384) && (HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT80_NATIVE_TYPE == 0))
        #define HYDRA_BOOST_CSTDFLOAT_FLOAT80_NATIVE_TYPE long double
        #undef  HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH
        #define HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH 80
        #undef  HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT80_NATIVE_TYPE
        #define HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT80_NATIVE_TYPE  1
        #define HYDRA_BOOST_FLOAT80_C(x)  (x ## L)
        #define HYDRA_BOOST_CSTDFLOAT_FLOAT_80_MIN  LDBL_MIN
        #define HYDRA_BOOST_CSTDFLOAT_FLOAT_80_MAX  LDBL_MAX
      #elif((LDBL_MANT_DIG == 113) && (LDBL_MAX_EXP == 16384) && (HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT128_NATIVE_TYPE == 0))
        #define HYDRA_BOOST_CSTDFLOAT_FLOAT128_NATIVE_TYPE long double
        #undef  HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH
        #define HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH 128
        #undef  HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT128_NATIVE_TYPE
        #define HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT128_NATIVE_TYPE  1
        #define HYDRA_BOOST_FLOAT128_C(x)  (x ## L)
        #define HYDRA_BOOST_CSTDFLOAT_FLOAT_128_MIN  LDBL_MIN
        #define HYDRA_BOOST_CSTDFLOAT_FLOAT_128_MAX  LDBL_MAX
      #endif
    #endif
  #endif

  // Check if quadruple-precision is supported. Here, we are checking
  // for the presence of __float128 from GCC's quadmath.h or _Quad
  // from ICC's /Qlong-double flag). To query these, we use the
  // HYDRA_BOOST_MATH_USE_FLOAT128 pre-processor definition from
  // <hydra/detail/external/hydra_boost/math/tools/config.hpp>.

  #if (HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT128_NATIVE_TYPE == 0) && defined(HYDRA_BOOST_MATH_USE_FLOAT128) && !defined(HYDRA_BOOST_CSTDFLOAT_NO_LIBQUADMATH_SUPPORT)

    // Specify the underlying name of the internal 128-bit floating-point type definition.
    namespace hydra_boost { namespace math { namespace cstdfloat { namespace detail {
    #if defined(__GNUC__)
      typedef __float128      float_internal128_t;
    #elif defined(__INTEL_COMPILER)
      typedef _Quad           float_internal128_t;
    #else
      #error "Sorry, the compiler is neither GCC, nor Intel, I don't know how to configure <hydra/detail/external/hydra_boost/cstdfloat.hpp>."
    #endif
    } } } } // hydra_boost::math::cstdfloat::detail

    #define HYDRA_BOOST_CSTDFLOAT_FLOAT128_NATIVE_TYPE hydra_boost::math::cstdfloat::detail::float_internal128_t
    #undef  HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH
    #define HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH 128
    #undef  HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT128_NATIVE_TYPE
    #define HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT128_NATIVE_TYPE  1
    #define HYDRA_BOOST_FLOAT128_C(x)  (x ## Q)
    #define HYDRA_BOOST_CSTDFLOAT_FLOAT128_MIN  3.36210314311209350626267781732175260e-4932Q
    #define HYDRA_BOOST_CSTDFLOAT_FLOAT128_MAX  1.18973149535723176508575932662800702e+4932Q
    #define HYDRA_BOOST_CSTDFLOAT_FLOAT128_EPS  1.92592994438723585305597794258492732e-0034Q
    #define HYDRA_BOOST_CSTDFLOAT_FLOAT128_DENORM_MIN  6.475175119438025110924438958227646552e-4966Q

  #endif // Not HYDRA_BOOST_CSTDFLOAT_NO_LIBQUADMATH_SUPPORT (i.e., the user would like to have libquadmath support)

  // This is the end of the preamble, and also the end of the
  // sections providing support for the C++ standard library
  // for quadruple-precision.

  // Now we use the results of the queries that have been obtained
  // in the preamble (far above) for the final type definitions in
  // the namespace hydra_boost.

  // Make sure that the compiler has any floating-point type(s) whatsoever.
  #if (   (HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT16_NATIVE_TYPE  == 0)  \
       && (HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT32_NATIVE_TYPE  == 0)  \
       && (HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT64_NATIVE_TYPE  == 0)  \
       && (HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT80_NATIVE_TYPE  == 0)  \
       && (HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT128_NATIVE_TYPE == 0))
    #error The compiler does not support any of the floating-point types required for <hydra/detail/external/hydra_boost/cstdfloat.hpp>.
  #endif

  // The following section contains the various min/max macros
  // for the *leastN and *fastN types.

  #if(HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT16_NATIVE_TYPE == 1)
    #define HYDRA_BOOST_FLOAT_FAST16_MIN   HYDRA_BOOST_CSTDFLOAT_FLOAT_16_MIN
    #define HYDRA_BOOST_FLOAT_LEAST16_MIN  HYDRA_BOOST_CSTDFLOAT_FLOAT_16_MIN
    #define HYDRA_BOOST_FLOAT_FAST16_MAX   HYDRA_BOOST_CSTDFLOAT_FLOAT_16_MAX
    #define HYDRA_BOOST_FLOAT_LEAST16_MAX  HYDRA_BOOST_CSTDFLOAT_FLOAT_16_MAX
  #endif

  #if(HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT32_NATIVE_TYPE == 1)
    #define HYDRA_BOOST_FLOAT_FAST32_MIN   HYDRA_BOOST_CSTDFLOAT_FLOAT_32_MIN
    #define HYDRA_BOOST_FLOAT_LEAST32_MIN  HYDRA_BOOST_CSTDFLOAT_FLOAT_32_MIN
    #define HYDRA_BOOST_FLOAT_FAST32_MAX   HYDRA_BOOST_CSTDFLOAT_FLOAT_32_MAX
    #define HYDRA_BOOST_FLOAT_LEAST32_MAX  HYDRA_BOOST_CSTDFLOAT_FLOAT_32_MAX
  #endif

  #if(HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT64_NATIVE_TYPE == 1)
    #define HYDRA_BOOST_FLOAT_FAST64_MIN   HYDRA_BOOST_CSTDFLOAT_FLOAT_64_MIN
    #define HYDRA_BOOST_FLOAT_LEAST64_MIN  HYDRA_BOOST_CSTDFLOAT_FLOAT_64_MIN
    #define HYDRA_BOOST_FLOAT_FAST64_MAX   HYDRA_BOOST_CSTDFLOAT_FLOAT_64_MAX
    #define HYDRA_BOOST_FLOAT_LEAST64_MAX  HYDRA_BOOST_CSTDFLOAT_FLOAT_64_MAX
  #endif

  #if(HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT80_NATIVE_TYPE == 1)
    #define HYDRA_BOOST_FLOAT_FAST80_MIN   HYDRA_BOOST_CSTDFLOAT_FLOAT_80_MIN
    #define HYDRA_BOOST_FLOAT_LEAST80_MIN  HYDRA_BOOST_CSTDFLOAT_FLOAT_80_MIN
    #define HYDRA_BOOST_FLOAT_FAST80_MAX   HYDRA_BOOST_CSTDFLOAT_FLOAT_80_MAX
    #define HYDRA_BOOST_FLOAT_LEAST80_MAX  HYDRA_BOOST_CSTDFLOAT_FLOAT_80_MAX
  #endif

  #if(HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT128_NATIVE_TYPE == 1)
    #define HYDRA_BOOST_CSTDFLOAT_HAS_INTERNAL_FLOAT128_T

    #define HYDRA_BOOST_FLOAT_FAST128_MIN   HYDRA_BOOST_CSTDFLOAT_FLOAT_128_MIN
    #define HYDRA_BOOST_FLOAT_LEAST128_MIN  HYDRA_BOOST_CSTDFLOAT_FLOAT_128_MIN
    #define HYDRA_BOOST_FLOAT_FAST128_MAX   HYDRA_BOOST_CSTDFLOAT_FLOAT_128_MAX
    #define HYDRA_BOOST_FLOAT_LEAST128_MAX  HYDRA_BOOST_CSTDFLOAT_FLOAT_128_MAX
  #endif

  // The following section contains the various min/max macros
  // for the *floatmax types.

  #if  (HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH == 16)
    #define HYDRA_BOOST_FLOATMAX_C(x) HYDRA_BOOST_FLOAT16_C(x)
    #define HYDRA_BOOST_FLOATMAX_MIN  HYDRA_BOOST_CSTDFLOAT_FLOAT_16_MIN
    #define HYDRA_BOOST_FLOATMAX_MAX  HYDRA_BOOST_CSTDFLOAT_FLOAT_16_MAX
  #elif(HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH == 32)
    #define HYDRA_BOOST_FLOATMAX_C(x) HYDRA_BOOST_FLOAT32_C(x)
    #define HYDRA_BOOST_FLOATMAX_MIN  HYDRA_BOOST_CSTDFLOAT_FLOAT_32_MIN
    #define HYDRA_BOOST_FLOATMAX_MAX  HYDRA_BOOST_CSTDFLOAT_FLOAT_32_MAX
  #elif(HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH == 64)
    #define HYDRA_BOOST_FLOATMAX_C(x) HYDRA_BOOST_FLOAT64_C(x)
    #define HYDRA_BOOST_FLOATMAX_MIN  HYDRA_BOOST_CSTDFLOAT_FLOAT_64_MIN
    #define HYDRA_BOOST_FLOATMAX_MAX  HYDRA_BOOST_CSTDFLOAT_FLOAT_64_MAX
  #elif(HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH == 80)
    #define HYDRA_BOOST_FLOATMAX_C(x) HYDRA_BOOST_FLOAT80_C(x)
    #define HYDRA_BOOST_FLOATMAX_MIN  HYDRA_BOOST_CSTDFLOAT_FLOAT_80_MIN
    #define HYDRA_BOOST_FLOATMAX_MAX  HYDRA_BOOST_CSTDFLOAT_FLOAT_80_MAX
  #elif(HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH == 128)
    #define HYDRA_BOOST_FLOATMAX_C(x) HYDRA_BOOST_FLOAT128_C(x)
    #define HYDRA_BOOST_FLOATMAX_MIN  HYDRA_BOOST_CSTDFLOAT_FLOAT_128_MIN
    #define HYDRA_BOOST_FLOATMAX_MAX  HYDRA_BOOST_CSTDFLOAT_FLOAT_128_MAX
  #else
    #error The maximum available floating-point width for <hydra/detail/external/hydra_boost/cstdfloat.hpp> is undefined.
  #endif

  // And finally..., we define the floating-point typedefs having
  // specified widths. The types are defined in the namespace hydra_boost.

  // For simplicity, the least and fast types are type defined identically
  // as the corresponding fixed-width type. This behavior may, however,
  // be modified when being optimized for a given compiler implementation.

  // In addition, a clear assessment of IEEE-754 conformance is carried out
  // using compile-time assertion.

  namespace hydra_boost
  {
    #if(HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT16_NATIVE_TYPE == 1)
      typedef HYDRA_BOOST_CSTDFLOAT_FLOAT16_NATIVE_TYPE float16_t;
      typedef hydra_boost::float16_t float_fast16_t;
      typedef hydra_boost::float16_t float_least16_t;

      static_assert(std::numeric_limits<hydra_boost::float16_t>::is_iec559    == true, "hydra_boost::float16_t has been detected in <hydra/detail/external/hydra_boost/cstdfloat>, but verification with std::numeric_limits fails");
      static_assert(std::numeric_limits<hydra_boost::float16_t>::radix        ==    2, "hydra_boost::float16_t has been detected in <hydra/detail/external/hydra_boost/cstdfloat>, but verification with std::numeric_limits fails");
      static_assert(std::numeric_limits<hydra_boost::float16_t>::digits       ==   11, "hydra_boost::float16_t has been detected in <hydra/detail/external/hydra_boost/cstdfloat>, but verification with std::numeric_limits fails");
      static_assert(std::numeric_limits<hydra_boost::float16_t>::max_exponent ==   16, "hydra_boost::float16_t has been detected in <hydra/detail/external/hydra_boost/cstdfloat>, but verification with std::numeric_limits fails");

      #undef HYDRA_BOOST_CSTDFLOAT_FLOAT_16_MIN
      #undef HYDRA_BOOST_CSTDFLOAT_FLOAT_16_MAX
    #endif

    #if(HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT32_NATIVE_TYPE == 1)
      typedef HYDRA_BOOST_CSTDFLOAT_FLOAT32_NATIVE_TYPE float32_t;
      typedef hydra_boost::float32_t float_fast32_t;
      typedef hydra_boost::float32_t float_least32_t;

      static_assert(std::numeric_limits<hydra_boost::float32_t>::is_iec559    == true, "hydra_boost::float32_t has been detected in <hydra/detail/external/hydra_boost/cstdfloat>, but verification with std::numeric_limits fails");
      static_assert(std::numeric_limits<hydra_boost::float32_t>::radix        ==    2, "hydra_boost::float32_t has been detected in <hydra/detail/external/hydra_boost/cstdfloat>, but verification with std::numeric_limits fails");
      static_assert(std::numeric_limits<hydra_boost::float32_t>::digits       ==   24, "hydra_boost::float32_t has been detected in <hydra/detail/external/hydra_boost/cstdfloat>, but verification with std::numeric_limits fails");
      static_assert(std::numeric_limits<hydra_boost::float32_t>::max_exponent ==  128, "hydra_boost::float32_t has been detected in <hydra/detail/external/hydra_boost/cstdfloat>, but verification with std::numeric_limits fails");

      #undef HYDRA_BOOST_CSTDFLOAT_FLOAT_32_MIN
      #undef HYDRA_BOOST_CSTDFLOAT_FLOAT_32_MAX
    #endif

#if (defined(__SGI_STL_PORT) || defined(_STLPORT_VERSION)) && defined(__SUNPRO_CC)
#undef HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT80_NATIVE_TYPE
#define HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT80_NATIVE_TYPE 0
#undef HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT128_NATIVE_TYPE
#define HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT128_NATIVE_TYPE 0
#undef HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH
#define HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH 64
#endif

    #if(HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT64_NATIVE_TYPE == 1)
      typedef HYDRA_BOOST_CSTDFLOAT_FLOAT64_NATIVE_TYPE float64_t;
      typedef hydra_boost::float64_t float_fast64_t;
      typedef hydra_boost::float64_t float_least64_t;

      static_assert(std::numeric_limits<hydra_boost::float64_t>::is_iec559    == true, "hydra_boost::float64_t has been detected in <hydra/detail/external/hydra_boost/cstdfloat>, but verification with std::numeric_limits fails");
      static_assert(std::numeric_limits<hydra_boost::float64_t>::radix        ==    2, "hydra_boost::float64_t has been detected in <hydra/detail/external/hydra_boost/cstdfloat>, but verification with std::numeric_limits fails");
      static_assert(std::numeric_limits<hydra_boost::float64_t>::digits       ==   53, "hydra_boost::float64_t has been detected in <hydra/detail/external/hydra_boost/cstdfloat>, but verification with std::numeric_limits fails");
      static_assert(std::numeric_limits<hydra_boost::float64_t>::max_exponent == 1024, "hydra_boost::float64_t has been detected in <hydra/detail/external/hydra_boost/cstdfloat>, but verification with std::numeric_limits fails");

      #undef HYDRA_BOOST_CSTDFLOAT_FLOAT_64_MIN
      #undef HYDRA_BOOST_CSTDFLOAT_FLOAT_64_MAX
    #endif

    #if(HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT80_NATIVE_TYPE == 1)
      typedef HYDRA_BOOST_CSTDFLOAT_FLOAT80_NATIVE_TYPE float80_t;
      typedef hydra_boost::float80_t float_fast80_t;
      typedef hydra_boost::float80_t float_least80_t;

      static_assert(std::numeric_limits<hydra_boost::float80_t>::is_iec559    ==  true, "hydra_boost::float80_t has been detected in <hydra/detail/external/hydra_boost/cstdfloat>, but verification with std::numeric_limits fails");
      static_assert(std::numeric_limits<hydra_boost::float80_t>::radix        ==     2, "hydra_boost::float80_t has been detected in <hydra/detail/external/hydra_boost/cstdfloat>, but verification with std::numeric_limits fails");
      static_assert(std::numeric_limits<hydra_boost::float80_t>::digits       ==    64, "hydra_boost::float80_t has been detected in <hydra/detail/external/hydra_boost/cstdfloat>, but verification with std::numeric_limits fails");
      static_assert(std::numeric_limits<hydra_boost::float80_t>::max_exponent == 16384, "hydra_boost::float80_t has been detected in <hydra/detail/external/hydra_boost/cstdfloat>, but verification with std::numeric_limits fails");

      #undef HYDRA_BOOST_CSTDFLOAT_FLOAT_80_MIN
      #undef HYDRA_BOOST_CSTDFLOAT_FLOAT_80_MAX
    #endif

    #if(HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT128_NATIVE_TYPE == 1)
      typedef HYDRA_BOOST_CSTDFLOAT_FLOAT128_NATIVE_TYPE float128_t;
      typedef hydra_boost::float128_t float_fast128_t;
      typedef hydra_boost::float128_t float_least128_t;

      #if defined(HYDRA_BOOST_CSTDFLOAT_HAS_INTERNAL_FLOAT128_T) && defined(HYDRA_BOOST_MATH_USE_FLOAT128) && !defined(HYDRA_BOOST_CSTDFLOAT_NO_LIBQUADMATH_SUPPORT)
      // This configuration does not *yet* support std::numeric_limits<hydra_boost::float128_t>.
      // Support for std::numeric_limits<hydra_boost::float128_t> is added in the detail
      // file <hydra/detail/external/hydra_boost/math/cstdfloat/cstdfloat_limits.hpp>.
      #else
      static_assert(std::numeric_limits<hydra_boost::float128_t>::is_iec559    ==  true, "hydra_boost::float128_t has been detected in <hydra/detail/external/hydra_boost/cstdfloat>, but verification with std::numeric_limits fails");
      static_assert(std::numeric_limits<hydra_boost::float128_t>::radix        ==     2, "hydra_boost::float128_t has been detected in <hydra/detail/external/hydra_boost/cstdfloat>, but verification with std::numeric_limits fails");
      static_assert(std::numeric_limits<hydra_boost::float128_t>::digits       ==   113, "hydra_boost::float128_t has been detected in <hydra/detail/external/hydra_boost/cstdfloat>, but verification with std::numeric_limits fails");
      static_assert(std::numeric_limits<hydra_boost::float128_t>::max_exponent == 16384, "hydra_boost::float128_t has been detected in <hydra/detail/external/hydra_boost/cstdfloat>, but verification with std::numeric_limits fails");
      #endif

      #undef HYDRA_BOOST_CSTDFLOAT_FLOAT_128_MIN
      #undef HYDRA_BOOST_CSTDFLOAT_FLOAT_128_MAX
    #endif

    #if  (HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH ==  16)
      typedef hydra_boost::float16_t  floatmax_t;
    #elif(HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH ==  32)
      typedef hydra_boost::float32_t  floatmax_t;
    #elif(HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH ==  64)
      typedef hydra_boost::float64_t  floatmax_t;
    #elif(HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH ==  80)
      typedef hydra_boost::float80_t  floatmax_t;
    #elif(HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH == 128)
      typedef hydra_boost::float128_t floatmax_t;
    #else
      #error The maximum available floating-point width for <hydra/detail/external/hydra_boost/cstdfloat.hpp> is undefined.
    #endif

    #undef HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT16_NATIVE_TYPE
    #undef HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT32_NATIVE_TYPE
    #undef HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT64_NATIVE_TYPE
    #undef HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT80_NATIVE_TYPE
    #undef HYDRA_BOOST_CSTDFLOAT_HAS_FLOAT128_NATIVE_TYPE

    #undef HYDRA_BOOST_CSTDFLOAT_MAXIMUM_AVAILABLE_WIDTH
  }
  // namespace hydra_boost

#endif // HYDRA_BOOST_MATH_CSTDFLOAT_BASE_TYPES_2014_01_09_HPP_

