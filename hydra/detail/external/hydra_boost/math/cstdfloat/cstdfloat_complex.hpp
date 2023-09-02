///////////////////////////////////////////////////////////////////////////////
// Copyright Christopher Kormanyos 2014.
// Copyright John Maddock 2014.
// Copyright Paul Bristow 2014.
// Distributed under the Boost Software License,
// Version 1.0. (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Implement quadruple-precision (and extended) support for <complex>.

#ifndef HYDRA_BOOST_MATH_CSTDFLOAT_COMPLEX_2014_02_15_HPP_
  #define HYDRA_BOOST_MATH_CSTDFLOAT_COMPLEX_2014_02_15_HPP_

  #include <hydra/detail/external/hydra_boost/math/cstdfloat/cstdfloat_types.hpp>
  #include <hydra/detail/external/hydra_boost/math/cstdfloat/cstdfloat_limits.hpp>
  #include <hydra/detail/external/hydra_boost/math/cstdfloat/cstdfloat_cmath.hpp>
  #include <hydra/detail/external/hydra_boost/math/cstdfloat/cstdfloat_iostream.hpp>

  #if defined(HYDRA_BOOST_CSTDFLOAT_NO_LIBQUADMATH_LIMITS)
  #error You can not use <hydra/detail/external/hydra_boost/math/cstdfloat/cstdfloat_complex.hpp> with HYDRA_BOOST_CSTDFLOAT_NO_LIBQUADMATH_LIMITS defined.
  #endif
  #if defined(HYDRA_BOOST_CSTDFLOAT_NO_LIBQUADMATH_CMATH)
  #error You can not use <hydra/detail/external/hydra_boost/math/cstdfloat/cstdfloat_complex.hpp> with HYDRA_BOOST_CSTDFLOAT_NO_LIBQUADMATH_CMATH defined.
  #endif
  #if defined(HYDRA_BOOST_CSTDFLOAT_NO_LIBQUADMATH_IOSTREAM)
  #error You can not use <hydra/detail/external/hydra_boost/math/cstdfloat/cstdfloat_complex.hpp> with HYDRA_BOOST_CSTDFLOAT_NO_LIBQUADMATH_IOSTREAM defined.
  #endif

  #if defined(HYDRA_BOOST_CSTDFLOAT_HAS_INTERNAL_FLOAT128_T) && defined(HYDRA_BOOST_MATH_USE_FLOAT128) && !defined(HYDRA_BOOST_CSTDFLOAT_NO_LIBQUADMATH_SUPPORT)

  #define HYDRA_BOOST_CSTDFLOAT_EXTENDED_COMPLEX_FLOAT_TYPE hydra_boost::math::cstdfloat::detail::float_internal128_t
  #include <hydra/detail/external/hydra_boost/math/cstdfloat/cstdfloat_complex_std.hpp>
  #undef HYDRA_BOOST_CSTDFLOAT_EXTENDED_COMPLEX_FLOAT_TYPE

  #endif // Not HYDRA_BOOST_CSTDFLOAT_NO_LIBQUADMATH_SUPPORT (i.e., the user would like to have libquadmath support)

#endif // HYDRA_BOOST_MATH_CSTDFLOAT_COMPLEX_2014_02_15_HPP_
