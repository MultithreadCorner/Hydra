///////////////////////////////////////////////////////////////////////////////
//  Copyright 2014 Anton Bikineev
//  Copyright 2014 Christopher Kormanyos
//  Copyright 2014 John Maddock
//  Copyright 2014 Paul Bristow
//  Distributed under the Boost
//  Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
#ifndef HYDRA_BOOST_MATH_HYPERGEOMETRIC_0F1_BESSEL_HPP
#define HYDRA_BOOST_MATH_HYPERGEOMETRIC_0F1_BESSEL_HPP

#include <hydra/detail/external/hydra_boost/math/special_functions/bessel.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/gamma.hpp>

  namespace hydra_boost { namespace math { namespace detail {

  template <class T, class Policy>
  inline T hypergeometric_0F1_bessel(const T& b, const T& z, const Policy& pol)
  {
    HYDRA_BOOST_MATH_STD_USING

    const bool is_z_nonpositive = z <= 0;

    const T sqrt_z = is_z_nonpositive ? T(sqrt(-z)) : T(sqrt(z));
    const T bessel_mult = is_z_nonpositive ?
      hydra_boost::math::cyl_bessel_j(b - 1, 2 * sqrt_z, pol) :
      hydra_boost::math::cyl_bessel_i(b - 1, 2 * sqrt_z, pol) ;

    if (b > hydra_boost::math::max_factorial<T>::value)
    {
       const T lsqrt_z = log(sqrt_z);
       const T lsqrt_z_pow_b = (b - 1) * lsqrt_z;
       T lg = (hydra_boost::math::lgamma(b, pol) - lsqrt_z_pow_b);
       lg = exp(lg);
       return lg * bessel_mult;
    }
    else
    {
       const T sqrt_z_pow_b = pow(sqrt_z, b - 1);
       return (hydra_boost::math::tgamma(b, pol) / sqrt_z_pow_b) * bessel_mult;
    }
  }

  } } } // namespaces

#endif // HYDRA_BOOST_MATH_HYPERGEOMETRIC_0F1_BESSEL_HPP
